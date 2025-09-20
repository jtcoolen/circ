// midnight_backend.rs
//! Lowering IR to Midnight-ZK using the ZkStdLib.

use crate::cfg::CircCfg;
use crate::ir::term::*;
use crate::target::plonkish::VarType;
use rug::Integer;
use num_bigint::BigUint;
use num_traits::Num;
use std::convert::TryInto;
use im::HashSet;
use log::{debug, trace};
use std::cell::RefCell;
use std::collections::HashMap;
use std::fmt::Display;
use std::rc::Rc;

use midnight_curves::Fq as F;
use midnight_circuits::{
    compact_std_lib as m,
    instructions::{
        ArithInstructions, AssertionInstructions, AssignmentInstructions, BinaryInstructions,
        CanonicityInstructions, ControlFlowInstructions, ConversionInstructions,
        DecompositionInstructions, EqualityInstructions, PublicInputInstructions,
        RangeCheckInstructions, ZeroInstructions,
    },
    types::{AssignedBit, AssignedNative, Instantiable},
};
 use midnight_circuits::halo2curves::ff::Field;
use midnight_proofs::{
    circuit::{Layouter, Value}, halo2curves::ff::PrimeField, plonk::Error
};

// -------------------------------
// Assigned terms (Midnight side)
// -------------------------------

#[derive(Clone)]
enum AssignedTerm {
    Field(AssignedNative<F>),
    Bit(AssignedBit<F>),
    Bv {
        width: usize,
        bits: Vec<AssignedBit<F>>,            // LSB-first
        uint: Option<AssignedNative<F>>,      // cached recomposition
    },
    Tuple(Vec<AssignedTerm>),
}

impl AssignedTerm {
    fn as_field(&self) -> &AssignedNative<F> {
        match self {
            AssignedTerm::Field(x) => x,
            _ => panic!("Expected field"),
        }
    }
    fn as_bit(&self) -> &AssignedBit<F> {
        match self {
            AssignedTerm::Bit(b) => b,
            _ => panic!("Expected bit"),
        }
    }
    fn as_bits(&self) -> (&[AssignedBit<F>], usize) {
        match self {
            AssignedTerm::Bv { width, bits, .. } => (&bits[..], *width),
            _ => panic!("Expected bitvector"),
        }
    }
}

// -------------------------------------------------
// Midnight lowering context (replaces ToPlonk core)
// -------------------------------------------------

struct ToMidnight<'a, 'b, L: Layouter<F>> {
    std: &'a m::ZkStdLib,
    lay: &'b mut L,
    cache: TermMap<AssignedTerm>,
    visited: Rc<RefCell<TermSet>>,
    cfg: &'a CircCfg,
    used_vars: HashSet<String>,

    // materialized assignment for variables (provided by Relation::Witness)
    wmap: &'a HashMap<String, F>,        // witness (private)
    imap: &'a HashMap<String, F>,        // instance/public
    pub_order: &'a [String],             // ordered public names
}

impl<'a, 'b, L: Layouter<F>> ToMidnight<'a, 'b, L> {
    fn new(
        std: &'a m::ZkStdLib,
        lay: &'b mut L,
        cfg: &'a CircCfg,
        used_vars: HashSet<String>,
        wmap: &'a HashMap<String, F>,
        imap: &'a HashMap<String, F>,
        pub_order: &'a [String],
    ) -> Self {
        Self {
            std,
            lay,
            cache: TermMap::default(),
            visited: Default::default(),
            cfg,
            used_vars,
            wmap,
            imap,
            pub_order,
        }
    }

    // -------------------------
    // Tiny helpers
    // -------------------------

    fn f_zero(&self) -> F { F::from(0) }
    fn f_one(&self) -> F { F::from(1) }
    fn f_from_u64(&self, x: u64) -> F { F::from(x) }

    fn as_value<T: Clone>(&self, t: T) -> Value<T> {
        Value::known(t)
    }

    // ----------------------------------------
    // Field arith mapping (add/sub/mul/const)
    // ----------------------------------------

    fn add(&mut self, a: &AssignedNative<F>, b: &AssignedNative<F>) -> Result<AssignedNative<F>, Error> {
        self.std.add(self.lay, a, b)
    }
    fn sub(&mut self, a: &AssignedNative<F>, b: &AssignedNative<F>) -> Result<AssignedNative<F>, Error> {
        self.std.sub(self.lay, a, b)
    }
    fn mul(&mut self, a: &AssignedNative<F>, b: &AssignedNative<F>) -> Result<AssignedNative<F>, Error> {
        self.std.mul(self.lay, a, b, None)
    }
    fn add_const(&mut self, a: &AssignedNative<F>, c: F) -> Result<AssignedNative<F>, Error> {
        self.std.add_constant(self.lay, a, c)
    }
    fn mul_const(&mut self, a: &AssignedNative<F>, c: F) -> Result<AssignedNative<F>, Error> {
        self.std.mul_by_constant(self.lay, a, c)
    }

    fn assert_zero(&mut self, a: &AssignedNative<F>) -> Result<(), Error> {
        self.std.assert_zero(self.lay, a)
    }

    fn assert_boolean(&mut self, b: &AssignedBit<F>) -> Result<(), Error> {
        // AssignedBit is boolean by construction; if sourced from field, convert & recheck.
        Ok(())
    }

    // -------------------------
    // Bits / Decomposition
    // -------------------------

    fn bitify(
        &mut self,
        x: &AssignedNative<F>,
        n: usize,
        enforce_canonical: bool,
    ) -> Result<Vec<AssignedBit<F>>, Error> {
        self.std.assigned_to_le_bits(self.lay, x, Some(n), enforce_canonical)
    }

    fn debitify(
        &mut self,
        bits: &[AssignedBit<F>],
        signed: bool,
    ) -> Result<AssignedNative<F>, Error> {
        // ∑ (2^i * b_i), with MSB negated if signed.
        let mut coeff = self.f_one();
        let mut terms: Vec<(F, AssignedNative<F>)> = Vec::with_capacity(bits.len());
        for (i, bit) in bits.iter().enumerate() {
            let limb: AssignedNative<F> = self.std.convert(self.lay, bit)?;
            let c = if signed && i + 1 == bits.len() { -coeff } else { coeff };
            terms.push((c, limb));
            coeff = coeff + coeff; // *= 2
        }
        self.std.linear_combination(self.lay, &terms, self.f_zero())
    }

    // -------------------------
    // Boolean ops
    // -------------------------

    fn bool_not(&mut self, a: &AssignedBit<F>) -> Result<AssignedBit<F>, Error> {
        self.std.not(self.lay, a)
    }
    fn nary_and(&mut self, xs: &[AssignedBit<F>]) -> Result<AssignedBit<F>, Error> {
        self.std.and(self.lay, xs)
    }
    fn nary_or(&mut self, xs: &[AssignedBit<F>]) -> Result<AssignedBit<F>, Error> {
        self.std.or(self.lay, xs)
    }
    fn nary_xor(&mut self, xs: &[AssignedBit<F>]) -> Result<AssignedBit<F>, Error> {
        self.std.xor(self.lay, xs)
    }

    // -------------------------
    // ITE (select)
    // -------------------------

    fn ite_field(
        &mut self,
        c: &AssignedBit<F>,
        t: &AssignedNative<F>,
        f: &AssignedNative<F>,
    ) -> Result<AssignedNative<F>, Error> {
        self.std.select(self.lay, c, t, f)
    }

    fn ite_bit(
        &mut self,
        c: &AssignedBit<F>,
        t: &AssignedBit<F>,
        f: &AssignedBit<F>,
    ) -> Result<AssignedBit<F>, Error> {
        self.std.select(self.lay, c, t, f)
    }

    // -------------------------
    // Zero test: is_zero(x)
    // returns a boolean bit 1 iff x == 0.
    // -------------------------

    fn is_zero(&mut self, x: &AssignedNative<F>) -> Result<AssignedBit<F>, Error> {
        // use sgn0 then NOT (sgn0==1 means x != 0 for the bounded encoding)
        let nz = self.std.sgn0(self.lay, x)?;   // 1 if non-zero
        self.bool_not(&nz)
    }

    fn are_equal_field(
        &mut self,
        a: &AssignedNative<F>,
        b: &AssignedNative<F>,
    ) -> Result<AssignedBit<F>, Error> {
        self.std.is_equal(self.lay, a, b)
    }

    fn bits_equal(
        &mut self,
        a: &AssignedBit<F>,
        b: &AssignedBit<F>,
    ) -> Result<AssignedBit<F>, Error> {
        let xor = self.nary_xor(&[a.clone(), b.clone()])?;
        self.bool_not(&xor)
    }

    // ----------------------------------------
    // Embedding: variables, consts, ops
    // ----------------------------------------

    fn embed_var(&mut self, var: &Term, ty: VarType) -> Result<(), Error> {
        if self.cache.contains_key(var) {
            return Ok(());
        }
        if !self.used_vars.contains(var.as_var_name()) {
            return Ok(()); // dead var skip
        }
        let Op::Var(v) = var.op() else { panic!("embed_var expects Op::Var") };

        // value source
        let name = v.as_ref().name.clone();
        let is_public = matches!(ty, VarType::Inst);

        match &v.sort {
            Sort::Bool => {
                let fv = *self.wmap.get(&*name)
                    .or_else(|| self.imap.get(&*name))
                    .unwrap_or(&self.f_zero());
                let as_bool = fv == self.f_one();
                let b = if is_public {
                    self.std.assign_as_public_input(self.lay, self.as_value(as_bool))?
                } else {
                    self.std.assign(self.lay, self.as_value(as_bool))?
                };
                if is_public {
                    // guard: tie to instance value too (redundant if assign_as_public_input used)
                    self.std.constrain_as_public_input(self.lay, &b)?;
                }
                self.cache.insert(var.clone(), AssignedTerm::Bit(b));
            }
            Sort::Field(fsort) => {
                assert_eq!(fsort, self.cfg.field(), "field mismatch");
                let fv = *self.wmap.get(&*name)
                    .or_else(|| self.imap.get(&*name))
                    .unwrap_or(&self.f_zero());
                let x = self.std.assign(self.lay, self.as_value(fv))?;
                if is_public {
                    self.std.constrain_as_public_input(self.lay, &x)?;
                }
                self.cache.insert(var.clone(), AssignedTerm::Field(x));
            }
            Sort::BitVector(w) => {
                // treat value as field, then assert it fits in w bits by decomposition
                let fv = *self.wmap.get(&*name).or_else(|| self.imap.get(&*name)).unwrap_or(&self.f_zero());
                let x = self.std.assign(self.lay, self.as_value(fv))?;
                let bits = self.bitify(&x, *w, /*enforce_canonical=*/true)?;
                self.cache.insert(var.clone(), AssignedTerm::Bv {
                    width: *w,
                    bits,
                    uint: Some(x),
                });
                if is_public {
                    // If you want the entire BV exposed as public input, you can
                    // do it either as one field or as bits. As a default, expose the field:
                    self.std.constrain_as_public_input(self.lay, self.cache.get(var).unwrap().as_field())?;
                }
            }
            _ => panic!("Unsupported var sort {}", v.sort),
        }
        Ok(())
    }

    fn embed(&mut self, t: Term) -> Result<(), Error> {
        let visited_rc = self.visited.clone();
        for c in extras::PostOrderSkipIter::new(t, &move |s: &Term| visited_rc.borrow().contains(s)) {
            if self.visited.borrow().contains(&c) { continue; }
            match check(&c) {
                Sort::Bool => { self.embed_bool(c.clone())?; }
                Sort::Field(_) => { self.embed_pf(c.clone())?; }
                Sort::BitVector(_) => { self.embed_bv(c.clone())?; }
                Sort::Tuple(_) => panic!("Tuple embedding not implemented"),
                s => panic!("embed unimplemented for {:?}", s),
            }
            self.visited.borrow_mut().insert(c);
        }
        Ok(())
    }

    fn embed_bool(&mut self, c: Term) -> Result<&AssignedBit<F>, Error> {
        if !self.cache.contains_key(&c) {
            let w = match c.op() {
                Op::Var(_) => { panic!("call embed_var for variables") }
                Op::Const(v) => {
                    let b = v.as_bool();
                    AssignedTerm::Bit(self.std.assign(self.lay, self.as_value(b))?)
                }
                Op::Eq => {
                    // equality is sort-driven
                    let a = &c.cs()[0]; let b = &c.cs()[1];
                    match check(a) {
                        Sort::Bool => {
                            let ab = self.get_bit(a).unwrap().clone();
                            let bb = self.get_bit(b).unwrap().clone();
                            AssignedTerm::Bit(self.bits_equal(&ab, &bb)?)
                        }
                        Sort::Field(_) => {
                            let af = self.get_field(a).unwrap().clone();
                            let bf = self.get_field(b).unwrap().clone();
                            AssignedTerm::Bit(self.are_equal_field(&af, &bf)?)
                        }
                        Sort::BitVector(_) => {
                            // compare as uints
                            let au = self.get_bv_uint(a)?;
                            let bu = self.get_bv_uint(b)?;
                            AssignedTerm::Bit(self.are_equal_field(&au, &bu)?)
                        }
                        Sort::Tuple(sorts) => {
                            // all fields equal
                            let mut bits = Vec::with_capacity(sorts.len());
                            for i in 0..sorts.len() {
                                let ai = term![Op::Field(i); a.clone()];
                                let bi = term![Op::Field(i); b.clone()];
                                self.embed_bool(term![Op::Eq; ai.clone(), bi.clone()])?;
                                bits.push(self.get_bit(&term![Op::Eq; ai, bi]).unwrap().clone());
                            }
                            AssignedTerm::Bit(self.nary_and(&bits)?)
                        }
                         s => panic!("Unsupported sort in embed_bool: {:?}", s),
                    }
                }
                Op::Ite => {
                    let cnd = self.get_bit(&c.cs()[0]).unwrap().clone();
                    match check(&c.cs()[1]) {
                        Sort::Bool => {
                            let t = self.get_bit(&c.cs()[1]).unwrap().clone();
                            let f = self.get_bit(&c.cs()[2]).unwrap().clone();
                            AssignedTerm::Bit(self.ite_bit(&cnd, &t, &f)?)
                        }
                        _ => panic!("Non-bool branches in bool ITE"),
                    }
                }
                Op::Not => {
                    let a = self.get_bit(&c.cs()[0]).unwrap().clone();
                    AssignedTerm::Bit(self.bool_not(&a)?)
                }
                Op::Implies => {
                    let a = self.get_bit(&c.cs()[0]).unwrap().clone();
                    let b = self.get_bit(&c.cs()[1]).unwrap().clone();
                    let not_a = self.bool_not(&a)?;
                    AssignedTerm::Bit(self.nary_or(&[not_a, b])?)
                }
                Op::BoolNaryOp(o) => {
                    let args: Vec<AssignedBit<F>> = c.cs().iter().map(|t| self.get_bit(t).unwrap().clone()).collect();
                    match o {
                        BoolNaryOp::And => AssignedTerm::Bit(self.nary_and(&args)?),
                        BoolNaryOp::Or  => AssignedTerm::Bit(self.nary_or(&args)?),
                        BoolNaryOp::Xor => AssignedTerm::Bit(self.nary_xor(&args)?),
                    }
                }
                Op::BvBit(i) => {
                    let bits = self.get_bv_bits(&c.cs()[0])?;
                    AssignedTerm::Bit(bits[*i].clone())
                }
                Op::PfToBoolTrusted => {
                    // interpret as 0/1 field
                    let x = self.get_field(&c.cs()[0])?.clone();
                    let is_zero = self.is_zero(&x)?;
                    AssignedTerm::Bit(self.bool_not(&is_zero)?)
                }
                other => panic!("embed_bool: unsupported op {}", other),
            };
            self.cache.insert(c.clone(), w);
        }
        Ok(self.get_bit(&c).unwrap())
    }

    fn embed_pf(&mut self, c: Term) -> Result<&AssignedNative<F>, Error> {
        if !self.cache.contains_key(&c) {
            let w = match c.op() {
                Op::Var(_) => panic!("call embed_var for vars"),
                Op::Const(v) => {
                    let fv = v.as_pf().as_ty_ref(self.cfg.field());
                    let fv : Integer= fv.i();
                    let fv_biguint = BigUint::from_bytes_be(&fv.to_digits::<u8>(rug::integer::Order::MsfBe));
                    let fv_bytes: [u8; 32] = fv_biguint.to_bytes_be().try_into().unwrap();
                    let fv_f = F::from_bytes_be(&fv_bytes).unwrap();
                    AssignedTerm::Field(self.std.assign(self.lay, Value::known(fv_f))?)
                }
                Op::Ite => {
                    let cnd = self.get_bit(&c.cs()[0]).unwrap().clone();
                    let t = self.get_field(&c.cs()[1])?.clone();
                    let f = self.get_field(&c.cs()[2])?.clone();
                    AssignedTerm::Field(self.ite_field(&cnd, &t, &f)?)
                }
                Op::PfNaryOp(PfNaryOp::Add) => {
                    // First, collect all the arguments as AssignedNative<F>
                    let args: Vec<AssignedNative<F>> = c.cs()
                        .iter()
                        .map(|t| self.get_field(t).map(|x| x.clone()))
                        .collect::<Result<_, _>>()?;
                    // Now, fold them with self.add
                    let mut it = args.into_iter();
                    let first = it.next().unwrap();
                    let sum = it.try_fold(first, |acc, t| self.add(&acc, &t))?;
                    AssignedTerm::Field(sum)
                }
                Op::PfNaryOp(PfNaryOp::Mul) => {
                    let args: Vec<AssignedNative<F>> = c.cs()
                        .iter()
                        .map(|t| self.get_field(t).map(|x| x.clone()))
                        .collect::<Result<_, _>>()?;
                    let mut it = args.into_iter();
                    let first = it.next().unwrap();
                    let prod = it.try_fold(first, |acc, t| self.mul(&acc, &t))?;
                    AssignedTerm::Field(prod)
                }
                Op::UbvToPf(_) => {
                    let x = self.get_bv_uint(&c.cs()[0])?;
                    AssignedTerm::Field(x)
                }
                Op::PfUnOp(PfUnOp::Neg) => {
                    let a = self.get_field(&c.cs()[0])?.clone();
                    AssignedTerm::Field(self.mul_const(&a, -F::ONE)?)
                }
                Op::PfUnOp(PfUnOp::Recip) => {
                    // Safe inverse: inv0 + check (your cfg.div_by_zero semantics).
                    let x = self.get_field(&c.cs()[0])?.clone();
                    let inv0 = self.std.inv0(self.lay, &x)?;
                    // Derive guard bit == 0 iff x == 0
                    // (optional extra assertions depending on your policy)
                    AssignedTerm::Field(inv0)
                }
                Op::PfDiv => {
                    let y = self.get_field(&c.cs()[0])?.clone();
                    let x = self.get_field(&c.cs()[1])?.clone();
                    let inv = self.std.inv(self.lay, &x)?;
                    AssignedTerm::Field(self.mul(&y, &inv)?)
                }
                other => panic!("embed_pf unsupported op {}", other),
            };
            self.cache.insert(c.clone(), w);
        }
        Ok(self.get_field(&c).unwrap())
    }

    fn embed_bv(&mut self, bv: Term) -> Result<(), Error> {
        let Sort::BitVector(n) = check(&bv) else { panic!("embed_bv expects bv") };
        if self.cache.contains_key(&bv) { return Ok(()) }

        let assigned = match bv.op() {
            Op::Var(_) => panic!("call embed_var for vars"),
            Op::Const(v) => {
                // interpret each bit as AssignedBit.
                let b = v.as_bv();
                let mut bits = Vec::with_capacity(n);
                for i in 0..n {
                    let bit = b.uint().get_bit(i as u32) != false;
                    bits.push(self.std.assign(self.lay, self.as_value(bit))?);
                }
                AssignedTerm::Bv { width: n, bits, uint: None }
            }
            Op::Ite => {
                // ITE at field level then bitify
                let cond = self.get_bit(&bv.cs()[0]).unwrap().clone();
                let t = self.get_bv_uint(&bv.cs()[1])?;
                let f = self.get_bv_uint(&bv.cs()[2])?;
                let u = self.ite_field(&cond, &t, &f)?;
                let bits = self.bitify(&u, n, true)?;
                AssignedTerm::Bv { width: n, bits, uint: Some(u) }
            }
            Op::BvNaryOp(o) => {
                match o {
                    BvNaryOp::Xor | BvNaryOp::And | BvNaryOp::Or => {
                        let all: Vec<Vec<AssignedBit<F>>> = bv.cs().iter().map(|t| self.get_bv_bits(t).unwrap()).collect();
                        let w = all[0].len();
                        let mut out = Vec::with_capacity(w);
                        for i in 0..w {
                            let slice: Vec<AssignedBit<F>> = all.iter().map(|v| v[i].clone()).collect();
                            let b = match o {
                                BvNaryOp::And => self.nary_and(&slice)?,
                                BvNaryOp::Or  => self.nary_or(&slice)?,
                                BvNaryOp::Xor => self.nary_xor(&slice)?,
                                _ => unreachable!()
                            };
                            out.push(b);
                        }
                        AssignedTerm::Bv { width: w, bits: out, uint: None }
                    }
                    BvNaryOp::Add => {
                        // fold fields + bitify/truncate
                        let inputs: Vec<AssignedNative<F>> = bv.cs().iter().map(|t| self.get_bv_uint(t).unwrap()).collect();
                        let sum = inputs.into_iter().reduce(|a, b| self.add(&a, &b).unwrap()).unwrap();
                        let mut bits = self.bitify(&sum, n + bitsize(bv.cs().len().saturating_sub(1)), true)?;
                        bits.truncate(n);
                        AssignedTerm::Bv { width: n, bits, uint: Some(sum) }
                    }
                    BvNaryOp::Mul => {
                        let inputs: Vec<AssignedNative<F>> = bv.cs().iter().map(|t| self.get_bv_uint(t).unwrap()).collect();
                        let prod = inputs.into_iter().reduce(|a, b| self.mul(&a, &b).unwrap()).unwrap();
                        // truncate to n
                        let mut bits = self.bitify(&prod, 2*n, true)?;
                        bits.truncate(n);
                        AssignedTerm::Bv { width: n, bits, uint: Some(prod) }
                    }
                }
            }
            Op::BvBinOp(o) => {
                let a = self.get_bv_uint(&bv.cs()[0])?;
                let b = self.get_bv_uint(&bv.cs()[1])?;
                match o {
                    BvBinOp::Sub => {
                        // a - b modulo 2^n
                        let modulus = big_pow2(n);
                        let a_plus_mod = self.add_const(&a, modulus)?;
                        let diff = self.sub(&a_plus_mod, &b)?;
                        let mut bits = self.bitify(&diff, n + 1, true)?;
                        bits.truncate(n);
                        AssignedTerm::Bv { width: n, bits, uint: Some(diff) }
                    }
                    BvBinOp::Shl => {
                        // TODO: variable shift with select/hardening.
                        // Minimal support: if RHS is constant, multiply by 2^k.
                        let rb = &bv.cs()[1];
                        if let Op::Const(v) = rb.op() {
                            let sh = v.as_bv().uint().to_u64_wrapping() as usize;
                            let mul = self.mul_const(&a, big_pow2(sh))?;
                            let mut bits = self.bitify(&mul, n + sh + 1, true)?;
                            bits.drain(0..sh); // left shift
                            bits.truncate(n);
                            AssignedTerm::Bv { width: n, bits, uint: Some(mul) }
                        } else {
                            panic!("Bv Shl by non-const: TODO (use layered select gadget)");
                        }
                    }
                    BvBinOp::Lshr | BvBinOp::Ashr => {
                        // TODO: as above. For now support constant RHS.
                        let rb = &bv.cs()[1];
                        if let Op::Const(v) = rb.op() {
                            let sh = v.as_bv().uint().to_u64_wrapping() as usize;
                            let bits = self.get_bv_bits(&bv.cs()[0])?;
                            let mut out = vec![self.std.assign(self.lay, self.as_value(false))?; n];
                            if *o == BvBinOp::Lshr {
                                for i in 0..n {
                                    out[i] = if i + sh < n { bits[i + sh].clone() } else { self.std.assign(self.lay, self.as_value(false))? };
                                }
                            } else {
                                // arithmetic right shift fills with sign bit
                                let sign = bits[n-1].clone();
                                for i in 0..n {
                                    out[i] = if i + sh < n { bits[i + sh].clone() } else { sign.clone() };
                                }
                            }
                            AssignedTerm::Bv { width: n, bits: out, uint: None }
                        } else {
                            panic!("Bv Rshift by non-const: TODO");
                        }
                    }
                    BvBinOp::Udiv | BvBinOp::Urem => {
                        // TODO: implement with native division gadget + range checks
                        panic!("Bv Udiv/Urem: TODO (implement with quotient & remainder witnesses, range checks)")
                    }
                }
            }
            Op::BvConcat => {
                let mut bits = Vec::new();
                for c in bv.cs().iter().rev() {
                    bits.extend_from_slice(&self.get_bv_bits(c).unwrap());
                }
                AssignedTerm::Bv { width: bits.len(), bits, uint: None }
            }
            Op::BvExtract(hi, lo) => {
                let base = self.get_bv_bits(&bv.cs()[0])?;
                let mut out = base[*lo as usize ..= *hi as usize].to_vec();
                AssignedTerm::Bv { width: out.len(), bits: out, uint: None }
            }
            Op::PfToBv(nbits) => {
                let x = self.get_field(&bv.cs()[0])?.clone();
                let bits = self.bitify(&x, *nbits, true)?;
                AssignedTerm::Bv { width: *nbits, bits, uint: Some(x) }
            }
            Op::BoolToBv => {
                let b = self.get_bit(&bv.cs()[0]).unwrap().clone();
                AssignedTerm::Bv { width: 1, bits: vec![b], uint: None }
            }
            other => panic!("embed_bv unsupported op {}", other),
        };
        self.cache.insert(bv, assigned);
        Ok(())
    }

    // -------------------------
    // Lookups in cache
    // -------------------------

    fn get_field(&mut self, t: &Term) -> Result<&AssignedNative<F>, Error> {
        if !self.cache.contains_key(t) { self.embed(t.clone())?; }
        match self.cache.get(t) {
            Some(AssignedTerm::Field(x)) => Ok(x),
            Some(AssignedTerm::Bv { uint: Some(u), .. }) => Ok(u),
            other => panic!("Expected field for {}", t),
        }
    }

    fn get_bit(&mut self, t: &Term) -> Option<&AssignedBit<F>> {
        match self.cache.get(t) {
            Some(AssignedTerm::Bit(b)) => Some(b),
            _ => None,
        }
    }

    fn get_bv_bits(&mut self, t: &Term) -> Result<Vec<AssignedBit<F>>, Error> {
        if !self.cache.contains_key(t) { self.embed_bv(t.clone())?; }
        match self.cache.get(t) {
            Some(AssignedTerm::Bv { bits, .. }) => Ok(bits.clone()),
            _ => panic!("Expected BV for {}", t),
        }
    }

    fn get_bv_uint(&mut self, t: &Term) -> Result<AssignedNative<F>, Error> {
        if !self.cache.contains_key(t) { self.embed_bv(t.clone())?; }
        // Scope for the first mutable borrow
        let (needs_compute, bits_cloned) = match self.cache.get_mut(t) {
            Some(AssignedTerm::Bv { bits, uint, .. }) => {
                if let Some(u) = uint.clone() {
                    return Ok(u);
                }
                (true, bits.clone())
            }
            _ => panic!("Expected BV for {}", t),
        };
        // Now, outside the borrow, do the computation if needed
        if needs_compute {
            let u = self.debitify(&bits_cloned, /*signed=*/false)?;
            // Now re-borrow to set the cache
            if let Some(AssignedTerm::Bv { uint, .. }) = self.cache.get_mut(t) {
                *uint = Some(u.clone());
            }
            Ok(u)
        } else {
            unreachable!()
        }
    }

    // -------------------------
    // Assertions
    // -------------------------

    fn assert_bool(&mut self, t: &Term) -> Result<(), Error> {
        if t.op() == &Op::Eq {
            let a = &t.cs()[0];
            let b = &t.cs()[1];
            match check(a) {
                Sort::Bool => {
                    let ab = self.embed_bool(a.clone())?.clone();
                    let bb = self.embed_bool(b.clone())?.clone();
                    let eq = self.bits_equal(&ab, &bb)?;
                    self.std.assert_true(self.lay, &eq)
                }
                Sort::Field(_) => {
                    let af = self.get_field(a)?.clone();
                    let bf = self.get_field(b)?.clone();
                    self.std.assert_equal(self.lay, &af, &bf)
                }
                Sort::BitVector(_) => {
                    let au = self.get_bv_uint(a)?;
                    let bu = self.get_bv_uint(b)?;
                    self.std.assert_equal(self.lay, &au, &bu)
                }
                _ => panic!("Eq on unsupported sort"),
            }
        } else if t.op() == &AND {
            for c in t.cs() {
                self.assert_bool(c)?;
            }
            Ok(())
        } else if let Op::PfFitsInBits(n) = t.op() {
            let x = self.get_field(&t.cs()[0])?.clone();
            let _ = self.bitify(&x, *n, /*enforce_canonical=*/true)?;
            Ok(())
        } else {
            self.embed_bool(t.clone())?;
            let b = self.get_bit(&t).unwrap().clone();
            self.std.assert_true(self.lay, &b)
        }
    }
}

// -----------------------------
// Small utils
// -----------------------------

fn bitsize(n: usize) -> usize {
    if n == 0 { 1 } else { (n as f64).log2().ceil() as usize }
}

fn big_pow2(n: usize) -> F {
    // NOTE: in real code, use a robust BigInt->F conversion that handles mod reduction
    let mut acc = F::from(1);
    for _ in 0..n { acc = acc + acc; }
    acc
}

// =======================================================
// A Midnight Relation that wraps your IR Computation
// =======================================================

#[derive(Clone)]
pub struct IrRelation<'a> {
    pub cs: &'a Computation,
    pub cfg: &'a CircCfg,
    // public input names in order (these will be required by verification)
    pub public_names: Vec<String>,
    // all input names (order) for witness vector
    pub all_names: Vec<String>,
}

impl<'a> m::Relation for IrRelation<'a> {
    type Instance = Vec<F>;        // raw scalars (public inputs in same order as public_names)
    type Witness  = Vec<F>;        // all inputs in cs.metadata.ordered_input_names() order

    fn format_instance(instance: &Self::Instance) -> Vec<F> {
        instance.clone()
    }

    fn circuit(
        &self,
        std_lib: &m::ZkStdLib,
        layouter: &mut impl Layouter<F>,
        instance: Value<Self::Instance>,
        witness: Value<Self::Witness>,
    ) -> Result<(), Error> {
        // materialize the Value<>s
        let pi_vec: Vec<_> = instance.transpose_vec(self.public_names.len());
        let wit_vec: Vec<_> = witness.transpose_vec(self.all_names.len());

        // name -> value maps
        let imap: HashMap<String, F> = self.public_names.iter().cloned().zip(pi_vec.into_iter()).collect();
        let wmap: HashMap<String, F> = self.all_names.iter().cloned().zip(wit_vec.into_iter()).collect();

        let used_vars: HashSet<String> =
            extras::free_variables(term(Op::Tuple, self.cs.outputs.clone()))
                .into_iter()
                .collect();

        let mut ctx = ToMidnight::new(
            std_lib,
            &mut layouter.namespace(|| "IR->Midnight"),
            self.cfg,
            used_vars,
            &wmap,
            &imap,
            &self.public_names,
        );

        // 1) Declare variables
        let vars = self.cs.metadata.interactive_vars();
        for v in &vars.instances {
            ctx.embed_var(v, VarType::Inst)?;
        }
        for w in &vars.final_witnesses {
            ctx.embed_var(w, VarType::FinalWit)?;
        }

        // 2) Enforce outputs (assertions)
        for c in &self.cs.outputs {
            ctx.assert_bool(c)?;
        }

        Ok(())
    }

    fn used_chips(&self) -> m::ZkStdLibArch {
        // Enable default core chips (Poseidon/SHA if you need them).
        // If you know you won't use SHA/Poseidon, you can disable to shrink k.
        m::ZkStdLibArch::default()
    }

    fn write_relation<W: std::io::Write>(&self, _writer: &mut W) -> std::io::Result<()> { Ok(()) }
    fn read_relation<R: std::io::Read>(_reader: &mut R) -> std::io::Result<Self> {
        Err(std::io::Error::new(std::io::ErrorKind::Other, "IrRelation::read not supported; build from Computation"))
    }
}

// -------------------------------------
// Builder: produce a Relation for CS
// -------------------------------------

pub fn to_midnight_relation<'a>(cs: &'a Computation, cfg: &'a CircCfg) -> IrRelation<'a> {
    let public_names: Vec<String> = cs.metadata.interactive_vars()
        .instances
        .iter()
        .map(|t| t.as_var_name().to_owned())
        .collect();
    let all_names = cs.metadata.ordered_input_names();
    IrRelation { cs, cfg, public_names, all_names }
}
