// midnight_backend.rs
//! Lowering IR to Midnight-ZK using the ZkStdLib, now using AssignedBigUint
//! for BitVectors (no bitify/debitify of AssignedNative for BVs).

use crate::cfg::CircCfg;
use crate::ir::term::*;
use crate::target::plonkish::VarType;
use ark_std::iterable::Iterable;
use im::HashSet;
use num_bigint::BigUint;
use num_traits::{Num, One};
use rug::Integer;
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use midnight_circuits::halo2curves::ff::Field;
use midnight_circuits::{
    // BigUint gadget + types
    biguint::{biguint_gadget::BigUintGadget, AssignedBigUint},
    compact_std_lib as m,
    instructions::{
        ArithInstructions, AssertionInstructions, AssignmentInstructions, BinaryInstructions,
        ControlFlowInstructions, DecompositionInstructions, EqualityInstructions,
        PublicInputInstructions, ZeroInstructions,
    },
    // Types from the stdlib side
    types::{AssignedBit, AssignedNative},
};

use ark_ff::Zero;
use midnight_circuits::field::decomposition::chip::P2RDecompositionChip;
use midnight_circuits::field::NativeChip;
use midnight_circuits::field::NativeGadget;
use midnight_curves::Fq as F;
use midnight_proofs::{
    circuit::{Layouter, Value},
    halo2curves::ff::PrimeField,
    plonk::Error,
};

// copied from midnight, make it public upstream?
pub(crate) const LOG2_BASE: u32 = 96;

pub fn modulus<F: PrimeField>() -> BigUint {
    BigUint::from_str_radix(&F::MODULUS[2..], 16).unwrap()
}

pub fn big_to_fe<F: PrimeField>(e: BigUint) -> F {
    let modulus = modulus::<F>();
    let e = e % modulus;
    F::from_str_vartime(&e.to_str_radix(10)[..]).unwrap()
}

pub fn big_to_limbs(nb_limbs: u32, base: &BigUint, value: &BigUint) -> Vec<BigUint> {
    use num_traits::Euclid;
    let mut output = vec![];
    let mut q = (*value).clone();
    let mut r;
    while output.len() < nb_limbs as usize {
        (q, r) = q.div_rem_euclid(base);
        output.push(r.clone());
    }
    if !BigUint::is_zero(&q) {
        panic!(
            "big_to_limbs: {} cannot be expressed in base {} with {} limbs",
            value, base, nb_limbs
        )
    };
    output
}
fn biguint_to_limbs<F: PrimeField>(value: &BigUint, nb_limbs: Option<u32>) -> Vec<F> {
    let nb_limbs = nb_limbs.unwrap_or(value.bits().div_ceil(LOG2_BASE as u64) as u32);
    big_to_limbs(nb_limbs, &(BigUint::from(1u8) << LOG2_BASE), value)
        .into_iter()
        .map(big_to_fe::<F>)
        .collect()
}

fn be32_from_biguint(n: &BigUint) -> Result<[u8; 32], &'static str> {
    let bytes = n.to_bytes_be();
    if bytes.len() > 32 {
        return Err("value does not fit in 32 bytes");
    }
    let mut out = [0u8; 32];
    let start = 32 - bytes.len();
    out[start..].copy_from_slice(&bytes); // left-pad with zeros
    Ok(out)
}

// ---------------------------------------
// InputValue: host-side mixed input type
// ---------------------------------------

#[derive(Clone, Debug)]
pub enum InputValue {
    Field(F),
    Big(BigUint),
    Bool(bool),
}

impl InputValue {
    fn as_field_or_default(&self) -> F {
        match self {
            InputValue::Field(x) => *x,
            InputValue::Bool(b) => {
                if *b {
                    F::ONE
                } else {
                    F::ZERO
                }
            }
            InputValue::Big(_bu) => {
                // Reduce biguint mod p into F as a fallback
                // (ONLY used if the IR says this variable is a field).
                // This path won't be used for BVs.
                unimplemented!("BigUint to Field input not implemented");
                //let bytes = bu.to_bytes_be().try_into().unwrap();
                //F::from_bytes_be(bytes).unwrap()
            }
        }
    }
    fn as_bool_or_default(&self) -> bool {
        match self {
            InputValue::Bool(b) => *b,
            InputValue::Field(f) => *f == F::ONE,
            InputValue::Big(bu) => !bu.is_zero(),
        }
    }
    fn as_big_or_default(&self) -> BigUint {
        match self {
            InputValue::Big(b) => b.clone(),
            InputValue::Bool(b) => {
                if *b {
                    BigUint::one()
                } else {
                    BigUint::default()
                }
            }
            InputValue::Field(f) => {
                // Interpret field as canonical BigUint
                // (ONLY used if the IR says this variable is a bit-vector and
                // host supplied it as a field; not typical).
                let mut buf = [0u8; 32];
                buf.copy_from_slice(&f.to_bytes_be());
                BigUint::from_bytes_be(&buf)
            }
        }
    }
}

fn pad_to_64(v: Vec<u8>) -> [u8; 64] {
    if v.len() > 64 {
        // truncate (mod p homomorphism anyway)
        let start = v.len() - 64;
        let slice = &v[start..];
        let mut out = [0u8; 64];
        out.copy_from_slice(slice);
        return out;
    }
    let mut out = [0u8; 64];
    let start = 64 - v.len();
    out[start..].copy_from_slice(&v);
    out
}

// -------------------------------
// Assigned terms (Midnight side)
// -------------------------------

#[derive(Clone)]
enum AssignedTerm {
    Field(AssignedNative<F>),
    Bit(AssignedBit<F>),
    /// BitVector represented natively as AssignedBigUint with an optional bit cache (LE).
    Bv {
        width: usize,
        big: AssignedBigUint<F>,
        bits_cache_le: Option<Vec<AssignedBit<F>>>, // LSB-first cache
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
    fn as_bv(&self) -> (&AssignedBigUint<F>, usize) {
        match self {
            AssignedTerm::Bv { width, big, .. } => (big, *width),
            _ => panic!("Expected bitvector"),
        }
    }
}

// -------------------------------------------------
// Midnight lowering context (replaces ToPlonk core)
// -------------------------------------------------

type NG = NativeGadget<F, P2RDecompositionChip<F>, NativeChip<F>>;

struct ToMidnight<'a, 'b, L: Layouter<F>> {
    std: &'a m::ZkStdLib,
    big: &'a BigUintGadget<F, NG>, // BigUint gadget over the same stdlib
    lay: &'b mut L,
    cache: TermMap<AssignedTerm>,
    visited: Rc<RefCell<TermSet>>,
    cfg: &'a CircCfg,
    used_vars: HashSet<String>,

    // materialized assignment for variables (provided by Relation::Witness/Instance)
    wmap: &'a HashMap<String, Value<InputValue>>, // witness (private)
    imap: &'a HashMap<String, Value<InputValue>>, // instance/public
    pub_order: &'a [String],                      // ordered public names
}

impl<'a, 'b, L: Layouter<F>> ToMidnight<'a, 'b, L> {
    fn new(
        std: &'a m::ZkStdLib,
        lay: &'b mut L,
        cfg: &'a CircCfg,
        used_vars: HashSet<String>,
        wmap: &'a HashMap<String, Value<InputValue>>,
        imap: &'a HashMap<String, Value<InputValue>>,
        pub_order: &'a [String],
    ) -> Self {
        Self {
            std,
            big: std.biguint(),
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

    fn f_zero(&self) -> F {
        F::from(0)
    }
    fn f_one(&self) -> F {
        F::from(1)
    }
    fn f_from_u64(&self, x: u64) -> F {
        F::from(x)
    }

    fn as_value<T: Clone>(&self, t: T) -> Value<T> {
        Value::known(t)
    }

    // ----------------------------------------
    // Field arith mapping (add/sub/mul/const)
    // ----------------------------------------

    fn add(
        &mut self,
        a: &AssignedNative<F>,
        b: &AssignedNative<F>,
    ) -> Result<AssignedNative<F>, Error> {
        self.std.add(self.lay, a, b)
    }
    fn sub(
        &mut self,
        a: &AssignedNative<F>,
        b: &AssignedNative<F>,
    ) -> Result<AssignedNative<F>, Error> {
        self.std.sub(self.lay, a, b)
    }
    fn mul(
        &mut self,
        a: &AssignedNative<F>,
        b: &AssignedNative<F>,
    ) -> Result<AssignedNative<F>, Error> {
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

    fn assert_boolean(&mut self, _b: &AssignedBit<F>) -> Result<(), Error> {
        // AssignedBit is boolean by construction; if sourced from field, convert & recheck.
        Ok(())
    }

    fn poseidon_hash(&mut self, input: &[AssignedNative<F>]) -> Result<AssignedNative<F>, Error> {
        self.std.poseidon(self.lay, input)
    }

    // -------------------------
    // Bits / Decomposition
    // (keep for true Pf<->bits, but never to represent BVs)
    // -------------------------

    fn bitify_true_pf_to_bits(
        &mut self,
        x: &AssignedNative<F>,
        n: usize,
        enforce_canonical: bool,
    ) -> Result<Vec<AssignedBit<F>>, Error> {
        self.std
            .assigned_to_le_bits(self.lay, x, Some(n), enforce_canonical)
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
    fn ite_big(
        &mut self,
        c: &AssignedBit<F>,
        t: &AssignedBigUint<F>,
        f: &AssignedBigUint<F>,
    ) -> Result<AssignedBigUint<F>, Error> {
        self.big.select(self.lay, c, t, f)
    }

    // -------------------------
    // Zero test: is_zero(x)
    // -------------------------

    fn is_zero(&mut self, x: &AssignedNative<F>) -> Result<AssignedBit<F>, Error> {
        let nz = self.std.sgn0(self.lay, x)?; // 1 if non-zero
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

    // -------------------------
    // BV helpers (BigUint based)
    // -------------------------

    /// Ensure (and cache) the LE bit representation for a cached BV.
    fn ensure_bv_bits_le(&mut self, t: &Term) -> Result<Vec<AssignedBit<F>>, Error> {
        if !self.cache.contains_key(t) {
            self.embed_bv(t.clone())?;
        }
        let (need, width, big) = match self.cache.get_mut(t) {
            Some(AssignedTerm::Bv {
                width,
                big,
                bits_cache_le,
            }) => {
                if bits_cache_le.is_none() {
                    (true, *width, big.clone())
                } else {
                    (false, *width, big.clone())
                }
            }
            _ => panic!("Expected BV for {}", t),
        };

        if need {
            let mut bits = self.big.to_le_bits(self.lay, &big)?;
            bits.truncate(width);
            if let Some(AssignedTerm::Bv { bits_cache_le, .. }) = self.cache.get_mut(t) {
                *bits_cache_le = Some(bits.clone());
            }
            Ok(bits)
        } else {
            match self.cache.get(t) {
                Some(AssignedTerm::Bv {
                    bits_cache_le: Some(b),
                    ..
                }) => Ok(b.clone()),
                _ => unreachable!(),
            }
        }
    }

    /// Return the BigUint handle for a BV term.
    fn get_bv_big(&mut self, t: &Term) -> Result<AssignedBigUint<F>, Error> {
        if !self.cache.contains_key(t) {
            self.embed_bv(t.clone())?;
        }
        match self.cache.get(t) {
            Some(AssignedTerm::Bv { big, .. }) => Ok(big.clone()),
            _ => panic!("Expected BV for {}", t),
        }
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
        let Op::Var(v) = var.op() else {
            panic!("embed_var expects Op::Var")
        };

        // value source
        let name = v.as_ref().name.clone();
        let is_public = matches!(ty, VarType::Inst);

        match &v.sort {
            Sort::Bool => {
                let vv = self
                    .wmap
                    .get(&*name)
                    .or_else(|| self.imap.get(&*name))
                    .cloned()
                    .unwrap_or(Value::known(InputValue::Bool(false)));
                let as_bool: Value<bool> = vv.map(|iv| iv.as_bool_or_default());
                let b = if is_public {
                    self.std.assign_as_public_input(self.lay, as_bool)?
                } else {
                    self.std.assign(self.lay, as_bool)?
                };
                if is_public {
                    self.std.constrain_as_public_input(self.lay, &b)?;
                }
                self.cache.insert(var.clone(), AssignedTerm::Bit(b));
            }
            Sort::Field(fsort) => {
                assert_eq!(fsort, self.cfg.field(), "field mismatch");
                let vv = self
                    .wmap
                    .get(&*name)
                    .or_else(|| self.imap.get(&*name))
                    .cloned()
                    .unwrap_or(Value::known(InputValue::Field(self.f_zero())));
                let fv: Value<F> = vv.map(|iv| iv.as_field_or_default());
                let x = self.std.assign(self.lay, fv)?;
                if is_public {
                    self.std.constrain_as_public_input(self.lay, &x)?;
                }
                self.cache.insert(var.clone(), AssignedTerm::Field(x));
            }
            Sort::BitVector(w) => {
                // Directly assign a BigUint bounded by w bits — no field-bitify path.
                let vv = self
                    .wmap
                    .get(&*name)
                    .or_else(|| self.imap.get(&*name))
                    .cloned()
                    .unwrap_or(Value::known(InputValue::Big(BigUint::default())));
                let bu_val: Value<BigUint> = vv.map(|iv| iv.as_big_or_default());
                let big = self.big.assign_biguint(self.lay, bu_val, *w as u32)?;
                if is_public {
                    // Constrain each limb as public input dynamically (no const-generic).
                    for limb in &big.limbs {
                        self.std.constrain_as_public_input(self.lay, limb)?;
                    }
                }
                self.cache.insert(
                    var.clone(),
                    AssignedTerm::Bv {
                        width: *w,
                        big,
                        bits_cache_le: None,
                    },
                );
            }
            _ => panic!("Unsupported var sort {}", v.sort),
        }
        Ok(())
    }

    fn embed(&mut self, t: Term) -> Result<(), Error> {
        let visited_rc = self.visited.clone();
        for c in extras::PostOrderSkipIter::new(t, &move |s: &Term| visited_rc.borrow().contains(s))
        {
            if self.visited.borrow().contains(&c) {
                continue;
            }
            match check(&c) {
                Sort::Bool => {
                    self.embed_bool(c.clone())?;
                }
                Sort::Field(_) => {
                    self.embed_pf(c.clone())?;
                }
                Sort::BitVector(_) => {
                    self.embed_bv(c.clone())?;
                }
                Sort::Tuple(_items) => {
                    for ch in c.cs() {
                        self.embed(ch.clone())?;
                    }
                }
                s => panic!("embed unimplemented for {:?}", s),
            }
            self.visited.borrow_mut().insert(c);
        }
        Ok(())
    }

    fn embed_bool(&mut self, c: Term) -> Result<&AssignedBit<F>, Error> {
        if !self.cache.contains_key(&c) {
            let w = match c.op() {
                Op::Var(_) => {
                    panic!("call embed_var for variables")
                }
                Op::Const(v) => {
                    let b = v.as_bool();
                    AssignedTerm::Bit(self.std.assign(self.lay, self.as_value(b))?)
                }
                Op::Eq => {
                    let a = &c.cs()[0];
                    let b = &c.cs()[1];
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
                            // Compare biguints limb-wise (normalized inside gadget).
                            let ax = self.get_bv_big(a)?;
                            let bx = self.get_bv_big(b)?;
                            AssignedTerm::Bit(self.big.is_equal(self.lay, &ax, &bx)?)
                        }
                        Sort::Tuple(sorts) => {
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
                    let args: Vec<AssignedBit<F>> = c
                        .cs()
                        .iter()
                        .map(|t| self.get_bit(t).unwrap().clone())
                        .collect();
                    match o {
                        BoolNaryOp::And => AssignedTerm::Bit(self.nary_and(&args)?),
                        BoolNaryOp::Or => AssignedTerm::Bit(self.nary_or(&args)?),
                        BoolNaryOp::Xor => AssignedTerm::Bit(self.nary_xor(&args)?),
                    }
                }
                Op::BvBit(i) => {
                    let bits = self.ensure_bv_bits_le(&c.cs()[0])?;
                    AssignedTerm::Bit(bits[*i].clone())
                }
                Op::PfToBoolTrusted => {
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
                    let fv: Integer = fv.i();
                    let fv_biguint =
                        BigUint::from_bytes_be(&fv.to_digits::<u8>(rug::integer::Order::MsfBe));
                    let fv_bytes: [u8; 32] = be32_from_biguint(&fv_biguint).unwrap();
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
                    let args: Vec<AssignedNative<F>> = c
                        .cs()
                        .iter()
                        .map(|t| self.get_field(t).map(|x| x.clone()))
                        .collect::<Result<_, _>>()?;
                    let mut it = args.into_iter();
                    let first = it.next().unwrap();
                    let sum = it.try_fold(first, |acc, t| self.add(&acc, &t))?;
                    AssignedTerm::Field(sum)
                }
                Op::PfNaryOp(PfNaryOp::Mul) => {
                    let args: Vec<AssignedNative<F>> = c
                        .cs()
                        .iter()
                        .map(|t| self.get_field(t).map(|x| x.clone()))
                        .collect::<Result<_, _>>()?;
                    let mut it = args.into_iter();
                    let first = it.next().unwrap();
                    let prod = it.try_fold(first, |acc, t| self.mul(&acc, &t))?;
                    AssignedTerm::Field(prod)
                }
                Op::UbvToPf(_) => {
                    // Recompose a field from the BV biguint without bitify/debitify of a native.
                    let big = self.get_bv_big(&c.cs()[0])?;
                    // x = sum_i limbs[i] * BASE^i
                    let base_fe = F::from_u128(1u128 << LOG2_BASE);
                    let mut coeff = F::ONE;
                    let mut terms: Vec<(F, AssignedNative<F>)> =
                        Vec::with_capacity(big.limbs.len());
                    for limb in big.limbs.iter() {
                        terms.push((coeff, limb.clone()));
                        coeff = coeff * base_fe;
                    }
                    let x = self.std.linear_combination(self.lay, &terms, F::ZERO)?;
                    AssignedTerm::Field(x)
                }
                Op::PfUnOp(PfUnOp::Neg) => {
                    let a = self.get_field(&c.cs()[0])?.clone();
                    AssignedTerm::Field(self.mul_const(&a, -F::ONE)?)
                }
                Op::PfUnOp(PfUnOp::Recip) => {
                    let x = self.get_field(&c.cs()[0])?.clone();
                    let inv0 = self.std.inv0(self.lay, &x)?;
                    AssignedTerm::Field(inv0)
                }
                Op::PfDiv => {
                    let y = self.get_field(&c.cs()[0])?.clone();
                    let x = self.get_field(&c.cs()[1])?.clone();
                    let inv = self.std.inv(self.lay, &x)?;
                    AssignedTerm::Field(self.mul(&y, &inv)?)
                }
                Op::UndefinedFnCall(call) => {
                    if !matches!(call.ret_sort, Sort::Field(_)) {
                        panic!(
                            "UndefinedFnCall '{}' returns non-field sort in embed_pf: {}",
                            call.name, call.ret_sort
                        );
                    }
                    let arg_terms = c.cs();

                    let mut flatten_field_args =
                        |terms: &[Term], sorts: &[Sort]| -> Result<Vec<AssignedNative<F>>, Error> {
                            let mut out = Vec::<AssignedNative<F>>::new();
                            for (i, t) in terms.iter().enumerate() {
                                match &sorts[i] {
                                    Sort::Field(_) => out.push(self.get_field(t)?.clone()),
                                    Sort::Array(inner) => {
                                        if !matches!(inner.key, Sort::Field(_)) {
                                            panic!(
                                                "Poseidon: array arg {} must have element sort Field, got {}",
                                                i, inner.key
                                            );
                                        }
                                        for e in t.cs() {
                                            out.push(self.get_field(&e)?.clone());
                                        }
                                    }
                                    other => {
                                        panic!(
                                            "Poseidon: unsupported arg sort at index {}: {} (expected Field, Array(Field,_))",
                                            i, other
                                        );
                                    }
                                }
                            }
                            Ok(out)
                        };

                    match call.name.as_str() {
                        "PlonkMul2" => {
                            if arg_terms.len() != 1 {
                                panic!("PlonkMul2 expects 1 argument, got {}", arg_terms.len());
                            }
                            let a = self.get_field(&arg_terms[0])?.clone();
                            let aa = self.mul(&a, &a)?;
                            let aa = self.add(&aa, &a)?;
                            AssignedTerm::Field(aa)
                        }
                        "midnight_poseidon" => {
                            let inputs = flatten_field_args(&arg_terms, &call.arg_sorts)?;
                            if inputs.is_empty() {
                                panic!("midnight_poseidon: need at least one field input");
                            }
                            let h = self.poseidon_hash(&inputs)?;
                            AssignedTerm::Field(h)
                        }
                        other => {
                            panic!("UndefinedFnCall '{}' not implemented in embed_pf", other);
                        }
                    }
                }
                other => panic!("embed_pf unsupported op {}", other),
            };
            self.cache.insert(c.clone(), w);
        }
        Ok(self.get_field(&c).unwrap())
    }

    fn embed_bv(&mut self, bv: Term) -> Result<(), Error> {
        let Sort::BitVector(n) = check(&bv) else {
            panic!("embed_bv expects bv")
        };
        if self.cache.contains_key(&bv) {
            return Ok(());
        }

        let assigned = match bv.op() {
            Op::Var(_) => panic!("call embed_var for vars"),
            Op::Const(v) => {
                // Direct BigUint assignment from constant BV.
                let b = v.as_bv().uint(); // TODO is it me or the type inference is struggling?
                let val = BigUint::from_str_radix(&b.to_string_radix(10), 10).unwrap();
                let big = self
                    .big
                    .assign_biguint(self.lay, Value::known(val), n as u32)?;
                AssignedTerm::Bv {
                    width: n,
                    big,
                    bits_cache_le: None,
                }
            }
            Op::Ite => {
                // Branch on biguint directly; produce BV as biguint, cache bits lazily.
                let cond = self.get_bit(&bv.cs()[0]).unwrap().clone();
                let t = self.get_bv_big(&bv.cs()[1])?;
                let f = self.get_bv_big(&bv.cs()[2])?;
                let chosen = self.ite_big(&cond, &t, &f)?;
                // Truncate to n bits at the bit level then rebuild a normalized biguint.
                let mut bits = self.big.to_le_bits(self.lay, &chosen)?;
                bits.truncate(n);
                let big = self.big.from_le_bits(self.lay, &bits)?;
                AssignedTerm::Bv {
                    width: n,
                    big,
                    bits_cache_le: Some(bits),
                }
            }
            Op::BvNaryOp(o) => {
                match o {
                    BvNaryOp::Xor | BvNaryOp::And | BvNaryOp::Or => {
                        // Bitwise ops in the bit domain, then rebuild a biguint.
                        let all_bits: Vec<Vec<AssignedBit<F>>> = bv
                            .cs()
                            .iter()
                            .map(|t| {
                                let mut b = self.ensure_bv_bits_le(t).unwrap();
                                b.truncate(n);
                                b
                            })
                            .collect();
                        let mut out_bits = Vec::with_capacity(n);
                        for i in 0..n {
                            let slice: Vec<AssignedBit<F>> =
                                all_bits.iter().map(|v| v[i].clone()).collect();
                            let b = match o {
                                BvNaryOp::And => self.nary_and(&slice)?,
                                BvNaryOp::Or => self.nary_or(&slice)?,
                                BvNaryOp::Xor => self.nary_xor(&slice)?,
                                _ => unreachable!(),
                            };
                            out_bits.push(b);
                        }
                        let big = self.big.from_le_bits(self.lay, &out_bits)?;
                        AssignedTerm::Bv {
                            width: n,
                            big,
                            bits_cache_le: Some(out_bits),
                        }
                    }
                    BvNaryOp::Add => {
                        let inputs: Vec<AssignedBigUint<F>> = bv
                            .cs()
                            .iter()
                            .map(|t| self.get_bv_big(t).unwrap())
                            .collect();
                        let sum = inputs
                            .into_iter()
                            .reduce(|a, b| self.big.add(self.lay, &a, &b).unwrap())
                            .unwrap();
                        let mut bits = self.big.to_le_bits(self.lay, &sum)?;
                        bits.truncate(n);
                        let big = self.big.from_le_bits(self.lay, &bits)?;
                        AssignedTerm::Bv {
                            width: n,
                            big,
                            bits_cache_le: Some(bits),
                        }
                    }
                    BvNaryOp::Mul => {
                        let inputs: Vec<AssignedBigUint<F>> = bv
                            .cs()
                            .iter()
                            .map(|t| self.get_bv_big(t).unwrap())
                            .collect();
                        let prod = inputs
                            .into_iter()
                            .reduce(|a, b| self.big.mul(self.lay, &a, &b).unwrap())
                            .unwrap();
                        let mut bits = self.big.to_le_bits(self.lay, &prod)?;
                        bits.truncate(n);
                        let big = self.big.from_le_bits(self.lay, &bits)?;
                        AssignedTerm::Bv {
                            width: n,
                            big,
                            bits_cache_le: Some(bits),
                        }
                    }
                }
            }
            Op::BvBinOp(o) => {
                let a = self.get_bv_big(&bv.cs()[0])?;
                let b = self.get_bv_big(&bv.cs()[1])?;
                match o {
                    BvBinOp::Sub => {
                        let diff = self.big.sub(self.lay, &a, &b)?;
                        let mut bits = self.big.to_le_bits(self.lay, &diff)?;
                        bits.truncate(n);
                        let big = self.big.from_le_bits(self.lay, &bits)?;
                        AssignedTerm::Bv {
                            width: n,
                            big,
                            bits_cache_le: Some(bits),
                        }
                    }
                    BvBinOp::Shl => {
                        // For now support constant RHS shifts via bit-level movement, then rebuild biguint.
                        let rb = &bv.cs()[1];
                        if let Op::Const(v) = rb.op() {
                            let sh = v.as_bv().uint().to_u64_wrapping() as usize;
                            let mut in_bits = self.ensure_bv_bits_le(&bv.cs()[0])?;
                            in_bits.truncate(n);
                            let mut out =
                                vec![self.std.assign(self.lay, self.as_value(false))?; sh];
                            out.extend(in_bits.into_iter().take(n.saturating_sub(sh)));
                            out.truncate(n);
                            let big = self.big.from_le_bits(self.lay, &out)?;
                            AssignedTerm::Bv {
                                width: n,
                                big,
                                bits_cache_le: Some(out),
                            }
                        } else {
                            panic!("Bv Shl by non-const: TODO (use layered select gadget)");
                        }
                    }
                    BvBinOp::Lshr | BvBinOp::Ashr => {
                        let rb = &bv.cs()[1];
                        if let Op::Const(v) = rb.op() {
                            let sh = v.as_bv().uint().to_u64_wrapping() as usize;
                            let bits = {
                                let mut b = self.ensure_bv_bits_le(&bv.cs()[0])?;
                                b.truncate(n);
                                b
                            };
                            let mut out = vec![self.std.assign(self.lay, self.as_value(false))?; n];
                            if *o == BvBinOp::Lshr {
                                for i in 0..n {
                                    out[i] = if i + sh < n {
                                        bits[i + sh].clone()
                                    } else {
                                        self.std.assign(self.lay, self.as_value(false))?
                                    };
                                }
                            } else {
                                // arithmetic right shift fills with sign bit (MSB)
                                let sign = bits[n - 1].clone();
                                for i in 0..n {
                                    out[i] = if i + sh < n {
                                        bits[i + sh].clone()
                                    } else {
                                        sign.clone()
                                    };
                                }
                            }
                            let big = self.big.from_le_bits(self.lay, &out)?;
                            AssignedTerm::Bv {
                                width: n,
                                big,
                                bits_cache_le: Some(out),
                            }
                        } else {
                            panic!("Bv Rshift by non-const: TODO");
                        }
                    }
                    BvBinOp::Udiv | BvBinOp::Urem => {
                        // Implement via biguint division gadget when available; otherwise TODO.
                        panic!("Bv Udiv/Urem: TODO (use BigUintGadget::div_rem)")
                    }
                }
            }
            Op::BvConcat => {
                // Concat in bit domain, then rebuild biguint.
                let mut out_bits = Vec::new();
                for c in bv.cs().iter().rev() {
                    let mut b = self.ensure_bv_bits_le(c).unwrap();
                    // Concatenation is high..low; ensure we don't exceed declared widths
                    b.truncate(check(c).as_bv());
                    out_bits.extend_from_slice(&b);
                }
                let w = out_bits.len();
                let big = self.big.from_le_bits(self.lay, &out_bits)?;
                AssignedTerm::Bv {
                    width: w,
                    big,
                    bits_cache_le: Some(out_bits),
                }
            }
            Op::BvExtract(hi, lo) => {
                let base = self.ensure_bv_bits_le(&bv.cs()[0])?;
                let out = base[*lo as usize..=*hi as usize].to_vec();
                let big = self.big.from_le_bits(self.lay, &out)?;
                AssignedTerm::Bv {
                    width: out.len(),
                    big,
                    bits_cache_le: Some(out),
                }
            }
            Op::PfToBv(nbits) => {
                // This op semantically decomposes a field to bits; use true decomposition.
                let x = self.get_field(&bv.cs()[0])?.clone();
                let bits = self.bitify_true_pf_to_bits(&x, *nbits, true)?;
                let big = self.big.from_le_bits(self.lay, &bits)?;
                AssignedTerm::Bv {
                    width: *nbits,
                    big,
                    bits_cache_le: Some(bits),
                }
            }
            Op::BoolToBv => {
                let b = self.get_bit(&bv.cs()[0]).unwrap().clone();
                let big = self.big.from_le_bits(self.lay, &[b.clone()])?;
                AssignedTerm::Bv {
                    width: 1,
                    big,
                    bits_cache_le: Some(vec![b]),
                }
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
        if !self.cache.contains_key(t) {
            self.embed(t.clone())?;
        }
        match self.cache.get(t) {
            Some(AssignedTerm::Field(x)) => Ok(x),
            _ => panic!("Expected field for {}", t),
        }
    }

    fn get_bit(&mut self, t: &Term) -> Option<&AssignedBit<F>> {
        match self.cache.get(t) {
            Some(AssignedTerm::Bit(b)) => Some(b),
            _ => None,
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
                    let ax = self.get_bv_big(a)?;
                    let bx = self.get_bv_big(b)?;
                    let eq = self.big.is_equal(self.lay, &ax, &bx)?;
                    self.std.assert_true(self.lay, &eq)
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
            let _ = self.bitify_true_pf_to_bits(&x, *n, /*enforce_canonical=*/ true)?;
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
    if n == 0 {
        1
    } else {
        (n as f64).log2().ceil() as usize
    }
}

fn big_pow2(n: usize) -> F {
    // NOTE: used only in legacy contexts; BigUint path doesn't rely on this for BVs
    let mut acc = F::from(1);
    for _ in 0..n {
        acc = acc + acc;
    }
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
    /// Mixed, host-side inputs (flattened later to F's).
    type Instance = Vec<InputValue>;
    type Witness = Vec<InputValue>;

    /// Flatten public inputs to raw field elements (ordering must match constraints).
    ///
    /// - Field(F) and Bool map to one F each (Bool -> 0/1).
    /// - Big(BigUint) maps to limbs in base 2^LOG2_BASE (least-significant limb first),
    ///   which is exactly how we constrain biguint public inputs in-circuit (one limb per public).
    fn format_instance(instance: &Self::Instance) -> Vec<F> {
        let mut out = Vec::new();
        for iv in instance {
            match iv {
                InputValue::Field(f) => out.push(*f),
                InputValue::Bool(b) => out.push(if *b { F::ONE } else { F::ZERO }),
                InputValue::Big(bu) => {
                    // Use the helper that matches the stdlib/base decomposition.
                    out.extend(biguint_to_limbs::<F>(bu, None));
                }
            }
        }
        out
    }

    fn circuit(
        &self,
        std_lib: &m::ZkStdLib,
        layouter: &mut impl Layouter<F>,
        instance: Value<Self::Instance>,
        witness: Value<Self::Witness>,
    ) -> Result<(), Error> {
        // Keep them as Value<InputValue>; don't "materialize".
        let pi_vec: Vec<Value<InputValue>> = instance.transpose_vec(self.public_names.len());
        let wit_vec: Vec<Value<InputValue>> = witness.transpose_vec(self.all_names.len());

        // name -> Value<InputValue> maps
        let imap: HashMap<String, Value<InputValue>> = self
            .public_names
            .iter()
            .cloned()
            .zip(pi_vec.into_iter())
            .collect();
        let wmap: HashMap<String, Value<InputValue>> = self
            .all_names
            .iter()
            .cloned()
            .zip(wit_vec.into_iter())
            .collect();

        let used_vars: HashSet<String> =
            extras::free_variables(term(Op::Tuple, self.cs.outputs.clone()))
                .into_iter()
                .collect();

        let mut binding = layouter.namespace(|| "IR->Midnight");
        let mut ctx = ToMidnight::new(
            std_lib,
            &mut binding,
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
        m::ZkStdLibArch::default()
    }

    fn write_relation<W: std::io::Write>(&self, _writer: &mut W) -> std::io::Result<()> {
        Ok(())
    }
    fn read_relation<R: std::io::Read>(_reader: &mut R) -> std::io::Result<Self> {
        Err(std::io::Error::new(
            std::io::ErrorKind::Other,
            "IrRelation::read not supported; build from Computation",
        ))
    }
}

// -------------------------------------
// Builder: produce a Relation for CS
// -------------------------------------

pub fn to_midnight_relation<'a>(cs: &'a Computation, cfg: &'a CircCfg) -> IrRelation<'a> {
    let public_names: Vec<String> = cs
        .metadata
        .interactive_vars()
        .instances
        .iter()
        .map(|t| t.as_var_name().to_owned())
        .collect();
    let all_names = cs.metadata.ordered_input_names();
    ark_std::println!(
        "Midnight relation: {} public inputs, {} total inputs",
        public_names.len(),
        all_names.len()
    );
    IrRelation {
        cs,
        cfg,
        public_names,
        all_names,
    }
}
