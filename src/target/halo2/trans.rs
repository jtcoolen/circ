// midnight_backend.rs
//! Lowering IR to Midnight-ZK using the ZkStdLib, now using AssignedBigUint
//! for BitVectors (no bitify/debitify of AssignedNative for BVs).

use crate::cfg::CircCfg;
use crate::ir::term::*;
use crate::target::plonkish::VarType;
use im::HashSet;
use itertools::Itertools;
use midnight_circuits::types::AssignedField;
use midnight_circuits::verifier::Accumulator;
use midnight_circuits::verifier::AssignedAccumulator;
use midnight_circuits::verifier::AssignedMsm;
use midnight_circuits::verifier::BlstrsEmulation;
use num_bigint::BigUint;
use num_traits::{Num, One};
use rsmt2::print;
use rug::Assign;
use rug::Integer;
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use ark_ff::Zero;
use log::{debug, info};
use midnight_circuits::field::decomposition::chip::P2RDecompositionChip;
use midnight_circuits::field::NativeChip;
use midnight_circuits::field::NativeGadget;
use midnight_circuits::halo2curves::ff::Field;
use midnight_circuits::instructions::ConversionInstructions;
use midnight_circuits::types::InnerValue;
use midnight_circuits::types::Instantiable;
use midnight_circuits::verifier::AssignedVk;
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
    types::{AssignedBit, AssignedByte, AssignedNative},
};
use midnight_curves::Fq as F;
use midnight_proofs::{
    circuit::{Layouter, Value},
    halo2curves::ff::PrimeField,
    plonk::Error,
};
use std::convert::TryInto;

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
    Byte(u8),
    Bool(bool),
    ByteArray(Vec<u8>),
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
            _ => panic!("as_field_or_default"),
        }
    }
    fn as_bool_or_default(&self) -> bool {
        match self {
            InputValue::Bool(b) => *b,
            InputValue::Field(f) => *f == F::ONE,
            InputValue::Big(bu) => !bu.is_zero(),
            _ => panic!("as_bool_or_default"),
        }
    }
    fn as_byte_or_default(&self) -> u8 {
        match self {
            InputValue::Byte(b) => *b,
            _e => panic!("as_byte_or_default {:?}", _e),
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
            _ => panic!("as_big_or_default"),
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
    Byte(AssignedByte<F>),
    Bytes(Vec<AssignedByte<F>>),
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

    fn as_byte(&self) -> &AssignedByte<F> {
        match self {
            AssignedTerm::Byte(byte) => byte,
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

    /// Flatten `terms` using their `sorts` into field wires.
    /// Accepts Field or Array(Field, _). Panics otherwise.
    fn flatten_field_args(
        &mut self,
        terms: &[Term],
        sorts: &[Sort],
    ) -> Result<Vec<AssignedNative<F>>, Error> {
        let mut out = Vec::<AssignedNative<F>>::new();
        for (i, t) in terms.iter().enumerate() {
            match &sorts[i] {
                Sort::Field(_) => out.push(self.get_field(t)?.clone()),
                Sort::Array(inner) => {
                    // Array element type must be Field
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
                        "Poseidon: unsupported arg sort at index {}: {} (expected Field or Array(Field,_))",
                        i, other
                    );
                }
            }
        }
        Ok(out)
    }

    /// Flatten any container of Fields (Field | Tuple-of-Field | Array-of-Field).
    fn flatten_fields_any(&mut self, root: &Term) -> Result<Vec<AssignedNative<F>>, Error> {
        let mut out = Vec::<AssignedNative<F>>::new();
        let mut stack: Vec<Term> = vec![root.clone()];
        while let Some(node) = stack.pop() {
            match check(&node) {
                Sort::Field(_) => out.push(self.get_field(&node)?.clone()),
                Sort::Tuple(items) => {
                    let cs = node.cs();
                    if items.len() != cs.len() {
                        panic!(
                            "tuple arity mismatch (decl {} vs term {})",
                            items.len(),
                            cs.len()
                        );
                    }
                    // push reversed to preserve original order when popping
                    for ch in cs.iter().rev() {
                        stack.push(ch.clone());
                    }
                }
                Sort::Array(_arr) => {
                    //if !matches!(arr.key, Sort::Field(_)) {
                    //    panic!("array element must be Field, got {}", arr.key);
                    //}
                    for ch in node.cs().iter().rev() {
                        stack.push(ch.clone());
                    }
                }
                Sort::Bool => {
                    // Coerce Bool -> Field(0/1)
                    let b = self.get_bit(&node)?.clone();
                    let one = self.std.assign_fixed(self.lay, F::ONE)?;
                    let zero = self.std.assign_fixed(self.lay, F::ZERO)?;
                    let as_f = self.std.select(self.lay, &b, &one, &zero)?;
                    out.push(as_f);
                }
                // NEW: coerce BV(8) byte -> Field(b)
                Sort::BitVector(8) => {
                    let by = self.get_byte(&node)?.clone();
                    // assigned_from_be_bytes accepts a slice of AssignedByte<F>
                    let as_f = self
                        .std
                        .assigned_from_be_bytes(self.lay, std::slice::from_ref(&by))?;
                    out.push(as_f);
                }

                // (optional) generic BV(n): recompose its BigUint to a Field
                Sort::BitVector(_) => {
                    let big = self.get_bv_big(&node)?; // AssignedBigUint<F>
                                                       // x = Σ_i limbs[i] * (2^LOG2_BASE)^i
                    let base = F::from_u128(1u128 << LOG2_BASE);
                    let mut coeff = F::ONE;
                    let mut terms: Vec<(F, AssignedNative<F>)> =
                        Vec::with_capacity(big.limbs.len());
                    for limb in big.limbs.iter() {
                        terms.push((coeff, limb.clone()));
                        coeff = coeff * base;
                    }
                    let x = self.std.linear_combination(self.lay, &terms, F::ZERO)?;
                    out.push(x);
                }

                other => panic!(
                    "expected Field/tuple/array of Field, got {} : {}",
                    node, other
                ),
            }
        }
        Ok(out)
    }

    /// Collect bytes (BV(8)) from any container, ensuring exactly 32 bytes.
    fn collect_32_bytes(&mut self, t: &Term) -> Result<[AssignedByte<F>; 32], Error> {
        let v = self.collect_bytes(t)?; // your existing helper returning Vec<AssignedByte<F>>
        if v.len() != 32 {
            panic!("expected exactly 32 bytes, got {}", v.len());
        }
        // Vec -> [T; 32]
        let boxed: Box<[AssignedByte<F>]> = v.into_boxed_slice();
        let boxed: Box<[AssignedByte<F>; 32]> = boxed.try_into().map_err(|_| Error::Synthesis)?;
        Ok(*boxed)
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

    /*/// Coerce a BV(8) term into an AssignedByte<F>:
    ///   BV(8) --UbvToPf--> Field (0..256) --tag bound--> AssignedByte<F>
    fn bv8_to_assigned_byte(&mut self, t: &Term) -> Result<AssignedByte<F>, Error> {
        // Sanity check
        match check(t) {
            Sort::BitVector(8) => {}
            other => panic!("expected BV(8), got {other}"),
        }

        let res = t.as_bv_opt().unwrap().uint().to_u8();
        // Mark it as 8-bit bounded (AssignedByte is an AssignedBounded with bound=8)
        Ok(AssignedByte::<F>::new(res))
    }*/

    /// Collect bytes from a term:
    /// accepts BV(8), Tuple(BV(8),...), or Array(BV(8), N).
    fn collect_bytes(&mut self, t: &Term) -> Result<Vec<AssignedByte<F>>, Error> {
        match check(t) {
            Sort::BitVector(8) => Ok(vec![self.get_byte(t).unwrap().clone()]),

            Sort::Tuple(items) => {
                if !items.iter().all(|s| matches!(s, Sort::BitVector(8))) {
                    panic!("tuple elements must be BV(8), got {items:?}");
                }
                Ok(t.cs()
                    .iter()
                    .map(|c| self.get_byte(c).unwrap().clone())
                    .collect())
            }

            Sort::Array(a) if matches!(a.val, Sort::BitVector(8)) => {
                // If your IR materializes array elements as children, this is enough.
                // If it uses a sparse map, adapt to iterate the domain.
                Ok(t.cs()
                    .iter()
                    .map(|c| self.get_byte(c).unwrap().clone())
                    .collect())
            }

            other => panic!(
                "midnight_sha256: expected bytes (BV(8)/tuple/array), got {}",
                other
            ),
        }
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

    fn get_bv_byte(&mut self, t: &Term) -> Result<AssignedByte<F>, Error> {
        if !self.cache.contains_key(t) {
            self.embed_bv(t.clone())?;
        }
        match self.cache.get(t) {
            Some(AssignedTerm::Byte(b)) => Ok(b.clone()),
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
        //if !self.used_vars.contains(var.as_var_name()) {
        //    return Ok(()); // dead var skip
        //}
        // Only skip dead *witness* vars. Keep all public vars.
        if !matches!(ty, VarType::Inst) && !self.used_vars.contains(var.as_var_name()) {
            return Ok(());
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
                //if is_public {
                //    self.std.constrain_as_public_input(self.lay, &b)?;
                //}
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
            Sort::BitVector(8) => {
                let vv = self
                    .wmap
                    .get(&*name)
                    .or_else(|| self.imap.get(&*name))
                    .cloned()
                    .unwrap_or(Value::known(InputValue::Byte(0u8)));
                let by_val: Value<u8> = vv.map(|iv| iv.as_byte_or_default());
                let byte = self.std.assign(self.lay, by_val)?;
                if is_public {
                    self.std.constrain_as_public_input(self.lay, &byte)?;
                }
                self.cache.insert(var.clone(), AssignedTerm::Byte(byte));
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
                        Sort::BitVector(8) => {
                            // Compare biguints limb-wise (normalized inside gadget).
                            let ax = self.get_bv_byte(a)?;
                            let bx = self.get_bv_byte(b)?;
                            AssignedTerm::Bit(self.std.is_equal(self.lay, &ax, &bx)?)
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
                    let arg_terms = c.cs();

                    match call.name.as_str() {
                        "PlonkMul2" => {
                            if !matches!(call.ret_sort, Sort::Field(_)) {
                                panic!(
                                    "UndefinedFnCall '{}' returns non-field sort in embed_pf: {}",
                                    call.name, call.ret_sort
                                );
                            }

                            if arg_terms.len() != 1 {
                                panic!("PlonkMul2 expects 1 argument, got {}", arg_terms.len());
                            }
                            let a = self.get_field(&arg_terms[0])?.clone();
                            let aa = self.mul(&a, &a)?;
                            let aa = self.add(&aa, &a)?;
                            AssignedTerm::Field(aa)
                        }
                        "midnight_poseidon" => {
                            if !matches!(call.ret_sort, Sort::Field(_)) {
                                panic!(
                                    "UndefinedFnCall '{}' returns non-field sort in embed_pf: {}",
                                    call.name, call.ret_sort
                                );
                            }
                            let inputs = self.flatten_field_args(&arg_terms, &call.arg_sorts)?;
                            if inputs.is_empty() {
                                panic!("midnight_poseidon: need at least one field input");
                            }
                            let h = self.poseidon_hash(&inputs)?;
                            AssignedTerm::Field(h)
                        }
                        "midnight_hybrid_mt" => {
                            // Expect 3 arguments: leaf_bytes, siblings, positions
                            if arg_terms.len() != 3 {
                                panic!(
                                    "midnight_hybrid_mt expects 3 arguments (leaf_bytes, siblings, positions); got {}",
                                    arg_terms.len()
                                );
                            }

                            // 1) Leaf bytes: exactly 32 BV(8)
                            let leaf_bytes: [AssignedByte<F>; 32] =
                                self.collect_32_bytes(&arg_terms[0])?;

                            // 2) Siblings: Field[]
                            let assigned_input_words: Vec<AssignedNative<F>> =
                                self.flatten_fields_any(&arg_terms[1])?;

                            // 3) Positions: Field[] (each 0/1) -> AssignedBit
                            let pos_fields: Vec<AssignedNative<F>> =
                                self.flatten_fields_any(&arg_terms[2])?;
                            if assigned_input_words.len() != pos_fields.len() {
                                panic!(
                                    "midnight_hybrid_mt: siblings and positions length mismatch ({} vs {})",
                                    assigned_input_words.len(),
                                    pos_fields.len()
                                );
                            }
                            let assigned_input_positions: Vec<AssignedBit<F>> = pos_fields
                                .iter()
                                .map(|p| self.std.convert(self.lay, p))
                                .collect::<Result<_, _>>()?;

                            // Compute SHA256(leaf_bytes)
                            let output: [AssignedByte<F>; 32] =
                                self.std.sha256(self.lay, &leaf_bytes)?;

                            // Convert digest bytes to 8 words (big-endian, 4 bytes each)
                            let output_words: Vec<AssignedNative<F>> = output
                                .chunks_exact(4)
                                .map(|word_bytes| {
                                    self.std.assigned_from_be_bytes(self.lay, word_bytes)
                                })
                                .collect::<Result<Vec<_>, _>>()?;

                            // lo = 2^96*w0 + 2^64*w1 + 2^32*w2 + w3
                            // hi = 2^96*w4 + 2^64*w5 + 2^32*w6 + w7
                            let lo = self.std.linear_combination(
                                self.lay,
                                &[
                                    (F::from_u128(2u128.pow(96)), output_words[0].clone()),
                                    (F::from_u128(2u128.pow(64)), output_words[1].clone()),
                                    (F::from_u128(2u128.pow(32)), output_words[2].clone()),
                                    (F::ONE, output_words[3].clone()),
                                ],
                                F::ZERO,
                            )?;
                            let hi = self.std.linear_combination(
                                self.lay,
                                &[
                                    (F::from_u128(2u128.pow(96)), output_words[4].clone()),
                                    (F::from_u128(2u128.pow(64)), output_words[5].clone()),
                                    (F::from_u128(2u128.pow(32)), output_words[6].clone()),
                                    (F::ONE, output_words[7].clone()),
                                ],
                                F::ZERO,
                            )?;

                            let zero: AssignedNative<F> =
                                self.std.assign_fixed(self.lay, F::ZERO)?;

                            // Leaf = Poseidon(lo, hi, 0)
                            let leaf = self.std.poseidon(self.lay, &[lo, hi, zero.clone()])?;

                            // Fold siblings up the tree using the exact select pattern:
                            // left  = select(pos, acc, sibling)  // if pos==1 -> current node is left; else sibling
                            // right = select(pos, sibling, acc)  // if pos==1 -> sibling is right; else current node
                            let root = assigned_input_words
                                .iter()
                                .zip(assigned_input_positions.iter())
                                .try_fold(leaf, |acc, (x, pos)| {
                                    let left = self.std.select(self.lay, pos, &acc, x)?;
                                    let right = self.std.select(self.lay, pos, x, &acc)?;
                                    self.std.poseidon(self.lay, &[left, right, zero.clone()])
                                })?;

                            AssignedTerm::Field(root)
                        }
                        "midnight_verify_recursive_proof" => {
                            // Return shape must be an array/tuple of fields (ACC_SIZE). If you encode
                            // arrays as tuples in your IR, this branch will return a tuple-of-fields.
                            if !matches!(&call.ret_sort, Sort::Array(arr) if matches!(arr.val, Sort::Field(_)))
                                && !matches!(call.ret_sort, Sort::Tuple(_))
                            {
                                panic!(
                                    "midnight_verify_recursive_proof must return field[ACC_SIZE] (array/tuple of fields), got {}",
                                    call.ret_sort
                                );
                            }

                            use midnight_circuits::types::AssignedNative;
                            use midnight_proofs::{
                                circuit::Value, plonk::ConstraintSystem, poly::EvaluationDomain,
                            };

                            // ---- Parse & sanity-check args ----------------------------------------------------------
                            let args = c.cs();
                            if args.len() != 5 {
                                panic!(
                                    "midnight_verify_recursive_proof expects 5 args: \
                                    (vk, is_genesis, prev_state, prev_acc[ACC_SIZE], prev_proof[PROOF_SIZE]); got {}",
                                    args.len()
                                );
                            }

                            println!("args.len() = {}", args.len());
                            ark_std::println!("args[0] = {}", args[0]);
                            ark_std::println!("args[1] = {}", args[1]);
                            ark_std::println!("args[2] = {}", args[2]);
                            ark_std::println!("args[3] = {}", args[3]);
                            ark_std::println!("args[4] = {}", args[4]);

                            // 0) vk : field
                            let vk_field = self.get_field(&args[0])?;

                            // Value<&F> → Value<F>
                            let vk_repr_val = vk_field.clone();

                            // 1) is_genesis : field(0/1) → AssignedBit via equality-to-one
                            let is_genesis_f = self.get_bit(&args[1]).unwrap().clone();
                            let is_not_genesis =
                                self.std.is_equal_to_fixed(self.lay, &is_genesis_f, false)?;
                            let is_genesis = self.std.not(self.lay, &is_not_genesis)?;

                            // 2) prev_state : field (becomes part of the verifier PI vector)
                            let prev_state = self.get_field(&args[2])?.clone();

                            // 3) prev_acc : field[ACC_SIZE] (flat PI encoding we can splice directly)
                            let prev_acc_fields: Vec<AssignedNative<F>> =
                                self.flatten_fields_any(&args[3])?;
                            if prev_acc_fields.is_empty() {
                                panic!("prev_acc must be non-empty");
                            }

                            // 4) prev_proof : field[PROOF_SIZE] → witness blob for the transcript
                            // If you have a helper to build Value<Vec<u8>> from your IR, use it here.
                            let proof_bytes_v: Vec<AssignedByte<F>> =
                                self.collect_bytes(&args[4])?;
                            let proof_bytes_v: Value<Vec<u8>> =
                                proof_bytes_v.iter().map(|b| b.value()).collect();
                            // ---- Local CS + domain (since we don't have self.self_cs / self.self_domain) ------------
                            let mut tmp_cs: ConstraintSystem<F> = ConstraintSystem::default();
                            midnight_circuits::compact_std_lib::ZkStdLib::configure(
                                &mut tmp_cs,
                                midnight_circuits::compact_std_lib::ZkStdLibArch::default(),
                            );
                            let domain = EvaluationDomain::new(tmp_cs.degree() as u32, 19);

                            // ---- Verifier wiring via std-lib gadgets ------------------------------------------------
                            // Assign self VK as public input
                            let self_vk_name = "self_vk";
                            // We expect a finalized cs with no selectors, i.e. whose selectors have been
                            // converted into fixed columns.
                            let selectors = vec![vec![false]; tmp_cs.num_selectors()];
                            let (processed_cs, _) = tmp_cs
                                .clone()
                                .directly_convert_selectors_to_fixed(selectors);

                            // Beware there be dragons!
                            let prev_acc_f: Vec<F> = prev_acc_fields
                                .clone()
                                .into_iter()
                                .map(|e| e.value().into_option().unwrap().clone())
                                .collect();
                            // off-circuit accumulator; we need to link

                            let prev_acc: Accumulator<_> =
                                AssignedAccumulator::from_public_input(prev_acc_f, 1);
                            let mut prev_acc = self.std.verifier_assign_accumulator_from_witness(
                                self.lay,
                                self_vk_name,
                                &tmp_cs,
                                Value::known(prev_acc),
                            )?;
                            // enforce equality -- we should maybe optimise this with less created variables
                            let res: Vec<AssignedNative<F>> =
                                self.std.verifier().as_public_input(self.lay, &prev_acc)?;
                            for e in res.into_iter().zip_eq(prev_acc_fields.iter()) {
                                self.std.assert_equal(self.lay, &e.0, e.1)?;
                            }

                            let assigned_vk = AssignedVk {
                                vk_name: self_vk_name.to_string(),
                                domain: domain.clone(),
                                cs: tmp_cs,
                                transcript_repr: vk_repr_val,
                            };

                            // Committed-instance binding point (default to avoid group trait version pinning)
                            let id_point = self.std.verifier_identity_point(self.lay)?;

                            // Build public inputs for the recursive verify: [vk_pub, prev_state, prev_acc_pub]
                            // We can pass prev_acc as its PI encoding (`prev_acc_fields`) directly.
                            let mut pi: Vec<AssignedNative<F>> = Vec::new();
                            pi.push(assigned_vk.transcript_repr.clone());
                            pi.push(prev_state.clone());
                            pi.extend(prev_acc_fields.clone());

                            // Partial verification → proof_acc
                            let mut proof_acc = self.std.verifier_prepare_partial_plonk(
                                self.lay,
                                &assigned_vk,
                                &[("com_instance", id_point.clone())],
                                &[&pi],
                                proof_bytes_v,
                            )?;

                            self.std.accumulator_scale_by_bit(
                                self.lay,
                                &is_not_genesis,
                                &mut proof_acc,
                            )?;
                            self.std.collapse_accumulator(self.lay, &mut proof_acc)?;

                            let mut next_acc =
                                self.std.accumulate(self.lay, &[proof_acc, prev_acc])?;
                            self.std.collapse_accumulator(self.lay, &mut next_acc)?;

                            let next_acc_field_elts =
                                self.std.verifier().as_public_input(self.lay, &next_acc)?;
                            // Return field[ACC_SIZE] (same encoding)
                            AssignedTerm::Tuple(
                                next_acc_field_elts
                                    .into_iter()
                                    .map(AssignedTerm::Field)
                                    .collect::<Vec<AssignedTerm>>(),
                            )
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
        let check_bv = check(&bv);
        let Sort::BitVector(n) = check_bv else {
            panic!("embed_bv expects bv")
        };
        if self.cache.contains_key(&bv) {
            return Ok(());
        }

        let assigned = match bv.op() {
            Op::Var(_) => panic!("call embed_var for vars"),
            Op::Const(v) => {
                if check_bv == Sort::BitVector(8) {
                    // Special case: BV(8) constant as AssignedByte
                    let b: u8 = v.as_bv().uint().to_u8().unwrap();
                    let byte: AssignedByte<F> = self.std.assign(self.lay, Value::known(b))?;
                    self.cache.insert(bv.clone(), AssignedTerm::Byte(byte));
                    return Ok(());
                } else {
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
            Op::UndefinedFnCall(call) => {
                if !matches!(call.ret_sort, Sort::BitVector(8)) {
                    panic!(
                        "UndefinedFnCall '{}' returns non-field sort in embed_pf: {}",
                        call.name, call.ret_sort
                    );
                }
                let arg_terms = bv.cs();

                match call.name.as_str() {
                    "midnight_sha256" => {
                        ark_std::println!("midnight_sha256: hey there",);
                        // Return sort must be Array(BV(8), 32)
                        if let Sort::Array(ret_arr) = &call.ret_sort {
                            assert!(
                                matches!(ret_arr.val, Sort::BitVector(8)) && ret_arr.size == 32,
                                "midnight_sha256: return sort must be Array(BV(8), 32), got {}",
                                call.ret_sort
                            );
                        } else {
                            panic!(
                                "midnight_sha256 must return Array(BV(8), 32), got {}",
                                call.ret_sort
                            );
                        }

                        // Gather all input bytes from args (BV(8), tuples/arrays of BV(8))
                        let mut bytes_in: Vec<AssignedByte<F>> = Vec::new();
                        for t in arg_terms {
                            bytes_in.extend(self.collect_bytes(t)?);
                        }
                        if bytes_in.is_empty() {
                            panic!("midnight_sha256: need at least one input byte");
                        }
                        ark_std::println!(
                            "midnight_sha256: hashing {} input bytes",
                            bytes_in.len()
                        );

                        // Call Midnight stdlib SHA256 gadget
                        let digest: [AssignedByte<F>; 32] = self.std.sha256(self.lay, &bytes_in)?;

                        AssignedTerm::Bytes(Vec::from(digest))
                    }

                    other => {
                        panic!("UndefinedFnCall '{}' not implemented in embed_pf", other);
                    }
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

    fn get_bit(&mut self, t: &Term) -> Result<&AssignedBit<F>, Error> {
        if !self.cache.contains_key(t) {
            self.embed(t.clone())?;
        }
        match self.cache.get(t) {
            Some(AssignedTerm::Bit(b)) => Ok(b),
            _ => panic!("Expected bit for {}", t),
        }
    }

    fn get_byte(&mut self, t: &Term) -> Result<&AssignedByte<F>, Error> {
        if !self.cache.contains_key(t) {
            self.embed(t.clone())?;
        }
        match self.cache.get(t) {
            Some(AssignedTerm::Byte(b)) => Ok(b),
            _ => panic!("Expected byte for {}", t),
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
                Sort::BitVector(8) => {
                    let ax = self.get_bv_byte(a)?;
                    let bx = self.get_bv_byte(b)?;
                    let eq = self.std.is_equal(self.lay, &ax, &bx)?;
                    self.std.assert_true(self.lay, &eq)
                }
                Sort::BitVector(_) => {
                    let ax = self.get_bv_big(a)?;
                    let bx = self.get_bv_big(b)?;
                    let eq = self.big.is_equal(self.lay, &ax, &bx)?;
                    self.std.assert_true(self.lay, &eq)
                }
                /*Sort::Array(_) => {
                    // TODO we support array of fields only for now
                    //println!("assert_bool Eq on arrays: {}", a);
                    println!("a {}", a.id());
                    println!("b {}", b.id());
                    let a_fields = self.flatten_fields_any(a)?;
                    let b_fields = self.flatten_fields_any(b)?;
                    if a_fields.len() != b_fields.len() {
                        panic!(
                            "assert_bool Eq on arrays of different lengths: {} vs {}",
                            a_fields.len(),
                            b_fields.len()
                        );
                    }
                    let mut eqs = Vec::with_capacity(a_fields.len());
                    for (x, y) in a_fields.iter().zip(b_fields.iter()) {
                        let e = self.std.is_equal(self.lay, x, y)?;
                        eqs.push(e);
                    }
                    let all_eq = self.nary_and(&eqs)?;
                    self.std.assert_true(self.lay, &all_eq)
                }*/
                Sort::Array(arr) => {
                    let acs = a.cs();
                    let bcs = b.cs();
                    assert!(
                        acs.len() == bcs.len(),
                        "assert_bool Eq on arrays of different lengths: {} vs {}",
                        acs.len(),
                        bcs.len()
                    );

                    match &arr.val {
                        // Field[] → flatten then field equality
                        Sort::Field(_) => {
                            let a_fields = self.flatten_fields_any(a)?;
                            let b_fields = self.flatten_fields_any(b)?;
                            let mut eqs = Vec::with_capacity(a_fields.len());
                            for (x, y) in a_fields.iter().zip(b_fields.iter()) {
                                eqs.push(self.std.is_equal(self.lay, x, y)?);
                            }
                            let all = self.nary_and(&eqs)?;
                            self.std.assert_true(self.lay, &all)
                        }

                        // u8[] → bytewise equality
                        Sort::BitVector(8) => {
                            let mut eqs = Vec::with_capacity(acs.len());
                            for i in 0..acs.len() {
                                let ax = self.get_byte(&acs[i])?.clone();
                                let bx = self.get_byte(&bcs[i])?.clone();
                                eqs.push(self.std.is_equal(self.lay, &ax, &bx)?);
                            }
                            let all = self.nary_and(&eqs)?;
                            self.std.assert_true(self.lay, &all)
                        }

                        // BV[n][] → biguint equality
                        Sort::BitVector(_) => {
                            let mut eqs = Vec::with_capacity(acs.len());
                            for i in 0..acs.len() {
                                let ax = self.get_bv_big(&acs[i])?;
                                let bx = self.get_bv_big(&bcs[i])?;
                                eqs.push(self.big.is_equal(self.lay, &ax, &bx)?);
                            }
                            let all = self.nary_and(&eqs)?;
                            self.std.assert_true(self.lay, &all)
                        }

                        // Bool[] → bit equality
                        Sort::Bool => {
                            let mut eqs = Vec::with_capacity(acs.len());
                            for i in 0..acs.len() {
                                let ab = self.get_bit(&acs[i])?.clone();
                                let bb = self.get_bit(&bcs[i])?.clone();
                                eqs.push(self.bits_equal(&ab, &bb)?);
                            }
                            let all = self.nary_and(&eqs)?;
                            self.std.assert_true(self.lay, &all)
                        }

                        // Nested tuples/arrays → recurse per element
                        Sort::Tuple(_) | Sort::Array(_) => {
                            let mut eq_bits = Vec::with_capacity(acs.len());
                            for i in 0..acs.len() {
                                let ei = term![Op::Eq; acs[i].clone(), bcs[i].clone()];
                                self.assert_bool(&ei)?; // builds constraints for nested equality
                                eq_bits.push(self.get_bit(&ei)?.clone());
                            }
                            let all = self.nary_and(&eq_bits)?;
                            self.std.assert_true(self.lay, &all)
                        }

                        other => panic!("Eq on unsupported array element sort: {}", other),
                    }
                }
                _sort => panic!("Eq on unsupported sort {}", _sort),
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
                InputValue::Byte(b) => {
                    let mut bytes = [0u8; 32];
                    bytes[31] = *b;
                    out.push(F::from_bytes_be(&bytes).unwrap())
                }
                InputValue::ByteArray(ba) => {
                    for b in ba {
                        let mut bytes = [0u8; 32];
                        bytes[31] = *b;
                        out.push(F::from_bytes_be(&bytes).unwrap())
                    }
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
        // TODO commented assert_bool seems dangerous
        for c in &self.cs.outputs {
            //println!("Asserting output {}", c);
            info!("Assert: {}", c);
            assert!(check(&c) == Sort::Bool, "Non bool in assert");
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
