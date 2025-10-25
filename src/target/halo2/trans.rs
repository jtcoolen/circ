// midnight_backend.rs
//! Lowering IR to Midnight-ZK using the ZkStdLib, now using AssignedBigUint
//! for BitVectors (no bitify/debitify of AssignedNative for BVs).

use crate::cfg::CircCfg;
use crate::ir::term::*;
use crate::target::plonkish::VarType;
use bellman::groth16::VerifyingKey;
use itertools::Itertools;
use midnight_circuits::compact_std_lib::MidnightVK;
use midnight_circuits::types::AssignedField;
use midnight_circuits::verifier;
use midnight_circuits::verifier::Accumulator;
use midnight_circuits::verifier::AssignedAccumulator;
use midnight_circuits::verifier::AssignedMsm;
use midnight_circuits::verifier::BlstrsEmulation;
use midnight_circuits::verifier::Msm;
use midnight_proofs::halo2curves;
use num_bigint::BigUint;
use num_traits::{Num, One};
use rsmt2::print;
use rug::Assign;
use rug::Integer;
use std::cell::RefCell;
use std::collections::HashMap;
use std::collections::HashSet;
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

pub fn lex_to_numeric_in_place<T>(elems: &mut [T]) {
    let n = elems.len();

    // order[pos] = numeric index where the element at `pos` should go.
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by_key(|&i| i.to_string()); // lexicographic by decimal string

    // Apply the permutation in-place using swaps. After each swap,
    // swap in `order` as well to keep "order[pos] is the target of the element at pos".
    for i in 0..n {
        while order[i] != i {
            let j = order[i];
            elems.swap(i, j);
            order.swap(i, j);
        }
    }
}

/// Convenience wrapper that takes ownership and returns the reordered vector.
pub fn reorder_lex_to_numeric<T>(mut elems: Vec<T>) -> Vec<T> {
    lex_to_numeric_in_place(&mut elems);
    elems
}

/// Reorder a slice from numeric index order (0,1,2,…) to
/// lexicographic index order ("0","1","10","11","2",…).
/// Works in-place, O(n log n) due to the sort.
pub fn reorder_numeric_to_lex_in_place<T>(elems: &mut [T]) {
    let n = elems.len();

    // lex = indices [0..n) sorted by their decimal-string representation
    let mut lex: Vec<usize> = (0..n).collect();
    lex.sort_by_key(|&i| i.to_string());

    // dest[i] = destination position (in lex order) for element currently at numeric index i
    let mut dest = vec![0usize; n];
    for (pos, &idx) in lex.iter().enumerate() {
        dest[idx] = pos;
    }

    // Apply permutation in-place using swap cycles
    for i in 0..n {
        // While the element at i doesn't belong at i, swap it to its destination
        while dest[i] != i {
            let j = dest[i];
            elems.swap(i, j);
            dest.swap(i, j); // keep mapping consistent with the swap
        }
    }
}

/// Convenience wrapper returning a reordered Vec
pub fn reorder_numeric_to_lex<T>(mut elems: Vec<T>) -> Vec<T> {
    reorder_numeric_to_lex_in_place(&mut elems);
    elems
}

/// Read a usize from an *Int* constant.
fn as_usize_int_const(t: &Term) -> usize {
    match t.op() {
        Op::Const(v) => v.as_int().to_i64().expect("index out of range") as usize,
        _ => panic!("array index must be an Int const, got {}", t),
    }
}

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

// helper: lexicographic index order: 0,1,10,11,...,19,2,20,...
fn lex_indices(n: usize) -> Vec<usize> {
    let mut idxs: Vec<usize> = (0..n).collect();
    idxs.sort_by(|a, b| a.to_string().cmp(&b.to_string()));
    idxs
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

    building_ivc: HashSet<usize>,

    // materialized assignment for variables (provided by Relation::Witness/Instance)
    wmap: &'a HashMap<String, Value<InputValue>>, // witness (private)
    imap: &'a HashMap<String, Value<InputValue>>, // instance/public
    pub_order: &'a [String],                      // ordered public names
    prev_acc: Option<Accumulator<BlstrsEmulation>>,
    vk: Option<MidnightVK>,
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
        prev_acc: Option<Accumulator<BlstrsEmulation>>,
        vk: Option<MidnightVK>,
    ) -> Self {
        Self {
            std,
            big: std.biguint(),
            lay,
            cache: TermMap::default(),
            visited: Default::default(),
            cfg,
            used_vars,
            building_ivc: HashSet::new(),
            wmap,
            imap,
            pub_order,
            prev_acc,
            vk,
        }
    }

    /*/// If `arr: Array(Field,N)`, materialize it once in cache as `AssignedTerm::Tuple([Field; N])`.
    fn materialize_field_array_in_cache(&mut self, arr: &Term) -> Result<(), Error> {
        // Already done?
        if matches!(self.cache.get(arr), Some(AssignedTerm::Tuple(_))) {
            return Ok(());
        }

        // If this array is the result of midnight_ivc(...), materialize it directly
        if let Op::UndefinedFnCall(call) = arr.op() {
            if call.name == "midnight_ivc" {
                // This will fill self.cache.insert(arr.clone(), AssignedTerm::Tuple(...))
                self.embed_ivc(arr.clone())?;
                debug_assert!(matches!(self.cache.get(arr), Some(AssignedTerm::Tuple(_))));
                return Ok(());
            }
        }

        // Regular array-of-field materialization
        let Sort::Array(a) = check(arr) else {
            panic!("expected Array(..), got {}", check(arr));
        };
        assert!(matches!(a.val, Sort::Field(_)), "only Field[] supported");

        let mut elts = Vec::with_capacity(a.size);

        if arr.cs().len() == a.size {
            // Elements are present as children: just fetch as fields
            for i in 0..a.size {
                elts.push(AssignedTerm::Field(self.get_field(&arr.cs()[i])?.clone()));
            }
        } else {
            // IMPORTANT: do NOT call get_field on Select(arr, i) here, it would recurse.
            // Instead, delegate element access to embed_pf(Select) via a single call later.
            // Easiest: create the selects, embed them to fields *without* rematerializing `arr`.
            for i in 0..a.size {
                let idx = int_lit(rug::Integer::from(i)); // Int const
                let sel = term![Op::Select; arr.clone(), idx];
                // `embed_pf` for Select WILL NOT re-enter materialize if `arr` is not midnight_ivc
                elts.push(AssignedTerm::Field(self.get_field(&sel)?.clone()));
            }
        }

        self.cache.insert(arr.clone(), AssignedTerm::Tuple(elts));
        Ok(())
    }*/

    fn materialize_field_array_in_cache(&mut self, arr: &Term) -> Result<(), Error> {
        if matches!(self.cache.get(arr), Some(AssignedTerm::Tuple(_))) {
            return Ok(());
        }

        // Special-case: result of midnight_ivc is an array-of-fields; materialize once.
        if let Op::UndefinedFnCall(call) = arr.op() {
            if call.name == "midnight_ivc" {
                if self.building_ivc.contains(&(arr.id().0 as usize)) {
                    return Ok(());
                }
                if !self.cache.contains_key(arr) {
                    //self.embed_ivc(arr.clone())?;
                }
                // After embed_ivc, cache MUST contain a Tuple-of-fields for `arr`.
                debug_assert!(matches!(self.cache.get(arr), Some(AssignedTerm::Tuple(_))));
                return Ok(());
            }
        }

        // Regular array-of-field path
        let Sort::Array(a) = check(arr) else {
            panic!("expected Array(..), got {}", check(arr));
        };
        assert!(matches!(a.val, Sort::Field(_)), "only Field[] supported");

        let mut elts = Vec::with_capacity(a.size);
        if arr.cs().len() == a.size {
            for i in 0..a.size {
                elts.push(AssignedTerm::Field(self.get_field(&arr.cs()[i])?.clone()));
            }
        } else {
            for i in 0..a.size {
                let idx = int_lit(rug::Integer::from(i));
                let sel = term![Op::Select; arr.clone(), idx];
                elts.push(AssignedTerm::Field(self.get_field(&sel)?.clone()));
            }
        }

        self.cache.insert(arr.clone(), AssignedTerm::Tuple(elts));
        Ok(())
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
    fn flatten_fields_any(
        &mut self,
        root: &Term,
        id: Option<usize>,
    ) -> Result<Vec<AssignedNative<F>>, Error> {
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
                /*Sort::Array(arr) if matches!(arr.val, Sort::Field(_)) => {
                    if let Op::UndefinedFnCall(call) = node.op() {
                        if call.name == "midnight_ivc" {
                            let id_ = node.id().0;
                            println!("id = {}", id_);
                            if self.building_ivc.contains(&(node.id().0 as usize)) || id.is_some() {

                                // Defer: someone else is building it; when they finish,
                                // callers will re-run and see the cached tuple.
                                continue;
                            }
                            //self.embed_ivc(node.clone())?;
                            self.materialize_field_array_in_cache(&node)?;
                        } else {
                            self.materialize_field_array_in_cache(&node)?;
                        }
                    } else {
                        self.materialize_field_array_in_cache(&node)?;
                    }

                    if let Some(AssignedTerm::Tuple(elts)) = self.cache.get(&node) {
                        // push reversed so pop preserves order
                        for e in elts.iter().rev() {
                            let AssignedTerm::Field(f) = e else {
                                panic!("non-field in field[]")
                            };
                            out.push(f.clone());
                        }
                        continue;
                    }
                }*/
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

    fn dump_cs<F: halo2curves::ff::Field>(
        label: &str,
        cs: &midnight_proofs::plonk::ConstraintSystem<F>,
    ) {
        let nb_perm_chunks =
            (cs.permutation().columns.len().saturating_sub(1) / cs.degree().saturating_sub(2)) + 1;

        println!("=== {label} ===");
        println!("degree                : {}", cs.degree());
        println!(
            "advice/fixed/inst     : {}/{}/{}",
            cs.num_advice_columns(),
            cs.num_fixed_columns(),
            cs.num_instance_columns()
        );
        println!("selectors             : {}", cs.num_selectors());
        println!(
            "perm columns/chunks   : {}/{}",
            cs.permutation().columns.len(),
            nb_perm_chunks
        );
        println!("lookups               : {}", cs.lookups().len());
        println!("minimum_rows          : {}", cs.minimum_rows());
    }

    // ----------------------------------------
    // Embedding: variables, consts, ops
    // ----------------------------------------

    /// Materialize the tuple/array-of-fields returned by `midnight_ivc(...)`
    /// and cache it as `AssignedTerm::Tuple(AssignedTerm::Field(..), ...)` on `c`.
    fn embed_ivc(&mut self, c: Term) -> Result<(), Error> {
        if !self.building_ivc.insert(c.id().0 as usize) {
            // already in progress somewhere up the call stack
            return Ok(());
        }

        if self.cache.contains_key(&c) {
            return Ok(());
        }
        let id = c.id().0 as usize;

        // Also mark as visited right now so generic embed() won’t try to re-embed it
        self.visited.borrow_mut().insert(c.clone());

        //println!("embed_ivc: building id={}", c.id()); // real work
        // Return type must be tuple/array of fields
        match check(&c) {
            Sort::Array(arr) => {
                assert!(
                    matches!(arr.val, Sort::Field(_)),
                    "midnight_ivc must return field[]"
                );
            }
            other => panic!(
                "midnight_ivc must return tuple/array of fields, got {}",
                other
            ),
        }

        // ---- Parse args ------------------------------------------------------
        let args = c.cs();
        assert!(args.len() == 5, "midnight_ivc expects 5 args");

        // 0) vk : Field
        let vk_field = self.get_field(&args[0])?.clone();
        println!("vk_field {:?}", vk_field.clone().value());

        // 1) is_genesis : Bool/Field(0/1) → bit
        let is_genesis = self.get_bit(&args[1])?.clone();
        println!("is_genesis {:?}", is_genesis.clone().value());
        let is_not_genesis = self.std.not(self.lay, &is_genesis)?;

        // 2) prev_state : Field
        let prev_state = self.get_field(&args[2])?.clone();
        println!("prev_state {:?}", prev_state.clone().value());

        // 3) prev_acc : field[ACC_SIZE]
        let prev_acc_fields: Vec<AssignedNative<F>> = self.flatten_fields_any(&args[3], None)?;

        //let prev_acc_fields: Vec<AssignedNative<F>> = reorder_lex_to_numeric(prev_acc_fields);
        //for e in &prev_acc_fields {
        //   println!("e  {:?}", e.value());
        //}
        /*assert!(
            !prev_acc_fields.is_empty(),
            "midnight_ivc: prev_acc must be non-empty"
        );*/

        // 4) prev_proof : u8[PROOF_SIZE]
        let proof_bytes_v: Vec<AssignedByte<F>> = self.collect_bytes(&args[4])?;

        //println!(
        //    "proof bytes circuit : {:?}",
        //    proof_bytes_v.iter().map(|e| e.value()).collect::<Vec<_>>()
        //);

        //let proof_bytes_v: Vec<AssignedByte<F>> = reorder_lex_to_numeric(proof_bytes_v);
        let proof_bytes_val: Value<Vec<u8>> = proof_bytes_v.iter().map(|b| b.value()).collect();

        //println!("proof bytes {:?}", proof_bytes_val);

        // ---- Build a local verifier context (same as before) -----------------
        use midnight_circuits::verifier::AssignedVk;
        use midnight_proofs::{plonk::ConstraintSystem, poly::EvaluationDomain};

        let mut raw_cs: ConstraintSystem<F> = ConstraintSystem::default();

        let mut arch = m::ZkStdLibArch::default();
        arch.verifier = true;
        arch.jubjub = false;
        arch.poseidon = true;
        arch.sha256 = false;
        arch.sha512 = false;
        arch.secp256k1 = false;
        arch.bls12_381 = false;
        arch.base64 = false;
        arch.nr_pow2range_cols = 1; // value does not matter, overridden
        arch.automaton = false;
        m::ZkStdLib::configure(&mut raw_cs, arch);

        Self::dump_cs("embed ivc config", &raw_cs);

        let (cs_no_selectors, _) =
            raw_cs
                .clone()
                .directly_convert_selectors_to_fixed(vec![vec![false]; raw_cs.num_selectors()]);
        /*let cs_no_selectors = if let Some(vk) = &self.vk {
            vk.vk().cs()
        } else {
            &cs_no_selectors
        };*/

        let domain = EvaluationDomain::new(raw_cs.degree() as u32, 19);

        let self_vk_name = "self_vk";

        let assigned_vk = if let Some(vk) = &self.vk {
            //let vk_repr: midnight_proofs::circuit::AssignedCell<F, F> = self
            //    .std
            //    .assign(self.lay, vk_field)?; //Value::known(vk.vk().transcript_repr()))?;

            println!("using SET key {:?}", vk_field.clone().value());
            //println!("domain = {:?}", vk.vk().get_domain().clone());
            //println!("cs = {:?}", vk.vk().cs().clone());
            //println!("cs_no_selectors = {:?}", cs_no_selectors.clone());
            AssignedVk {
                vk_name: self_vk_name.to_string(),
                //domain: domain.clone(),
                //cs: cs_no_selectors.clone(),
                domain: vk.vk().get_domain().clone(),
                cs: vk.vk().cs().clone(),
                transcript_repr: vk_field,
            }
        } else {
            AssignedVk {
                vk_name: self_vk_name.to_string(),
                domain: domain.clone(),
                cs: cs_no_selectors.clone(),
                transcript_repr: vk_field,
            }
        };

        println!("vk_repr circ = {:?}", assigned_vk.transcript_repr.value());

        // Identity point for committed-instance binding
        let id_point = self.std.verifier_identity_point(self.lay)?;

        let prev_acc = if let Some(prev_acc) = &self.prev_acc {
            //println!(
            //    "prev acc circuit {:?}",
            //    AssignedAccumulator::as_public_input(&prev_acc)
            //);
            Value::known(prev_acc.clone())
        } else {
            Value::unknown()
        };

        let prev_acc_assigned: AssignedAccumulator<BlstrsEmulation> =
            self.std.verifier_assign_accumulator_from_witness(
                self.lay,
                self_vk_name,
                &assigned_vk.cs, // <-- processed, selector-free CS
                prev_acc,        // <-- IMPORTANT: do NOT try to build it from concrete F's
            )?;

        let mut fixed_base_names = vec![String::from("com_instance")];
        fixed_base_names.extend(verifier::fixed_base_names::<BlstrsEmulation>(
            self_vk_name,
            raw_cs.num_fixed_columns() + raw_cs.num_selectors(),
            raw_cs.permutation().columns.len(),
        ));
        println!("fixed_bases_circ = {:?}", fixed_base_names);

        let prev_acc_pi_enc = self
            .std
            .verifier()
            .as_public_input(self.lay, &prev_acc_assigned)?;
        let eqs: Vec<AssignedBit<F>> = prev_acc_pi_enc
            .iter()
            .zip(prev_acc_fields.iter())
            .map(|(a, b)| self.std.is_equal(self.lay, a, b))
            .collect::<Result<Vec<_>, _>>()?;
        let all = self.nary_and(&eqs)?;
        self.std.assert_true(self.lay, &all)?;

        // Build PI vector: [vk_pub, prev_state, prev_acc_pub...]
        // TODO take the PI from the IR
        let mut pi: Vec<AssignedNative<F>> = Vec::new();

        //pi.extend(prev_acc_fields.clone())
        //let order = lex_indices(prev_acc_fields.len() as usize);
        //let prev_acc_fields_lex: Vec<AssignedNative<F>> =
        //    order.iter().map(|&i| prev_acc_fields[i].clone()).collect();
        pi.extend(
            self.std
                .verifier()
                .as_public_input(self.lay, &prev_acc_assigned)
                .unwrap(),
        ); //prev_acc_fields);
        pi.push(prev_state.clone());
        pi.push(assigned_vk.transcript_repr.clone());
        println!("Pi vk {:?}", assigned_vk.transcript_repr.clone().value());

        println!(
            "circuit pis = {:?}",
            pi.iter().map(|e| e.value()).collect::<Vec<_>>().clone()
        );
        println!("circuit proof = {:?}", proof_bytes_val.clone());

        // ---- Partial verify + accumulate ------------------------------------
        let mut proof_acc = self.std.verifier_prepare_partial_plonk(
            self.lay,
            &assigned_vk,
            &[("com_instance", id_point.clone())],
            &[&pi],
            proof_bytes_val,
        )?;

        // For genesis: scale proof_acc by 0 (we already computed is_not_genesis)
        self.std
            .accumulator_scale_by_bit(self.lay, &is_not_genesis, &mut proof_acc)?;
        self.std.collapse_accumulator(self.lay, &mut proof_acc)?;

        /*println!(
            "prev acc fields (len = {}) {:?}",
            prev_acc_fields.len(),
            prev_acc_fields
        );*/
        // Link the provided prev_acc (as witness) to its expected PI encoding
        //let prev_acc_f: Vec<F> = prev_acc_fields
        //    .iter()
        //    .map(|e| e.value().into_option().unwrap().clone())
        ///    .collect();
        // let prev_acc_pi_enc =
        //    AssignedAccumulator::<BlstrsEmulation>::from_public_input(prev_acc_f, 1);
        //let mut prev_acc = self.std.verifier_assign_accumulator_from_witness(
        //    self.lay,
        //   self_vk_name,
        //   &assigned_vk.cs,
        //   Value::known(prev_acc_pi_enc),
        //)?;
        //let prev_acc_vec_val: Value<Vec<F>> =
        //    prev_acc_fields.iter().map(|a| a.value().copied()).collect();

        // 2) Turn that into a host witness Accumulator (batch = 1).
        //let prev_acc_witness: Value<Accumulator<BlstrsEmulation>> = prev_acc_vec_val
        //    .map(|fs| AssignedAccumulator::<BlstrsEmulation>::from_public_input(fs, 1));

        /*println!(
            "cs fixed cols {}, num selectors {}, perm cols len {}",
            cs.num_fixed_columns(),
            cs.num_selectors(),
            cs.permutation().columns.len()
        );*/
        // println!("ivc fixed bases len = {}", fixed_base_names.len());

        // Accumulate and collapse to next_acc
        let mut next_acc = self
            .std
            .accumulate(self.lay, &[proof_acc, prev_acc_assigned])?;
        self.std.collapse_accumulator(self.lay, &mut next_acc)?;

        println!("collapse accu for node");
        // Expose as tuple-of-fields
        let next_acc_field_elts = self.std.verifier().as_public_input(self.lay, &next_acc)?;
        //let next_acc_elts_lex: Vec<AssignedNative<F>> =
        //    order.iter().map(|&i| next_acc_field_elts[i].clone()).collect();

        let tuple: Vec<AssignedTerm> = next_acc_field_elts
            .clone()
            .into_iter()
            .map(AssignedTerm::Field)
            .collect();

        println!(
            "next_acc {:?}",
            next_acc_field_elts
                .iter()
                .map(|e| e.value())
                .collect::<Vec<_>>()
        );

        self.cache.insert(c.clone(), AssignedTerm::Tuple(tuple));
        Ok(())
    }

    fn embed_var(&mut self, var: &Term, ty: VarType) -> Result<(), Error> {
        //if self.cache.contains_key(var) {
        //    return Ok(());
        //}
        assert!(
            !self.cache.contains_key(var),
            "already have var {}",
            var.op()
        );
        assert!(!matches!(ty, VarType::CWit), "Unimplemented");
        if !self.used_vars.contains(var.as_var_name()) {
            return Ok(());
        }
        //if !self.used_vars.contains(var.as_var_name()) {
        //    return Ok(()); // dead var skip
        //}
        // Only skip dead *witness* vars. Keep all public vars.
        //if !matches!(ty, VarType::Inst) && !self.used_vars.contains(var.as_var_name()) {
        //    return Ok(());
        // }
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
            Sort::Array(a) => {
                // Only support Field[] here (extend as needed)
                if !matches!(a.val, Sort::Field(_)) {
                    panic!(
                        "embed_var: only Array(Field, _) supported for '{}', got {}",
                        name, v.sort
                    );
                }

                // Build element-by-element from host inputs: "<name>.<idx>"
                let mut elts: Vec<AssignedTerm> = Vec::with_capacity(a.size);
                for i in 0..a.size {
                    let sub = format!("{}.{}", name, i);

                    // Pull from witness map first, else instance map, else default 0
                    let vv = self
                        .wmap
                        .get(&sub)
                        .or_else(|| self.imap.get(&sub))
                        .cloned()
                        .unwrap_or(Value::known(InputValue::Field(self.f_zero())));

                    // Convert InputValue -> F and assign
                    let fv: Value<F> = vv.map(|iv| iv.as_field_or_default());
                    let x = self.std.assign(self.lay, fv)?;
                    if is_public {
                        self.std.constrain_as_public_input(self.lay, &x)?;
                    }

                    elts.push(AssignedTerm::Field(x));
                }

                // Cache the whole array as a tuple of fields so later code can flatten it
                self.cache.insert(var.clone(), AssignedTerm::Tuple(elts));
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

            // NEW: always catch tuple/array-returning calls up-front
            if let Op::UndefinedFnCall(call) = c.op() {
                if call.name == "midnight_ivc" {
                    self.embed_ivc(c.clone())?;
                    //self.embed_ivc(c.clone())?;
                    self.visited.borrow_mut().insert(c);
                    continue; // already materialized as a Tuple of fields
                }
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
                Sort::Tuple(_) | Sort::Array(_) => {
                    // Still embed all argument children (they can be fields/bytes/etc.)
                    for ch in c.cs() {
                        self.embed(ch.clone())?;
                    }
                }
                Sort::Int => {}
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
                        /*Sort::Tuple(sorts) => {
                            let mut bits = Vec::with_capacity(sorts.len());
                            for i in 0..sorts.len() {
                                let ai = term![Op::Field(i); a.clone()];
                                let bi = term![Op::Field(i); b.clone()];
                                self.embed_bool(term![Op::Eq; ai.clone(), bi.clone()])?;
                                bits.push(self.get_bit(&term![Op::Eq; ai, bi]).unwrap().clone());
                            }
                            AssignedTerm::Bit(self.nary_and(&bits)?)
                        }*/
                        Sort::Tuple(sorts) => {
                            let a = &c.cs()[0];
                            let b = &c.cs()[1];

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
                Op::Select => {
                    let arr = &c.cs()[0];
                    let idx = as_usize_int_const(&c.cs()[1]); // Int const

                    if let Op::UndefinedFnCall(call) = arr.op() {
                        // normal path
                        //self.materialize_field_array_in_cache(arr)?;
                        let AssignedTerm::Tuple(elts) = self.cache.get(arr).unwrap().clone() else {
                            unreachable!()
                        };
                        let AssignedTerm::Field(f) = elts[idx].clone() else {
                            panic!("element not a Field")
                        };
                        AssignedTerm::Field(f)
                    } else {
                        // normal path
                        // self.materialize_field_array_in_cache(arr)?;
                        let AssignedTerm::Tuple(elts) = self.cache.get(arr).unwrap().clone() else {
                            unreachable!()
                        };
                        let AssignedTerm::Field(f) = elts[idx].clone() else {
                            panic!("element not a Field")
                        };
                        AssignedTerm::Field(f)
                    }
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
                                self.flatten_fields_any(&arg_terms[1], None)?;

                            // 3) Positions: Field[] (each 0/1) -> AssignedBit
                            let pos_fields: Vec<AssignedNative<F>> =
                                self.flatten_fields_any(&arg_terms[2], None)?;
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
                    /*assert!(
                        acs.len() == bcs.len(),
                        "assert_bool Eq on arrays of different lengths: {} vs {}",
                        acs.len(),
                        bcs.len()
                    );*/

                    match &arr.val {
                        // Field[] → flatten then field equality
                        Sort::Field(_) => {
                            let a_fields = self.flatten_fields_any(a, None)?;
                            let b_fields = self.flatten_fields_any(b, None)?;
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
    pub prev_acc: Option<Accumulator<BlstrsEmulation>>,
    pub vk: Option<MidnightVK>,
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
        println!("imap {:?}", imap);
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
            self.prev_acc.clone(),
            self.vk.clone(),
        );

        // 1) Declare variables
        let vars = self.cs.metadata.interactive_vars();

        let mut inst_vars = vars.instances.clone();
        inst_vars.sort_by(|a, b| natural_cmp(a.as_var_name(), b.as_var_name()));
        for v in &inst_vars {
            ctx.embed_var(v, VarType::Inst)?;
        }

        let mut wit_vars = vars.final_witnesses.clone();
        wit_vars.sort_by(|a, b| natural_cmp(a.as_var_name(), b.as_var_name()));
        for w in &wit_vars {
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
        let mut arch = m::ZkStdLibArch::default();
        arch.verifier = true;
        arch.jubjub = false;
        arch.poseidon = true;
        arch.sha256 = false; // disable SHA tables completely
        arch.secp256k1 = false;
        arch.bls12_381 = false; // keep OFF; the verifier uses the *self* version below
        arch.base64 = false;
        arch.automaton = false;
        //arch.nr_pow2range_cols = 6;     // or 7 (max); bump parallelism

        arch
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

impl<'a> IrRelation<'a> {
    pub fn set_prev_acc(&mut self, acc: Accumulator<BlstrsEmulation>) {
        self.prev_acc = Some(acc);
    }

    pub fn set_vk(&mut self, vk: MidnightVK) {
        self.vk = Some(vk);
    }
}

use std::cmp::Ordering;

fn split_numeric_suffix(s: &str) -> (&str, Option<u64>) {
    // dot form: foo.bar.12
    if let Some((pre, suf)) = s.rsplit_once('.') {
        if let Ok(n) = suf.parse::<u64>() {
            return (pre, Some(n));
        }
    }
    // bracket form: foo[12]
    if let (Some(l), true) = (s.rfind('['), s.ends_with(']')) {
        if let Ok(n) = s[l + 1..s.len() - 1].parse::<u64>() {
            return (&s[..l], Some(n));
        }
    }
    (s, None)
}

fn natural_cmp(a: &str, b: &str) -> Ordering {
    let (pa, ia) = split_numeric_suffix(a);
    let (pb, ib) = split_numeric_suffix(b);
    match pa.cmp(pb) {
        Ordering::Equal => match (ia, ib) {
            (Some(na), Some(nb)) => na.cmp(&nb), // numeric order within same prefix
            _ => a.cmp(b),                       // fallback to normal lexicographic
        },
        other => other, // different prefixes: lexicographic
    }
}

// -------------------------------------
// Builder: produce a Relation for CS
// -------------------------------------

pub fn to_midnight_relation<'a>(
    cs: &'a Computation,
    cfg: &'a CircCfg,
    prev_acc: Option<Accumulator<BlstrsEmulation>>,
    vk: Option<MidnightVK>,
) -> IrRelation<'a> {
    let mut public_names: Vec<String> = cs
        .metadata
        .interactive_vars()
        .instances
        .iter()
        .map(|t| t.as_var_name().to_owned())
        .collect();

    public_names.sort_by(|a, b| natural_cmp(a, b));

    let mut all_names = cs.metadata.ordered_input_names();
    all_names.sort_by(|a, b| natural_cmp(a, b));

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
        prev_acc,
        vk,
    }
}
