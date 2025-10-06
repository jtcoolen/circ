//! IR Evaluation

use super::{
    check, const_, extras, term, Array, BitVector, BoolNaryOp, BvBinOp, BvBinPred, BvNaryOp,
    BvUnOp, FieldToBv, FxHashMap, IntBinOp, IntBinPred, IntNaryOp, IntUnOp, Integer, Node, Op,
    PfNaryOp, PfUnOp, Sort, Term, TermMap, Value,
};
use log::trace;

use crate::cfg::cfg_or_default;
use circ_fields::{FieldT, FieldV};
use once_cell::sync::Lazy;
use rug::integer::Order;
use sha2::{Digest, Sha256};

use midnight_circuits::halo2curves::ff::Field;
use midnight_circuits::halo2curves::ff::PrimeField;
use midnight_circuits::hash::poseidon::{round_skips::PreComputedRoundCPU, PoseidonChip};
use midnight_circuits::instructions::hash::HashCPU;
use std::convert::TryInto;

type F = midnight_curves::Fq;
const WIDTH: usize = PoseidonChip::<F>::register_size();
static POSEIDON_PRE: Lazy<PreComputedRoundCPU<F>> = Lazy::new(PreComputedRoundCPU::init);

// ---------- FieldV <-> F helpers (ff 0.13) ----------

fn pf_to_f(x: &FieldV) -> F {
    // Decimal -> field element (reduced mod p). Provided by ff 0.13 PrimeField.
    F::from_str_vartime(&x.i().to_string()).expect("invalid scalar for F")
}

fn f_to_pf(fty: &FieldT, x: &F) -> FieldV {
    // Canonical little-endian bytes -> big integer -> IR FieldV
    let repr = x.to_repr(); // needs PrimeField in scope
    let n = Integer::from_digits(repr.as_ref(), Order::Lsf);
    fty.new_v(n)
}
// Recursively flatten a Value into a Vec<FieldV>, accepting:
//   - Field
//   - Tuple of Fields (how arrays are often lowered in your pipeline)
//   - Array of Fields
fn push_fields_from_value(v: &crate::ir::term::Value, out: &mut Vec<crate::ir::term::Value>) {
    use crate::ir::term::Value;
    match v {
        Value::Field(_) | Value::BitVector(_) => out.push(v.clone()),
        Value::Tuple(ts) => {
            for t in ts {
                push_fields_from_value(t, out);
            }
        }
        Value::Array(a) => {
            let iter = a.key_sort.clone().elems_iter_values().take(a.size);
            for idxval in iter {
                let el = a.select(&idxval);
                push_fields_from_value(&el, out);
            }
        }
        other => panic!(
            "push_fields_from_value: expected Field or tuple/array of Fields, got {:?}",
            other.sort()
        ),
    }
}

/// Recursively flatten a Value into a Vec<u8>, accepting:
///  - BV(8)
///  - Tuple of BV(8)
///  - Array of BV(8)
pub fn push_bytes_from_value(v: &Value, out: &mut Vec<u8>) {
    match v {
        // single byte
        Value::BitVector(bv) if bv.width() == 8 => {
            let b = bv
                .uint()
                .to_u8()
                .expect("push_bytes_from_value: BV(8) value out of range");
            out.push(b);
        }

        // tuple-of-bytes (or nested tuples)
        Value::Tuple(ts) => {
            for t in ts {
                push_bytes_from_value(t, out);
            }
        }

        // array-of-bytes
        Value::Array(a) => {
            // iterate indices in order and collect elements
            let iter = a.key_sort.clone().elems_iter_values().take(a.size);
            for idx in iter {
                let el = a.select(&idx);
                push_bytes_from_value(&el, out);
            }
        }

        // anything else is a type error for the sha/merkle APIs
        other => panic!(
            "push_bytes_from_value: expected BV(8) or tuple/array of BV(8); got {:?}",
            other.sort()
        ),
    }
}

/// Recursively the term `t`, using variable values in `h` and storing intermediate evaluations in
/// the cache `vs`.
pub fn eval_cached<'a>(
    t: &Term,
    h: &FxHashMap<String, Value>,
    vs: &'a mut TermMap<Value>,
) -> &'a Value {
    // the custom traversal (rather than [PostOrderIter]) allows us to break early based on the cache

    // (children pushed, term)
    let mut stack = vec![(false, t.clone())];
    while let Some((children_pushed, node)) = stack.pop() {
        if vs.contains_key(&node) {
            continue;
        }
        if children_pushed {
            eval_value(vs, h, node);
        } else {
            stack.push((true, node.clone()));
            for c in node.cs() {
                // vs doubles as our visited set.
                if !vs.contains_key(c) {
                    stack.push((false, c.clone()));
                }
            }
        }
    }
    vs.get(t).unwrap()
}

/// Recursively evaluate the term `t`, using variable values in `h`.
pub fn eval(t: &Term, h: &FxHashMap<String, Value>) -> Value {
    let mut vs = TermMap::<Value>::default();
    eval_cached(t, h, &mut vs).clone()
}

/// Helper function for eval function. Handles a single term
fn eval_value(vs: &mut TermMap<Value>, h: &FxHashMap<String, Value>, t: Term) -> Value {
    let args: Vec<&Value> = t.cs().iter().map(|c| vs.get(c).unwrap()).collect();
    trace!("Eval {} on {:?}", t.op(), args);
    let v = eval_op(t.op(), &args, h);
    trace!("=> {}", v);
    if let Value::Bool(false) = &v {
        trace!("term {}", t);
        for v in extras::free_variables(t.clone()) {
            trace!("  {} = {}", v, h.get(&v).unwrap());
        }
    }
    vs.insert(t, v.clone());
    v
}

/// Helper function for eval function. Handles a single op
#[allow(clippy::uninlined_format_args)]
pub fn eval_op(op: &Op, args: &[&Value], var_vals: &FxHashMap<String, Value>) -> Value {
    match op {
        Op::Var(var) => var_vals
            .get(&*var.name)
            .unwrap_or_else(|| panic!("Missing var: {} in {:?}", var.name, var_vals))
            .clone(),
        Op::Eq => Value::Bool(args[0] == args[1]),
        Op::Not => Value::Bool(!args[0].as_bool()),
        Op::Implies => Value::Bool(!args[0].as_bool() || args[1].as_bool()),
        Op::BoolNaryOp(BoolNaryOp::Or) => Value::Bool(args.iter().any(|a| a.as_bool())),
        Op::BoolNaryOp(BoolNaryOp::And) => Value::Bool(args.iter().all(|a| a.as_bool())),
        Op::BoolNaryOp(BoolNaryOp::Xor) => Value::Bool(
            args.iter()
                .map(|a| a.as_bool())
                .fold(false, std::ops::BitXor::bitxor),
        ),
        Op::BvBit(i) => Value::Bool(args[0].as_bv().uint().get_bit(*i as u32)),
        Op::BoolMaj => {
            let c0 = args[0].as_bool() as u8;
            let c1 = args[1].as_bool() as u8;
            let c2 = args[2].as_bool() as u8;
            Value::Bool(c0 + c1 + c2 > 1)
        }
        Op::BvConcat => Value::BitVector({
            let mut it = args.iter().map(|a| a.as_bv().clone());
            let f = it.next().unwrap();
            it.fold(f, BitVector::concat)
        }),
        Op::BvExtract(h, l) => Value::BitVector(args[0].as_bv().clone().extract(*h, *l)),
        Op::Const(v) => (**v).clone(),
        Op::BvBinOp(o) => Value::BitVector({
            let a = args[0].as_bv().clone();
            let b = args[1].as_bv().clone();
            match o {
                BvBinOp::Udiv => a / &b,
                BvBinOp::Urem => a % &b,
                BvBinOp::Sub => a - b,
                BvBinOp::Ashr => a.ashr(&b),
                BvBinOp::Lshr => a.lshr(&b),
                BvBinOp::Shl => a << &b,
            }
        }),
        Op::BvUnOp(o) => Value::BitVector({
            let a = args[0].as_bv().clone();
            match o {
                BvUnOp::Not => !a,
                BvUnOp::Neg => -a,
            }
        }),
        Op::BvNaryOp(o) => Value::BitVector({
            let mut xs = args.iter().map(|a| a.as_bv().clone());
            let f = xs.next().unwrap();
            xs.fold(
                f,
                match o {
                    BvNaryOp::Add => std::ops::Add::add,
                    BvNaryOp::Mul => std::ops::Mul::mul,
                    BvNaryOp::Xor => std::ops::BitXor::bitxor,
                    BvNaryOp::Or => std::ops::BitOr::bitor,
                    BvNaryOp::And => std::ops::BitAnd::bitand,
                },
            )
        }),
        Op::BvSext(w) => Value::BitVector({
            let a = args[0].as_bv().clone();
            let mask = ((Integer::from(1) << *w as u32) - 1)
                * Integer::from(a.uint().get_bit(a.width() as u32 - 1));
            BitVector::new(a.uint() | (mask << a.width() as u32), a.width() + w)
        }),
        Op::PfToBv(w) => Value::BitVector({
            let i = args[0].as_pf().i();
            if let FieldToBv::Panic = cfg_or_default().ir.field_to_bv {
                assert!(
                    (i.significant_bits() as usize) <= *w,
                    "{}",
                    "oversized input to Op::PfToBv({w})",
                );
            }
            BitVector::new(i % (Integer::from(1) << *w), *w)
        }),
        Op::BvUext(w) => Value::BitVector({
            let a = args[0].as_bv().clone();
            BitVector::new(a.uint().clone(), a.width() + w)
        }),
        Op::Ite => args[if args[0].as_bool() { 1 } else { 2 }].clone(),
        Op::BvBinPred(o) => Value::Bool({
            let a = args[0].as_bv();
            let b = args[1].as_bv();
            match o {
                BvBinPred::Sge => a.as_sint() >= b.as_sint(),
                BvBinPred::Sgt => a.as_sint() > b.as_sint(),
                BvBinPred::Sle => a.as_sint() <= b.as_sint(),
                BvBinPred::Slt => a.as_sint() < b.as_sint(),
                BvBinPred::Uge => a.uint() >= b.uint(),
                BvBinPred::Ugt => a.uint() > b.uint(),
                BvBinPred::Ule => a.uint() <= b.uint(),
                BvBinPred::Ult => a.uint() < b.uint(),
            }
        }),
        Op::BoolToBv => Value::BitVector(BitVector::new(Integer::from(args[0].as_bool()), 1)),
        Op::PfUnOp(o) => Value::Field({
            let a = args[0].as_pf().clone();
            match o {
                PfUnOp::Recip => {
                    if a.is_zero() {
                        a.ty().zero()
                    } else {
                        a.recip()
                    }
                }
                PfUnOp::Neg => -a,
            }
        }),
        Op::PfDiv => Value::Field({
            let a = args[0].as_pf().clone();
            let b = args[1].as_pf().clone();
            a * b.recip()
        }),
        Op::PfNaryOp(o) => Value::Field({
            let mut xs = args.iter().map(|a| a.as_pf().clone());
            let f = xs.next().unwrap();
            xs.fold(
                f,
                match o {
                    PfNaryOp::Add => std::ops::Add::add,
                    PfNaryOp::Mul => std::ops::Mul::mul,
                },
            )
        }),
        Op::IntBinPred(o) => Value::Bool({
            let a = args[0].as_int();
            let b = args[1].as_int();
            match o {
                IntBinPred::Ge => a >= b,
                IntBinPred::Gt => a > b,
                IntBinPred::Le => a <= b,
                IntBinPred::Lt => a < b,
            }
        }),
        Op::IntNaryOp(o) => Value::Int({
            let mut xs = args.iter().map(|a| a.as_int().clone());
            let f = xs.next().unwrap();
            xs.fold(
                f,
                match o {
                    IntNaryOp::Add => std::ops::Add::add,
                    IntNaryOp::Mul => std::ops::Mul::mul,
                },
            )
        }),

        Op::IntBinOp(o) => Value::Int({
            let a = args[0].as_int().clone();
            let b = args[1].as_int().clone();
            match o {
                IntBinOp::Sub => a - b,
                IntBinOp::Div => a / b,
                IntBinOp::Rem => a % b,
                IntBinOp::ModInv => a.invert(&b).expect("unable to find modular inverse"),
            }
        }),
        Op::IntSize => Value::BitVector(BitVector::new(
            Integer::from(args[0].as_int().significant_bits()),
            32,
        )),
        Op::IntToBv(a) => Value::BitVector(BitVector::new(args[0].as_int().clone(), *a)),
        Op::IntToPf(fty) => Value::Field(fty.new_v(args[0].as_int())),
        Op::PfToInt => Value::Int(args[0].as_pf().i()),
        Op::IntUnOp(o) => Value::Int({
            let a = args[0].as_int().clone();
            match o {
                IntUnOp::Neg => -a,
            }
        }),
        Op::UbvToPf(fty) => Value::Field(fty.new_v(args[0].as_bv().uint())),
        Op::PfChallenge(c) => Value::Field(eval_pf_challenge(&c.name, &c.field)),
        Op::Witness(_) => args[0].clone(),
        Op::PfFitsInBits(n_bits) => {
            Value::Bool(args[0].as_pf().i().signed_bits() <= *n_bits as u32)
        }
        // tuple
        Op::Tuple => Value::Tuple(args.iter().map(|a| (*a).clone()).collect()),
        Op::Field(i) => {
            let t = args[0].as_tuple();
            assert!(i < &t.len(), "{} out of bounds for {} on {:?}", i, op, args);
            t[*i].clone()
        }
        Op::Update(i) => {
            let mut t = Vec::from(args[0].as_tuple()).into_boxed_slice();
            assert!(i < &t.len(), "{} out of bounds for {} on {:?}", i, op, args);
            let e = args[1].clone();
            assert_eq!(t[*i].sort(), e.sort());
            t[*i] = e;
            Value::Tuple(t)
        }
        // array
        Op::Store => {
            let a = args[0].as_array().clone();
            let i = args[1].clone();
            let v = args[2].clone();
            Value::Array(a.store(i, v))
        }
        Op::CStore => {
            let a = args[0].as_array().clone();
            let i = args[1].clone();
            let v = args[2].clone();
            let c = args[3].as_bool();
            if c {
                Value::Array(a.store(i, v))
            } else {
                Value::Array(a)
            }
        }
        Op::Fill(f) => {
            let v = args[0].clone();
            Value::Array(Array::new(
                f.key_sort.clone(),
                Box::new(v),
                Default::default(),
                f.size,
            ))
        }
        Op::Array(a) => Value::Array(Array::from_vec(
            a.key.clone(),
            a.val.clone(),
            args.iter().cloned().cloned().collect(),
        )),
        Op::Select => {
            let a = args[0].as_array();
            let i = args[1];
            a.select(i)
        }
        Op::Map(inner_op) => {
            //  term_vecs[i] will store a vector of all the i-th index entries of the array arguments
            let mut arg_vecs: Vec<Vec<Value>> = vec![Vec::new(); args[0].as_array().size];

            for arg in args {
                let arr = arg.as_array().clone();
                let iter = match arg.sort() {
                    Sort::Array(a) => a.key.clone().elems_iter_values().take(a.size).enumerate(),
                    _ => panic!("Input type should be Array"),
                };
                for (j, jval) in iter {
                    arg_vecs[j].push(arr.select(&jval))
                }
            }
            let term = term(
                op.clone(),
                args.iter().map(|a| const_((*a).clone())).collect(),
            );
            let (mut res, iter) = match check(&term) {
                Sort::Array(a) => (
                    Array::default(a.key.clone(), &a.val, a.size),
                    a.key.clone().elems_iter_values().take(a.size).enumerate(),
                ),
                _ => panic!("Output type of map should be array"),
            };

            for (i, idxval) in iter {
                let args: Vec<&Value> = arg_vecs[i].iter().collect();
                let val = eval_op(inner_op, &args, var_vals);
                res.map.insert(idxval, val);
            }
            Value::Array(res)
        }
        Op::Rot(i) => {
            let a = args[0].as_array().clone();
            let (mut res, iter, len) = match args[0].sort() {
                Sort::Array(a) => (
                    Array::default(a.key.clone(), &a.val, a.size),
                    a.key.clone().elems_iter_values().take(a.size).enumerate(),
                    a.size,
                ),
                _ => panic!("Input type should be Array"),
            };

            // calculate new rotation amount
            let rot = *i % len;
            for (idx, idx_val) in iter {
                let w = idx_val.as_bv().width();
                let new_idx = Value::BitVector(BitVector::new(Integer::from((idx + rot) % len), w));
                let new_val = a.select(&idx_val);
                res.map.insert(new_idx, new_val);
            }
            Value::Array(res)
        }
        Op::PfToBoolTrusted => {
            let v = args[0].as_pf().i();
            assert!(v == 0 || v == 1);
            Value::Bool(v == 1)
        }
        Op::ExtOp(o) => o.eval(args),

        Op::UndefinedFnCall(call) => {
            match call.name.as_str() {
                "PlonkMul2" => {
                    assert!(
                        args.len() == 1,
                        "PlonkMul2 expects 1 argument, got {}",
                        args.len()
                    );
                    let x = args[0].as_pf().clone(); // FieldV
                    let sq = x.clone() * x.clone(); // a^2
                    Value::Field(sq + x) // a^2 + a
                }

                "midnight_poseidon" => {
                    // 1) Flatten inputs (tuple/array/field) to a linear Vec<FieldV>
                    let mut flat_vals: Vec<crate::ir::term::Value> = Vec::new();
                    for a in args {
                        push_fields_from_value(a, &mut flat_vals);
                    }
                    assert!(
                        !flat_vals.is_empty(),
                        "midnight_poseidon: need at least one Field input"
                    );

                    // 2) Ensure all fields are the same type; capture that type for the result
                    let mut fty_opt: Option<crate::ir::term::FieldT> = None;
                    let mut field_vs = Vec::with_capacity(flat_vals.len());
                    for v in flat_vals {
                        let f = v.as_pf().clone();
                        if let Some(fty) = &fty_opt {
                            assert_eq!(
                                f.ty(),
                                *fty,
                                "midnight_poseidon: mixed field types in inputs"
                            );
                        } else {
                            fty_opt = Some(f.ty());
                        }
                        field_vs.push(f);
                    }

                    let inputs_f: Vec<F> = field_vs.iter().map(pf_to_f).collect();

                    let out_pf = PoseidonChip::hash(&inputs_f);

                    // 2) Grab the IR field type from the first input (all should share the same type)
                    let fty: FieldT = field_vs[0].ty().clone();

                    // 3) Convert F -> FieldV and wrap into Value::Field
                    let out_v: FieldV = f_to_pf(&fty, &out_pf);

                    crate::ir::term::Value::Field(out_v)
                }

                "midnight_sha256" => {
                    // 1) Flatten inputs (tuple/array/field) to a linear Vec<FieldV>
                    let mut flat_vals: Vec<crate::ir::term::Value> = Vec::new();
                    for a in args {
                        push_fields_from_value(a, &mut flat_vals);
                    }
                    assert!(
                        !flat_vals.is_empty(),
                        "midnight_sha256: need at least one Field input"
                    );

                    // 2) Ensure all fields are the same type; capture that type for the result
                    let mut u8_vs = Vec::with_capacity(flat_vals.len());
                    for v in flat_vals {
                        let f = v.as_bv().clone();

                        u8_vs.push(f.uint().to_u8().unwrap());
                    }

                    let mut hasher = Sha256::new();
                    hasher.update(&u8_vs);
                    let out: [u8; 32] = hasher.finalize().into();

                    let items = out
                        .iter()
                        .map(|b| {
                            crate::ir::term::Value::BitVector(BitVector::new(Integer::from(*b), 8))
                        })
                        .collect::<Vec<Value>>();

                    // TODO check key_sort correct
                    crate::ir::term::Value::Array(Array::from_vec(
                        Sort::Field(FieldT::FBls12381), // hackish, we use bls in the prover
                        Sort::BitVector(8),
                        items,
                    ))
                }
                "midnight_hybrid_mt" => {
                    // Args: (leaf_bytes, siblings, positions)
                    assert!(
                        args.len() == 3,
                        "midnight_hybrid_mt expects 3 arguments (leaf_bytes, siblings, positions), got {}",
                        args.len()
                    );

                    // --- 1) Leaf bytes (must be exactly 32 bytes) ---
                    let mut leaf_bytes: Vec<u8> = Vec::new();
                    push_bytes_from_value(args[0], &mut leaf_bytes);
                    assert!(
                        leaf_bytes.len() == 32,
                        "midnight_hybrid_mt: leaf_bytes must be exactly 32 bytes, got {}",
                        leaf_bytes.len()
                    );

                    // SHA256(leaf_bytes)
                    let mut hasher = Sha256::new();
                    hasher.update(&leaf_bytes);
                    let digest: [u8; 32] = hasher.finalize().into();

                    // Split digest into two u128 (big-endian by 16-byte chunks).
                    // hi = digest[16..32], lo = digest[0..16], each as u128::from_be_bytes
                    let lo_u128 = u128::from_be_bytes(digest[0..16].try_into().unwrap());
                    let hi_u128 = u128::from_be_bytes(digest[16..32].try_into().unwrap());

                    let lo_f = F::from_u128(lo_u128);
                    let hi_f = F::from_u128(hi_u128);
                    let zero_f = F::ZERO;

                    // --- 2) Siblings: flatten fields (must be same field type) ---
                    let mut sib_vals: Vec<crate::ir::term::Value> = Vec::new();
                    push_fields_from_value(args[1], &mut sib_vals);
                    assert!(
                        !sib_vals.is_empty(),
                        "midnight_hybrid_mt: need at least one sibling"
                    );

                    // Capture field type for the output from the first sibling
                    let first_sib_pf = sib_vals[0].as_pf().clone();
                    let fty: FieldT = first_sib_pf.ty().clone();

                    let siblings_f: Vec<F> = sib_vals
                        .iter()
                        .map(|v| {
                            let pf = v.as_pf().clone();
                            // Ensure uniform field type
                            assert_eq!(
                                pf.ty(),
                                fty,
                                "midnight_hybrid_mt: mixed field types in siblings"
                            );
                            pf_to_f(&pf)
                        })
                        .collect();

                    // --- 3) Positions: flatten fields; must be same length as siblings; each ∈ {0,1} ---
                    let mut pos_vals: Vec<crate::ir::term::Value> = Vec::new();
                    push_fields_from_value(args[2], &mut pos_vals);
                    assert_eq!(
                        siblings_f.len(),
                        pos_vals.len(),
                        "midnight_hybrid_mt: siblings and positions length mismatch ({} vs {})",
                        siblings_f.len(),
                        pos_vals.len()
                    );

                    let positions_bits: Vec<bool> = pos_vals
                        .iter()
                        .map(|v| {
                            let pf = v.as_pf().clone();
                            // Allow positions to be any FieldT but force them to be 0 or 1 as integers.
                            let i = pf.i();
                            assert!(
                                i == 0 || i == 1,
                                "midnight_hybrid_mt: position not in {{0,1}}"
                            );
                            i == 1
                        })
                        .collect();

                    // --- 4) Compute leaf Poseidon(lo, hi, 0) ---
                    let mut acc = PoseidonChip::<F>::hash(&[lo_f, hi_f, zero_f]);

                    // --- 5) Fold along path: if pos==1 (Right) => hash(acc, sib, 0); else (Left) => hash(sib, acc, 0)
                    for (sib, is_right) in siblings_f.into_iter().zip(positions_bits.into_iter()) {
                        acc = if is_right {
                            PoseidonChip::<F>::hash(&[acc, sib, zero_f])
                        } else {
                            PoseidonChip::<F>::hash(&[sib, acc, zero_f])
                        };
                    }

                    // --- 6) Return field element with the same IR field type as siblings ---
                    let out_v: FieldV = f_to_pf(&fty, &acc);
                    Value::Field(out_v)
                }

                other => {
                    panic!("UndefinedFnCall '{}' not implemented in interpreter", other);
                }
            }
        }

        o => unimplemented!("eval: {:?}", o),
    }
}

/// Compute a (deterministic) prime-field challenge.
pub fn eval_pf_challenge(name: &str, field: &FieldT) -> FieldV {
    use rand::SeedableRng;
    use rand_chacha::ChaChaRng;
    use std::hash::{Hash, Hasher};
    // hash the string
    let mut hasher = fxhash::FxHasher::default();
    name.hash(&mut hasher);
    let hash: u64 = hasher.finish();
    // seed ChaCha with the hash
    let mut seed = [0u8; 32];
    seed[0..8].copy_from_slice(&hash.to_le_bytes());
    let mut rng = ChaChaRng::from_seed(seed);
    // sample from ChaCha
    field.random_v(&mut rng)
}
