use ark_bls12_381::Bls12_381;

/*
use bellman::gadgets::test::TestConstraintSystem;
use bellman::groth16::{
    create_random_proof, generate_parameters, generate_random_parameters, prepare_verifying_key,
    verify_proof, Parameters, Proof, VerifyingKey,
};
use bellman::Circuit;
use bls12_381::{Bls12, Scalar};
*/
use ark_bls12_381::Fr;
use circ::front::zsharp::{self, ZSharpFE};
use circ::front::{FrontEnd, Mode};
use circ::ir::opt::{opt, Opt};
use circ::ir::term::{Op, Term};
use circ::target::r1cs::wit_comp;
use circ_fields::FullFieldV::FBls12381;
use fxhash::FxHashMap;
use rug::Integer;

/*
use circ::target::r1cs::bellman::parse_instance;
*/
use circ::ir::term::*;
use circ::target::plonkish::trans::{to_plonk, PlonkConstraint, PlonkCs};
use circ::target::r1cs::opt::reduce_linearities;
use circ::target::r1cs::trans::to_r1cs;
/*
use std::fs::File;
use std::io::Read;
use std::io::Write;
*/
use circ::cfg::{
    cfg,
    clap::{self, Parser, ValueEnum},
    CircOpt,
};

use std::path::PathBuf;
use std::time::Instant;

use ark_ff::{BigInt, BigInteger, PrimeField};
use circ::target::plonkish::trans::Wire;
use circ_fields::{FieldT, FieldV};
use hyperplonk::structs::HyperPlonkParams;
use rug::integer::Order;
use std::collections::HashMap;

use rayon::prelude::*;

use ark_std::iterable::Iterable;
use std::cmp::max;
use std::collections::HashSet;

use ark_std::log2;

/// A row of selector of width `#selectors`
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SelectorRow<F: PrimeField>(pub Vec<F>);

/// A column of selectors of length `#constraints`
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct SelectorColumn<F: PrimeField>(pub Vec<F>);

impl<F: PrimeField> SelectorColumn<F> {
    /// the number of variables of the multilinear polynomial that presents a
    /// column.
    pub fn get_nv(&self) -> usize {
        log2(self.0.len()) as usize
    }

    /// Append a new element to the selector column
    pub fn append(&mut self, new_element: F) {
        self.0.push(new_element)
    }

    /// Build selector columns from rows
    pub fn from_selector_rows(selector_rows: &[SelectorRow<F>]) -> Result<Vec<Self>, String> {
        if selector_rows.is_empty() {
            return Err("empty witness rows".to_string());
        }

        let mut res = Vec::with_capacity(selector_rows.len());
        let num_colnumns = selector_rows[0].0.len();

        for i in 0..num_colnumns {
            let mut cur_column = Vec::new();
            for row in selector_rows.iter() {
                cur_column.push(row.0[i])
            }
            res.push(Self(cur_column))
        }

        Ok(res)
    }
}

/// Customized gate is a list of tuples of
///     (coefficient, selector_index, wire_indices)
///
/// Example:
///     q_L(X) * W_1(X)^5 - W_2(X) = 0
/// is represented as
/// vec![
///     ( 1,    Some(id_qL),    vec![id_W1, id_W1, id_W1, id_W1, id_W1]),
///     (-1,    None,           vec![id_W2])
/// ]
///
/// CustomizedGates {
///     gates: vec![
///         (1, Some(0), vec![0, 0, 0, 0, 0]),
///         (-1, None, vec![1])
///     ],
/// };
/// where id_qL = 0 // first selector
/// id_W1 = 0 // first witness
/// id_w2 = 1 // second witness
///
/// NOTE: here coeff is a signed integer, instead of a field element
///
/// Customized gates structure from HyperPlonk.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CustomizedGates {
    pub gates: Vec<(i64, Option<usize>, Vec<usize>)>,
}

impl CustomizedGates {
    /// The degree of the algebraic customized gate
    pub fn degree(&self) -> usize {
        let mut res = 0;
        for x in self.gates.iter() {
            res = max(res, x.2.len() + (x.1.is_some() as usize))
        }
        res
    }

    /// The number of selectors in a customized gate
    pub fn num_selector_columns(&self) -> usize {
        let mut res = 0;
        for (_coeff, q, _ws) in self.gates.iter() {
            // a same selector must not be used for multiple monomials.
            if q.is_some() {
                res += 1;
            }
        }
        res
    }

    /// The number of witnesses in a customized gate
    pub fn num_witness_columns(&self) -> usize {
        let mut res = 0;
        for (_coeff, _q, ws) in self.gates.iter() {
            // witness list must be ordered
            // so we just need to compare with the last one
            if let Some(&p) = ws.last() {
                if res < p {
                    res = p
                }
            }
        }
        // add one here because index starts from 0
        res + 1
    }

    pub fn evaluate<F: PrimeField>(&self, selectors: &[F], witness: &[F]) -> F {
        let mut res = F::zero();
        for (coeff, q, ws) in self.gates.iter() {
            let mut term = if *coeff < 0 {
                -F::from((-coeff) as u64)
            } else {
                F::from(*coeff as u64)
            };
            if let Some(selector_idx) = q {
                term *= selectors[*selector_idx];
            }
            for witness_idx in ws {
                term *= witness[*witness_idx];
            }
            res += term;
        }
        res
    }

    /// Return a vanilla plonk gate:
    /// ``` ignore
    ///   q_L w_1 + q_R w_2 + q_O w_3 + q_M w1w2 + q_C = 0
    /// ```
    /// which is
    /// ``` ignore
    ///     (1,    Some(id_qL),     vec![id_W1]),
    ///     (1,    Some(id_qR),     vec![id_W2]),
    ///     (1,    Some(id_qO),     vec![id_W3]),
    ///     (1,    Some(id_qM),     vec![id_W1, id_w2]),
    ///     (1,    Some(id_qC),     vec![]),
    /// ```
    pub fn vanilla_plonk_gate() -> Self {
        Self {
            gates: vec![
                (1, Some(0), vec![0]),
                (1, Some(1), vec![1]),
                (1, Some(2), vec![2]),
                (1, Some(3), vec![0, 1]),
                (1, Some(4), vec![]),
            ],
        }
    }

    /// Return a jellyfish turbo plonk gate:
    /// ```ignore
    ///     q_1 w_1   + q_2 w_2   + q_3 w_3   + q_4 w4
    ///   + q_M1 w1w2 + q_M2 w3w4
    ///   + q_H1 w1^5 + q_H2 w2^5 + q_H3 w3^5 + q_H4 w4^5
    ///   + q_E w1w2w3w4
    ///   + q_O w5
    ///   + q_C
    ///   = 0
    /// ```
    /// with
    /// - w = [w1, w2, w3, w4, w5]
    /// - q = [ q_1, q_2, q_3, q_4, q_M1, q_M2, q_H1, q_H2, q_H3, q_H4, q_E,
    ///   q_O, q_c ]
    ///
    /// which is
    /// ```ignore
    ///     (1,    Some(q[0]),     vec![w[0]]),
    ///     (1,    Some(q[1]),     vec![w[1]]),
    ///     (1,    Some(q[2]),     vec![w[2]]),
    ///     (1,    Some(q[3]),     vec![w[3]]),
    ///     (1,    Some(q[4]),     vec![w[0], w[1]]),
    ///     (1,    Some(q[5]),     vec![w[2], w[3]]),
    ///     (1,    Some(q[6]),     vec![w[0], w[0], w[0], w[0], w[0]]),
    ///     (1,    Some(q[7]),     vec![w[1], w[1], w[1], w[1], w[1]]),
    ///     (1,    Some(q[8]),     vec![w[2], w[2], w[2], w[2], w[2]]),
    ///     (1,    Some(q[9]),     vec![w[3], w[3], w[3], w[3], w[3]]),
    ///     (1,    Some(q[10]),    vec![w[0], w[1], w[2], w[3]]),
    ///     (1,    Some(q[11]),    vec![w[4]]),
    ///     (1,    Some(q[12]),    vec![]),
    /// ```
    pub fn jellyfish_turbo_plonk_gate() -> Self {
        CustomizedGates {
            gates: vec![
                (1, Some(0), vec![0]),
                (1, Some(1), vec![1]),
                (1, Some(2), vec![2]),
                (1, Some(3), vec![3]),
                (1, Some(4), vec![0, 1]),
                (1, Some(5), vec![2, 3]),
                (1, Some(6), vec![0, 0, 0, 0, 0]),
                (1, Some(7), vec![1, 1, 1, 1, 1]),
                (1, Some(8), vec![2, 2, 2, 2, 2]),
                (1, Some(9), vec![3, 3, 3, 3, 3]),
                (1, Some(10), vec![0, 1, 2, 3]),
                (1, Some(11), vec![4]),
                (1, Some(12), vec![]),
            ],
        }
    }

    /// Generate a random gate for `num_witness` with a highest degree =
    /// `degree`
    pub fn mock_gate(num_witness: usize, degree: usize) -> Self {
        let mut gates = vec![];

        let mut high_degree_term = vec![0; degree - 1];
        high_degree_term.push(1);

        gates.push((1, Some(0), high_degree_term));
        for i in 0..num_witness {
            gates.push((1, Some(i + 1), vec![i]))
        }
        gates.push((1, Some(num_witness + 1), vec![]));

        CustomizedGates { gates }
    }

    /// Return a plonk gate where #selector > #witness * 2
    /// ``` ignore
    ///   q_1 w_1   + q_2 w_2   + q_3 w_3   +
    ///   q_4 w1w2  + q_5 w1w3  + q_6 w2w3  +
    ///   q_7 = 0
    /// ```
    /// which is
    /// ``` ignore
    ///     (1,    Some(id_qL),     vec![id_W1]),
    ///     (1,    Some(id_qR),     vec![id_W2]),
    ///     (1,    Some(id_qO),     vec![id_W3]),
    ///     (1,    Some(id_qM),     vec![id_W1, id_w2]),
    ///     (1,    Some(id_qC),     vec![]),
    /// ```
    pub fn super_long_selector_gate() -> Self {
        Self {
            gates: vec![
                (1, Some(0), vec![0]),
                (1, Some(1), vec![1]),
                (1, Some(2), vec![2]),
                (1, Some(3), vec![0, 1]),
                (1, Some(4), vec![0, 2]),
                (1, Some(5), vec![1, 2]),
                (1, Some(6), vec![]),
            ],
        }
    }

    pub fn super_long_selector_gate_with_output() -> Self {
        Self {
            gates: vec![
                (1, Some(0), vec![0]),
                (1, Some(1), vec![1]),
                (1, Some(2), vec![2]),
                (1, Some(3), vec![0, 1]),
                (1, Some(4), vec![0, 2]),
                (1, Some(5), vec![1, 2]),
                (1, Some(6), vec![3]),
                (1, Some(7), vec![]),
            ],
        }
    }
}

// Gate structure that we use
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct GateInfo {
    pub gates: Vec<Vec<(usize, usize)>>,
    pub is_linear: Vec<bool>,
    // This describes the 'equivalence' of the variables
    // Assuming that the variables are sorted, e.g. [1, 2, 3, 4], which
    // orders should we try to cover all possiblilities? If all variables
    // are equivalent, there is only one order required ([0, 1, 2, 3])
    pub orders: Vec<Vec<usize>>,
    // (var, selector)
    pub linear_terms: Vec<(usize, usize)>,
    // For linear-only; (var1, var2, selector_1, selector_2, selector_mul)
    pub vanilla_compatibility_info: (usize, usize, usize, usize, usize),
}

impl GateInfo {
    /// The number of selectors in a customized gate
    pub fn num_selector_columns(&self) -> usize {
        // Output gate and constant gate
        self.gates.len() + 2
    }

    /// The number of witnesses in a customized gate
    pub fn num_witness_columns(&self) -> usize {
        let mut res = 0;
        for ws in self.gates.iter() {
            // witness list must be ordered
            // so we just need to compare with the last one
            if let Some(&(var, _)) = ws.last() {
                if res < var {
                    res = var
                }
            }
        }
        // add one here because index starts from 0
        // and one more here because the output is not included
        res + 2
    }

    fn next_permutation<T: Ord>(v: &mut [T]) -> bool {
        if v.len() == 1 {
            return false;
        }
        let mut i = v.len() - 1;
        while i > 0 {
            i -= 1;
            if v[i] < v[i + 1] {
                let mut j = v.len() - 1;
                while v[i] >= v[j] {
                    j -= 1;
                }
                v.swap(i, j);

                let mut low = i + 1;
                let mut high = v.len() - 1;
                while low < high {
                    v.swap(low, high);
                    low += 1;
                    high -= 1;
                }
                return true;
            }
        }
        return false;
    }

    pub fn new(gate: &CustomizedGates) -> Result<Self, String> {
        let mut gates = vec![];
        for (coeff, selector, variables) in &gate.gates {
            if *coeff != 1 {
                return Err("Non-1 coeff is currently unsupported".to_string());
            }
            if let Some(selector_idx) = selector {
                if *selector_idx != gates.len() {
                    return Err("Some selector indices appear to be skipped".to_string());
                }
                let mut out_gate = vec![];
                for i in 0..variables.len() {
                    if i == 0 || variables[i] != variables[i - 1] {
                        out_gate.push((variables[i], 0usize));
                    }
                    out_gate.last_mut().unwrap().1 += 1;
                }
                gates.push(out_gate);
            } else {
                return Err("Missing selector is currently unsupported".to_string());
            }
        }
        if !gates.last().unwrap().is_empty() {
            return Err("Missing constant term".to_string());
        }
        let output_term = gates[gates.len() - 2].clone();
        if output_term.len() != 1
            || output_term[0].1 != 1
            || output_term[0].0 != gate.num_witness_columns() - 1
        {
            return Err("Output term is not in proper form".to_string());
        }
        gates.truncate(gates.len() - 2);

        let is_linear = gates
            .iter()
            .map(|gate| gate.len() == 1 && gate[0].1 == 1)
            .collect::<Vec<_>>();
        let linear_terms = gates
            .iter()
            .enumerate()
            .flat_map(|(selector_idx, gate)| {
                if gate.len() == 1 && gate[0].1 == 1 {
                    Some((gate[0].0, selector_idx))
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();
        let var_a = linear_terms[0].0;
        let var_b = linear_terms[1].0;
        let mul_selector = gates
            .iter()
            .position(|gate| *gate == vec![(var_a, 1), (var_b, 1)])
            .ok_or("Failed to find multiplication term".to_string())?;
        let vanilla_compatibility_info = (
            var_a,
            var_b,
            linear_terms[0].1,
            linear_terms[1].1,
            mul_selector,
        );

        let mut perm = (0..(gate.num_witness_columns() - 1)).collect::<Vec<_>>();
        let mut orders = vec![];
        let mut effective_gates_set = HashSet::new();
        loop {
            let mut effective_gates = gates
                .iter()
                .map(|gate| {
                    let mut new_gate = gate
                        .iter()
                        .flat_map(|(var, power)| {
                            if *var == perm.len() {
                                None
                            } else {
                                Some((perm[*var], *power))
                            }
                        })
                        .collect::<Vec<_>>();
                    new_gate.sort();
                    new_gate
                })
                .collect::<Vec<_>>();
            effective_gates.sort();
            if effective_gates_set.insert(effective_gates) {
                orders.push(perm.clone());
            }

            if !Self::next_permutation(&mut perm) {
                break;
            }
        }

        Ok(GateInfo {
            gates,
            is_linear,
            orders,
            linear_terms,
            vanilla_compatibility_info,
        })
    }

    pub fn jellyfish_turbo_plonk_gate() -> Self {
        Self {
            gates: vec![
                (vec![(0, 1)]),
                (vec![(1, 1)]),
                (vec![(2, 1)]),
                (vec![(3, 1)]),
                (vec![(0, 1), (1, 1)]),
                (vec![(2, 1), (3, 1)]),
                (vec![(0, 5)]),
                (vec![(1, 5)]),
                (vec![(2, 5)]),
                (vec![(3, 5)]),
                (vec![(0, 1), (1, 1), (2, 1), (3, 1)]),
            ],
            is_linear: vec![
                true, true, true, true, false, false, false, false, false, false, false,
            ],
            // gate_priority: vec![6, 7, 8, 9, 10, 4, 5, 0, 1, 2, 3],
            orders: vec![vec![0, 1, 2, 3], vec![0, 2, 1, 3], vec![1, 2, 0, 3]],
            linear_terms: vec![(0, 0), (1, 1), (2, 2), (3, 3)],
            vanilla_compatibility_info: (0, 1, 0, 1, 4),
        }
    }

    pub fn evaluate_no_output<F: PrimeField>(
        &self,
        selectors: &[F],
        witness: &[F],
        variables: &[usize],
    ) -> F {
        self.gates
            .iter()
            .zip(selectors.iter())
            .map(|(gate, selector)| {
                if selector.is_zero() {
                    F::zero()
                } else {
                    gate.iter()
                        .map(|(idx, power)| witness[variables[*idx]].pow([*power as u64]))
                        .product::<F>()
                        * selector
                }
            })
            .sum::<F>()
            + selectors.last().unwrap()
    }

    pub fn evaluate<F: PrimeField>(
        &self,
        selectors: &[F],
        witness: &[F],
        variables: &[usize],
    ) -> F {
        let selector_len = selectors.len();
        self.gates
            .iter()
            .zip(selectors.iter())
            .map(|(gate, selector)| {
                if selector.is_zero() {
                    F::zero()
                } else {
                    gate.iter()
                        .map(|(idx, power)| witness[variables[*idx]].pow([*power as u64]))
                        .product::<F>()
                        * selector
                }
            })
            .sum::<F>()
            + selectors[selector_len - 2] * witness[*variables.last().unwrap()]
            + selectors.last().unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gate_info() {
        let generated_gate_info =
            GateInfo::new(&CustomizedGates::jellyfish_turbo_plonk_gate()).unwrap();
        let expected_gate_info = GateInfo {
            gates: vec![
                (vec![(0, 1)]),
                (vec![(1, 1)]),
                (vec![(2, 1)]),
                (vec![(3, 1)]),
                (vec![(0, 1), (1, 1)]),
                (vec![(2, 1), (3, 1)]),
                (vec![(0, 5)]),
                (vec![(1, 5)]),
                (vec![(2, 5)]),
                (vec![(3, 5)]),
                (vec![(0, 1), (1, 1), (2, 1), (3, 1)]),
            ],
            is_linear: vec![
                true, true, true, true, false, false, false, false, false, false, false,
            ],
            orders: vec![vec![0, 1, 2, 3], vec![0, 2, 1, 3], vec![0, 3, 1, 2]],
            linear_terms: vec![(0, 0), (1, 1), (2, 2), (3, 3)],
            vanilla_compatibility_info: (0, 1, 0, 1, 4),
        };
        assert_eq!(generated_gate_info, expected_gate_info);
    }

    #[test]
    fn test_gate_info_2() {
        let generated_gate_info =
            GateInfo::new(&CustomizedGates::super_long_selector_gate_with_output()).unwrap();
        let expected_gate_info = GateInfo {
            gates: vec![
                (vec![(0, 1)]),
                (vec![(1, 1)]),
                (vec![(2, 1)]),
                (vec![(0, 1), (1, 1)]),
                (vec![(0, 1), (2, 1)]),
                (vec![(1, 1), (2, 1)]),
            ],
            is_linear: vec![true, true, true, false, false, false],
            orders: vec![vec![0, 1, 2]],
            linear_terms: vec![(0, 0), (1, 1), (2, 2)],
            vanilla_compatibility_info: (0, 1, 0, 1, 3),
        };
        assert_eq!(generated_gate_info, expected_gate_info);
    }
}

/// The HyperPlonk instance parameters, consists of the following:
///   - the number of constraints
///   - number of public input columns
///   - the customized gate function
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PlonkishCircuitParams {
    /// the number of constraints for gate_func
    pub num_constraints: usize,
    /// number of public input
    // public input is only 1 column and is implicitly the first witness column.
    // this size must not exceed number of total constraints.
    // Beware that public input must be wired to regular gates. If public input
    // needs to be wired to lookup gates an no-op regular gate is necessary
    pub num_pub_input: usize,
    /// customized gate function
    pub gate_func: CustomizedGates,
}

/// The HyperPlonk index, consists of the following:
///   - HyperPlonk parameters
///   - the wire permutation
///   - the selector vectors
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PlonkishCircuit<F: PrimeField> {
    // HyperPlonkIndex struct in Hyperplonk repository
    pub params: PlonkishCircuitParams, // called HyperPlonkParams in Hyperplonk repository
    pub permutation: Vec<F>,
    pub selectors: Vec<SelectorColumn<F>>,
}

// HyperPlonk circuit is
// pub struct MockCircuit<F: PrimeField> {
//     pub public_inputs: Vec<F>,
//     pub witnesses: Vec<WitnessColumn<F>>,
//     pub index: HyperPlonkIndex<F>,
// }

impl<F: PrimeField> PlonkishCircuit<F> {
    fn witness_row(&self, values: &[F], idx: usize) -> Vec<F> {
        let mut witness_values = vec![F::zero(); self.params.gate_func.num_witness_columns()];
        for i in 0..witness_values.len() {
            witness_values[i] = values[i * self.params.num_constraints + idx];
        }
        witness_values
    }

    fn selector_row(&self, idx: usize) -> Vec<F> {
        let mut selector_values = vec![F::zero(); self.selectors.len()];
        for (i, column) in self.selectors.iter().enumerate() {
            selector_values[i] = column.0[idx];
        }
        selector_values
    }

    pub fn is_satisfied(&self, values: &[F]) -> bool {
        //println!("\n\nvalues = {:?}\n\n", values);
        //println!("\n\nselectors = {:?}\n\n", self.selectors);

        let gate_constraint = (0..self.params.num_constraints).into_par_iter().all(|i| {
            let res = self
                .params
                .gate_func
                .evaluate(&self.selector_row(i), &self.witness_row(values, i))
                == F::zero();
            //println!("selectors = {:?}", self.selector_row(i));
            //println!("wits = {:?}", self.witness_row(values, i));
            //println!("res = {}", res);
            res
        });
        if !gate_constraint {
            return false;
        }

        let wiring_constraint = (0..self.permutation.len()).into_par_iter().all(|i| {
            let next_idx_val = self.permutation[i].into_bigint();
            let next_idx = next_idx_val.as_ref();
            if !next_idx.iter().skip(1).all(|&e| e == 0) {
                return false;
            }
            if values[i] != values[next_idx[0] as usize] {
                println!("i = {}, next_idx = {}", i, next_idx[0]);
                return false;
            }
            true
        });
        wiring_constraint
    }
}

/// Maps a traditional Plonk constraint system to HyperPlonk circuit representation
pub struct PlonkToHyperPlonkMapper<F: PrimeField> {
    wire_to_index: HashMap<Wire, usize>,
    next_witness_index: usize,
    _marker: std::marker::PhantomData<F>,
    plonk_cs: PlonkCs,
    memo: std::collections::HashMap<Term, F>,
    var_vals: FxHashMap<String, F>,
    terms: FxHashMap<Var, Term>,
    precompute: precomp::PreComp,
    inputs: FxHashMap<String, Value>,
}

impl<F: PrimeField> PlonkToHyperPlonkMapper<F> {
    pub fn new(plonk_cs: PlonkCs) -> Self {
        let wire_values = plonk_cs.wire_values.clone();
        // extract terms: HashMap<Var, Term> from wire_values
        let terms = wire_values
            .iter()
            .filter_map(|(_, term)| {
                if let Op::Var(var) = term.op() {
                    println!("var {:?} = {:?}", var.name, term);
                    Some((var.as_ref().clone(), term.clone()))
                } else {
                    None
                }
            })
            .collect::<FxHashMap<_, _>>();
        println!("public inputs = {:?}", plonk_cs.public_inputs);
        println!("wits = {:?}", plonk_cs.witness);
        let mut rng = rand::thread_rng();
        let input_names = plonk_cs.all_inputs.clone();
        let mut inputs = FxHashMap::<String, Value>::default();
        // add (x, 1) and (return, 3) to inputs
        inputs.insert(
            "x".to_string(),
            Value::Field(FieldV::new_ty(1i64, FieldT::FBls12381)),
        );
        inputs.insert(
            "return".to_string(),
            Value::Field(FieldV::new_ty(3i64, FieldT::FBls12381)),
        );
        // Map input_names to random Value in the inputs FxHashMap<String, Value>
        /*for input_name in input_names {

            let random_value = Value::Field(FieldV::random(FieldT::FBls12381, &mut rng));
            inputs.insert(input_name, random_value);
        }*/
        println!("terms = {:?}", terms.clone().into_values());
        Self {
            wire_to_index: HashMap::new(),
            next_witness_index: 0,
            _marker: std::marker::PhantomData,
            plonk_cs,
            memo: std::collections::HashMap::new(),
            var_vals: FxHashMap::default(),
            terms,
            precompute: precomp::PreComp::new(),
            inputs,
        }
    }

    /// Convert PlonkCs to PlonkishCircuit
    pub fn convert(&mut self) -> Result<PlonkishCircuit<F>, String> {
        // Step 1: Build wire index mapping
        let plonk_cs_clone = self.plonk_cs.clone();
        self.build_wire_mapping(&plonk_cs_clone)?;

        // Step 2: Create selector columns from constraints
        let selector_columns = self.create_selector_columns(&self.plonk_cs)?;

        // Step 3: Create permutation vector from copy constraints
        let permutation = self.create_permutation_vector(&self.plonk_cs)?;

        // Step 4: Create circuit parameters
        let params = PlonkishCircuitParams {
            num_constraints: self.plonk_cs.constraints.len(),
            num_pub_input: self.plonk_cs.public_inputs.len(),
            gate_func: CustomizedGates::vanilla_plonk_gate(), // Standard Plonk gate
        };

        Ok(PlonkishCircuit {
            params,
            permutation,
            selectors: selector_columns,
        })
    }

    /// Build mapping from Wire to witness column index
    fn build_wire_mapping(&mut self, plonk_cs: &PlonkCs) -> Result<(), String> {
        // Collect all unique wires
        let mut all_wires = std::collections::HashSet::new();

        // Add public input wires
        for wire in &plonk_cs.public_inputs {
            all_wires.insert(wire.clone());
        }

        // Add witness wires
        for wire in &plonk_cs.witness {
            all_wires.insert(wire.clone());
        }

        // Add wires from constraints
        for constraint in &plonk_cs.constraints {
            all_wires.insert(constraint.a.clone());
            all_wires.insert(constraint.b.clone());
            all_wires.insert(constraint.c.clone());
        }

        // Create mapping (public inputs come first)
        let mut index = 0;

        // Map public inputs first
        for wire in &plonk_cs.public_inputs {
            self.wire_to_index.insert(wire.clone(), index);
            index += 1;
        }

        // Map remaining wires
        for wire in all_wires {
            if !self.wire_to_index.contains_key(&wire) {
                self.wire_to_index.insert(wire, index);
                index += 1;
            }
        }

        self.next_witness_index = index;
        println!("next_witness_index = {}", index);
        Ok(())
    }

    /// Create selector columns from Plonk constraints
    fn create_selector_columns(
        &self,
        plonk_cs: &PlonkCs,
    ) -> Result<Vec<SelectorColumn<F>>, String> {
        let num_constraints = plonk_cs.constraints.len();

        // Initialize selector columns: [q_L, q_R, q_O, q_M, q_C]
        let mut selector_columns = vec![
            SelectorColumn::default(), // q_L
            SelectorColumn::default(), // q_R
            SelectorColumn::default(), // q_O
            SelectorColumn::default(), // q_M
            SelectorColumn::default(), // q_C
        ];

        // Fill selector columns from constraints
        for constraint in &plonk_cs.constraints {
            // Convert FieldV values to F (assuming you have a conversion method)
            let q_l = self.field_v_to_f(&constraint.q_l)?;
            let q_r = self.field_v_to_f(&constraint.q_r)?;
            let q_o = self.field_v_to_f(&constraint.q_o)?;
            let q_m = self.field_v_to_f(&constraint.q_m)?;
            let q_c = self.field_v_to_f(&constraint.q_c)?;

            selector_columns[0].append(q_l);
            selector_columns[1].append(q_r);
            selector_columns[2].append(q_o);
            selector_columns[3].append(q_m);
            selector_columns[4].append(q_c);
        }

        Ok(selector_columns)
    }

    /// Create permutation vector from copy constraints
    /*fn create_permutation_vector(&self, plonk_cs: &PlonkCs) -> Result<Vec<F>, String> {
        let total_vars = 3 * plonk_cs.constraints.len();
        let mut permutation = Vec::with_capacity(total_vars);

        // Initialize permutation as identity
        for i in 0..total_vars {
            permutation.push(F::from(i as u64));
        }

        println!("#copy constraints = {}", plonk_cs.copy_constraints.len());
        // Apply copy constraints to create cycles in permutation
        for copy_constraint in &plonk_cs.copy_constraints {
            let wire1_idx = self
                .wire_to_index
                .get(&copy_constraint.wire1)
                .ok_or("Wire not found in mapping")?;
            let wire2_idx = self
                .wire_to_index
                .get(&copy_constraint.wire2)
                .ok_or("Wire not found in mapping")?;

            // For each constraint row, create permutation cycle
            for row in 0..plonk_cs.constraints.len() {
                let var1_global = wire1_idx * plonk_cs.constraints.len() + row;
                let var2_global = wire2_idx * plonk_cs.constraints.len() + row;

                // Create cycle: var1 -> var2 -> ... -> var1
                let temp = permutation[var1_global];
                permutation[var1_global] = permutation[var2_global];
                permutation[var2_global] = temp;
            }
        }

        println!("#permutation = {}", permutation.len());
        panic!();
        Ok(permutation)
    }*/

    pub fn create_permutation_vector(&self, plonk_cs: &PlonkCs) -> Result<Vec<F>, String> {
        // Step 1: Build the witness table
        let mut witness_table = Vec::new();
        let mut wire_to_indices = HashMap::new();

        for (i, constraint) in plonk_cs.constraints.iter().enumerate() {
            // Add (a, b, c) to the witness table
            witness_table.push((
                constraint.a.clone(),
                constraint.b.clone(),
                constraint.c.clone(),
            ));

            // Map wire IDs to their positions in the witness table
            wire_to_indices.insert(constraint.a.index, i * 3);
            wire_to_indices.insert(constraint.b.index, i * 3 + 1);
            wire_to_indices.insert(constraint.c.index, i * 3 + 2);
        }

        // Step 2: Build the permutation vector
        let mut permutation = (0..(3 * plonk_cs.constraints.len()))
            .map(|i| F::from(i as u64))
            .collect::<Vec<_>>();

        for copy_constraint in plonk_cs.copy_constraints.clone() {
            let idx1 = *wire_to_indices.get(&copy_constraint.wire1.index).unwrap();
            let idx2 = *wire_to_indices.get(&copy_constraint.wire2.index).unwrap();

            // Link the two indices in the permutation vector
            permutation[idx1] = F::from(idx2 as u64);
            permutation[idx2] = F::from(idx1 as u64);
        }

        Ok(permutation)
    }

    /// Helper to convert FieldV to F (you'll need to implement this based on your types)
    fn field_v_to_f(&self, field_v: &FieldV) -> Result<F, String> {
        // This is a placeholder - you'll need to implement the actual conversion
        // based on how your FieldV type relates to the PrimeField F
        F::from_bigint(F::BigInt::from_bits_be(
            field_v.i().to_digits(Order::MsfBe).as_slice(),
        ))
        .ok_or_else(|| "Failed to convert FieldV to F".to_string())
        //todo!("Implement conversion from FieldV to F based on your type system")
    }

    /// Get the witness column index for a wire
    pub fn get_wire_index(&self, wire: &Wire) -> Option<usize> {
        self.wire_to_index.get(wire).copied()
    }

    /// Create witness values matrix from PlonkCs
    pub fn create_witness_values(&mut self) -> Result<Vec<F>, String> {
        let num_constraints = self.plonk_cs.constraints.clone().len();
        //let num_witnesses = self.next_witness_index.clone();
        println!("num_witnesses: {}", self.next_witness_index);
        println!("num_constraints = {}", self.plonk_cs.constraints.len());
        let mut values = vec![F::zero(); 3 * num_constraints];

        let vars: HashMap<Var, FieldV> = self.eval_all_vars(&self.inputs);
        for (var, field_v) in vars {
            let field_f = self.field_v_to_f(&field_v)?;
            self.var_vals.insert(var.name.to_string(), field_f);
        }

        // Fill witness values
        for (row_idx, constraint) in self.plonk_cs.constraints.clone().into_iter().enumerate() {
            let a_term = self
                .plonk_cs
                .wire_values
                .get(&constraint.a)
                .ok_or("Wire a value not found")?;
            let a_term = a_term.clone();
            // Get values from wire_values map and convert Term to F
            let a_val = self.term_to_f(&a_term)?;

            let b_term = self
                .plonk_cs
                .wire_values
                .get(&constraint.b)
                .ok_or("Wire b value not found")?;
            let b_term = b_term.clone();
            let b_val = self.term_to_f(&b_term)?;

            let c_term = self
                .plonk_cs
                .wire_values
                .get(&constraint.c)
                .ok_or("Wire c value not found")?;
            let c_term = c_term.clone();
            let c_val = self.term_to_f(&c_term)?;

            let a_idx = self
                .wire_to_index
                .get(&constraint.a)
                .ok_or("Wire a not found")?;

            let b_idx = self
                .wire_to_index
                .get(&constraint.b)
                .ok_or("Wire b not found")?;

            let c_idx = self
                .wire_to_index
                .get(&constraint.c)
                .ok_or("Wire c not found")?;

            // Check constraint holds
            let q_l = self.field_v_to_f(&constraint.q_l)?;
            let q_r = self.field_v_to_f(&constraint.q_r)?;
            let q_o = self.field_v_to_f(&constraint.q_o)?;
            let q_m = self.field_v_to_f(&constraint.q_m)?;
            let q_c = self.field_v_to_f(&constraint.q_c)?;

            //println!("row idx {} , q_o = {:?}, {}", row_idx, q_o, q_o.to_string());
            let constraint_value =
                q_l * a_val + q_r * b_val + q_o * c_val + q_m * a_val * b_val + q_c;

            /*println!(
                "\n\nSelector values: q_l = {:?}, q_r = {:?}, q_o = {:?}, q_m = {:?}, q_c = {:?}\n\n",
                q_l, q_r, q_o, q_m, q_c
            );
            println!(
                "\n\nWitness values: a_val = {:?}, b_val = {:?}, c_val = {:?}\n\n",
                a_val, b_val, c_val
            );*/
            if constraint_value != F::zero() {
                println!(
                    "Constraint computation: q_l * a_val = {:?}, q_r * b_val = {:?}, q_o * c_val = {:?}, q_m * a_val * b_val = {:?}, q_c = {:?}",
                    q_l * a_val,
                    q_r * b_val,
                    q_o * c_val,
                    q_m * a_val * b_val,
                    q_c
                );
                println!(
                    "row index {} out of {} = {}",
                    row_idx,
                    self.plonk_cs.constraints.len(),
                    constraint_value
                );
                return Err(format!("Constraint not satisfied at row {}", row_idx));
            }

            // Store in column-major format
            values[0 * num_constraints + row_idx] = a_val;
            values[1 * num_constraints + row_idx] = b_val;
            values[2 * num_constraints + row_idx] = c_val;
        }
        Ok(values)
    }

    /// Helper to convert Term to F (you'll need to implement this)
    /// Convert a Term to field value F by evaluating it

    // Add witness_values field to your struct
    // Helper function to convert field element to Integer
    fn field_to_integer(&self, field_val: &F) -> Integer {
        // Convert field element to bytes then to Integer
        let mut bytes = Vec::new();
        field_val.serialize_compressed(&mut bytes).unwrap();
        Integer::from_digits(&bytes, rug::integer::Order::Lsf)
    }

    // Helper function to convert Integer to field element
    fn integer_to_field(&self, int_val: &Integer) -> Result<F, String> {
        // Convert Integer to bytes then to field element
        let mut bytes = int_val.to_digits(rug::integer::Order::Lsf);
        let field_byte_size = (F::MODULUS_BIT_SIZE + 7) / 8; // Calculate the expected byte size
        if bytes.len() < field_byte_size as usize {
            bytes.resize(field_byte_size as usize, 0); // Pad with zeros
        }
        let res = F::deserialize_compressed(&bytes[..])
            .map_err(|e| format!("Failed to convert Integer to field: {:?}", e));
        //println!("int = {}, bytes = {:?}, is_ok = {}", int_val, bytes, res.is_ok());
        res
    }

    fn eval_all_vars(&self, inputs: &FxHashMap<String, Value>) -> HashMap<Var, FieldV> {
        let after_precompute = self.precompute.eval(inputs);
        let mut cache = Default::default();
        self.terms
            .iter()
            .map(|(var, term)| {
                let val = eval_cached(term, &after_precompute, &mut cache);
                if let Value::Field(f) = val {
                    (var.clone(), f.clone())
                } else {
                    panic!("Non-field");
                }
            })
            .collect()
    }

    // Memoization map to store computed results
    // Use Precomp module instead with supplied circuit's inputs/outputs variables assignments
    fn term_to_f(&mut self, term: &Term) -> Result<F, String> {
        if let Some(cached_result) = self.memo.get(term) {
            return Ok(*cached_result);
        }

        let result = match term.op() {
            // Direct field constant
            Op::Const(boxed_val) => {
                match boxed_val.as_ref() {
                    Value::Field(field_val) => Ok(self.field_v_to_f(field_val)?),
                    Value::Bool(b) => Ok(if *b { F::one() } else { F::zero() }),
                    Value::Int(i) => {
                        // Convert to Integer first, then to field
                        let int_val = Integer::from(i.clone());
                        self.integer_to_field(&int_val)
                    }
                    Value::BitVector(bv) => {
                        // Convert bitvector to Integer then to field element
                        let uint_val = Integer::from(bv.uint().clone());
                        self.integer_to_field(&uint_val)
                    }
                    _ => Err(format!("Unsupported constant type: {:?}", boxed_val)),
                }
            }

            // Variable lookup
            Op::Var(var) => {
                /*if let Some(assigned_value) = self.plonk_cs.witness.iter().find(|wire| {
                    wire.name.starts_with(&*var.name)
                        && wire.name[var.name.len()..].starts_with("_n")
                        && wire.name[var.name.len() + 2..]
                            .chars()
                            .filter(|c| c.is_digit(10))
                            .count()
                            >= 1
                }) {
                    let t = self.plonk_cs.wire_values[&assigned_value].clone();
                    println!("var name {}, {:?}", var.name, t);
                    //Err(format!("No assignment for variable: {}", var.name))
                    //Ok(self.term_to_f(&t)?)
                    panic!("")
                } else {
                    Err(format!("No assignment for variable: {}", var.name))
                }*/
                if let Some(value) = self.var_vals.get(&var.as_ref().name.as_ref().to_string()) {
                    Ok(*value)
                } else {
                    Err(format!("No assignment for variable: {}", var.name))
                }
                //todo!("Implement variable lookup for Term: {}", var.name)
            }

            // If-then-else
            Op::Ite => {
                let children = term.cs();
                if children.len() != 3 {
                    return Err("Ite expects 3 operands".to_string());
                }
                let condition = self.term_to_f(&children[0])?;
                let then_val = self.term_to_f(&children[1])?;
                let else_val = self.term_to_f(&children[2])?;

                // If condition is non-zero, use then_val, else use else_val
                Ok(if condition != F::zero() {
                    then_val
                } else {
                    else_val
                })
            }

            // Equality comparison
            Op::Eq => {
                let children = term.cs();
                if children.len() != 2 {
                    return Err("Eq expects 2 operands".to_string());
                }
                let left = self.term_to_f(&children[0])?;
                let right = self.term_to_f(&children[1])?;
                Ok(if left == right { F::one() } else { F::zero() })
            }

            // Bit-vector binary operators
            Op::BvBinOp(bop) => {
                let children = term.cs();
                if children.len() != 2 {
                    return Err(format!("{:?} expects 2 operands", bop));
                }
                let left = self.term_to_f(&children[0])?;
                let right = self.term_to_f(&children[1])?;

                // Convert to Integer for arbitrary precision operations
                let left_int = self.field_to_integer(&left);
                let right_int = self.field_to_integer(&right);

                let result = match bop {
                    BvBinOp::Sub => left_int - &right_int,
                    BvBinOp::Udiv => {
                        if right_int == 0 {
                            return Err("Division by zero".to_string());
                        }
                        left_int / &right_int
                    }
                    BvBinOp::Urem => {
                        if right_int == 0 {
                            return Err("Modulo by zero".to_string());
                        }
                        left_int % &right_int
                    }
                    BvBinOp::Shl => {
                        let shift_amount = right_int.to_u32().unwrap_or(0);
                        left_int << shift_amount
                    }
                    BvBinOp::Lshr => {
                        let shift_amount = right_int.to_u32().unwrap_or(0);
                        left_int >> shift_amount
                    }
                    _ => return Err(format!("Unsupported BvBinOp: {:?}", bop)),
                };
                self.integer_to_field(&result)
            }

            // Bit-vector binary predicates
            Op::BvBinPred(pred) => {
                let children = term.cs();
                if children.len() != 2 {
                    return Err(format!("{:?} expects 2 operands", pred));
                }
                let left = self.term_to_f(&children[0])?;
                let right = self.term_to_f(&children[1])?;

                let left_int = self.field_to_integer(&left);
                let right_int = self.field_to_integer(&right);

                let result = match pred {
                    BvBinPred::Ult => left_int < right_int,
                    BvBinPred::Ule => left_int <= right_int,
                    BvBinPred::Ugt => left_int > right_int,
                    BvBinPred::Uge => left_int >= right_int,
                    BvBinPred::Slt => {
                        // For signed comparison, we need to handle the sign bit properly
                        // This is a simplified version - you may need more sophisticated handling
                        left_int.cmp(&right_int) == std::cmp::Ordering::Less
                    }
                    BvBinPred::Sle => left_int <= right_int,
                    BvBinPred::Sgt => left_int > right_int,
                    BvBinPred::Sge => left_int >= right_int,
                    _ => return Err(format!("Unsupported BvBinPred: {:?}", pred)),
                };
                Ok(if result { F::one() } else { F::zero() })
            }

            // Bit-vector n-ary operators
            Op::BvNaryOp(nop) => {
                let children = term.cs();
                match nop {
                    BvNaryOp::Add => {
                        let mut result = Integer::new();
                        for child in children {
                            let val = self.term_to_f(child)?;
                            let val_int = self.field_to_integer(&val);
                            result += val_int;
                        }
                        self.integer_to_field(&result)
                    }
                    BvNaryOp::Mul => {
                        let mut result = Integer::from(1);
                        for child in children {
                            let val = self.term_to_f(child)?;
                            let val_int = self.field_to_integer(&val);
                            result *= val_int;
                        }
                        self.integer_to_field(&result)
                    }
                    BvNaryOp::Or => {
                        let mut result = Integer::new();
                        for child in children {
                            let val = self.term_to_f(child)?;
                            let val_int = self.field_to_integer(&val);
                            result |= val_int;
                        }
                        self.integer_to_field(&result)
                    }
                    BvNaryOp::And => {
                        let mut result = Integer::from(-1); // All bits set
                        for child in children {
                            let val = self.term_to_f(child)?;
                            let val_int = self.field_to_integer(&val);
                            result &= val_int;
                        }
                        self.integer_to_field(&result)
                    }
                    BvNaryOp::Xor => {
                        let mut result = Integer::new();
                        for child in children {
                            let val = self.term_to_f(child)?;
                            let val_int = self.field_to_integer(&val);
                            result ^= val_int;
                        }
                        self.integer_to_field(&result)
                    }
                    _ => Err(format!("Unsupported BvNaryOp: {:?}", nop)),
                }
            }

            // Bit-vector unary operators
            Op::BvUnOp(uop) => {
                let children = term.cs();
                if children.len() != 1 {
                    return Err(format!("{:?} expects 1 operand", uop));
                }
                let operand = self.term_to_f(&children[0])?;
                let operand_int = self.field_to_integer(&operand);

                let result = match uop {
                    BvUnOp::Not => !operand_int,
                    BvUnOp::Neg => -operand_int,
                    _ => return Err(format!("Unsupported BvUnOp: {:?}", uop)),
                };
                self.integer_to_field(&result)
            }

            // Boolean to bit-vector conversion
            Op::BoolToBv => {
                let children = term.cs();
                if children.len() != 1 {
                    return Err("BoolToBv expects 1 operand".to_string());
                }
                let operand = self.term_to_f(&children[0])?;
                Ok(if operand != F::zero() {
                    F::one()
                } else {
                    F::zero()
                })
            }

            // Bit extraction
            Op::BvExtract(high, low) => {
                let children = term.cs();
                if children.len() != 1 {
                    return Err("BvExtract expects 1 operand".to_string());
                }
                let operand = self.term_to_f(&children[0])?;
                let operand_int = self.field_to_integer(&operand);

                let width = high - low + 1;
                let mask = (Integer::from(1) << width) - 1;
                let extracted = (operand_int >> low) & mask;
                self.integer_to_field(&extracted)
            }

            // Bit-vector concatenation
            Op::BvConcat => {
                let children = term.cs();
                if children.is_empty() {
                    return Err("BvConcat expects at least 1 operand".to_string());
                }

                let mut result = Integer::new();
                let mut shift = 0u32;

                // Process children from right to left (low-order to high-order)
                for child in children.iter().rev() {
                    let val = self.term_to_f(child)?;
                    let val_int = self.field_to_integer(&val);
                    result |= val_int << shift;
                    // Assume each child contributes some number of bits (you may need to track this)
                    shift += 8; // This is a simplification - you'd need actual bit widths
                }
                self.integer_to_field(&result)
            }

            // Zero extension
            Op::BvUext(_bits) => {
                let children = term.cs();
                if children.len() != 1 {
                    return Err("BvUext expects 1 operand".to_string());
                }
                // Zero extension is essentially a no-op for witness computation
                self.term_to_f(&children[0])
            }

            // Sign extension
            Op::BvSext(_bits) => {
                let children = term.cs();
                if children.len() != 1 {
                    return Err("BvSext expects 1 operand".to_string());
                }
                let operand = self.term_to_f(&children[0])?;
                // For simplicity, treat as no-op (proper sign extension would need bit width info)
                Ok(operand)
            }

            // Prime field to bit-vector
            Op::PfToBv(_width) => {
                let children = term.cs();
                if children.len() != 1 {
                    return Err("PfToBv expects 1 operand".to_string());
                }
                // For witness computation, this is essentially a no-op
                self.term_to_f(&children[0])
            }

            // Boolean implication
            Op::Implies => {
                let children = term.cs();
                if children.len() != 2 {
                    return Err("Implies expects 2 operands".to_string());
                }
                let left = self.term_to_f(&children[0])?;
                let right = self.term_to_f(&children[1])?;

                let result = (left == F::zero()) || (right != F::zero());
                Ok(if result { F::one() } else { F::zero() })
            }

            // Boolean n-ary operations
            Op::BoolNaryOp(nop) => {
                let children = term.cs();
                match nop {
                    BoolNaryOp::And => {
                        for child in children {
                            let val = self.term_to_f(child)?;
                            if val == F::zero() {
                                return Ok(F::zero());
                            }
                        }
                        Ok(F::one())
                    }
                    BoolNaryOp::Or => {
                        for child in children {
                            let val = self.term_to_f(child)?;
                            if val != F::zero() {
                                return Ok(F::one());
                            }
                        }
                        Ok(F::zero())
                    }
                    BoolNaryOp::Xor => {
                        let mut result = false;
                        for child in children {
                            let val = self.term_to_f(child)?;
                            result ^= val != F::zero();
                        }
                        Ok(if result { F::one() } else { F::zero() })
                    }
                    _ => Err(format!("Unsupported BoolNaryOp: {:?}", nop)),
                }
            }

            // Boolean not
            Op::Not => {
                let children = term.cs();
                if children.len() != 1 {
                    return Err("Not expects 1 operand".to_string());
                }
                let operand = self.term_to_f(&children[0])?;
                Ok(if operand == F::zero() {
                    F::one()
                } else {
                    F::zero()
                })
            }

            // Get bit from bit-vector
            Op::BvBit(bit_index) => {
                let children = term.cs();
                if children.len() != 1 {
                    return Err("BvBit expects 1 operand".to_string());
                }
                let operand = self.term_to_f(&children[0])?;
                let operand_int = self.field_to_integer(&operand);
                let bit = (operand_int >> bit_index) & 1;
                self.integer_to_field(&bit)
            }

            // Boolean majority
            Op::BoolMaj => {
                let children = term.cs();
                if children.len() != 3 {
                    return Err("BoolMaj expects 3 operands".to_string());
                }
                let a = self.term_to_f(&children[0])? != F::zero();
                let b = self.term_to_f(&children[1])? != F::zero();
                let c = self.term_to_f(&children[2])? != F::zero();

                let result = (a && b) || (a && c) || (b && c);
                Ok(if result { F::one() } else { F::zero() })
            }

            // Prime field unary operations
            Op::PfUnOp(unop) => {
                let children = term.cs();
                if children.len() != 1 {
                    return Err(format!("{:?} expects 1 operand", unop));
                }
                let operand = self.term_to_f(&children[0])?;

                match unop {
                    PfUnOp::Neg => Ok(-operand),
                    PfUnOp::Recip => {
                        if operand == F::zero() {
                            Err("Division by zero in reciprocal".to_string())
                        } else {
                            Ok(operand.inverse().unwrap())
                        }
                    }
                    _ => Err(format!("Unsupported PfUnOp: {:?}", unop)),
                }
            }

            // Prime field n-ary operations
            Op::PfNaryOp(nop) => {
                let children = term.cs();
                match nop {
                    PfNaryOp::Add => {
                        let mut result = F::zero();
                        for child in children {
                            result += self.term_to_f(child)?;
                        }
                        Ok(result)
                    }
                    PfNaryOp::Mul => {
                        let mut result = F::one();
                        for child in children {
                            result *= self.term_to_f(child)?;
                        }
                        Ok(result)
                    }
                    _ => Err(format!("Unsupported PfNaryOp: {:?}", nop)),
                }
            }

            // Unsigned bit-vector to prime field
            Op::UbvToPf(_field) => {
                let children = term.cs();
                if children.len() != 1 {
                    return Err("UbvToPf expects 1 operand".to_string());
                }
                // Convert and take modulus - for simplicity, just pass through
                self.term_to_f(&children[0])
            }

            // Prime field challenge
            Op::PfChallenge(challenge_op) => {
                // In witness computation, this would typically be a predetermined value
                // You'd need to have challenges pre-computed or use a deterministic method
                /*if let Some(challenge_value) = self.challenges.get(&challenge_op.name) {
                    Ok(*challenge_value)
                } else {
                    Err(format!("No challenge value for: {}", challenge_op.name))
                }*/
                todo!("Implement challenge lookup for challenge")
            }

            // Prime field fits in bits check
            Op::PfFitsInBits(bits) => {
                let children = term.cs();
                if children.len() != 1 {
                    return Err("PfFitsInBits expects 1 operand".to_string());
                }
                let operand = self.term_to_f(&children[0])?;

                // Check if the field element fits in the specified number of bits
                let max_val = (Integer::from(1) << bits) - 1;
                let operand_int = self.field_to_integer(&operand);
                Ok(if operand_int <= max_val {
                    F::one()
                } else {
                    F::zero()
                })
            }

            // Prime field division
            Op::PfDiv => {
                let children = term.cs();
                if children.len() != 2 {
                    return Err("PfDiv expects 2 operands".to_string());
                }
                let numerator = self.term_to_f(&children[0])?;
                let denominator = self.term_to_f(&children[1])?;

                if denominator == F::zero() {
                    Err("Division by zero in PfDiv".to_string())
                } else {
                    Ok(numerator * denominator.inverse().unwrap())
                }
            }

            // Witness value
            Op::Witness(name) => {
                /*if let Some(witness_value) = self.witness_values.get(name.as_ref()) {
                    Ok(*witness_value)
                } else {
                    Err(format!("No witness value for: {}", name))
                }*/
                todo!("Implement witness value lookup for witness: {}", name)
            }

            // Integer operations
            Op::IntBinPred(pred) => {
                let children = term.cs();
                if children.len() != 2 {
                    return Err(format!("{:?} expects 2 operands", pred));
                }
                let left = self.term_to_f(&children[0])?;
                let right = self.term_to_f(&children[1])?;

                let left_int = self.field_to_integer(&left);
                let right_int = self.field_to_integer(&right);

                let result = match pred {
                    IntBinPred::Lt => left_int < right_int,
                    IntBinPred::Le => left_int <= right_int,
                    IntBinPred::Gt => left_int > right_int,
                    IntBinPred::Ge => left_int >= right_int,
                    _ => return Err(format!("Unsupported IntBinPred: {:?}", pred)),
                };
                Ok(if result { F::one() } else { F::zero() })
            }

            Op::IntNaryOp(nop) => {
                let children = term.cs();
                match nop {
                    IntNaryOp::Add => {
                        let mut result = Integer::new();
                        for child in children {
                            let val = self.term_to_f(child)?;
                            let val_int = self.field_to_integer(&val);
                            result += val_int;
                        }
                        self.integer_to_field(&result)
                    }
                    IntNaryOp::Mul => {
                        let mut result = Integer::from(1);
                        for child in children {
                            let val = self.term_to_f(child)?;
                            let val_int = self.field_to_integer(&val);
                            result *= val_int;
                        }
                        self.integer_to_field(&result)
                    }
                    _ => Err(format!("Unsupported IntNaryOp: {:?}", nop)),
                }
            }

            Op::IntBinOp(iop) => {
                let children = term.cs();
                if children.len() != 2 {
                    return Err(format!("{:?} expects 2 operands", iop));
                }
                let left = self.term_to_f(&children[0])?;
                let right = self.term_to_f(&children[1])?;

                let left_int = self.field_to_integer(&left);
                let right_int = self.field_to_integer(&right);

                let result = match iop {
                    IntBinOp::Sub => left_int - &right_int,
                    IntBinOp::Div => {
                        if right_int == 0 {
                            return Err("Division by zero".to_string());
                        }
                        left_int / &right_int
                    }
                    IntBinOp::Rem => {
                        if right_int == 0 {
                            return Err("Modulo by zero".to_string());
                        }
                        left_int % &right_int
                    }
                    IntBinOp::ModInv => {
                        if right_int == 0 {
                            return Err("Division by zero in ModInv".to_string());
                        }
                        // Compute modular inverse using Extended Euclidean algorithm
                        let gcd = left_int.clone().gcd(&right_int);
                        if gcd != 1 {
                            return Err("No modular inverse exists".to_string());
                        }
                        left_int.invert(&right_int).unwrap()
                        // Should not happen due to gcd check
                    }
                    _ => return Err(format!("Unsupported IntBinOp: {:?}", iop)),
                };
                self.integer_to_field(&result)
            }

            Op::IntUnOp(uop) => {
                let children = term.cs();
                if children.len() != 1 {
                    return Err(format!("{:?} expects 1 operand", uop));
                }
                let operand = self.term_to_f(&children[0])?;
                let operand_int = self.field_to_integer(&operand);

                let result = match uop {
                    IntUnOp::Neg => -operand_int,
                    // Note: IntUnOp::Abs doesn't exist, removed it
                    _ => return Err(format!("Unsupported IntUnOp: {:?}", uop)),
                };
                self.integer_to_field(&result)
            }

            // Integer to bit-vector conversion
            Op::IntToBv(width) => {
                let children = term.cs();
                if children.len() != 1 {
                    return Err("IntToBv expects 1 operand".to_string());
                }
                let operand = self.term_to_f(&children[0])?;
                let operand_int = self.field_to_integer(&operand);
                // Mask to the specified width
                let mask = (Integer::from(1) << width) - 1;
                let result = operand_int & mask;
                self.integer_to_field(&result)
            }

            // Integer to prime field
            Op::IntToPf(_field) => {
                let children = term.cs();
                if children.len() != 1 {
                    return Err("IntToPf expects 1 operand".to_string());
                }
                // Direct conversion
                self.term_to_f(&children[0])
            }

            // Prime field to integer
            Op::PfToInt => {
                let children = term.cs();
                if children.len() != 1 {
                    return Err("PfToInt expects 1 operand".to_string());
                }
                // Direct conversion
                self.term_to_f(&children[0])
            }

            // Prime field to boolean (trusted)
            Op::PfToBoolTrusted => {
                let children = term.cs();
                if children.len() != 1 {
                    return Err("PfToBoolTrusted expects 1 operand".to_string());
                }
                let operand = self.term_to_f(&children[0])?;
                // Assume the field element is 0 or 1
                Ok(operand)
            }

            // Floating point operations (simplified - you may need proper IEEE 754 handling)
            Op::FpBinOp(_)
            | Op::FpBinPred(_)
            | Op::FpUnPred(_)
            | Op::FpUnOp(_)
            | Op::BvToFp
            | Op::UbvToFp(_)
            | Op::SbvToFp(_)
            | Op::FpToFp(_) => {
                Err("Floating point operations not implemented in witness computation".to_string())
            }

            // Array operations (simplified)
            Op::Select | Op::Store | Op::CStore | Op::Fill(_) | Op::Array(_) => {
                Err("Array operations not implemented in witness computation".to_string())
            }

            // Tuple operations
            Op::Tuple | Op::Field(_) | Op::Update(_) => {
                Err("Tuple operations not implemented in witness computation".to_string())
            }

            // Map operation
            Op::Map(_) => Err("Map operation not implemented in witness computation".to_string()),

            // Function calls
            Op::Call(_) => Err("Function calls not implemented in witness computation".to_string()),

            // Array rotation
            Op::Rot(_) => Err("Array rotation not implemented in witness computation".to_string()),

            // Extension operations
            Op::ExtOp(_) => {
                Err("Extension operations not supported in witness computation".to_string())
            }

            // Integer size
            Op::IntSize => {
                let children = term.cs();
                if children.len() != 1 {
                    return Err("IntSize expects 1 operand".to_string());
                }
                // Return some measure of the integer size - this is domain-specific
                Ok(F::from(64u64)) // Assuming 64-bit integers
            }
        };

        if let Ok(value) = result {
            self.memo.insert(term.clone(), value);
        }

        result
    }

    /*fn term_to_f(&self, term: &Term) -> Result<F, String> {
        match term.op() {
            // Direct field constant
            Op::Const(boxed_val) => {
                match boxed_val.as_ref() {
                    Value::Field(field_val) => {
                        // Convert FieldV to your F type
                        Ok(self.field_v_to_f(field_val)?)
                    }
                    Value::Bool(b) => {
                        Ok(if *b { F::one() } else { F::zero() })
                    }
                    Value::Int(i) => {
                        // Convert integer to field element
                        Ok(F::from(i.to_u64().ok_or("Integer too large")?))
                    }
                    _ => Err(format!("Unsupported constant type: {:?}", boxed_val))
                }
            }

            // Variable lookup - need to evaluate recursively or from assignment
            Op::Var(var) => {
                // If you have variable assignments stored somewhere
                /*if let Some(assigned_value) = self.variable_assignments.get(&var.name) {
                    Ok(*assigned_value)
                } else {
                    Err(format!("No assignment for variable: {}", var.name))
                }*/
                todo!("unimplemented")
            }

            // Arithmetic operations - evaluate recursively
            &PF_ADD => {
                let children = term.cs();
                if children.len() != 2 {
                    return Err("PfAdd expects 2 operands".to_string());
                }
                let left = self.term_to_f(&children[0])?;
                let right = self.term_to_f(&children[1])?;
                Ok(left + right)
            }

            &PF_MUL => {
                let children = term.cs();
                if children.len() != 2 {
                    return Err("PfMul expects 2 operands".to_string());
                }
                let left = self.term_to_f(&children[0])?;
                let right = self.term_to_f(&children[1])?;
                Ok(left * right)
            }

            Op::PfUnOp(PfUnOp::Neg) => {
                let children = term.cs();
                if children.len() != 1 {
                    return Err("PfNeg expects 1 operand".to_string());
                }
                let operand = self.term_to_f(&children[0])?;
                Ok(-operand)
            }

            Op::Ite => {
                let children = term.cs();
                if children.len() != 3 {
                    return Err("Ite expects 3 operands".to_string());
                }
                let condition = self.term_to_f(&children[0])?;
                let then_val = self.term_to_f(&children[1])?;
                let else_val = self.term_to_f(&children[2])?;

                // If condition is non-zero, use then_val, else use else_val
                Ok(if condition != F::zero() { then_val } else { else_val })
            }

            _ => Err(format!("Unsupported term operation: {:?}", term.op()))
        }
    }*/
}

/// Helper function to create a complete HyperPlonk circuit from PlonkCs
pub fn plonk_to_hyperplonk<F: PrimeField>(
    plonk_cs: PlonkCs,
) -> Result<(PlonkishCircuit<F>, Vec<F>), String> {
    let mut mapper = PlonkToHyperPlonkMapper::new(plonk_cs);
    let circuit = mapper.convert()?;
    let witness_values = mapper.create_witness_values()?;

    Ok((circuit, witness_values))
}

fn convert_selectors(
    selectors: Vec<SelectorColumn<Fr>>,
) -> Vec<hyperplonk::selectors::SelectorColumn<Fr>> {
    use ark_std::Zero;
    selectors
        .into_iter()
        .map(|mut s| {
            // Calculate the next power of two
            let next_power_of_two = s.0.len().next_power_of_two();

            // Pad with zeros if necessary
            if s.0.len() < next_power_of_two {
                s.0.resize(next_power_of_two, Fr::zero());
            }

            hyperplonk::selectors::SelectorColumn(s.0)
        })
        .collect()
}

fn convert_gates(gates: CustomizedGates) -> hyperplonk::custom_gate::CustomizedGates {
    hyperplonk::custom_gate::CustomizedGates { gates: gates.gates }
}

fn convert_params(params: PlonkishCircuitParams) -> hyperplonk::structs::HyperPlonkParams {
    hyperplonk::structs::HyperPlonkParams {
        num_constraints: params.num_constraints,
        num_pub_input: params.num_pub_input,
        gate_func: convert_gates(params.gate_func),
    }
}

//=======

#[derive(Debug, Parser)]
#[command(name = "zxc", about = "CirC: the circuit compiler")]
struct Options {
    /// Input file
    #[arg(name = "PATH")]
    path: PathBuf,

    /*
    #[arg(long, default_value = "P", parse(from_os_str))]
    prover_key: PathBuf,

    #[arg(long, default_value = "V", parse(from_os_str))]
    verifier_key: PathBuf,

    #[arg(long, default_value = "pi", parse(from_os_str))]
    proof: PathBuf,

    #[arg(long, default_value = "x", parse(from_os_str))]
    instance: PathBuf,
    */
    #[arg(short = 'L')]
    /// skip linearity reduction entirely
    skip_linred: bool,

    #[command(flatten)]
    /// CirC options
    circ: CircOpt,

    #[arg(long, default_value = "count")]
    action: ProofAction,

    #[arg(short = 'q')]
    /// quiet mode: don't print R1CS at the end
    quiet: bool,
}

#[derive(PartialEq, Eq, Debug, Clone, ValueEnum)]
enum ProofAction {
    Count,
    Setup,
    Prove,
    Verify,
}

#[derive(PartialEq, Debug, Clone, ValueEnum)]
enum ProofOption {
    Count,
    Prove,
}

// ====

use ark_ff::Field;

use hyperplonk::witness;

#[derive(Parser)]
#[command(version, about, long_about = None)]
struct Cli {
    /// Optimization level
    #[arg(short = 'O', default_value_t = 1, value_parser = clap::value_parser!(u8).range(..3))]
    optimize: u8,

    /// Whether to use jellyfish turboplonk gates
    #[arg(long)]
    general: bool,

    /// R1CS circuit file (e.g. circuit.r1cs)
    circuit: String,

    /// JSON witness file (e.g. witness.json)
    witness: String,
}

pub fn pad_permutation_field<F: PrimeField>(
    mut permutation: Vec<F>,
    num_rows: usize,
    padding: usize,
    expected_length: usize,
) -> Vec<F> {
    let mut new_permutation = permutation;
    let mut current_offset = 0;

    println!("original permutation length = {}", new_permutation.len());

    let mut chunk_start = 0;
    while chunk_start + num_rows <= new_permutation.len() {
        let insert_at = chunk_start + num_rows + current_offset;
        println!(
            "inserting at {}, new_permutation.len() = {}",
            insert_at,
            new_permutation.len()
        );

        // Insert padding block
        for i in 0..padding {
            new_permutation.insert(insert_at + i, F::zero()); // placeholder
        }

        // Shift all values >= insert_at by `padding`
        for val in new_permutation.iter_mut() {
            let v = val.into_bigint().as_ref()[0] as usize;
            if v >= insert_at {
                *val = F::from((v + padding) as u64);
            }
        }

        // Fix the inserted padding block into a cycle
        for i in 0..padding {
            let idx = insert_at + i;
            let next = insert_at + ((i + 1) % padding);
            new_permutation[idx] = F::from(next as u64);
        }

        current_offset += padding;
        chunk_start += num_rows;
    }

    // Final padding if needed
    let final_pad_start = new_permutation.len();
    while new_permutation.len() < expected_length {
        new_permutation.push(F::zero());
    }
    let final_pad_indices: Vec<usize> = (final_pad_start..expected_length).collect();
    for (i, &idx) in final_pad_indices.iter().enumerate() {
        let next = final_pad_indices[(i + 1) % final_pad_indices.len()];
        new_permutation[idx] = F::from(next as u64);
    }

    new_permutation
}

/// Checks that the permutation is a valid reordering of the witnesses with correct cycles.
///
/// A correct permutation must:
/// - Be the same length as `witnesses`
/// - Only include valid indices (i.e., < witnesses.len())
/// - Consist of disjoint cycles that map every index in the witness array
pub fn check_permutation<F: PrimeField>(
    witnesses: &[F],
    permutation: &[F],
    num_rows: usize,
) -> bool {
    let len = witnesses.len();
    if permutation.len() != len {
        println!(
            "Permutation length mismatch: expected {}, got {}",
            len,
            permutation.len()
        );
        return false;
    }

    let zero = F::zero();
    let zero_indices: Vec<usize> = permutation
        .iter()
        .enumerate()
        .filter_map(|(i, &val)| if val == zero { Some(i) } else { None })
        .collect();

    // Step 1: Check that all permutation values are valid indices
    let mut seen = vec![false; len];
    for &perm in permutation {
        let idx = perm.into_bigint().as_ref()[0] as usize;
        if idx >= len {
            println!("Expected index < {}, got {}", len, idx);
            return false;
        }
    }

    // Step 2: Follow each unvisited cycle and mark entries
    for start in 0..len {
        if seen[start] {
            continue;
        }

        let mut i = start;
        let mut cycle_len = 0;
        let next = 0;
        loop {
            if seen[i] {
                // Cycle looped to an already seen value before completing — error
                println!("cycle looped to an already seen value before completing, index {}, len {}, value {}", i, cycle_len, next);
                //continue;
                return false;
            }
            seen[i] = true;
            let next = permutation[i].into_bigint().as_ref()[0] as usize;
            cycle_len += 1;
            if start == next {
                break;
            }
            i = next;
        }

        if cycle_len == 0 {
            println!("cycle of len 0");
            return false;
        }
    }

    // Step 3: All entries must be seen
    seen.into_iter().all(|v| v)
}

fn split_flat_witness<F: Clone + ark_std::Zero>(
    flat_witness: &[F],
    num_columns: usize,
    num_rows: usize,
    num_pub_inputs: usize,
) -> Vec<Vec<F>> {
    let padding = num_pub_inputs.next_power_of_two() - num_pub_inputs;
    let padded_num_rows = (num_rows + padding).next_power_of_two();

    let mut columns = vec![Vec::with_capacity(padded_num_rows); num_columns];

    for wire in 0..num_columns {
        // 1. Public inputs
        for row in 0..num_pub_inputs {
            columns[wire].push(flat_witness[wire * num_rows + row].clone());
        }

        // 2. Padding
        for _ in 0..padding {
            columns[wire].push(F::zero());
        }

        // 3. Private inputs
        for row in num_pub_inputs..num_rows {
            columns[wire].push(flat_witness[wire * num_rows + row].clone());
        }

        // 4. Padding to the next power of two
        while columns[wire].len() < padded_num_rows {
            columns[wire].push(F::zero());
        }
        // Sanity check
        assert_eq!(columns[wire].len(), padded_num_rows);
    }

    columns
}

pub fn flatten_witness_matrix_preserve_padding<F: Clone>(columns: &[Vec<F>]) -> Vec<F> {
    let num_columns = columns.len();
    let padded_num_rows = columns
        .first()
        .map(|col| col.len())
        .expect("Empty columns vector");

    // Sanity check: all columns should have the same length
    for col in columns {
        assert_eq!(col.len(), padded_num_rows, "Column length mismatch");
    }

    let mut flat = Vec::with_capacity(num_columns * padded_num_rows);

    for col in columns {
        for row in col {
            flat.push(row.clone());
        }
    }

    flat
}

// ===

fn main() {
    env_logger::Builder::from_default_env()
        .format_level(false)
        .format_timestamp(None)
        .init();
    let options = Options::parse();
    circ::cfg::set(&options.circ);
    println!("{options:?}");

    let cs = {
        let inputs = zsharp::Inputs {
            file: options.path,
            mode: Mode::Proof,
        };
        ZSharpFE::gen(inputs)
    };

    print!("Optimizing IR... ");
    let cs = opt(
        cs,
        vec![
            Opt::ScalarizeVars,
            Opt::Flatten,
            Opt::Sha,
            Opt::ConstantFold(Box::new([])),
            Opt::Flatten,
            Opt::Inline,
            // Tuples must be eliminated before oblivious array elim
            Opt::Tuple,
            Opt::ConstantFold(Box::new([])),
            Opt::Obliv,
            // The obliv elim pass produces more tuples, that must be eliminated
            Opt::Tuple,
            Opt::LinearScan,
            // The linear scan pass produces more tuples, that must be eliminated
            Opt::Tuple,
            Opt::Flatten,
            Opt::ConstantFold(Box::new([])),
            Opt::Inline,
        ],
    );
    println!("done.");

    let action = options.action;
    /*
    let proof = options.proof;
    let prover_key = options.prover_key;
    let verifier_key = options.verifier_key;
    let instance = options.instance;
    */

    //println!("Converting to r1cs");
    //let r1cs = to_r1cs(cs.get("main"), cfg());
    /*let r1cs = if options.skip_linred {
        println!("Skipping linearity reduction, as requested.");
        r1cs
    } else {
        println!(
            "R1cs size before linearity reduction: {}",
            r1cs.constraints().len()
        );
        reduce_linearities(r1cs, cfg())
    };
    println!("Final r1cs: {} constraints", r1cs.constraints().len());
    println!("{:?}", r1cs.num_vars());*/
    let plonk = to_plonk(cs.get("main"), cfg());
    println!("plonk constraints: {:?}", plonk.constraints.len());
    println!("{:?}", plonk.public_inputs.len());
    println!("{:?}", plonk.witness.len());
    println!("{:?}", plonk.copy_constraints.len());
    println!("{:?}", plonk.wire_values.len());
    use ark_bls12_381::Fr;
    let (mut plonkish_circuit, plonkish_witness) =
        plonk_to_hyperplonk::<Fr>(plonk.clone()).unwrap();
    //println!("wits2 = {:?}", plonkish_witness);
    println!(
        "plonkish circuit permutation length = {}",
        plonkish_circuit.permutation.len()
    );
    println!("plonkish circuit wits length = {}", plonkish_witness.len());
    println!("nb copy constraints = {}", plonk.copy_constraints.len());

    assert!(plonkish_circuit.is_satisfied(&plonkish_witness));

    // =========

    let num_rows: usize = plonkish_circuit.params.num_constraints; //num_constraints
    let num_columns = plonkish_circuit.params.gate_func.num_witness_columns();
    let num_pub_inputs = plonkish_circuit.params.num_pub_input;

    let witnesses: Vec<hyperplonk::witness::WitnessColumn<_>> =
        split_flat_witness(&plonkish_witness, num_columns, num_rows, num_pub_inputs)
            .into_iter()
            .map(hyperplonk::witness::WitnessColumn::new)
            .collect();

    let witnesses_vec: Vec<Vec<_>> = witnesses
        .iter()
        .map(|w| w.coeff_ref().to_vec()) // Convert each slice into an owned Vec
        .collect();

    let witnesses_flattened = flatten_witness_matrix_preserve_padding(&witnesses_vec);

    use ark_std::log2;

    use ark_std::Zero;

    let selectors = plonkish_circuit.selectors.clone();
    println!("#selectors = {}", selectors.len());

    plonkish_circuit.params.num_constraints =
        plonkish_circuit.params.num_constraints.next_power_of_two();
    plonkish_circuit.params.num_pub_input =
        plonkish_circuit.params.num_pub_input.next_power_of_two();

    let padding = num_pub_inputs.next_power_of_two() - num_pub_inputs;

    // TODO Padding function broken
    let num_priv_inputs = num_rows - num_pub_inputs;
    let pub_padding = num_pub_inputs.next_power_of_two() - num_pub_inputs;
    let total_len = num_pub_inputs + pub_padding + num_priv_inputs;
    println!(
        "num_priv_inputs = {}, pub_padding = {}, total_len = {}",
        num_priv_inputs, pub_padding, total_len
    );

    let mut padded_selectors: Vec<
        Vec<ark_ff::Fp<ark_ff::MontBackend<ark_bls12_381::FrConfig, 4>, 4>>,
    > = vec![vec![Fr::zero(); total_len]; selectors.len()];

    for (i, sel_column) in selectors.iter().enumerate() {
        // Copy public inputs
        for j in 0..num_pub_inputs {
            padded_selectors[i][j] = sel_column.0[j].clone();
        }
        // Padding remains zero (implicitly)

        // Copy private inputs after padding
        for j in 0..num_priv_inputs {
            padded_selectors[i][num_pub_inputs + pub_padding + j] =
                sel_column.0[num_pub_inputs + j].clone();
        }
    }

    let padded_selectors: Vec<
        SelectorColumn<ark_ff::Fp<ark_ff::MontBackend<ark_bls12_381::FrConfig, 4>, 4>>,
    > = padded_selectors
        .into_iter()
        .map(|col| SelectorColumn(col))
        .collect();

    let new_num_rows = num_rows + padding;
    let padded_num_rows = num_rows.next_power_of_two();
    let pad = padded_num_rows - num_rows;

    println!(
        "new_num_rows = {}, padded_num_rows = {}, pad = {}",
        new_num_rows, padded_num_rows, pad
    );

    let chunk_size = 1 << log2(plonkish_circuit.params.num_constraints) as usize;
    assert_eq!(chunk_size, padded_num_rows);
    let expected_length = chunk_size * num_columns;
    let mut permutation = plonkish_circuit.permutation.clone();

    let mut new_permutation =
        pad_permutation_field(permutation.clone(), num_rows, padding, expected_length);

    assert_eq!(
        plonkish_circuit.params.num_constraints,
        witnesses[0].coeff_ref().len()
    );

    assert!(
        check_permutation(&plonkish_witness, &permutation, num_rows),
        "Permutation check failed"
    );
    // TODO investigate permutation length mismatch + not passing
    assert!(
        check_permutation(
            &witnesses_flattened,
            &new_permutation,
            num_rows.next_power_of_two()
        ),
        "Permutation check failed"
    );

    let circuit: HyperPlonkIndex<ark_ff::Fp<ark_ff::MontBackend<ark_bls12_381::FrConfig, 4>, 4>> =
        HyperPlonkIndex {
            params: convert_params(plonkish_circuit.params.clone()),
            permutation: new_permutation,
            selectors: convert_selectors(padded_selectors),
        };
    assert_eq!(
        plonkish_circuit.params.num_constraints,
        circuit.selectors[0].0.len()
    );

    println!("Num gates: {}", num_columns);
    println!(
        "Num constraints (after padding): {}",
        circuit.params.num_constraints
    );
    println!(
        "Num public inputs (after padding): {}",
        circuit.params.num_pub_input
    );

    use ark_ff::PrimeField;
    use std::str::FromStr;

    use hyperplonk::structs::{HyperPlonkIndex, HyperPlonkParams};
    use hyperplonk::HyperPlonkSNARK;

    use ark_std::test_rng;
    use subroutines::{
        pcs::{
            prelude::{MultilinearKzgPCS, MultilinearUniversalParams},
            PolynomialCommitmentScheme,
        },
        poly_iop::PolyIOP,
    };

    const SUPPORTED_SIZE: usize = 20;

    let mut rng = test_rng();
    let pcs_srs = {
        let srs =
            MultilinearKzgPCS::<Bls12_381>::gen_srs_for_testing(&mut rng, SUPPORTED_SIZE).unwrap();
        //write_srs(&srs);
        srs
    };
    use ark_ff::BigInt;

    let mut public_inputs = plonkish_witness[..num_pub_inputs].to_vec();
    public_inputs.resize(num_pub_inputs.next_power_of_two(), Fr::zero());

    let start = Instant::now();

    let (pk, vk) =
        <PolyIOP<Fr> as HyperPlonkSNARK<Bls12_381, MultilinearKzgPCS<Bls12_381>>>::preprocess(
            &circuit, &pcs_srs,
        )
        .unwrap();

    println!("key extraction: {:?}", start.elapsed());

    //==========================================================
    // generate a proof
    let start = Instant::now();

    let _proof = <PolyIOP<Fr> as HyperPlonkSNARK<Bls12_381, MultilinearKzgPCS<Bls12_381>>>::prove(
        &pk,
        &public_inputs,
        &witnesses,
    )
    .unwrap();

    println!("proving: {:?}", start.elapsed());

    let proof = <PolyIOP<Fr> as HyperPlonkSNARK<Bls12_381, MultilinearKzgPCS<Bls12_381>>>::prove(
        &pk,
        &public_inputs,
        &witnesses,
    )
    .unwrap();
    //==========================================================
    // verify a proof
    let start = Instant::now();
    //println!("proof : {:?}", proof.perm_check_proof.zero_check_proof);

    let verify = <PolyIOP<Fr> as HyperPlonkSNARK<Bls12_381, MultilinearKzgPCS<Bls12_381>>>::verify(
        &vk,
        &public_inputs,
        &proof,
    )
    .unwrap();
    assert!(verify);

    println!("verifying: {:?}", start.elapsed());

    // implement optimizer
    match action {
        ProofAction::Count => {
            if !options.quiet {
                //eprintln!("{:#?}", r1cs.constraints());
            }
        }
        ProofAction::Prove => {
            unimplemented!()
            /*
            println!("Proving");
            r1cs.check_all();
            let rng = &mut rand::thread_rng();
            let mut pk_file = File::open(prover_key).unwrap();
            let pk = Parameters::<Bls12>::read(&mut pk_file, false).unwrap();
            let pf = create_random_proof(&r1cs, &pk, rng).unwrap();
            let mut pf_file = File::create(proof).unwrap();
            pf.write(&mut pf_file).unwrap();
            */
        }
        ProofAction::Setup => {
            unimplemented!()
            /*
            let rng = &mut rand::thread_rng();
            let p =
                generate_random_parameters::<bls12_381::Bls12, _, _>(&r1cs, rng).unwrap();
            let mut pk_file = File::create(prover_key).unwrap();
            p.write(&mut pk_file).unwrap();
            let mut vk_file = File::create(verifier_key).unwrap();
            p.vk.write(&mut vk_file).unwrap();
            */
        }
        ProofAction::Verify => {
            unimplemented!()
            /*
            println!("Verifying");
            let mut vk_file = File::open(verifier_key).unwrap();
            let vk = VerifyingKey::<Bls12>::read(&mut vk_file).unwrap();
            let pvk = prepare_verifying_key(&vk);
            let mut pf_file = File::open(proof).unwrap();
            let pf = Proof::read(&mut pf_file).unwrap();
            let instance_vec = parse_instance(&instance);
            verify_proof(&pvk, &pf, &instance_vec).unwrap();
            */
        }
    };
}
