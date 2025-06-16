use bls12_381::Bls12;
/*
use bellman::gadgets::test::TestConstraintSystem;
use bellman::groth16::{
    create_random_proof, generate_parameters, generate_random_parameters, prepare_verifying_key,
    verify_proof, Parameters, Proof, VerifyingKey,
};
use bellman::Circuit;
use bls12_381::{Bls12, Scalar};
*/
use circ::front::zsharp::{self, ZSharpFE};
use circ::front::{FrontEnd, Mode};
use circ::ir::opt::{opt, Opt};
use circ::ir::term::{Term, Op};
/*
use circ::target::r1cs::bellman::parse_instance;
*/
use circ::target::plonkish::trans::{to_plonk, PlonkConstraint, PlonkCs};
use circ::target::r1cs::opt::reduce_linearities;
use circ::target::r1cs::trans::to_r1cs;
use circ::ir::term::*;
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

use ark_ff::{BigInt, BigInteger, PrimeField};
use circ::target::plonkish::trans::Wire;
use circ_fields::FieldV;
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
        let gate_constraint = (0..self.params.num_constraints).into_par_iter().all(|i| {
            self.params
                .gate_func
                .evaluate(&self.selector_row(i), &self.witness_row(values, i))
                == F::zero()
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
}

impl<F: PrimeField> PlonkToHyperPlonkMapper<F> {
    pub fn new() -> Self {
        Self {
            wire_to_index: HashMap::new(),
            next_witness_index: 0,
            _marker: std::marker::PhantomData,
        }
    }

    /// Convert PlonkCs to PlonkishCircuit
    pub fn convert(&mut self, plonk_cs: &PlonkCs) -> Result<PlonkishCircuit<F>, String> {
        // Step 1: Build wire index mapping
        self.build_wire_mapping(plonk_cs)?;

        // Step 2: Create selector columns from constraints
        let selector_columns = self.create_selector_columns(plonk_cs)?;

        // Step 3: Create permutation vector from copy constraints
        let permutation = self.create_permutation_vector(plonk_cs)?;

        // Step 4: Create circuit parameters
        let params = PlonkishCircuitParams {
            num_constraints: plonk_cs.constraints.len(),
            num_pub_input: plonk_cs.public_inputs.len(),
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
    fn create_permutation_vector(&self, plonk_cs: &PlonkCs) -> Result<Vec<F>, String> {
        let total_vars = self.next_witness_index * plonk_cs.constraints.len();
        let mut permutation = Vec::with_capacity(total_vars);

        // Initialize permutation as identity
        for i in 0..total_vars {
            permutation.push(F::from(i as u64));
        }

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

        Ok(permutation)
    }

    /// Helper to convert FieldV to F (you'll need to implement this based on your types)
    fn field_v_to_f(&self, field_v: &FieldV) -> Result<F, String> {
        // This is a placeholder - you'll need to implement the actual conversion
        // based on how your FieldV type relates to the PrimeField F
        F::from_bigint(F::BigInt::from_bits_be(field_v.i().to_digits(Order::LsfBe).as_slice()))
            .ok_or_else(|| "Failed to convert FieldV to F".to_string())
        //todo!("Implement conversion from FieldV to F based on your type system")
    }

    /// Get the witness column index for a wire
    pub fn get_wire_index(&self, wire: &Wire) -> Option<usize> {
        self.wire_to_index.get(wire).copied()
    }

    /// Create witness values matrix from PlonkCs
    pub fn create_witness_values(&self, plonk_cs: &PlonkCs) -> Result<Vec<F>, String> {
        let num_constraints = plonk_cs.constraints.len();
        let num_witnesses = self.next_witness_index;
        let mut values = vec![F::zero(); num_witnesses * num_constraints];

        // Fill witness values
        for (row_idx, constraint) in plonk_cs.constraints.iter().enumerate() {
            // Get wire indices
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

            // Get values from wire_values map and convert Term to F
            let a_val = self.term_to_f(
                plonk_cs
                    .wire_values
                    .get(&constraint.a)
                    .ok_or("Wire a value not found")?,
            )?;
            let b_val = self.term_to_f(
                plonk_cs
                    .wire_values
                    .get(&constraint.b)
                    .ok_or("Wire b value not found")?,
            )?;
            let c_val = self.term_to_f(
                plonk_cs
                    .wire_values
                    .get(&constraint.c)
                    .ok_or("Wire c value not found")?,
            )?;

            // Store in column-major format
            values[a_idx * num_constraints + row_idx] = a_val;
            values[b_idx * num_constraints + row_idx] = b_val;
            values[c_idx * num_constraints + row_idx] = c_val;
        }

        Ok(values)
    }

    /// Helper to convert Term to F (you'll need to implement this)
      /// Convert a Term to field value F by evaluating it
      fn term_to_f(&self, term: &Term) -> Result<F, String> {
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
    }
    
}

/// Helper function to create a complete HyperPlonk circuit from PlonkCs
pub fn plonk_to_hyperplonk<F: PrimeField>(
    plonk_cs: &PlonkCs,
) -> Result<(PlonkishCircuit<F>, Vec<F>), String> {
    let mut mapper = PlonkToHyperPlonkMapper::new();
    let circuit = mapper.convert(plonk_cs)?;
    let witness_values = mapper.create_witness_values(plonk_cs)?;

    Ok((circuit, witness_values))
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

    println!("Converting to r1cs");
    let r1cs = to_r1cs(cs.get("main"), cfg());
    let r1cs = if options.skip_linred {
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
    println!("{:?}", r1cs.num_vars());
    let plonk = to_plonk(cs.get("main"), cfg());
    println!("plonk constraints: {:?}", plonk.constraints.len());
    println!("{:?}", plonk.public_inputs.len());
    println!("{:?}", plonk.witness.len());
    println!("{:?}", plonk.copy_constraints.len());
    println!("{:?}", plonk.wire_values.len());
    use ark_bls12_381::Fr;
    let _res = plonk_to_hyperplonk::<Fr>(&plonk);
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
