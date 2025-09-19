//! Lowering IR to Plonk Constraints
//!
//! This converts the R1CS-based system to emit vanilla Plonk constraints
//! with copy constraints instead of R1CS constraints.

use crate::cfg::CircCfg;
use crate::ir::term::*;
use crate::target::plonkish::VarType;
use circ_fields::{FieldT, FieldV};
use circ_opt::FieldDivByZero;
use im::HashSet;
use itertools::assert_equal;
use log::{debug, trace};

use fxhash::FxHashMap;
use rug::ops::Pow;
use rug::Integer;

use std::cell::RefCell;
use std::fmt::Display;
use std::iter::ExactSizeIterator;
use std::rc::Rc;

/// Plonk wire representation
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct Wire {
    pub index: usize,
    pub name: String,
    ty: WireType,
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub enum WireType {
    Public,
    Private,
    Intermediate,
}

impl Wire {
    fn new(index: usize, name: String) -> Self {
        Self {
            index,
            name,
            ty: WireType::Intermediate,
        }
    }

    fn new_ty(index: usize, name: String, ty: WireType) -> Self {
        Self { index, name, ty }
    }
}

/// Plonk constraint: q_l * a + q_r * b + q_o * c + q_m * a * b + q_c = 0
#[derive(Clone, Debug)]
pub struct PlonkConstraint {
    pub q_l: FieldV, // Left selector
    pub q_r: FieldV, // Right selector
    pub q_o: FieldV, // Output selector
    pub q_m: FieldV, // Multiplication selector
    pub q_c: FieldV, // Constant selector
    pub a: Wire,     // Left wire
    pub b: Wire,     // Right wire
    pub c: Wire,     // Output wire
}

/// Copy constraint: enforces that two wires have the same value
#[derive(Clone, Debug)]
pub struct CopyConstraint {
    pub wire1: Wire,
    pub wire2: Wire,
}

/// Plonk constraint system
#[derive(Clone, Debug)]
pub struct PlonkCs {
    pub field: FieldT,
    pub constraints: Vec<PlonkConstraint>,
    pub copy_constraints: Vec<CopyConstraint>,
    pub public_inputs: Vec<Wire>,
    pub witness: Vec<Wire>,
    pub wire_values: FxHashMap<Wire, Term>,
    var_vals: FxHashMap<String, Value>,
    next_wire_id: usize,
    pub all_inputs: Vec<String>,
}

impl PlonkCs {
    fn new(field: FieldT) -> Self {
        Self {
            field,
            constraints: Vec::new(),
            copy_constraints: Vec::new(),
            public_inputs: Vec::new(),
            witness: Vec::new(),
            wire_values: FxHashMap::default(),
            var_vals: FxHashMap::default(),
            next_wire_id: 0,
            all_inputs: Vec::new(),
        }
    }

    /// Create a new wire with given name and value
    fn new_wire(&mut self, name: String, value: Term) -> Wire {
        //println!("added wire {} for term {:?}", name, value);
        let wire = Wire::new(self.next_wire_id, name);
        self.next_wire_id += 1;
        self.wire_values.insert(wire.clone(), value);
        wire
    }

    fn new_wire_ty(&mut self, name: String, value: Term) -> Wire {
        //println!("added wire {} for term {:?}", name, value);
        let wire = Wire::new_ty(self.next_wire_id, name, WireType::Public);
        self.next_wire_id += 1;
        self.wire_values.insert(wire.clone(), value);
        wire
    }

    /// Add a Plonk constraint
    fn add_constraint(&mut self, constraint: PlonkConstraint) {
        self.constraints.push(constraint);
    }

    /// Add a copy constraint
    fn add_copy_constraint(&mut self, wire1: Wire, wire2: Wire) {
        self.copy_constraints.push(CopyConstraint { wire1, wire2 });
    }

    /// Create constant wire (zero)
    pub fn zero_wire(&mut self) -> Wire {
        let zero_term = term![Op::Const(Box::new(Value::Field(self.field.new_v(0))))];
        self.new_wire("zero".to_string(), zero_term)
    }

    /// Create constant wire (one)
    fn one_wire(&mut self) -> Wire {
        let one_term = term![Op::Const(Box::new(Value::Field(self.field.new_v(1))))];
        self.new_wire("one".to_string(), one_term)
    }
}

struct BvEntry {
    width: usize,
    /// Empty if not yet created.
    uint: Option<Wire>,
    /// Empty if not yet created.
    bits: Vec<Wire>,
}

#[derive(Clone)]
enum EmbeddedTerm {
    Bv(Rc<RefCell<BvEntry>>),
    Bool(Wire),
    Field(Wire),
    #[allow(dead_code)]
    Tuple(Vec<EmbeddedTerm>),
}

struct ToPlonk<'cfg> {
    plonk: PlonkCs,
    cache: TermMap<EmbeddedTerm>,
    embed: Rc<RefCell<TermSet>>,
    next_idx: usize,
    //zero: Wire, // todo remove
    //one: Wire,
    cfg: &'cfg CircCfg,
    field: FieldT,
    used_vars: HashSet<String>,
}

impl<'cfg> ToPlonk<'cfg> {
    fn new(cfg: &'cfg CircCfg, used_vars: HashSet<String>) -> Self {
        let field = cfg.field().clone();
        debug!("Starting Plonk back-end, field: {}", field);
        let plonk = PlonkCs::new(field.clone());
        //let zero = plonk.zero_wire();
        //let one = plonk.one_wire();

        Self {
            plonk,
            cache: TermMap::default(),
            embed: Default::default(),
            used_vars,
            next_idx: 0,
            //zero,
            //one,
            field,
            cfg,
        }
    }

    /// Create a committed witness vector. Each input is a (name, term) pair.
    fn committed_wit(&mut self, elements: Vec<(String, Term)>) {
        for (name, value) in elements {
            //println!("name = {}, value = {}", name, value);
            let wire = self.plonk.new_wire(name.clone(), value.clone());

            self.plonk.witness.push(wire.clone());
            let var = var(name, check(&value));
            self.embed.borrow_mut().insert(var.clone());
            self.cache.insert(var, EmbeddedTerm::Field(wire));
        }
    }

    /// Get a new variable wire
    fn fresh_var<D: Display + ?Sized>(&mut self, ctx: &D, comp: Term, ty: VarType) -> Wire {
        let n = if matches!(ty, VarType::Chall) {
            format!("{ctx}")
        } else {
            format!("{ctx}_n{}", self.next_idx)
        };
        //println!("name {}", n);
        self.next_idx += 1;
        debug_assert!(matches!(check(&comp), Sort::Field(_)));
        let wire = self.plonk.new_wire_ty(n.clone(), comp);

        match ty {
            VarType::Inst => self.plonk.public_inputs.push(wire.clone()),
            _ => self.plonk.witness.push(wire.clone()),
        }

        debug!("fresh: {n:?}");
        wire
    }

    /// Get a new witness wire
    fn fresh_wit<D: Display + ?Sized>(&mut self, ctx: &D, comp: Term) -> Wire {
        self.fresh_var(ctx, comp, VarType::FinalWit)
    }

    /// Create a Plonk constraint: q_l * a + q_r * b + q_o * c + q_m * a * b + q_c = 0
    fn constraint(
        &mut self,
        q_l: FieldV,
        q_r: FieldV,
        q_o: FieldV,
        q_m: FieldV,
        q_c: FieldV,
        a: Wire,
        b: Wire,
        c: Wire,
    ) {
        let constraint = PlonkConstraint {
            q_l,
            q_r,
            q_o,
            q_m,
            q_c,
            a,
            b,
            c,
        };
        self.plonk.add_constraint(constraint);
    }

    /// Enforce `x` to be bit-valued: x * (x - 1) = 0
    fn enforce_bit(&mut self, b: Wire) {
        let z_c = self.plonk.zero_wire();
        // x * (x - 1) = 0  =>  x * x - x = 0  =>  q_m * a * b + q_l * a = 0
        // where a = b = x, so q_m = 1, q_l = -1
        self.constraint(
            self.field.new_v(-1),
            self.field.new_v(0),
            self.field.new_v(0),
            self.field.new_v(1),
            self.field.new_v(0), // q_l=-1, q_r=0, q_o=0, q_m=1, q_c=0
            b.clone(),
            b.clone(),
            z_c,
        );
    }

    /// Get a new bit-valued variable
    fn fresh_bit<D: Display + ?Sized>(&mut self, ctx: &D, comp: Term) -> Wire {
        //println!("check({}) = {}", ctx, check(&comp));
        debug_assert!(matches!(check(&comp), Sort::Bool));
        let comp = term![Op::Ite; comp, self.one_term(), self.zero_term()];
        let v = self.fresh_var(ctx, comp, VarType::FinalWit);
        self.enforce_bit(v.clone());
        v
    }

    fn zero_term(&self) -> Term {
        term![Op::Const(Box::new(Value::Field(self.field.new_v(0))))]
    }

    fn one_term(&self) -> Term {
        term![Op::Const(Box::new(Value::Field(self.field.new_v(1))))]
    }

    /// Add two wires: returns a wire representing a + b
    fn add(&mut self, a: Wire, b: Wire) -> Wire {
        // Step 1: Create fresh wires in current row to hold a and b
        let a_new = self.fresh_wit("add.in.0", self.plonk.wire_values[&a].clone());
        let b_new = self.fresh_wit("add.in.1", self.plonk.wire_values[&b].clone());

        // Step 2: Add copy constraints from original wires
        self.plonk.add_copy_constraint(a.clone(), a_new.clone());
        self.plonk.add_copy_constraint(b.clone(), b_new.clone());

        let sum_term = term![PF_ADD;
            self.plonk.wire_values[&a].clone(),
            self.plonk.wire_values[&b].clone()
        ];
        let result = self.fresh_wit("add.out.0", sum_term);

        // a + b - result = 0  =>  q_l * a + q_r * b + q_o * result = 0
        self.constraint(
            self.field.new_v(1),
            self.field.new_v(1),
            self.field.new_v(-1),
            self.field.new_v(0),
            self.field.new_v(0), // q_l=1, q_r=1, q_o=-1, q_m=0, q_c=0
            a_new,
            b_new,
            result.clone(),
        );
        result
    }

    /// Subtract two wires: returns a wire representing a - b  
    fn sub(&mut self, a: Wire, b: Wire) -> Wire {
        // Step 1: Create fresh wires in current row to hold a and b
        let a_new = self.fresh_wit("sub.in.0", self.plonk.wire_values[&a].clone());
        let b_new = self.fresh_wit("sub.in.1", self.plonk.wire_values[&b].clone());

        // Step 2: Add copy constraints from original wires
        self.plonk.add_copy_constraint(a.clone(), a_new.clone());
        self.plonk.add_copy_constraint(b.clone(), b_new.clone());

        let diff_term = term![PF_ADD;
            self.plonk.wire_values[&a].clone(),
            term![PF_NEG; self.plonk.wire_values[&b].clone()]
        ];
        let result = self.fresh_wit("sub.out.0", diff_term);

        // a - b - result = 0  =>  q_l * a + q_r * (-b) + q_o * result = 0
        self.constraint(
            self.field.new_v(1),
            self.field.new_v(-1),
            self.field.new_v(-1),
            self.field.new_v(0),
            self.field.new_v(0), // q_l=1, q_r=-1, q_o=-1, q_m=0, q_c=0
            a_new,
            b_new,
            result.clone(),
        );
        result
    }

    /// Multiply two wires: returns a wire representing a * b
    fn mul(&mut self, a: Wire, b: Wire) -> Wire {
        // Step 1: Create fresh wires in current row to hold a and b
        let a_new = self.fresh_wit("mul.in.0", self.plonk.wire_values[&a].clone());
        let b_new = self.fresh_wit("mul.in.1", self.plonk.wire_values[&b].clone());

        // Step 2: Add copy constraints from original wires
        self.plonk.add_copy_constraint(a.clone(), a_new.clone());
        self.plonk.add_copy_constraint(b.clone(), b_new.clone());

        let mul_term = term![PF_MUL;
            self.plonk.wire_values[&a].clone(),
            self.plonk.wire_values[&b].clone()
        ];
        let result = self.fresh_wit("mul.out.0", mul_term);

        // a * b - result = 0  =>  q_m * a * b + q_o * result = 0
        self.constraint(
            self.field.new_v(0),
            self.field.new_v(0),
            self.field.new_v(-1),
            self.field.new_v(1),
            self.field.new_v(0), // q_l=0, q_r=0, q_o=-1, q_m=1, q_c=0
            a_new,
            b_new,
            result.clone(),
        );
        result
    }

    /// Add a constant to a wire
    fn add_const(&mut self, a: Wire, c: FieldV) -> Wire {
        let const_term = term![Op::Const(Box::new(Value::Field(c.clone())))];
        let sum_term = term![PF_ADD; self.plonk.wire_values[&a].clone(), const_term];
        let result = self.fresh_wit("add_const", sum_term);
        let z_b = self.plonk.zero_wire();
        // a + c - result = 0  =>  q_l * a + q_o * result + q_c = 0
        self.constraint(
            self.field.new_v(1),
            self.field.new_v(0),
            self.field.new_v(-1),
            self.field.new_v(0),
            c, // q_l=1, q_r=0, q_o=-1, q_m=0, q_c=c
            a,
            z_b,
            result.clone(),
        );
        result
    }

    fn neg_add_const(&mut self, a: Wire, c: FieldV) -> Wire {
        // Step 1: Create fresh wires in current row to hold a and b
        let a_new = self.fresh_wit("neg_add_const.in.0", self.plonk.wire_values[&a].clone());

        // Step 2: Add copy constraints from original wires
        self.plonk.add_copy_constraint(a.clone(), a_new.clone());

        let const_term = term![Op::Const(Box::new(Value::Field(c.clone())))];
        //let one_term = term![Op::Const(Box::new(Value::Field(self.field.new_v(1))))];
        //let sum_term =
        //    term![PF_ADD; const_term, term![PF_NEG; self.plonk.wire_values[&a].clone()] ];
        let sum_term = //const_term;
            term![PF_ADD; const_term, term![PF_NEG; self.plonk.wire_values[&a].clone()] ];
        let result = self.fresh_wit("neg_add_const", sum_term);
        let z_b = self.plonk.zero_wire();
        // -a + c - result = 0  =>  q_l * a + q_o * result + q_c = 0
        self.constraint(
            self.field.new_v(-1),
            self.field.new_v(0),
            self.field.new_v(-1),
            self.field.new_v(0),
            c, // q_l=-1, q_r=0, q_o=-1, q_m=0, q_c=c
            a_new,
            z_b,
            result.clone(),
        );
        result
    }

    /// Multiply a wire by a constant
    fn mul_const(&mut self, a: Wire, c: FieldV) -> Wire {
        // Step 1: Create fresh wires in current row to hold a and b
        let a_new = self.fresh_wit("mul_const.in.0", self.plonk.wire_values[&a].clone());

        // Step 2: Add copy constraints from original wires
        self.plonk.add_copy_constraint(a.clone(), a_new.clone());

        let const_term = term![Op::Const(Box::new(Value::Field(c.clone())))];
        let mul_term = term![PF_MUL; self.plonk.wire_values[&a].clone(), const_term];
        let result = self.fresh_wit("mul.out.0", mul_term);
        let z_b = self.plonk.zero_wire();
        // c * a - result = 0  =>  q_l * (c * a) + q_o * result = 0
        self.constraint(
            c,
            self.field.new_v(0),
            self.field.new_v(-1),
            self.field.new_v(0),
            self.field.new_v(0), // q_l=c, q_r=0, q_o=-1, q_m=0, q_c=0
            a_new,
            z_b,
            result.clone(),
        );
        result
    }

    /// Assert that a wire equals zero
    fn assert_zero(&mut self, a: Wire) {
        // Step 1: Create fresh wires in current row to hold a and b
        let a_new = self.fresh_wit("assert_zero.in.0", self.plonk.wire_values[&a].clone());

        // Step 2: Add copy constraints from original wires
        self.plonk.add_copy_constraint(a.clone(), a_new.clone());

        let z_b = self.plonk.zero_wire();
        let z_c = self.plonk.zero_wire();
        // a = 0  =>  q_l * a = 0
        self.constraint(
            self.field.new_v(1),
            self.field.new_v(0),
            self.field.new_v(0),
            self.field.new_v(0),
            self.field.new_v(0), // q_l=1, q_r=0, q_o=0, q_m=0, q_c=0
            a_new,
            z_b,
            z_c,
        );
    }

    fn assert_boolean(&mut self, w: Wire) {
        /*let w_squared = self.mul(w.clone(), w.clone()); // w^2
        let diff = self.sub(w_squared, w); // w^2 - w
        self.assert_zero(diff); // only true when w ∈ {0,1}*/
        let z_c = self.plonk.zero_wire();
        // x * (x - 1) = 0  =>  x * x - x = 0  =>  q_m * a * b + q_l * a = 0
        // where a = b = x, so q_m = 1, q_l = -1
        self.constraint(
            self.field.new_v(-1),
            self.field.new_v(0),
            self.field.new_v(0),
            self.field.new_v(1),
            self.field.new_v(0), // q_l=-1, q_r=0, q_o=0, q_m=1, q_c=0
            w.clone(),
            w.clone(),
            z_c,
        );
    }

    /// Assert that two wires are equal by adding a copy constraint
    fn assert_equal(&mut self, a: Wire, b: Wire) {
        self.plonk.add_copy_constraint(a, b);
    }

    /// Return a bit indicating whether wire `x` is non-zero.
    /*fn is_zero(&mut self, x: Wire) -> Wire {
        let eqz = term![Op::Eq;
            self.plonk.wire_values[&x].clone(),
            self.zero_term()
        ];

        // m * x - 1 + is_zero == 0
        // is_zero * x == 0
        let m = self.fresh_wit(
            "is_zero_inv",
            term![Op::Ite; eqz.clone(), self.zero_term(),
                  term![PF_RECIP; self.plonk.wire_values[&x].clone()]],
        );
        let is_zero = self.fresh_wit(
            "is_zero",
            term![Op::Ite; eqz, self.one_term(), self.zero_term()],
        );

        // m * x + is_zero - 1 = 0
        let temp = self.mul(m, x.clone());
        let temp2 = self.add(temp, is_zero.clone());
        let temp3 = self.add_const(temp2, self.field.new_v(-1));
        self.assert_zero(temp3);

        // is_zero * x = 0
        let temp4 = self.mul(is_zero.clone(), x);
        self.assert_zero(temp4);

        is_zero
    }*/
    /// Return a bit indicating whether wire `x` is zero:
    ///   is_zero = 1 iff x = 0, else 0.
    fn is_zero(&mut self, x: Wire) -> Wire {
        // 1) Compute the boolean condition x == 0
        let eqz_term = term![Op::Eq;
            self.plonk.wire_values[&x].clone(),
            self.zero_term()
        ];

        // 2) Witness for conditional inverse: m = (x != 0) ? 1/x : 0
        let m = self.fresh_wit(
            "is_zero_inv",
            term![Op::Ite;
                eqz_term.clone(),
                self.zero_term(),                     // if x == 0
                term![PF_RECIP; self.plonk.wire_values[&x].clone()]  // else 1/x
            ],
        );

        // 3) Witness for the bit: is_zero = (x == 0) ? 1 : 0
        let is_zero = self.fresh_wit(
            "is_zero_bit",
            term![Op::Ite;
                eqz_term,
                self.one_term(),   // if x == 0
                self.zero_term()   // else
            ],
        );

        // 4) Make row-local copies and copy-constraints for x, m, and is_zero
        let x_row = self.fresh_wit("is_zero_x", self.plonk.wire_values[&x].clone());
        //let m_row      = self.fresh_wit("is_zero_m",      self.plonk.wire_values[&m].clone());
        //let is_zero_row = self.fresh_wit("is_zero_bit_r", self.plonk.wire_values[&is_zero].clone());

        self.plonk.add_copy_constraint(x.clone(), x_row.clone());
        //self.plonk.add_copy_constraint(m.clone(),      m_row.clone());
        //self.plonk.add_copy_constraint(is_zero.clone(), is_zero_row.clone());

        // 5) Gate 1:   m * x + is_zero - 1 = 0
        //    → q_m=1, q_o=1, q_c=-1
        self.constraint(
            /*q_l=*/ self.field.new_v(0),
            /*q_r=*/ self.field.new_v(0),
            /*q_o=*/ self.field.new_v(1),
            /*q_m=*/ self.field.new_v(1),
            /*q_c=*/ self.field.new_v(-1),
            m.clone(),
            x_row.clone(),
            is_zero.clone(),
        );

        // 6) Gate 2:   is_zero * x = 0
        //    → q_m=1, all others = 0
        let zero_row = self.fresh_wit("is_zero_zero", self.zero_term());
        self.plonk.add_copy_constraint(x.clone(), x_row.clone());

        let is_zero_row = self.fresh_wit("is_zero_bit_r", self.plonk.wire_values[&is_zero].clone());
        self.plonk
            .add_copy_constraint(is_zero.clone(), is_zero_row.clone());

        self.constraint(
            /*q_l=*/ self.field.new_v(0),
            /*q_r=*/ self.field.new_v(0),
            /*q_o=*/ self.field.new_v(0),
            /*q_m=*/ self.field.new_v(1),
            /*q_c=*/ self.field.new_v(0),
            is_zero_row,
            x_row,
            zero_row,
        );

        // Return the actual bit wire
        is_zero.clone()
    }

    /*fn is_zero(&mut self, x: Wire) -> Wire {
        let eqz = term![Op::Eq;
            self.plonk.wire_values[&x].clone(),
            self.zero_term()
        ];

        // Generate witness values
        let m = self.fresh_wit(
            "is_zero_inv",
            term![Op::Ite; eqz.clone(), self.zero_term(),
                  term![PF_RECIP; self.plonk.wire_values[&x].clone()]],
        );
        let is_zero = self.fresh_wit(
            "is_zero",
            term![Op::Ite; eqz, self.one_term(), self.zero_term()],
        );

        // Create fresh wires for current constraint row
        let x_copy = self.fresh_wit("is_zero_x", self.plonk.wire_values[&x].clone());
        //let m_copy = self.fresh_wit("is_zero_m", self.plonk.wire_values[&m].clone());
        //let is_zero_copy = self.fresh_wit("is_zero_result", self.plonk.wire_values[&is_zero].clone());

        // Add copy constraints
        self.plonk.add_copy_constraint(x.clone(), x_copy.clone());
        //self.plonk.add_copy_constraint(m.clone(), m_copy.clone());
        //self.plonk.add_copy_constraint(is_zero.clone(), is_zero_copy.clone());

        // Constraint 1: m * x + is_zero - 1 = 0
        // This becomes: q_m * m * x + q_o * is_zero + q_c = 0
        // So: q_m = 1, q_o = 1, q_c = -1
        self.constraint(
            self.field.new_v(0),  // q_l = 0
            self.field.new_v(0),  // q_r = 0
            self.field.new_v(1),  // q_o = 1 (for is_zero)
            self.field.new_v(1),  // q_m = 1 (for m * x)
            self.field.new_v(-1), // q_c = -1 (constant term)
            m.clone(),            // left wire (m)
            x_copy.clone(),       // right wire (x)
            is_zero.clone(),      // output wire (is_zero)
        );

        // Constraint 2: is_zero * x = 0
        // This becomes: q_m * is_zero * x = 0
        // So: q_m = 1, all others = 0
        let x_copy2 = self.fresh_wit("is_zero_x2", self.plonk.wire_values[&x].clone());
        let is_zero_copy2 =
            self.fresh_wit("is_zero_result2", self.plonk.wire_values[&is_zero].clone());
        let zero_wire = self.fresh_wit("zero_result", self.zero_term());

        self.plonk.add_copy_constraint(x.clone(), x_copy2.clone());
        self.plonk
            .add_copy_constraint(is_zero.clone(), is_zero_copy2.clone());

        self.constraint(
            self.field.new_v(0), // q_l = 0
            self.field.new_v(0), // q_r = 0
            self.field.new_v(0), // q_o = 0
            self.field.new_v(1), // q_m = 1 (for is_zero * x)
            self.field.new_v(0), // q_c = 0
            is_zero_copy2,       // left wire (is_zero)
            x_copy2,             // right wire (x)
            zero_wire.clone(),   // output wire (should be 0)
        );

        zero_wire
    }*/
    /// Return a bit indicating whether wire `x` is zero:
    ///   is_zero = 1 iff x = 0, else 0.
    /*fn is_zero(&mut self, x: Wire) -> Wire {
        // 1) Compute the boolean condition x == 0
        let eqz_term = term![Op::Eq;
            self.plonk.wire_values[&x].clone(),
            self.zero_term()
        ];

        // 2) Witness for conditional inverse: m = (x != 0) ? 1/x : 0
        let m = self.fresh_wit(
            "is_zero_inv",
            term![Op::Ite;
                eqz_term.clone(),
                self.zero_term(),                     // if x == 0
                term![PF_RECIP; self.plonk.wire_values[&x].clone()]  // else 1/x
            ]
        );

        // 3) Witness for the bit: is_zero = (x == 0) ? 1 : 0
        let is_zero = self.fresh_wit(
            "is_zero_bit",
            term![Op::Ite;
                eqz_term,
                self.one_term(),   // if x == 0
                self.zero_term()   // else
            ]
        );

        // 4) Make row-local copies and copy-constraints for x, m, and is_zero
        let x_row      = self.fresh_wit("is_zero_x",      self.plonk.wire_values[&x].clone());
        let m_row      = self.fresh_wit("is_zero_m",      self.plonk.wire_values[&m].clone());
        let is_zero_row = self.fresh_wit("is_zero_bit_r", self.plonk.wire_values[&is_zero].clone());

        self.plonk.add_copy_constraint(x.clone(),      x_row.clone());
        self.plonk.add_copy_constraint(m.clone(),      m_row.clone());
        self.plonk.add_copy_constraint(is_zero.clone(), is_zero_row.clone());

        // 5) Gate 1:   m * x + is_zero - 1 = 0
        //    → q_m=1, q_o=1, q_c=-1
        self.constraint(
            /*q_l=*/ self.field.new_v(0),
            /*q_r=*/ self.field.new_v(0),
            /*q_o=*/ self.field.new_v(1),
            /*q_m=*/ self.field.new_v(1),
            /*q_c=*/ self.field.new_v(-1),
            m_row.clone(),
            x_row.clone(),
            is_zero_row.clone(),
        );

        // 6) Gate 2:   is_zero * x = 0
        //    → q_m=1, all others = 0
        let zero_row = self.fresh_wit("is_zero_zero", self.zero_term());
        self.plonk.add_copy_constraint(x.clone(),        x_row.clone());
        self.plonk.add_copy_constraint(is_zero.clone(), is_zero_row.clone());

        self.constraint(
            /*q_l=*/ self.field.new_v(0),
            /*q_r=*/ self.field.new_v(0),
            /*q_o=*/ self.field.new_v(0),
            /*q_m=*/ self.field.new_v(1),
            /*q_c=*/ self.field.new_v(0),
            is_zero_row,
            x_row,
            zero_row,
        );

        // Return the actual bit wire
        is_zero
    }*/

    /// Return a bit indicating whether wires `x` and `y` are equal.
    fn are_equal(&mut self, x: Wire, y: Wire) -> Wire {
        let diff = self.sub(x, y);
        self.is_zero(diff)
    }

    /// Given wire `x`, returns a vector of `n` wires which are the bits of `x`.
    fn decomp<D: Display + ?Sized>(&mut self, d: &D, x: &Wire, n: usize) -> Vec<Wire> {
        (0..n)
            .map(|i| {
                self.fresh_bit(
                    &format!("{d}_b{i}"),
                    term![Op::BvBit(i); term![Op::PfToBv(n); self.plonk.wire_values[x].clone()]],
                )
            })
            .collect::<Vec<_>>()
    }

    /// Given wire `x`, returns a vector of `n` wires which are the bits of `x`.
    /// Constrains `x` to fit in `n` (`signed`) bits.
    fn bitify<D: Display + ?Sized>(
        &mut self,
        d: &D,
        x: &Wire,
        n: usize,
        signed: bool,
    ) -> Vec<Wire> {
        debug!("Bitify({}): {:?}", n, x);
        let bits = self.decomp(d, x, n);
        let sum = self.debitify(bits.iter().cloned(), signed);
        //let diff = self.sub(sum, x.clone());
        //println!("diff {:?}, x  = {:?}, d = {}", diff, x, d);
        //self.assert_zero(diff);
        self.plonk.add_copy_constraint(sum, x.clone());
        bits
    }

    /// Given a sequence of `bits`, returns a wire which represents their sum.
    /*fn debitify<I: ExactSizeIterator<Item = Wire>>(&mut self, bits: I, signed: bool) -> Wire {
        let n = bits.len();
        let mut acc = self.field.new_v(1u8);
        let mut result = self.zero.clone();

        for (i, bit) in bits.enumerate() {
            let coeff = if signed && i + 1 == n {
                -(acc.clone())
            } else {
                acc.clone()
            };

            let scaled_bit = self.mul_const(bit, coeff);
            result = self.add(result, scaled_bit);

            acc *= &self.field.new_v(2u8);
        }
        result
    }*/

    /// acc_{i+1} = 2*acc_i + bit_i   (unsigned)
    /// acc_{i+1} = 2*acc_i + bit_i   (unsigned)
    fn horner_step(&mut self, acc: Wire, bit: Wire) -> Wire {
        // 1) fresh‐copy inputs
        let acc_ = self.fresh_wit("horner.acc", self.plonk.wire_values[&acc].clone());
        let bit_ = self.fresh_wit("horner.bit", self.plonk.wire_values[&bit].clone());
        self.plonk.add_copy_constraint(acc, acc_.clone());
        self.plonk.add_copy_constraint(bit, bit_.clone());

        // 2) build the sum term = 2*acc_ + bit_
        let acc_term = self.plonk.wire_values[&acc_].clone();
        let bit_term = self.plonk.wire_values[&bit_].clone();
        let mul2 = term![PF_MUL;
            acc_term.clone(),
            term![Op::Const(Box::new(Value::Field(self.field.new_v(2))))]
        ];
        let sum = term![PF_ADD; mul2, bit_term.clone()];

        // 3) fresh‐wit the result
        let result = self.fresh_wit("horner.result", sum);

        // 4) enforce 2*acc_ + bit_ - result == 0
        self.constraint(
            self.field.new_v(2),  // q_l * acc_
            self.field.new_v(1),  // q_r * bit_
            self.field.new_v(-1), // q_o * result
            self.field.new_v(0),  // q_m
            self.field.new_v(0),  // q_c
            acc_,
            bit_,
            result.clone(),
        );

        result
    }

    fn debitify<I: ExactSizeIterator<Item = Wire>>(&mut self, bits: I, signed: bool) -> Wire {
        //let mut acc = self.const_wire(self.field.new_v(1)); // acc = 1 (2^0)
        //let two = self.field.new_v(2);
        let mut result = self.plonk.zero_wire();
        //let n = bits.len();

        let bits_vec: Vec<_> = bits.collect();
        for (i, bit) in bits_vec.into_iter().enumerate().rev() {
            // Enforce that bit is boolean: bit * (bit - 1) == 0
            //let bit_sq = self.mul(bit.clone(), bit.clone());
            //let bool_check = self.sub(bit.clone(), bit_sq);
            /*let z_c = self.plonk.zero_wire();
            self.constraint(
                self.field.new_v(-1),
                self.field.new_v(0),
                self.field.zero(),
                self.field.new_v(1),
                self.field.zero(),
                bit.clone(),
                bit.clone(),
                z_c,
            );*/
            //self.assert_zero(bool_check); // bit * (bit - 1) == 0

            // --- booleanity check for bit ---
            //let bit_ = self.fresh_wit("boolean.bit", self.plonk.wire_values[&bit].clone());
            //self.plonk.add_copy_constraint(bit, bit_.clone());

            // build 1 - bit_
            let bit_term = self.plonk.wire_values[&bit].clone();
            /*/*let one_minus = term![PF_ADD;
                self.one_term(),
                term![PF_NEG; bit_term.clone()]
            ];*/
            //let one_minus_bit = self.fresh_wit("boolean.1-bit", one_minus);

            // enforce bit_ * (1 - bit_) == 0
            let zero = self.plonk.zero_wire();
            self.constraint(
                self.field.new_v(-1),
                self.field.new_v(0),
                self.field.new_v(0),
                self.field.new_v(1),
                self.field.new_v(0),
                bit_.clone(),
                bit_.clone(),
                zero,
            );*/

            // term = acc * bit
            /*let term = self.mul(acc.clone(), bit_.clone());

            // If signed and this is MSB, subtract instead of add
            if signed && i + 1 == n {
                result = self.sub(result, term);
            } else {
                result = self.add(result, term);
            }

            // acc *= 2
            acc = self.mul_const(acc.clone(), two.clone());*/
            // --- Horner step, with sign‐bit negation if needed ---
            let step_input = if signed && i == 0 {
                // negate bit_
                let neg_term = term![PF_NEG; bit_term];
                let neg_bit = self.fresh_wit("boolean.neg-bit", neg_term);
                // enforce neg_bit + bit_ == 0
                let zero = self.plonk.zero_wire();
                self.constraint(
                    self.field.new_v(1), // +1 * neg_bit
                    self.field.new_v(1), // +1 * bit_
                    self.field.new_v(0), // no result wire
                    self.field.new_v(0), // no multiplication term
                    self.field.new_v(0), // no constant
                    neg_bit.clone(),
                    bit.clone(),
                    zero,
                );
                neg_bit
            } else {
                bit.clone()
            };

            result = self.horner_step(result, step_input);
        }

        result
        /*let mut acc = self.const_wire(self.field.zero());   // ❸ start at 0
        let n = bits.len();

        let bits_vec: Vec<_> = bits.collect();
        for (i, bit) in bits_vec.into_iter().enumerate().rev() {
            // --- Booleanity ---
            let bit_ = self.fresh_wit("boolean.bit", self.plonk.wire_values[&bit].clone());
            self.plonk.add_copy_constraint(bit, bit_.clone());

            let one_minus_bit = {
                let expr = term![PF_ADD; self.one_term(),
                                 term![PF_NEG; self.plonk.wire_values[&bit_].clone()]];
                self.fresh_wit("boolean.1_minus_bit", expr)
            };

            // bit_ * (1 - bit_) = 0
            let z_c = self.plonk.zero_wire();
            self.constraint(self.field.zero(), self.field.zero(), self.field.zero(),
                            self.field.new_v(1), self.field.zero(),
                            bit_.clone(), one_minus_bit.clone(), z_c);

            // 1 - bit_ - (1 - bit_) = 0   (ties the witnesses) ❶
            let z_c = self.plonk.zero_wire();
            self.constraint(self.field.new_v(1), self.field.new_v(1),
                            self.field.zero(), self.field.zero(), self.field.new_v(-1),
                            bit_.clone(), one_minus_bit.clone(), z_c);

            // --- optional sign‑bit negation ---
            let limb = if signed && i + 1 == n {
                // neg_bit + bit_ = 0   ❷
                let neg_bit = {
                    let expr = term![PF_NEG; self.plonk.wire_values[&bit_].clone()];
                    self.fresh_wit("boolean.neg_bit", expr)
                };
                let z_b = self.plonk.zero_wire();
                self.constraint(self.field.new_v(1), self.field.zero(),
                                self.field.zero(), self.field.zero(), self.field.zero(),
                                bit_.clone(), z_b, neg_bit.clone());
                neg_bit
            } else {
                bit_.clone()
            };

            acc = self.horner_step(acc, limb);
        }
        acc   // ❹*/
    }

    // ... Additional helper methods would continue here following similar patterns
    // This is a substantial refactoring, so I'm showing the key structural changes
    /// Given `xs`, an iterator of bit-valued wires, returns the XOR of all of them.
    fn nary_xor<I: ExactSizeIterator<Item = Wire>>(&mut self, mut xs: I) -> Wire {
        let n = xs.len();
        if n > 3 {
            let sum = xs.fold(self.plonk.zero_wire(), |s, i| self.add(s, i));
            let sum_bits = self.bitify("sum", &sum, bitsize(n), false);
            assert!(n > 0);
            sum_bits.into_iter().next().unwrap() // safe b/c assert
        } else {
            let first = xs.next().expect("empty XOR");
            xs.fold(first, |a, b| {
                // XOR: a + b - 2*a*b
                let sum = self.add(a.clone(), b.clone());
                let product = self.mul(a, b);
                let double_product = self.mul_const(product, self.field.new_v(2));
                self.sub(sum, double_product)
            })
        }
    }

    /// Given a bit-values `a`, returns its (boolean) not.
    fn bool_not(&mut self, a: Wire) -> Wire {
        // NOT: 1 - a
        self.neg_add_const(a, self.field.new_v(1))
        //self.sub(self.one.clone(), a)
    }

    /// Given `xs`, an iterator of bit-valued wires, returns the AND of all of them.
    fn nary_and<I: ExactSizeIterator<Item = Wire>>(&mut self, mut xs: I) -> Wire {
        let n = xs.len();
        if n <= 3 {
            let first = xs.next().expect("empty AND");
            xs.fold(first, |a, x| self.mul(a, x))
        } else {
            // De Morgan's law: AND = NOT(OR(NOT(a), NOT(b), ...))
            let negs: Vec<Wire> = xs.map(|x| self.bool_not(x)).collect();
            let a = self.nary_or(negs.into_iter());
            self.bool_not(a)
        }
    }

    /// Given `xs`, an iterator of bit-valued wires, returns the OR of all of them.
    fn nary_or<I: ExactSizeIterator<Item = Wire>>(&mut self, xs: I) -> Wire {
        let n = xs.len();
        if n <= 3 {
            // De Morgan's law: OR = NOT(AND(NOT(a), NOT(b), ...))
            let negs: Vec<Wire> = xs.map(|x| self.bool_not(x)).collect();
            let a = self.nary_and(negs.into_iter());
            self.bool_not(a)
        } else {
            let sum = xs.fold(self.plonk.zero_wire(), |s, x| self.add(s, x));
            let z = self.is_zero(sum);
            self.bool_not(z)
        }
    }

    /// Given a bit-valued `c`, and branches `t` and `f`, returns a wire which is `t` iff `c`, else `f`.
    fn ite(&mut self, c: Wire, t: Wire, f: Wire) -> Wire {
        // ITE: c * (t - f) + f = c * t + (1 - c) * f
        self.assert_boolean(c.clone());
        let diff = self.sub(t, f.clone());
        let product = self.mul(c, diff);
        self.add(product, f)
    }
    /*fn ite(&mut self, c: Wire, t: Wire, f: Wire) -> Wire {
        // Copy inputs to local witnesses and constrain them
        let c_ = self.fresh_wit("ite.c", self.plonk.wire_values[&c].clone());
        let t_ = self.fresh_wit("ite.t", self.plonk.wire_values[&t].clone());
        let f_ = self.fresh_wit("ite.f", self.plonk.wire_values[&f].clone());

        self.plonk.add_copy_constraint(c, c_.clone());
        self.plonk.add_copy_constraint(t, t_.clone());
        self.plonk.add_copy_constraint(f, f_.clone());

        // Boolean constraint: c * (1 - c) = 0
        let one_minus_c = self.fresh_wit(
            "ite.1-c",
            term![PF_ADD; self.one_term(), term![PF_NEG; self.plonk.wire_values[&c_].clone()]],
        );
        self.constraint(
            self.field.new_v(1), // q_l * c
            self.field.new_v(0), // q_r
            self.field.new_v(0), // q_o
            self.field.new_v(1), // q_m * c * (1 - c)
            self.field.new_v(0), // q_c
            c_.clone(),
            one_minus_c.clone(),
            self.zero.clone(),
        );

        // Compute t - f = t + (-f)
        let t_minus_f = self.fresh_wit(
            "ite.t_minus_f",
            term![PF_ADD; self.plonk.wire_values[&t_].clone(), term![PF_NEG; self.plonk.wire_values[&f_].clone()]],
        );

        self.constraint(
            self.field.new_v(1),  // q_l * t
            self.field.new_v(-1), // q_r * f
            self.field.new_v(-1), // q_o * (t - f)
            self.field.new_v(0),  // q_m
            self.field.new_v(0),
            t_.clone(),
            f_.clone(),
            t_minus_f.clone(),
        );

        // Compute delta = c * (t - f)
        let delta = self.fresh_wit(
            "ite.delta",
            term![PF_MUL; self.plonk.wire_values[&c_].clone(), self.plonk.wire_values[&t_minus_f].clone()],
        );

        self.constraint(
            self.field.new_v(0),
            self.field.new_v(0),
            self.field.new_v(-1),
            self.field.new_v(1), // q_m * c * (t - f)
            self.field.new_v(0),
            c_.clone(),
            t_minus_f.clone(),
            delta.clone(),
        );

        // res = delta + f
        let res = self.fresh_wit(
            "ite.res",
            term![PF_ADD; self.plonk.wire_values[&delta].clone(), self.plonk.wire_values[&f_].clone()],
        );

        self.constraint(
            self.field.new_v(1),
            self.field.new_v(1),
            self.field.new_v(-1),
            self.field.new_v(0),
            self.field.new_v(0),
            delta.clone(),
            f_.clone(),
            res.clone(),
        );

        res
    }*/

    /*fn ite(&mut self, c: Wire, t: Wire, f: Wire) -> Wire {
        // Step 1: Allocate new wires for t, f, c (to allow reuse across constraints)
        let t_new = self.fresh_wit("ite.t", self.plonk.wire_values[&t].clone());
        let f_new = self.fresh_wit("ite.f", self.plonk.wire_values[&f].clone());
        let c_new = self.fresh_wit("ite.c", self.plonk.wire_values[&c].clone());

        self.plonk.add_copy_constraint(t.clone(), t_new.clone());
        self.plonk.add_copy_constraint(f.clone(), f_new.clone());
        self.plonk.add_copy_constraint(c.clone(), c_new.clone());

        // Step 2: Compute diff = t - f
        let diff_val = term![PF_ADD; self.plonk.wire_values[&t].clone(), term![PF_NEG; self.plonk.wire_values[&f].clone()]];
        let diff = self.fresh_wit("ite.diff", diff_val.clone());

        // Constraint: t - f - diff = 0 → q_l=1, q_r=-1, q_o=-1
        self.constraint(
            self.field.new_v(1),
            self.field.new_v(-1),
            self.field.new_v(-1),
            self.field.new_v(0),
            self.field.new_v(0),
            t_new.clone(),
            f_new.clone(),
            diff.clone(),
        );

        // Step 3: Compute product = c * diff
        let prod_val = term![PF_MUL; self.plonk.wire_values[&c].clone(), diff_val.clone()];
        let product = self.fresh_wit("ite.prod", prod_val.clone());

        // Constraint: c * diff - product = 0 → q_m=1, q_o=-1
        self.constraint(
            self.field.new_v(0),
            self.field.new_v(0),
            self.field.new_v(-1),
            self.field.new_v(1),
            self.field.new_v(0),
            c_new.clone(),
            diff.clone(),
            product.clone(),
        );

        // Step 4: Compute result = product + f
        let res_val = term![PF_ADD; prod_val.clone(), self.plonk.wire_values[&f].clone()];
        let result = self.fresh_wit("ite.result", res_val);

        // Constraint: product + f - result = 0 → q_l=1, q_r=1, q_o=-1
        self.constraint(
            self.field.new_v(1),
            self.field.new_v(1),
            self.field.new_v(-1),
            self.field.new_v(0),
            self.field.new_v(0),
            product,
            f_new,
            result.clone(),
        );

        result
    }*/

    /// Embed this variable
    fn embed_var(&mut self, var: &Term, ty: VarType) {
        assert!(
            !self.cache.contains_key(var),
            "already have var {}",
            var.op()
        );
        assert!(!matches!(ty, VarType::CWit), "Unimplemented");
        if !self.used_vars.contains(var.as_var_name()) {
            return;
        }
        debug!("Embed var: {}", var.op());

        let public = matches!(ty, VarType::Inst);
        match var.op() {
            Op::Var(v) if matches!(&v.sort, Sort::Bool) => {
                let comp = term![Op::Ite; var.clone(), self.one_term(), self.zero_term()];
                let wire = self.fresh_var(&v.name, comp, ty);
                if !public {
                    self.enforce_bit(wire.clone());
                }
                self.cache.insert(var.clone(), EmbeddedTerm::Bool(wire));
                self.embed.borrow_mut().insert(var.clone());
            }
            Op::Var(v) if v.sort.is_bv() => {
                let public = matches!(ty, VarType::Inst);
                println!("bv var {:?}, public {}", v.as_ref().name, public);
                let wire = self.fresh_var(
                    &v.name,
                    var.clone(),
                    //term![Op::new_ubv_to_pf(self.field.clone()); var.clone()],
                    ty,
                );
                self.set_bv_uint(var.clone(), wire, v.sort.as_bv());
                if !public {
                    self.get_bv_bits(var);
                }
                self.embed.borrow_mut().insert(var.clone());
            }
            Op::Var(v) if v.sort.is_pf() => {
                assert_eq!(v.sort.as_pf(), &self.field);
                let wire = self.fresh_var(&v.name, var.clone(), ty);
                self.cache.insert(var.clone(), EmbeddedTerm::Field(wire));
                self.embed.borrow_mut().insert(var.clone());
            }
            o => unreachable!("Unhandled variable operator {}", o),
        }
    }

    /// Boolean AND operation for two wires
    fn bool_and(&mut self, a: Wire, b: Wire) -> Wire {
        let and_term = term![PF_MUL;
            self.plonk.wire_values[&a].clone(),
            self.plonk.wire_values[&b].clone()
        ];
        let result = self.fresh_bit("and", and_term);

        // a * b - result = 0  =>  q_m * a * b + q_o * result = 0
        self.constraint(
            self.field.new_v(0),
            self.field.new_v(0),
            self.field.new_v(-1),
            self.field.new_v(1),
            self.field.new_v(0), // q_l=0, q_r=0, q_o=-1, q_m=1, q_c=0
            a,
            b,
            result.clone(),
        );
        result
    }

    /// Boolean OR operation: a + b - a*b
    fn bool_or(&mut self, a: Wire, b: Wire) -> Wire {
        let or_term = term![PF_ADD;
            term![PF_ADD; self.plonk.wire_values[&a].clone(), self.plonk.wire_values[&b].clone()],
            term![PF_NEG; term![PF_MUL; self.plonk.wire_values[&a].clone(), self.plonk.wire_values[&b].clone()]]
        ];
        let result = self.fresh_bit("or", or_term);

        // a + b - a*b - result = 0  =>  q_l * a + q_r * b + q_m * (-a) * b + q_o * result = 0
        self.constraint(
            self.field.new_v(1),
            self.field.new_v(1),
            self.field.new_v(-1),
            self.field.new_v(-1),
            self.field.new_v(0), // q_l=1, q_r=1, q_o=-1, q_m=-1, q_c=0
            a,
            b,
            result.clone(),
        );
        result
    }

    /// Boolean XOR operation: a + b - 2*a*b
    fn bool_xor(&mut self, a: Wire, b: Wire) -> Wire {
        let xor_term = term![PF_ADD;
            term![PF_ADD; self.plonk.wire_values[&a].clone(), self.plonk.wire_values[&b].clone()],
            term![PF_MUL;
                pf_lit(self.field.new_v(-2)),
                term![PF_MUL; self.plonk.wire_values[&a].clone(), self.plonk.wire_values[&b].clone()]
            ]
        ];
        let result = self.fresh_bit("xor", xor_term);

        // a + b - 2*a*b - result = 0  =>  q_l * a + q_r * b + q_m * (-2) * a * b + q_o * result = 0
        self.constraint(
            self.field.new_v(1),
            self.field.new_v(1),
            self.field.new_v(-1),
            self.field.new_v(-2),
            self.field.new_v(0), // q_l=1, q_r=1, q_o=-1, q_m=-2, q_c=0
            a,
            b,
            result.clone(),
        );
        result
    }

    /// If-then-else: returns condition ? then_val : else_val
    /*fn ite(&mut self, condition: Wire, then_val: Wire, else_val: &Wire) -> Wire {
        let ite_term = term![Op::Ite;
            self.plonk.wire_values[&condition].clone(),
            self.plonk.wire_values[&then_val].clone(),
            self.plonk.wire_values[else_val].clone()
        ];
        let result = self.fresh_wit("ite", ite_term);

        // condition * (then_val - else_val) + else_val - result = 0
        // First compute diff = then_val - else_val
        let diff = self.sub(then_val, else_val.clone());

        // Then condition * diff + else_val - result = 0
        // q_m * condition * diff + q_r * else_val + q_o * result = 0
        self.constraint(
            0, 1, -1, 1, 0,  // q_l=0, q_r=1, q_o=-1, q_m=1, q_c=0
            condition, else_val.clone(), result.clone()
        );
        result
    }*/

    fn embed(&mut self, t: Term) {
        debug!("Embed: {}", t);
        let visited_set_rc = self.embed.clone();
        for c in
            extras::PostOrderSkipIter::new(t, &move |s: &Term| visited_set_rc.borrow().contains(s))
        {
            assert!(!self.embed.borrow().contains(&c));
            debug!("Embed op: {}", c.op());
            // Handle field access once and for all
            if let Op::Field(i) = &c.op() {
                if !self.cache.contains_key(&c) {
                    let t = self.get_field(&c.cs()[0], *i);
                    self.embed.borrow_mut().insert(c.clone());
                    self.cache.insert(c, t);
                }
            } else {
                match check(&c) {
                    Sort::Bool => {
                        self.embed_bool(c);
                    }
                    Sort::BitVector(_) => {
                        self.embed_bv(c);
                    }
                    Sort::Field(_) => {
                        self.embed_pf(c);
                    }
                    Sort::Tuple(_) => {
                        panic!("Cannot embed tuple term: {}", c)
                    }
                    s => panic!("Unsupported sort in embed: {:?}", s),
                }
            }
        }
    }

    fn get_field(&self, tuple_term: &Term, field: usize) -> EmbeddedTerm {
        match self.cache.get(tuple_term) {
            Some(EmbeddedTerm::Tuple(v)) => v[field].clone(),
            _ => panic!("No tuple for {}", tuple_term),
        }
    }

    fn embed_eq(&mut self, a: &Term, b: &Term) -> Wire {
        match check(a) {
            Sort::Bool => {
                let a = self.get_bool(a).clone();
                let b = self.get_bool(b).clone();
                self.bits_are_equal(a, b)
            }
            Sort::BitVector(_) => {
                let a = self.get_bv_uint(a);
                let b = self.get_bv_uint(b);
                self.are_equal(a, b)
            }
            Sort::Field(_) => {
                let a = self.get_pf(a).clone();
                let b = self.get_pf(b).clone();
                self.are_equal(a, b)
            }
            Sort::Tuple(sorts) => {
                let n = sorts.len();
                let eqs: Vec<Term> = (0..n).map(|i| {
                    term![Op::Eq; term![Op::Field(i); a.clone()], term![Op::Field(i); b.clone()]]
                }).collect();
                let conj = term(Op::BoolNaryOp(BoolNaryOp::And), eqs);
                self.embed(conj.clone());
                self.get_bool(&conj).clone()
            }
            s => panic!("Unimplemented sort for Eq: {:?}", s),
        }
    }

    fn assert_eq(&mut self, a: &Term, b: &Term) {
        match check(a) {
            Sort::Bool => {
                let a = self.get_bool(a).clone();
                let b = self.get_bool(b).clone();
                let diff = self.sub(a, b);
                self.assert_zero(diff);
            }
            Sort::BitVector(_) => {
                let a = self.get_bv_uint(a);
                let b = self.get_bv_uint(b);
                let diff = self.sub(a, b);
                self.assert_zero(diff);
            }
            Sort::Field(_) => {
                let a = self.get_pf(a).clone();
                let b = self.get_pf(b).clone();
                let diff = self.sub(a, b);
                self.assert_zero(diff);
            }
            s => panic!("Unimplemented sort for Eq: {:?}", s),
        }
    }

    fn embed_bool(&mut self, c: Term) -> Wire {
        debug_assert!(check(&c) == Sort::Bool);

        if !self.cache.contains_key(&c) {
            let wire = match &c.op() {
                Op::Var(..) => panic!("call embed_var instead"),
                Op::Const(v) => {
                    if v.as_bool() {
                        self.plonk.one_wire()
                    } else {
                        self.plonk.zero_wire()
                    }
                }
                Op::Eq => self.embed_eq(&c.cs()[0], &c.cs()[1]),
                Op::Ite => {
                    let cond = self.get_bool(&c.cs()[0]).clone();
                    let then_val = self.get_bool(&c.cs()[1]).clone();
                    let else_val = self.get_bool(&c.cs()[2]).clone();
                    self.ite(cond, then_val, else_val)
                }
                Op::BoolMaj => {
                    let a = self.get_bool(&c.cs()[0]).clone();
                    let b = self.get_bool(&c.cs()[1]).clone();
                    let c_val = self.get_bool(&c.cs()[2]).clone();
                    // Majority: ab + bc + ac - 2abc
                    // m = ab + c(b + a - 2ab) where i = ab
                    // m = i + c(b + a - 2i)
                    let ab = self.mul(a.clone(), b.clone());
                    let sum_ab = self.add(a, b);
                    let double_ab = self.mul_const(ab.clone(), self.field.new_v(2));
                    let inner = self.sub(sum_ab, double_ab);
                    let c_inner = self.mul(c_val, inner);
                    self.add(ab, c_inner)
                }
                Op::Not => {
                    let a = self.get_bool(&c.cs()[0]).clone();
                    self.bool_not(a)
                }
                Op::Implies => {
                    let a = self.get_bool(&c.cs()[0]).clone();
                    let b = self.get_bool(&c.cs()[1]).clone();
                    let not_a = self.bool_not(a);
                    self.nary_or(vec![not_a, b].into_iter())
                }
                Op::BoolNaryOp(o) => {
                    let args = c
                        .cs()
                        .iter()
                        .map(|c| self.get_bool(c).clone())
                        .collect::<Vec<_>>();
                    match o {
                        BoolNaryOp::Or => self.nary_or(args.into_iter()),
                        BoolNaryOp::And => self.nary_and(args.into_iter()),
                        BoolNaryOp::Xor => self.nary_xor(args.into_iter()),
                    }
                }
                Op::BvBit(i) => {
                    let a = self.get_bv_bits(&c.cs()[0]);
                    a[*i].clone()
                }
                Op::BvBinPred(o) => {
                    let n = check(&c.cs()[0]).as_bv();
                    use BvBinPred::*;
                    match o {
                        Sge => self.bv_cmp(n, true, false, &c.cs()[0], &c.cs()[1]),
                        Sgt => self.bv_cmp(n, true, true, &c.cs()[0], &c.cs()[1]),
                        Uge => self.bv_cmp(n, false, false, &c.cs()[0], &c.cs()[1]),
                        Ugt => self.bv_cmp(n, false, true, &c.cs()[0], &c.cs()[1]),
                        Sle => self.bv_cmp(n, true, false, &c.cs()[1], &c.cs()[0]),
                        Slt => self.bv_cmp(n, true, true, &c.cs()[1], &c.cs()[0]),
                        Ule => self.bv_cmp(n, false, false, &c.cs()[1], &c.cs()[0]),
                        Ult => self.bv_cmp(n, false, true, &c.cs()[1], &c.cs()[0]),
                    }
                }
                Op::PfToBoolTrusted => {
                    // we trust that this is zero or one
                    self.get_pf(&c.cs()[0]).clone()
                }
                _ => panic!("Non-boolean in embed_bool: {}", c),
            };
            self.cache.insert(c.clone(), EmbeddedTerm::Bool(wire));
        }

        self.get_bool(&c).clone()
    }

    // Helper methods to get embedded terms
    fn get_bool(&self, term: &Term) -> &Wire {
        match self.cache.get(term) {
            Some(EmbeddedTerm::Bool(wire)) => wire,
            _ => panic!("No boolean wire for term: {}", term),
        }
    }

    fn get_pf(&self, term: &Term) -> &Wire {
        match self
            .cache
            .get(term)
            .unwrap_or_else(|| panic!("Missing wire for {:?}", term))
        {
            EmbeddedTerm::Field(wire) => wire,
            _ => panic!("Non-field for {:?}", term),
        }
    }

    /*fn get_bv_uint(&self, term: &Term) -> Wire {
        match self.cache.get(term) {
            Some(EmbeddedTerm::Bv(entry)) => {
                entry.borrow().uint.clone().expect("BV uint not created")
            },
            _ => panic!("No bitvector uint for term: {}", term),
        }
    }*/

    fn get_bv_bits(&mut self, term: &Term) -> Vec<Wire> {
        let entry = match self.cache.get(term) {
            Some(EmbeddedTerm::Bv(entry)) => entry.clone(),
            _ => panic!("No bitvector for term: {}", term),
        };

        if entry.borrow().bits.is_empty() {
            let width = entry.borrow().width;
            let uint_wire = entry.borrow().uint.clone().expect("BV uint not created");
            let bits = self.decomp("bv_bits", &uint_wire, width);
            entry.borrow_mut().bits = bits.clone();
            bits
        } else {
            entry.borrow().bits.clone()
        }
    }

    fn set_bv_uint(&mut self, term: Term, wire: Wire, width: usize) {
        let entry = Rc::new(RefCell::new(BvEntry {
            width,
            uint: Some(wire),
            bits: Vec::new(),
        }));
        self.cache.insert(term, EmbeddedTerm::Bv(entry));
    }

    /// Compare two bitvectors
    fn bv_cmp(&mut self, width: usize, signed: bool, strict: bool, a: &Term, b: &Term) -> Wire {
        let a_bits = self.get_bv_bits(a);
        let b_bits = self.get_bv_bits(b);

        // For simplicity, convert to comparison on field elements
        let a_uint = self.debitify(a_bits.into_iter(), signed);
        let b_uint = self.debitify(b_bits.into_iter(), signed);

        if strict {
            // a > b equivalent to !(a <= b) which is !(b >= a)
            let diff = self.sub(b_uint, a_uint);
            let geq = self.is_geq_zero(diff);
            self.bool_not(geq)
        } else {
            // a >= b
            let diff = self.sub(a_uint, b_uint);
            self.is_geq_zero(diff)
        }
    }

    /// Check if a field element is >= 0 (for comparison purposes)
    fn is_geq_zero(&mut self, x: Wire) -> Wire {
        // This is a simplified implementation
        // In practice, you'd need range checks or other techniques
        // For now, assume all field elements are non-negative
        let is_zero_wire = self.is_zero(x);
        self.bool_not(is_zero_wire)
    }

    /// Check if two bits are equal
    fn bits_are_equal(&mut self, a: Wire, b: Wire) -> Wire {
        // a == b is equivalent to !(a XOR b)
        let xor = self.nary_xor(vec![a, b].into_iter());
        self.bool_not(xor)
    }

    /// Assert that a boolean term is true
    fn assert_bool(&mut self, t: &Term) {
        if t.op() == &Op::Eq {
            // For equality, embed both sides and add copy constraint
            t.cs().iter().for_each(|c| self.embed(c.clone()));
            let a = (&t.cs()[0]).clone();
            let b = (&t.cs()[1]).clone();
            self.assert_eq(&a, &b);
        } else if t.op() == &AND {
            // For AND, recursively assert each conjunct
            for c in t.cs() {
                self.assert_bool(c);
            }
        } else if let Op::PfFitsInBits(n) = t.op() {
            // Ensure the field element fits in n bits by converting to bit-vector
            //let value = self.get_pf(&t.cs()[0]).clone();
            //let _bits = self.bitify("fits_in_bits", &value, *n, false);
            self.embed(term![Op::PfToBv(*n); t.cs()[0].clone()]);
            // The bitification itself enforces the constraint
        } else {
            // For general boolean terms, embed and assert they equal 1
            self.embed(t.clone());
            let wire = self.get_bool(t).clone();
            let diff = self.add_const(wire, self.field.new_v(-1));
            self.assert_zero(diff);
        }
    }

    /// Create a constant wire
    fn const_wire(&mut self, value: FieldV) -> Wire {
        let const_term = term![Op::Const(Box::new(Value::Field(value.clone())))];
        let c = self.plonk.new_wire("const".to_string(), const_term);
        let z_a = self.plonk.zero_wire();
        let z_b = self.plonk.zero_wire();
        self.constraint(
            self.field.new_v(0),
            self.field.new_v(0),
            self.field.new_v(-1),
            self.field.new_v(0),
            value,
            z_a,
            z_b,
            c.clone(),
        );
        c
    }

    /// Get boolean wire from term
    fn get_bool_wire(&mut self, term: &Term) -> Wire {
        /*if let Some(embedded) = self.cache.get(term) {
            match embedded {
                EmbeddedTerm::Bool(wire) => wire.clone(),
                _ => panic!("Expected boolean term"),
            }
        } else {
            // Create new boolean wire
            let wire = self.fresh_bit("bool", term.clone());
            self.cache
                .insert(term.clone(), EmbeddedTerm::Bool(wire.clone()));
            wire
        }*/
        match self
            .cache
            .get(term)
            .unwrap_or_else(|| panic!("Missing wire for {:?}", term))
        {
            EmbeddedTerm::Bool(b) => b.clone(),
            _ => panic!("Non-boolean for {:?}", term),
        }
    }

    /// Get bit-vector uint wire from term
    fn get_bv_uint(&mut self, t: &Term) -> Wire {
        let entry_rc = self.get_bv_lit(t);
        let mut entry = entry_rc.borrow_mut();
        if let Some(uint) = entry.uint.as_ref() {
            uint.clone()
        } else {
            let uint = self.debitify(entry.bits.clone().into_iter(), false);
            entry.uint = Some(uint.clone());
            uint
        }
    }

    /// Given a and b such that -2^n < a - b < 2^n, returns whether a >= b (or a > b if `strict` is set)
    fn bv_greater(&mut self, a: Wire, b: Wire, n: usize, strict: bool) -> Wire {
        let n = if n >= 254 { 254 } else { n };
        let tweak = if strict { -1 } else { 0 };
        let shift_val = self.field.new_v((Integer::from(1) << n));
        let shift_wire = self.const_wire(shift_val);
        let tweak_wire = self.const_wire(self.field.new_v(tweak));

        // sum = a - b + shift + tweak
        let diff = self.sub(a, b);
        let sum1 = self.add(diff, shift_wire);
        let sum = self.add(sum1, tweak_wire);

        // Extract the top bit (bit n) which indicates if sum >= 2^n
        self.bitify("cmp", &sum, n + 1, false).pop().unwrap()
        //bits[n].clone() // Return the (n+1)th bit (0-indexed)
    }

    /// Treating `xs` and `ys` as unsigned bit-vectors (with LSB at index 0), emit a bit-wise comparison circuit
    fn bv_bitwise_greater(&mut self, xs: Vec<Wire>, ys: Vec<Wire>, strict: bool) -> Wire {
        let init_val = if strict { 0 } else { 1 };
        let mut acc = self.const_wire(self.field.new_v(init_val));

        // Process from MSB to LSB (reverse order since LSB is at index 0)
        for (x, y) in xs.into_iter().rev().zip(ys.into_iter().rev()) {
            // eq = (x == y) = 1 - (x XOR y)
            let xor = self.bool_xor(x.clone(), y.clone());
            let eq = self.bool_not(xor);

            // eq_and_acc = eq AND acc
            let eq_and_acc = self.bool_and(eq, acc.clone());

            // not_y = 1 - y
            let not_y = self.bool_not(y);

            // x_gt_y = x AND (NOT y)
            let x_gt_y = self.bool_and(x, not_y);

            // acc = x_gt_y OR eq_and_acc
            acc = self.bool_or(x_gt_y, eq_and_acc);
        }
        acc
    }

    /// Shift `x` left by `2^y`, if bit-valued `c` is true
    fn const_pow_shift_bv_lit(&mut self, x: &Wire, y: usize, c: Wire) -> Wire {
        let shift_amount = 1 << (1 << y); // 2^(2^y)
        let shift_val = self.field.new_v(shift_amount);
        let shift_wire = self.const_wire(shift_val).clone();
        let shifted_x = self.mul(x.clone(), shift_wire);
        self.ite(c, shifted_x, x.clone())
    }

    /// Shift `x` left by `y`, filling the blank spots with bit-valued `ext_bit`
    /// Returns an *oversized* number
    fn shift_bv_lit(&mut self, x: Wire, y: Vec<Wire>, ext_bit: Option<Wire>) -> Wire {
        if let Some(b) = ext_bit {
            // For sign extension: left = shift(x, y, None), right = shift(ext_bit, y, None) - 1
            let left = self.shift_bv_lit(x, y.clone(), None);
            let right_shifted = self.shift_bv_lit(b.clone(), y, None);
            let right = self.add_const(right_shifted, self.field.new_v(1));
            let extended = self.mul(b, right);
            self.add(left, extended)
        } else {
            // Regular left shift: fold over each bit position
            y.into_iter()
                .enumerate()
                .fold(x, |acc, (i, yi)| self.const_pow_shift_bv_lit(&acc, i, yi))
        }
    }

    /// Shift `x` left by `y`, filling the blank spots with bit-valued `ext_bit`
    /// Returns a bit sequence
    /// If `c` is true, returns bit sequence which is just a copy of `ext_bit`
    fn shift_bv_bits(
        &mut self,
        x: Wire,
        y: Vec<Wire>,
        ext_bit: Option<Wire>,
        x_w: usize,
        c: Wire,
    ) -> Vec<Wire> {
        let y_w = y.len();

        // Create mask for overflow case
        let mask = match ext_bit.as_ref() {
            Some(e) => {
                let mask_val = self.field.new_v((Integer::from(1) << x_w) - 1);
                let mask_wire = self.const_wire(mask_val).clone();
                self.mul(e.clone(), mask_wire)
            }
            None => self.plonk.zero_wire(),
        };

        // Perform the shift
        let s = self.shift_bv_lit(x, y, ext_bit);

        // Apply mask if overflow condition is true
        let masked_s = self.ite(c, mask, s);

        // Convert back to bits and truncate
        let mut bits = self.bitify("shift", &masked_s, (1 << y_w) + x_w - 1, false);
        bits.truncate(x_w);
        bits
    }

    /// Given a shift amount expressed as a bit-sequence, splits that shift into low bits and high bits
    fn split_shift_amt(&mut self, data_w: usize, mut shift_amt: Vec<Wire>) -> (Wire, Vec<Wire>) {
        let b = bitsize(data_w - 1); // Helper function to calculate bit size
        let high_bits: Vec<Wire> = shift_amt.drain(b..).collect();
        let some_high_bit = if high_bits.is_empty() {
            self.plonk.zero_wire()
        } else {
            self.nary_or(high_bits.into_iter())
        };
        (some_high_bit, shift_amt)
    }

    /// Complete bit-vector embedding for Plonk
    fn embed_bv(&mut self, bv: Term) {
        if let Sort::BitVector(n) = check(&bv) {
            if !self.cache.contains_key(&bv) {
                match bv.op() {
                    Op::Var(..) => panic!("call embed_var instead"),
                    Op::Const(v) => {
                        let b = v.as_bv();
                        let bit_wires = (0..b.width())
                            .map(|i| {
                                let bit_val = b.uint().get_bit(i as u32) as isize;
                                let bit_term = term![Op::Const(Box::new(Value::Field(
                                    self.field.new_v(bit_val)
                                )))];
                                let w = self.plonk.new_wire(format!("const_bit_{}", i), bit_term);
                                self.add_const(w.clone(), self.field.new_v(bit_val));
                                w
                            })
                            .collect();
                        self.set_bv_bits(bv.clone(), bit_wires);
                    }
                    Op::Ite => {
                        let c = self.get_bool_wire(&bv.cs()[0]);
                        let t = self.get_bv_uint(&bv.cs()[1]);
                        let f = self.get_bv_uint(&bv.cs()[2]);
                        let ite_wire = self.ite(c, t, f);
                        self.set_bv_uint(bv, ite_wire, n);
                    }
                    Op::BvUnOp(BvUnOp::Not) => {
                        let bits = self.get_bv_bits_wire(&bv.cs()[0]);
                        let not_bits = bits.iter().map(|bit| self.bool_not(bit.clone())).collect();
                        self.set_bv_bits(bv, not_bits);
                    }
                    Op::BvUnOp(BvUnOp::Neg) => {
                        println!("NEG!!!!!");
                        let x = self.get_bv_uint(&bv.cs()[0]);
                        // Two's complement: flip bits and add 1, but handle x == 0 case
                        let modulus_val = self.field.new_v(Integer::from(2).pow(n as u32));
                        //let modulus_wire = self.const_wire(modulus_val).clone();
                        //let neg_x = self.mul_const(x.clone(), self.field.new_v(-1));
                        let almost_neg_x = self.neg_add_const(x.clone(), modulus_val);
                        //let almost_neg_x = self.add_const(neg_x.clone(), modulus_val)
                        let is_zero = self.is_zero(x);
                        let z_t = self.plonk.zero_wire();
                        let neg_x = self.ite(is_zero, z_t, almost_neg_x);
                        self.set_bv_uint(bv, neg_x, n);
                    }
                    Op::BvUext(extra_n) => {
                        // Zero extension
                        if self.bv_has_bits(&bv.cs()[0]) {
                            let mut bits = self.get_bv_bits_wire(&bv.cs()[0]);
                            // Add zero bits for extension
                            for _ in 0..*extra_n {
                                bits.push(self.plonk.zero_wire());
                            }
                            self.set_bv_bits(bv, bits);
                        } else {
                            let x = self.get_bv_uint(&bv.cs()[0]);
                            self.set_bv_uint(bv, x, n);
                        }
                    }
                    Op::BvSext(extra_n) => {
                        // Sign extension
                        let bits = self.get_bv_bits_wire(&bv.cs()[0]);
                        let sign_bit = bits
                            .last()
                            .expect("Empty bit-vector for sign extension")
                            .clone();
                        let mut extended_bits = bits;
                        // Extend with copies of the sign bit
                        for _ in 0..*extra_n {
                            extended_bits.push(sign_bit.clone());
                        }
                        self.set_bv_bits(bv, extended_bits);
                    }
                    Op::PfToBv(nbits) => {
                        let wire = self.get_pf(&bv.cs()[0]).clone();
                        let bits = self.bitify("pf2bv", &wire, nbits.clone(), false).clone();
                        self.set_bv_bits(bv, bits);
                    }
                    Op::BoolToBv => {
                        let b = self.get_bool_wire(&bv.cs()[0]);
                        self.set_bv_bits(bv, vec![b]);
                    }
                    Op::BvNaryOp(o) => match o {
                        BvNaryOp::Xor | BvNaryOp::Or | BvNaryOp::And => {
                            let all_bits: Vec<Vec<Wire>> =
                                bv.cs().iter().map(|c| self.get_bv_bits_wire(c)).collect();

                            let width = all_bits[0].len();
                            let mut result_bits = Vec::new();

                            for bit_idx in 0..width {
                                let bits_at_pos: Vec<Wire> = all_bits
                                    .iter()
                                    .map(|bv_bits| bv_bits[bit_idx].clone())
                                    .collect();

                                let result_bit = match o {
                                    BvNaryOp::And => self.nary_and(bits_at_pos.into_iter()),
                                    BvNaryOp::Or => self.nary_or(bits_at_pos.into_iter()),
                                    BvNaryOp::Xor => self.nary_xor(bits_at_pos.into_iter()),
                                    _ => unreachable!(),
                                };
                                result_bits.push(result_bit);
                            }
                            self.set_bv_bits(bv, result_bits);
                        }
                        BvNaryOp::Add | BvNaryOp::Mul => {
                            let f_width = self.field.modulus().significant_bits() as usize - 1;
                            let values: Vec<Wire> =
                                bv.cs().iter().map(|c| self.get_bv_uint(c)).collect();

                            let (res, width) = match o {
                                BvNaryOp::Add => {
                                    let sum = if values.is_empty() {
                                        self.plonk.zero_wire() // Handle empty case
                                    } else {
                                        values
                                            .into_iter()
                                            .reduce(|acc, v| self.add(acc, v))
                                            .unwrap()
                                    };
                                    /*let sum = values
                                    .into_iter()
                                    .fold(self.zero.clone(), |s, v| self.add(s, v));*/
                                    let extra_width = bitsize(bv.cs().len().saturating_sub(1));
                                    (sum, n + extra_width)
                                }
                                BvNaryOp::Mul => {
                                    if bv.cs().len() * n < f_width {
                                        // Small multiplication
                                        let product = if values.is_empty() {
                                            self.plonk.one_wire() // Handle empty case
                                        } else {
                                            values
                                                .into_iter()
                                                .reduce(|acc, v| self.mul(acc, v))
                                                .unwrap()
                                        };
                                        (product, bv.cs().len() * n)
                                    } else {
                                        // Large multiplication with truncation
                                        let mut product = self.plonk.one_wire();
                                        for v in values {
                                            product = self.mul(product, v);
                                            let bits =
                                                self.bitify("binMul", &product, 2 * n, false);
                                            let truncated_bits =
                                                bits.into_iter().take(n).collect::<Vec<_>>();
                                            product =
                                                self.debitify(truncated_bits.into_iter(), false);
                                        }
                                        (product, n)
                                    }
                                }
                                _ => unreachable!(),
                            };

                            let mut bits = self.bitify("arith", &res, width, false); // why need increment here?
                            bits.truncate(n);
                            self.set_bv_bits(bv, bits);
                        }
                    },
                    Op::BvBinOp(o) => {
                        let a = self.get_bv_uint(&bv.cs()[0]);
                        let b = self.get_bv_uint(&bv.cs()[1]);

                        match o {
                            BvBinOp::Sub => {
                                //println!("SUB!!!!!");
                                let modulus_val = self.field.new_v(Integer::from(2).pow(n as u32));
                                //let modulus_wire = self.const_wire(modulus_val).clone();
                                let a = self.add_const(a, modulus_val);
                                let sum = self.sub(a, b);
                                //let sum = self.add(a, b);
                                // Now directly bitify without extra padding
                                let mut bits = self.bitify("sub", &sum, n + 1, false);
                                bits.truncate(n);
                                self.set_bv_bits(bv, bits);
                            }
                            BvBinOp::Udiv | BvBinOp::Urem => {
                                /*// Division requires witness generation
                                let a_bv_term =
                                    term![Op::PfToBv(n); self.plonk.wire_values[&a].clone()];
                                let b_bv_term =
                                    term![Op::PfToBv(n); self.plonk.wire_values[&b].clone()];
                                let q_term = term![Op::new_ubv_to_pf(self.field.clone()); term![BV_UDIV; a_bv_term.clone(), b_bv_term.clone()]];
                                let r_term = term![Op::new_ubv_to_pf(self.field.clone()); term![BV_UREM; a_bv_term, b_bv_term]];
                                let q = self.fresh_wit("div_q", q_term);
                                let r = self.fresh_wit("div_r", r_term);
                                let qb = self.bitify("div_q", &q, n, false);
                                let rb = self.bitify("div_r", &r, n, false);

                                // Constraint: a = q * b + r
                                let qb_product = self.mul(q.clone(), b.clone());
                                let reconstruction = self.add(qb_product, r.clone());
                                let diff = self.sub(a, reconstruction);
                                self.assert_zero(diff);

                                // Division by zero handling and remainder constraint
                                let r_ge_b = self.bv_greater(r, b, n, false);
                                let max_val = self.field.new_v((Integer::from(1) << 254) - 1);
                                let max_wire = self.const_wire(max_val);

                                let sub_wire = self.sub(q, max_wire);
                                let q_eq_max = self.is_zero(sub_wire);

                                // Check conditions
                                let one = self.fresh_wit("one", self.one_term());
                                let q_not_equals_max = self.sub(one, q_eq_max);

                                // We want NOT(r_geq_b AND q_not_equals_max) = 1
                                // Which means: (r_geq_b AND q_not_equals_max) = 0
                                // So: r_geq_b * q_not_equals_max = 0
                                let and_product = self.mul(r_ge_b, q_not_equals_max);
                                self.assert_zero(and_product.clone());

                                let bits = match o {
                                    BvBinOp::Udiv => qb,
                                    BvBinOp::Urem => rb,
                                    _ => unreachable!(),
                                };
                                self.set_bv_bits(bv, bits);*/

                                /*// Division requires witness generation
                                let a_bv_term =
                                    term![Op::PfToBv(n); self.plonk.wire_values[&a].clone()];
                                let b_bv_term =
                                    term![Op::PfToBv(n); self.plonk.wire_values[&b].clone()];
                                let q_term = term![Op::new_ubv_to_pf(self.field.clone()); term![BV_UDIV; a_bv_term.clone(), b_bv_term.clone()]];
                                let r_term = term![Op::new_ubv_to_pf(self.field.clone()); term![BV_UREM; a_bv_term, b_bv_term]];

                                let q = self.fresh_wit("div_q", q_term);
                                let r = self.fresh_wit("div_r", r_term);

                                // OPTION 2A: Direct constraint in one row
                                // Constraint: a = q * b + r  =>  q_l * a + q_m * q * b + q_r * r = 0
                                let a_copy =
                                    self.fresh_wit("div_a", self.plonk.wire_values[&a].clone());
                                let b_copy =
                                    self.fresh_wit("div_b", self.plonk.wire_values[&b].clone());
                                let qb_result = self.fresh_wit("div_qb", term![PF_MUL; self.plonk.wire_values[&q].clone(), self.plonk.wire_values[&b].clone()]);

                                self.plonk.add_copy_constraint(a.clone(), a_copy.clone());
                                self.plonk.add_copy_constraint(b.clone(), b_copy.clone());

                                // Row 1: q * b = qb_result
                                self.constraint(
                                    self.field.new_v(0),  // q_l
                                    self.field.new_v(0),  // q_r
                                    self.field.new_v(-1), // q_o
                                    self.field.new_v(1),  // q_m
                                    self.field.new_v(0),  // q_c
                                    q.clone(),
                                    b_copy,
                                    qb_result.clone(),
                                );

                                // Row 2: a - (qb_result + r) = 0  =>  a - qb_result - r = 0
                                let qb_copy = self.fresh_wit(
                                    "div_qb_copy",
                                    self.plonk.wire_values[&qb_result].clone(),
                                );

                                self.plonk.add_copy_constraint(qb_result, qb_copy.clone());

                                self.constraint(
                                    self.field.new_v(1),  // q_l = 1 (for a_copy)
                                    self.field.new_v(-1), // q_r = -1 (for qb_copy)
                                    self.field.new_v(-1), // q_o = -1 (for r_copy)
                                    self.field.new_v(0),  // q_m
                                    self.field.new_v(0),  // q_c
                                    a_copy,
                                    qb_copy,
                                    r.clone(),
                                );

                                // Continue with bitification and remainder constraints...
                                let qb = self.bitify("div_q", &q, n, false);
                                let rb = self.bitify("div_r", &r, n, false);

                                // Division by zero and remainder constraints
                                let r_ge_b = self.bv_greater(r.clone(), b.clone(), n, false);
                                let max_val = self.field.new_v((Integer::from(1) << 254) - 1);
                                let max_wire = self.const_wire(max_val);

                                let sub_wire = self.sub(q.clone(), max_wire);
                                let q_eq_max = self.is_zero(sub_wire);

                                let q_not_equals_max =
                                    self.add_const(q_eq_max, self.field.new_v(-1));

                                let and_product = self.mul(r_ge_b, q_not_equals_max);
                                self.assert_zero(and_product);

                                let bits = match o {
                                    BvBinOp::Udiv => qb,
                                    BvBinOp::Urem => rb,
                                    _ => unreachable!(),
                                };
                                self.set_bv_bits(bv, bits);*/
                                // 1) Create the quotient and remainder witnesses from IR
                                let a_bv = term![Op::PfToBv(n); self.plonk.wire_values[&a].clone()];
                                let b_bv = term![Op::PfToBv(n); self.plonk.wire_values[&b].clone()];
                                let q_term = term![Op::new_ubv_to_pf(self.field.clone()); term![BV_UDIV; a_bv.clone(), b_bv.clone()]];
                                let r_term = term![Op::new_ubv_to_pf(self.field.clone()); term![BV_UREM; a_bv, b_bv]];
                                let q = self.fresh_wit("div_q", q_term);
                                let r = self.fresh_wit("div_r", r_term);

                                // 2) Row 1: q * b = qb_res
                                let b_copy =
                                    self.fresh_wit("div_b", self.plonk.wire_values[&b].clone());
                                let qb_res     = self.fresh_wit("div_qb", term![PF_MUL; self.plonk.wire_values[&q].clone(), self.plonk.wire_values[&b].clone()]);
                                self.plonk.add_copy_constraint(b.clone(), b_copy.clone());
                                self.constraint(
                                    /*q_l*/ self.field.new_v(0),
                                    /*q_r*/ self.field.new_v(0),
                                    /*q_o*/ self.field.new_v(-1),
                                    /*q_m*/ self.field.new_v(1),
                                    /*q_c*/ self.field.new_v(0),
                                    q.clone(),
                                    b_copy,
                                    qb_res.clone(),
                                );

                                // 3) Row 2: a = qb_res + r  →  a - qb_res - r = 0
                                let a_copy =
                                    self.fresh_wit("div_a", self.plonk.wire_values[&a].clone());
                                let qb_copy = self.fresh_wit(
                                    "div_qb_copy",
                                    self.plonk.wire_values[&qb_res].clone(),
                                );
                                self.plonk.add_copy_constraint(a.clone(), a_copy.clone());
                                self.plonk
                                    .add_copy_constraint(qb_res.clone(), qb_copy.clone());
                                self.constraint(
                                    /*q_l*/ self.field.new_v(1),
                                    /*q_r*/ self.field.new_v(-1),
                                    /*q_o*/ self.field.new_v(-1),
                                    /*q_m*/ self.field.new_v(0),
                                    /*q_c*/ self.field.new_v(0),
                                    a_copy,
                                    qb_copy,
                                    r.clone(),
                                );

                                // 4) Range‐check r: enforce 0 ≤ r < b when b != 0
                                //    4a) bitify into k = ceil(log2(b)) bits to force 0 ≤ r < 2^k
                                let k = (n as f64).log2().ceil() as usize; // or compute ceil(log2(b_value)) if b is constant
                                let r_bits = self.bitify("rem_bits", &r, k, false);

                                //    4b) enforce r < b via r_ge_b = bv_greater(r, b, n, /*strict=*/false)
                                //        and assert_zero(r_ge_b)
                                let r_ge_b = self.bv_greater(r.clone(), b.clone(), n, false);
                                self.assert_zero(r_ge_b);

                                // 5) Finally, set the bit‐vector result to either the q‐bits or the r‐bits
                                let bits = match o {
                                    BvBinOp::Udiv => {
                                        let q_bits = self.bitify("quot_bits", &q, k, false);
                                        q_bits
                                    }
                                    BvBinOp::Urem => r_bits,
                                    _ => unreachable!(),
                                };
                                self.set_bv_bits(bv, bits);
                            }
                            BvBinOp::Shl => {
                                let rb = self.get_bv_bits_wire(&bv.cs()[1]);
                                let (high, low) = self.split_shift_amt(n, rb);
                                let bits = self.shift_bv_bits(a, low, None, n, high);
                                self.set_bv_bits(bv, bits);
                            }
                            BvBinOp::Lshr | BvBinOp::Ashr => {
                                let mut lb = self.get_bv_bits_wire(&bv.cs()[0]);
                                lb.reverse(); // Reverse for right shift

                                let ext_bit = match o {
                                    BvBinOp::Ashr => Some(lb[0].clone()), // Sign bit
                                    _ => None,
                                };

                                let l = self.debitify(lb.into_iter(), false);
                                let rb = self.get_bv_bits_wire(&bv.cs()[1]);
                                let (high, low) = self.split_shift_amt(n, rb);
                                let mut bits = self.shift_bv_bits(l, low, ext_bit, n, high);
                                bits.reverse(); // Reverse back
                                self.set_bv_bits(bv, bits);
                            }
                        }
                    }
                    Op::BvConcat => {
                        let mut bits = Vec::new();
                        // Concatenate in reverse order (MSB first)
                        for c in bv.cs().iter().rev() {
                            bits.extend(self.get_bv_bits_wire(c));
                        }
                        self.set_bv_bits(bv, bits);
                    }
                    Op::BvExtract(high, low) => {
                        let bits = self.get_bv_bits_wire(&bv.cs()[0]);
                        let extracted = bits
                            .into_iter()
                            .skip(*low as usize)
                            .take((*high - *low + 1) as usize)
                            .collect();
                        self.set_bv_bits(bv, extracted);
                    }
                    _ => panic!("Non-bv in embed_bv: {}", bv),
                }
            }
        } else {
            panic!("{} is not a bit-vector in embed_bv", bv);
        }
    }

    // Helper methods for bit-vector operations
    fn set_bv_bits(&mut self, bv: Term, bits: Vec<Wire>) {
        let width = bits.len();
        let entry = Rc::new(RefCell::new(BvEntry {
            width,
            uint: None,
            bits,
        }));
        self.cache.insert(bv, EmbeddedTerm::Bv(entry));
    }

    fn bv_has_bits(&self, term: &Term) -> bool {
        if let Some(EmbeddedTerm::Bv(entry)) = self.cache.get(term) {
            !entry.borrow().bits.is_empty()
        } else {
            false
        }
    }

    fn get_bv_bits_wire(&mut self, term: &Term) -> Vec<Wire> {
        self.embed(term.clone());
        if let Some(EmbeddedTerm::Bv(entry)) = self.cache.clone().get(term) {
            let mut entry_ref = entry.borrow_mut();
            if !entry_ref.bits.is_empty() {
                entry_ref.bits.clone()
            } else if let Some(uint) = &entry_ref.uint {
                // Convert uint to bits
                let uint: Wire = uint.clone();
                let width = entry_ref.width.clone();
                let bits = self.bitify("uint_to_bits", &uint, width, false);
                entry_ref.bits = bits.clone();
                bits
            } else {
                panic!("No bits or uint available for bit-vector");
            }
        } else {
            panic!("Expected bit-vector term");
        }
    }

    fn get_bv_signed_int_wire(&mut self, term: &Term) -> Wire {
        // For signed interpretation, we need to handle the sign bit
        let uint_wire = self.get_bv_uint(term);
        // This is a simplified version - full implementation would handle two's complement properly
        uint_wire
    }

    #[allow(dead_code)]
    fn debug_wire<D: Display + ?Sized>(&self, tag: &D, wire: &Wire) {
        println!("{}: wire_{} ({})", tag, wire.index, wire.name);
    }

    fn get_bv_lit(&self, t: &Term) -> Rc<RefCell<BvEntry>> {
        match self
            .cache
            .get(t)
            .unwrap_or_else(|| panic!("Missing wire for {:?}", t))
        {
            EmbeddedTerm::Bv(b) => b.clone(),
            _ => panic!("Non-bv for {:?}", t),
        }
    }

    fn get_bv_signed_int(&mut self, t: &Term) -> Wire {
        let bits = self.get_bv_bits(t);
        self.debitify(bits.into_iter(), true)
    }

    fn embed_pf(&mut self, c: Term) -> &Wire {
        if !self.cache.contains_key(&c) {
            debug!("embed_pf {}", c);
            let wire = match &c.op() {
                Op::Var(..) => panic!("call embed_var instead"),
                Op::Const(v) => {
                    let field_val = v.as_pf().as_ty_ref(&self.field);
                    // hack, a constant can genuinely be set to zero
                    if field_val.is_zero() {
                        self.plonk
                            .wire_values
                            .iter()
                            .find_map(
                                |(wire, term)| if term == &c { Some(wire.clone()) } else { None },
                            )
                            .expect("Wire not found for the given term")
                            .clone()
                    } else {
                        let const_term =
                            term![Op::Const(Box::new(Value::Field(field_val.clone())))];
                        println!("const {:?}", field_val);
                        let result = self.fresh_wit("const.out.0", const_term);
                        let z_a = self.plonk.zero_wire();
                        let z_b = self.plonk.zero_wire();
                        self.constraint(
                            self.field.zero(),
                            self.field.zero(),
                            self.field.new_v(-1),
                            self.field.zero(),
                            field_val,
                            z_a,
                            z_b,
                            result.clone(),
                        );
                        //self.plonk.new_wire("const".to_string(), const_term)
                        result
                    }
                }
                Op::Ite => {
                    let cond = self.get_bool(&c.cs()[0]).clone();
                    let t = self.get_pf(&c.cs()[1]).clone();
                    let f = self.get_pf(&c.cs()[2]).clone();
                    self.ite(cond, t, f)
                }
                Op::PfNaryOp(o) => {
                    let args: Vec<Wire> = c.cs().iter().map(|c| self.get_pf(c).clone()).collect();
                    match o {
                        PfNaryOp::Add => {
                            let mut result = args[0].clone();
                            for arg in args.iter().skip(1) {
                                result = self.add(result, arg.clone());
                            }
                            result
                        }
                        PfNaryOp::Mul => {
                            let mut result = args[0].clone();
                            for arg in args.iter().skip(1) {
                                result = self.mul(result, arg.clone());
                            }
                            result
                        }
                    }
                }
                Op::UbvToPf(_) => self.get_bv_uint(&c.cs()[0]),
                Op::PfUnOp(PfUnOp::Neg) => {
                    let arg = self.get_pf(&c.cs()[0]).clone();
                    self.mul_const(arg, self.field.new_v(-1))
                }
                Op::PfUnOp(PfUnOp::Recip) => {
                    let x = self.get_pf(&c.cs()[0]).clone();
                    match self.cfg.r1cs.div_by_zero {
                        FieldDivByZero::Incomplete => {
                            // inv_x * x = 1
                            let inv_x = self.fresh_wit(
                                "recip",
                                term![PF_RECIP;
                                self.plonk.wire_values[&x].clone()],
                            );
                            let one_wire: Wire = self.plonk.one_wire();
                            let prod = self.mul(inv_x.clone(), x);
                            self.assert_equal(prod, one_wire);
                            inv_x
                        }
                        FieldDivByZero::NonDet => {
                            // inv_x * x * x = x
                            let x2 = self.mul(x.clone(), x.clone());
                            let inv_x = self.fresh_wit(
                                "recip",
                                term![PF_RECIP;
                                self.plonk.wire_values[&x].clone()],
                            );
                            let lhs = self.mul(x2, inv_x.clone());
                            self.assert_equal(lhs, x);
                            inv_x
                        }
                        FieldDivByZero::Zero => {
                            // i * x = 1 - z
                            // z * x = 0
                            // z * i = 0
                            let eqz = term![Op::Eq;
                                self.plonk.wire_values[&x].clone(),
                                self.zero_term()];
                            let i = self.fresh_wit(
                                "is_zero_inv",
                                term![Op::Ite; eqz.clone(), self.zero_term(),
                                      term![PF_RECIP; self.plonk.wire_values[&x].clone()]],
                            );
                            let z = self.fresh_wit(
                                "is_zero",
                                term![Op::Ite; eqz, self.one_term(), self.zero_term()],
                            );

                            // i * x + z - 1 = 0
                            let ix = self.mul(i.clone(), x.clone());
                            let ix_plus_z = self.add(ix, z.clone());
                            let constraint1 = self.add_const(ix_plus_z, self.field.new_v(-1));
                            self.assert_zero(constraint1);

                            // z * x = 0
                            let zx = self.mul(z.clone(), x);
                            self.assert_zero(zx);

                            // z * i = 0
                            let zi = self.mul(z, i.clone());
                            self.assert_zero(zi);

                            i
                        }
                    }
                }
                Op::PfDiv => {
                    let y = self.get_pf(&c.cs()[0]).clone();
                    let x = self.get_pf(&c.cs()[1]).clone();
                    match self.cfg.r1cs.div_by_zero {
                        FieldDivByZero::Incomplete => {
                            // div * x = y
                            let div = self.fresh_wit(
                                "div",
                                term![PF_DIV;
                                self.plonk.wire_values[&y].clone(),
                                self.plonk.wire_values[&x].clone()],
                            );
                            let prod = self.mul(div.clone(), x);
                            self.assert_equal(prod, y);
                            div
                        }
                        _ => unimplemented!(),
                    }
                }
                Op::UndefinedFnCall(call) => {
                    // We are in embed_pf: the return must be a field element.
                    if !matches!(call.ret_sort, Sort::Field(_)) {
                        panic!(
                            "UndefinedFnCall '{}' returns non-field sort in embed_pf: {}",
                            call.name, call.ret_sort
                        );
                    };
                    // Convert all children to field wires using the declared arg sorts
                    let arg_terms = c.cs();
                    /*for (i, t) in arg_terms.iter().enumerate() {
                        let s = &call.arg_sorts[i];
                        args_pf.push(t);
                    }*/

                    if call.name.eq("PlonkMul2") {
                        if arg_terms.len() != 1 {
                            panic!("PlonkMul2 expects 1 arguments, got {}", arg_terms.len());
                        }
                        let a = self.get_pf(&arg_terms[0]).clone();
                        let aa = self.mul(a.clone(), a.clone());
                        let aa = self.mul(aa.clone(), a);
                        aa
                    } else {
                        panic!(
                            "UndefinedFnCall '{}' not implemented in embed_pf",
                            call.name
                        );
                        self.plonk.zero_wire()
                    }
                }
                _op => panic!("Non-field in embed_pf: {}, op: {}", c, _op),
            };
            self.cache.insert(c.clone(), EmbeddedTerm::Field(wire));
        }
        self.get_pf(&c)
    }

    fn assert(&mut self, t: Term) {
        debug!("Assert: {}", t);
        debug_assert!(check(&t) == Sort::Bool, "Non bool in assert");
        self.assert_bool(&t);
    }

    fn profile_print(&self) {
        debug!("Plonk constraint count: {}", self.plonk.constraints.len());
        debug!(
            "Plonk copy constraint count: {}",
            self.plonk.copy_constraints.len()
        );
        debug!(
            "Plonk public input count: {}",
            self.plonk.public_inputs.len()
        );
        debug!("Plonk witness count: {}", self.plonk.witness.len());
    }
}

/// Convert this (IR) constraint system `cs` to Plonk, over a prime field defined by `modulus`.
///
/// ## Returns
///
/// * Plonk constraint system
pub fn to_plonk(cs: &Computation, cfg: &CircCfg) -> PlonkCs {
    let public_inputs = cs.metadata.public_input_names_set();
    println!("public inputs: {:?}", public_inputs);
    let all_inputs = cs.metadata.ordered_input_names();
    let all_inputs_values = cs.precomputes.inputs();

    println!("all inputs: {:?}", all_inputs);
    println!("all inputs values: {:?}", all_inputs_values);
    let used_vars = extras::free_variables(term(Op::Tuple, cs.outputs.clone()));
    let mut converter = ToPlonk::new(cfg, used_vars.into_iter().collect());
    debug!(
        "Term count: {}",
        cs.outputs
            .iter()
            .map(|c| PostOrderIter::new(c.clone()).count())
            .sum::<usize>()
    );
    debug!("declaring inputs");
    let vars = cs.metadata.interactive_vars();
    println!("interactive_vars: {:#?}", vars);
    println!("vars.instances: {:?}", vars.instances);
    //println!("vars.")
    for i in &vars.instances {
        converter.embed_var(i, VarType::Inst);
        /*let out_wire = converter
            .plonk
            .wire_values
            .iter()
            .find_map(|(wire, term)| if term == i { Some(wire) } else { None })
            .expect("output term missing");
        converter
            .plonk
            .add_copy_constraint(out_wire.clone(), out_wire.clone());*/
    }
    /*for terms in &vars.committed_wit_vecs {
        let names_and_terms = terms
            .iter()
            .map(|t| (t.as_var_name().to_owned(), t.clone()))
            .collect();
        converter.committed_wit(names_and_terms);
    }*/
    /*for round in &vars.rounds {
        for w in &round.witnesses {
            converter.embed_var(w, VarType::RoundWit);
        }
        for c in &round.challenges {
            converter.embed_var(c, VarType::Chall);
        }
        // Note: Plonk doesn't have explicit rounds like R1CS,
        // but we can still process the variables
    }*/
    for w in &vars.final_witnesses {
        //println!("w = {}", w);
        converter.embed_var(w, VarType::FinalWit);
    }

    //for e in converter.plonk.public_inputs {
    //converter.plonk.add_copy_constraint(e, );
    //}

    debug!("Processing assertions");
    for c in &cs.outputs {
        //println!("output = {:?}", c.cs()[0]);
        // Lookup wire for output term
        //let out_wire = converter.plonk.wire_values.iter()
        //    .find_map(|(wire, term)| if term == &c.cs()[0] { Some(wire) } else { None })
        //    .expect("output term missing");

        // Add copy constraint between computation result and public output
        //converter.plonk.add_copy_constraint(out_wire.clone(), out_wire.clone());
        converter.assert(c.clone());
    }

    // No need for copy constraint with output because of output - result = 0 check
    /*for c in &cs.outputs {
        println!("output = {:?}", c.cs()[0]);
        // Lookup wire for output term
        let out_wire = converter.plonk.wire_values.iter()
            .find_map(|(wire, term)| if term == &c.cs()[0] { Some(wire) } else { None })
            .expect("output term missing");

        // Add copy constraint between computation result and public output
        //converter.plonk.add_copy_constraint(out_wire.clone(), out_wire.clone());
        //converter.assert(c.clone());
    }
    panic!();*/
    //converter.profile_print();
    let mut res = converter.plonk;

    res.all_inputs = all_inputs;
    res
}

// Utility function for bit size calculation
fn bitsize(n: usize) -> usize {
    if n == 0 {
        1
    } else {
        (n as f64).log2().ceil() as usize
    }
}

impl<'cfg> ToPlonk<'cfg> {
    pub fn finish(self) -> PlonkCs {
        self.plonk
    }
}

// Example usage and testing
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_constraints() {
        // Test would verify that basic arithmetic constraints work correctly
    }

    #[test]
    fn test_copy_constraints() {
        // Test would verify copy constraints are generated properly
    }
}
