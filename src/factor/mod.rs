///! Definition of the factor module
///!
///! A `Factor` represents a relationship between some set of `Variable`s.

use super::{Result, JeromeError};
use super::variable::{Variable, Assignment};
use super::ndarray::prelude as nd;

/// Alias f64 ndarray::Array as Table
pub type Table = nd::ArrayD<f64>;


#[derive(Clone, Debug)]
pub enum Factor {
    /// The empty, identity `Factor` with no scope. This type exists for dealing with arithmetic
    /// operations of `Factor`s
    Identity,

    /// A `Factor` over some scope of variables. Represented as a table-CPD as described in Koller
    /// & Friedman.
    TableFactor {
        /// The scope of the `Factor`
        scope: Vec<Variable>,

        /// The values of the `Factor` table.
        table: Table,

        /// `true`, if the `Factor` is a conditional probability distribution (i.e. is normalized)
        cpd: bool
    }
}


impl Factor {

    /// Get the identity factor
    pub fn identity() -> Self {
        Factor::Identity 
    }


    /// Create a new `Factor`
    pub fn new(scope: Vec<Variable>, table: Table, cpd: bool) -> Result<Self> {
        if scope.len() == 0 {
            return Err(
                JeromeError::General(
                    String::from("Invalid arguments. Scope may not be empty")
                )
            );
        } else if scope.len() != table.ndim() {
            return Err(
                JeromeError::General(
                    String::from("Invalid arguments. Cardinality of scope must match number of table dimensions")
                )
            );
        }

        for (v, t) in scope.iter().map(|&v| v.cardinality()).zip(table.shape().iter()) {
            if v != *t {
                return Err(
                    JeromeError::General(
                        String::from("Invalid arguments. Dimensions do not match")
                    )
                );
            }
        }

        // verify the table represents a cpd if the caller says it does
        if cpd && (table.scalar_sum() - 1.0).abs() > 0.01 {
            return Err(
                JeromeError::General(
                    String::from("Invalid arguments. Requested a CPD, but the values do not represent a CPD")
                )
            );
        }

        Ok(Factor::TableFactor { scope, table, cpd })
    }


    /// Check if the `Factor` is the identity `Factor`
    pub fn is_identity(&self) -> bool {
        match self {
            &Factor::Identity => true,
            _ => false
        }
    }


    /// Check if the `Factor` is a Conditional Probability Distribution - i.e. if the values in the
    /// `Factor` are normalized.
    ///
    /// # Note
    /// A conditional probability distribution is a specialization of a `Factor`. All CPDs are
    /// `Factor`s, but not all `Factor`s are CPDs. The identity `Factor` is considered a CPD.
    pub fn is_cpd(&self) -> bool {
        match self {
            &Factor::Identity => true,
            &Factor::TableFactor { cpd, .. } => cpd
        }
    }


    /// Retrieve the scope of the `Factor`.
    ///
    /// # Note
    /// This method returns a clone of the `Factor`'s scope. `Variable`'s are lightweight and
    /// therefore this is an acceptable overhead
    pub fn scope(&self) -> Vec<Variable> {
        match self {
            &Factor::Identity => vec![],
            &Factor::TableFactor { ref scope, .. } => scope.clone()
        }
    }


    /// Retrieve the value for a complete assignment over the scope of this `Factor`
    /// 
    /// This operation is defined only on non-indentity `Factor`s. 
    ///
    /// # Args
    /// assignment: a full assignment to the scope of a `Factor`. The assignment's scope  may be a
    ///             superset  of the `Factor`s scope.
    ///
    /// # Returns
    /// the value of the assignment, or an error.
    ///
    /// # Errors
    /// * `JeromeError::General` if the `Factor` is the identity
    /// * `JeromeError::IncompleteAssignment`, if assignment is not a complete assignment to the
    ///   scope of the `Factor`
    pub fn value(&self, assignment: &Assignment) -> Result<f64> {
        match self {
            &Factor::Identity => {
                Err(JeromeError::General(String::from("The identity factor has no value")))
            },
            &Factor::TableFactor { ref scope, ref table, .. } => {
                let idxs: Vec<Option<&usize>> = scope.iter().map(|v| assignment.get(v)).collect();
                if idxs.iter().any(|&v| v.is_none()) {
                    return Err(JeromeError::IncompleteAssignment);
                }

                let idxs: Vec<usize> = idxs.iter().map(|&v| *(v.unwrap())).collect();
                Ok(table[nd::IxDyn(&idxs)])
            }
        }
    }


    /// Product of this `Factor` and another `Factor` that have intersecting scope.
    ///
    /// Defined in Koller & Friedman Section 4.2.1
    ///
    /// # Args
    /// other: the `Factor` to multiply with.
    ///
    /// # Returns
    /// A new `Factor` of scope union(self.scope(), other.scope()) 
    ///
    /// # Errors
    /// * `JeromeError::InvalidScope`, if intersection(self.scope(), other.scope()) = []
    pub fn product(&self, other: &Self) -> Result<Self> {
        if let &Factor::Identity = self {
            return Ok(other.clone());
        } else if let &Factor::Identity = other {
            return Ok(self.clone());
        }

        Ok(Factor::Identity)
    }


    /// `Factor` division
    ///
    /// Defined in Koller & Friedman TODO
    ///
    /// # Args
    ///
    /// # Returns
    ///
    pub fn divide(&self, _other: &Self) -> Self {
        Factor::Identity
    }


    /// Reduce the `Factor` to over the given partial assignment
    ///
    /// Defined in Koller & Friedman 4.2.3
    ///
    /// # Args
    /// assignment: a partial assignment to the `Factor`
    ///
    /// # Returns
    /// A new `Factor` reduced over the given assignment
    pub fn reduce(&self, assignment: &Assignment) -> Self {
        match self {
            &Factor::Identity => Factor::Identity,
            &Factor::TableFactor { ref scope, ref table, .. } => {
                // reduce table based on assignment
                let mut view = table.view();
                let mut new_shape: Vec<usize> = Vec::new();
                let mut new_scope: Vec<Variable> = Vec::new();

                for (i, &v) in scope.iter().enumerate() {
                    if let Some(&val) = assignment.get(&v) {
                        view.subview_inplace(nd::Axis(i), val);
                    } else {
                        new_shape.push(table.len_of(nd::Axis(i)));
                        new_scope.push(v);
                    }
                }

                if new_scope.len() == 0 {
                    // complete assignment
                    Factor::Identity
                } else if new_scope.len() == scope.len() {
                    // empty assignment (relative to scope)
                    self.clone()
                } else {
                    // TODO - what to do if we are reducing a CPD? Renormalize?  For now, returning a
                    // non-cpd factor
                    Factor::new(
                        new_scope, 
                        view.to_owned().into_shape(new_shape).expect("reduce encountered error"), 
                        false
                    ).expect(
                        "reduce encountered unexpected error"
                    )
                }
            }
        }
    }


    /// Marginalize the `Factor` over the given `Variable`
    ///
    /// Defined in Koller & Friedman 9.3.1
    ///
    /// # Args
    /// other: the `Variable` over which to marginalize
    ///
    /// # Returns
    /// another `Factor`, marginalized over the given `Variable`
    pub fn marginalize(&self, other: Variable) -> Self {
        match self {
            // the identity factor marginalized over anything is the identity
            &Factor::Identity => Factor::Identity,

            &Factor::TableFactor { ref scope, ref table, cpd } => {
                if let Some(idx) = scope.iter().position(|&v| v == other) {
                    let new_table = table.sum_axis(nd::Axis(idx));
                    let new_scope = scope.clone().into_iter().filter(|&v| v != other).collect();

                    Factor::new(new_scope, new_table, cpd).expect(
                        "marginalize encountered error that should never occur"
                    )
                } else {
                    // variable not in the scope of this factor, so the factor is already
                    // marginalized over the factor
                    // TODO should this be an error instead?
                    self.clone()
                }
            }
        }
    }

}

// Unit tests
mod tests {
    use super::*;

    #[test]
    fn identity() {
        let f = Factor::identity();
        let f2 = Factor::identity();

        assert!(f.is_identity());
        assert!(f2.is_identity());
    }

    #[test]
    fn table_factor() {
        let vars = vec![ Variable::binary(), Variable::discrete(5), Variable::discrete(3) ];
        let mut table = Table::ones(vec![2, 5, 3]);
        table[[1, 1, 1].as_ref()] = 5.;

        // assert table holds correct values
        let f = Factor::new(vars.clone(), table, false).unwrap();

        assert!(! f.is_identity());
        for (x, y, z) in izip!(0..2, 0..5, 0..3) {
            let mut assn = Assignment::new();
            assn.set(&vars[0], x);
            assn.set(&vars[1], y);
            assn.set(&vars[2], z);

            let val = f.value(&assn).unwrap();
            if x == 1 && y == 1 && z == 1 {
                assert_eq!(5., val);
            } else {
                assert_eq!(1., val);
            }
        }

        assert!(! f.is_cpd());
    }

    #[test]
    fn table_factor_errs() {
        // empty scope
        let vars = vec![];
        let table = Table::ones(vec![2, 5, 3]);
        let f = Factor::new(vars, table, false);
        assert!(f.is_err());
        match f.expect_err("missing error") {
            JeromeError::General(_) => assert!(true),
            _ => panic!("wrong error type")
        };

        // mismatched number of dimensions
        let vars = vec![ Variable::binary(), Variable::binary() ];
        let table = Table::ones(vec![2, 2, 2]);
        let f = Factor::new(vars.clone(), table, false);
        assert!(f.is_err());
        match f.expect_err("missing error") {
            JeromeError::General(_) => assert!(true),
            _ => panic!("wrong error type")
        };

        // wrong cardinality
        let table = Table::ones(vec![2, 3]);
        let f = Factor::new(vars.clone(), table, false);
        assert!(f.is_err());
        match f.expect_err("missing error") {
            JeromeError::General(_) => assert!(true),
            _ => panic!("wrong error type")
        };

        // not a cpd
        let table = Table::ones(vec![2, 2]);
        let f = Factor::new(vars.clone(), table, true);
        assert!(f.is_err());
        match f.expect_err("missing error") {
            JeromeError::General(_) => assert!(true),
            _ => panic!("wrong error type")
        };
    }

    #[test]
    fn table_factor_cpd() {
        let vars = vec![ Variable::binary(), Variable::binary() ];
        let table = Table::ones(vec![2, 2]) / 4.;

        let f = Factor::new(vars, table, true).expect("unexpected error");
        assert!(f.is_cpd());
    }

    #[test]
    fn value() {
        let vars = vec![ Variable::binary(), Variable::binary() ];
        let mut table = Table::ones(vec![2, 2]);

        for (i, (x, y)) in (0..2).zip(0..2).enumerate() {
            table[[x, y].as_ref()] = i as f64;
        }

        let f = Factor::new(vars.clone(), table, false).expect("Unexpected error");

        // verify behavior on precise assignment
        for (i, (x, y)) in (0..2).zip(0..2).enumerate() {
            let mut assn = Assignment::new();
            assn.set(&vars[0], x);
            assn.set(&vars[1], y);

            assert_eq!(i as f64, f.value(&assn).expect("unexpected error"));
        }

        // verify behavior on full assignment with out of scope values
        let v3 = Variable::binary();

        for (i, (x, y)) in (0..2).zip(0..2).enumerate() {
            let mut assn = Assignment::new();
            assn.set(&vars[0], x);
            assn.set(&vars[1], y);
            assn.set(&v3, 0);

            assert_eq!(i as f64, f.value(&assn).expect("unexpected error"));
        }

        // verify behavior on incomplete assignment
        let mut assn = Assignment::new();
        assn.set(&vars[0], 0);
        assn.set(&v3, 0);

        let res = f.value(&assn);
        assert!(res.is_err());
        match res.expect_err("") {
            JeromeError::IncompleteAssignment => assert!(true),
            _ => panic!("incorrect error")
        };
    }

    #[test]
    /// Example taken from Koller & Friedman Figure 4.3
    fn product() {
        let a = Variable::discrete(3);
        let b = Variable::binary();
        let c = Variable::binary();

        let tbl1 = nd::Array::from_shape_vec(
            (3, 2), 
            vec![ 0.5, 0.8, 0.1, 0., 0.3, 0.9 ]
        ).expect("Unexpected error").into_dyn();
        let phi1 = Factor::new(vec![ a, b ], tbl1, false).expect("Unexpected error");

        let tbl2 = nd::Array::from_shape_vec(
            (2, 2), 
            vec![ 0.5, 0.7, 0.1, 0.2 ]
        ).expect("Unexpected error").into_dyn();
        let phi2 = Factor::new(vec![ b, c ], tbl2, false).expect("Unexpected error");

        let phi = phi1.product(&phi2).expect("Unexpected error");

        let expected = nd::Array::from_shape_vec(
            (3, 2, 2), 
            vec![ 0.25, 0.35, 0.08, 0.16, 0.05, 0.07, 0., 0., 0.15, 0.21, 0.09, 0.18 ]
        ).expect("Unexpected error").into_dyn();

        for (x, y, z) in izip!(0..3, 0..2, 0..2) {
            let mut assn = Assignment::new();
            assn.set(&a, x);
            assn.set(&b, y);
            assn.set(&c, z);

            let idx = vec![x, y, z];
            assert_eq!(expected[nd::IxDyn(&idx)], phi.value(&assn).expect("unexpected error"));
        }
    }

    #[test]
    /// Example take from Koller & Friedman Figure 4.5
    fn reduce_simple() {
        let a = Variable::discrete(3);
        let b = Variable::binary();
        let c = Variable::binary();
       
        let table = nd::Array::from_shape_vec(
            (3, 2, 2), 
            vec![ 0.25, 0.35, 0.08, 0.16, 0.05, 0.07, 0., 0., 0.15, 0.21, 0.09, 0.18 ]
        ).expect("Unexpected error").into_dyn();

        let phi = Factor::new(vec![a, b, c], table, false).expect("Unexpected error");

        let mut assn = Assignment::new();
        assn.set(&c, 0);
        
        let expected = nd::Array::from_shape_vec(
            (3, 2), 
            vec![ 0.25, 0.08, 0.05, 0., 0.15, 0.09 ]
        ).expect("Unexpected error").into_dyn();

        let reduced = phi.reduce(&assn);
        assert_eq!(vec![a, b], reduced.scope());
        for (x, y) in (0..3).zip(0..2) {
            let mut assn = Assignment::new();
            assn.set(&a, x);
            assn.set(&b, y);

            let idx = [x, y];
            assert_eq!(expected[nd::IxDyn(&idx)], reduced.value(&assn).expect("unexpected error"));
        }
    }

    #[test]
    fn reduce_empty() {
        let a = Variable::binary();
        let b = Variable::binary();
        let c = Variable::binary();

        let table = array![[ 1., 0. ], [ 0., 1. ]].into_dyn();
        let phi = Factor::new(vec![a, b], table.clone(), false).expect("Unexpected error");

        let mut assn = Assignment::new();
        assn.set(&c, 1);

        let reduced = phi.reduce(&assn);
        assert_eq!(vec![a, b], reduced.scope());
        for (x, y) in (0..2).zip(0..2) {
            let mut asn = Assignment::new();
            asn.set(&a, x);
            asn.set(&b, y);

            let idx = [x, y];
            assert_eq!(table[nd::IxDyn(&idx)], reduced.value(&asn).expect("Unexpected error"));
        }
    }

    #[test]
    fn reduce_full() {
        let a = Variable::binary();
        let b = Variable::binary();
        let c = Variable::binary();

        let table = array![[ 1., 0. ], [ 0., 1. ]].into_dyn();
        let phi = Factor::new(vec![a, b], table.clone(), false).expect("Unexpected error");

        let mut assn = Assignment::new();
        assn.set(&a, 0);
        assn.set(&b, 0);
        assn.set(&c, 1);

        let reduced = phi.reduce(&assn);
        assert!(reduced.is_identity());
    }

    #[test]
    fn reduce_multiple() {
        let a = Variable::discrete(3);
        let b = Variable::binary();
        let c = Variable::binary();
       
        let table = nd::Array::from_shape_vec(
            (3, 2, 2), 
            vec![ 0.25, 0.35, 0.08, 0.16, 0.05, 0.07, 0., 0., 0.15, 0.21, 0.09, 0.18 ]
        ).expect("Unexpected error").into_dyn();

        let phi = Factor::new(vec![a, b, c], table, false).expect("Unexpected error");

        let mut assn = Assignment::new();
        assn.set(&c, 0);
        assn.set(&a, 2);

        let expected = array![0.15, 0.09].into_dyn();

        let reduced = phi.reduce(&assn);
        assert_eq!(vec![b], reduced.scope());
        for x in 0..2 {
            let mut assn = Assignment::new();
            assn.set(&b, x);

            let idx = [x];
            assert_eq!(expected[nd::IxDyn(&idx)], reduced.value(&assn).expect("unexpected error"));
        }

    }

    #[test]
    fn marginalize() {
        let a = Variable::discrete(3);
        let b = Variable::binary();
        let c = Variable::binary();
       
        let table = nd::Array::from_shape_vec(
            (3, 2, 2), 
            vec![ 0.25, 0.35, 0.08, 0.16, 0.05, 0.07, 0., 0., 0.15, 0.21, 0.09, 0.18 ]
        ).expect("Unexpected error").into_dyn();

        let phi = Factor::new(vec![a, b, c], table, false).expect("Unexpected error");

        let marginalized = phi.marginalize(b);
        assert_eq!(vec![a, c], marginalized.scope());

        let expected = array![[0.33, 0.51], [0.05, 0.07], [0.24, 0.39]].into_dyn();
        for (x, y) in (0..3).zip(0..2) {
            let mut assn = Assignment::new();
            assn.set(&a, x);
            assn.set(&c, y);

            let idx = [ x, y ];
            assert_eq!(expected[nd::IxDyn(&idx)], marginalized.value(&assn).expect("unexpected error"));
        }
    }
}
