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


    pub fn is_cpd(&self) -> bool {
        match self {
            &Factor::Identity => true,
            &Factor::TableFactor { cpd, .. } => cpd
        }
    }


    /// Retrieve all of the operations defined by a `Factor`
    pub fn scope(&self) -> Vec<Variable> {
        match self {
            &Factor::Identity => vec![],
            &Factor::TableFactor { ref scope, .. } => scope.clone()
        }
    }


    /// Retrieve the value for a complete assignment over the scope of this `Factor`
    pub fn value(&self, assignment: &Assignment) -> Result<f64> {
        match self {
            &Factor::Identity => {
                Err(JeromeError::General(String::from("The identity factor has no value")))
            },
            &Factor::TableFactor { ref scope, ref table, .. } => {
                let idxs: Vec<Option<&usize>> = self.scope().iter().map(|v| assignment.get(v)).collect();
                if idxs.iter().any(|&v| v.is_none()) {
                    return Err(JeromeError::IncompleteAssignment);
                }

                let idxs: Vec<usize> = idxs.iter().map(|&v| *(v.unwrap())).collect();
                Ok(table[nd::IxDyn(&idxs)])
            }
        }
    }


    /// `Factor` multiplication
    ///
    /// Defined in Koller & Friedman TODO
    ///
    /// # Args
    ///
    /// # Returns
    ///
    pub fn multiply(&self, _other: &Self) -> Self {
        Factor::Identity
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
    /// Defined in Koller & Friedman TODO
    ///
    /// # Args
    ///
    /// # Returns
    ///
    pub fn reduce(&self, _assignment: &Assignment) -> Result<Self> {
        match self {
            &Factor::Identity => Ok(Factor::Identity),
            &Factor::TableFactor { .. } => {
                Ok(Factor::Identity)
            }
        }
    }


    /// Marginalize the `Factor` over the given `Variable`
    ///
    /// Defined in Koller & Friedman TODO
    ///
    /// # Args
    ///
    /// # Returns
    ///
    pub fn marginalize(&self, _other: &Variable) -> Result<Self> {
        if let &Factor::Identity = self {
            return Ok(Factor::Identity);
        }

        Ok(Factor::Identity)
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
}
