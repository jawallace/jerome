///! Definition of the factor module
///!
///! A `Factor` represents a relationship between some set of `Variable`s.

extern crate ndarray;

use super::{Result, JeromeError};
use super::variable::{Variable, Assignment};
use self::ndarray::prelude as nd;


/// Alias f64 ndarray::Array as Table
pub type Table<D> = nd::Array<f64, D>;


#[derive(Debug, Clone)]
pub enum Factor<D: nd::Dimension> {
    /// The empty, identity `Factor` with no scope. This type exists for dealing with arithmetic
    /// operations of `Factor`s
    Identity,

    /// A `Factor` over some scope of variables. Represented as a table-CPD as described in Koller
    /// & Friedman.
    TableFactor {
        /// The scope of the `Factor`
        scope: Vec<Variable>,

        /// The values of the `Factor` table.
        table: Table<D>,

        /// `true`, if the `Factor` is a conditional probability distribution (i.e. is normalized)
        cpd: bool
    }
}


impl<D: nd::Dimension> Factor<D> {


    /// Get the identity factor
    pub fn identity() -> Self {
        Factor::Identity 
    }


    /// Create a new `Factor`
    pub fn new(scope: Vec<Variable>, table: Table<D>, cpd: bool) -> Result<Self> {
        if scope.len() == 0 && table.len() == 0 {
            // return the identity factor if this is an empty factor
            return Ok(Factor::Identity);
        } else if scope.len() == 0 || table.len() == 0 {
            // if values or scope is nonempty, then return an error
            return Err(
                JeromeError::General(
                    String::from("Invalid arguments. Scope and values must both be empty or nonempty")
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


    /// Retrieve all of the operations defined by a `Factor`
    pub fn scope(&self) -> Vec<Variable> {
        match self {
            &Factor::Identity => vec![],
            &Factor::TableFactor { ref scope, ref table, cpd } => scope.clone()
        }
    }


    /// Retrieve the value for a complete assignment over the scope of this `Factor`
    pub fn value(&self, assignment: &Assignment) -> Result<f64> {
        match self {
            &Factor::Identity => {
                Err(JeromeError::General(String::from("The identity factor has no value")))
            },
            &Factor::TableFactor { ref scope, ref table, cpd } => {
                let idxs: Vec<Option<&usize>> = self.scope().iter().map(|&v| assignment.get(v)).collect();
                if ! idxs.iter().any(|&v| v.is_none()) {
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
    pub fn multiply<E, F>(&self, other: &Factor<E>) -> Factor<F> 
        where E: nd::Dimension,
              F: nd::Dimension
    {
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
    pub fn divide<E, F>(&self, other: &Factor<E>) -> Factor<F>
        where E: nd::Dimension,
              F: nd::Dimension
    {
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
    pub fn reduce<E>(&self, assignment: &Assignment) -> Result<Factor<E>> 
        where E: nd::Dimension
    {
        match self {
            &Factor::Identity => Ok(Factor::Identity),
            &Factor::TableFactor { ref scope, ref table, cpd } => {
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
    pub fn marginalize<E>(&self, other: &Variable) -> Result<Factor<E>>
        where E: nd::Dimension
    {
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
        assert_eq!(true, true);
    }

}
