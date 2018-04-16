//! Defines a trait that represents the operations defined on a `Factor`

use error::{Error, Result};
use variable::Variable;

/// Defines all the operations over a `Factor`
trait FactorLike {

    /// Retrieve all of the operations defined by a `Factor`
    pub fn scope(&self) -> Vec<Rc<Variable>>;


    /// Retrieve the value for a complete assignment over the scope of this `Factor`
    /// 
    /// # Returns
    ///
    pub fn value(&self, Vec<Rc<Variable>>) -> Result<f64>;


    /// Reduce the `FactorLike` to over the given partial assignment
    pub fn reduce(&self, Vec<Rc<Variable>>) -> Self;


    ///
    pub fn multiply(&self, other: &Factor) -> Self;


    pub fn divide(&self, other: &Factor) -> Factor {

    }

    pub fn marginalize(&self, other: &Variable) -> Factor {

    }

}

