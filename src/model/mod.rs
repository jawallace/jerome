//! Defines a `Model`, which is a Bayesian (directed) or Markovian (undirected) graphical model
//! representing the factorization of a probability distribution P.

use variable::{Assignment, Variable};
use util::Result;

use std::collections::HashSet;

/// The `Model` trait represents a Probabilistic Graphical Model.
pub trait Model{

    /// The concrete type of `PGM` that is returned by the `condition` operation. 
    type Model_Type;


    /// Lookup a `Variable` in the `DirectedModel` based on the name
    fn lookup_variable(&self, name: &str) -> Option<&Variable>;


    /// Lookup a `Variable`'s name in the `DirectedModel`.
    fn lookup_name(&self, var: &Variable) -> Option<&String>;


    /// Get all `Variable`s in the model.
    fn variables(&self) -> HashSet<Variable>;


    /// Get the number of `Variable`s in the the `DirectedModel`
    fn num_variables(&self) -> usize;


    /// Condition the `Model` given the evidence.
    ///
    /// # Args
    /// * `evidence`: a partial `Assignment` of the `Variable`s in this `Model`.
    ///
    /// # Returns:
    /// a new `Model` with scope ```self.vars() - evidence.keys()``` that represents the
    /// conditional distribution ```P(self.scope() - evidence.keys() | evidence.keys())```
    fn condition(&self, evidence: &Assignment) -> Self::Model_Type;


    /// Determine the probability of a full `Assignment` to the `Variable`s in the `Model`.
    ///
    /// Specifically, this computes ```P(zeta)```, where ```zeta``` is a full assignment.
    ///
    /// # Args
    /// * `assignment`: a full `Assignment` to the `Model`
    ///
    /// # Returns
    /// the probability of the `Assignment` given the `Model`
    fn probability(&self, assignment: &Assignment) -> Result<f64>;
}

pub mod directed;
pub mod undirected;
