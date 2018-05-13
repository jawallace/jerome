//! Defines the interface to inference engines

use factor::Factor;
use variable::{Assignment, Variable};
use super::Result;

use std::collections::HashSet;

pub mod variable_elimination;
pub use self::variable_elimination::VariableEliminationEngine;


/// A `ConditionalInferenceEngine` is capable of answering Conditional Probability Queries of the form:
///     ```P(Y | E = e)``` 
///
/// `ConditionalInferenceEngine`s are stateful and must take the evidence `e` as an argument to whatever
/// construction mechanism they employ.
trait ConditionalInferenceEngine {

    /// Infer the joint distribution ```P(variables | evidence)```
    fn infer(&self, variables: &HashSet<Variable>) -> Result<Factor>;

}


/// A `MapInferenceEngine` is cable of answering Maximum a posteiori queries:
///     ```MAP(Y | E = e) = argmax_y P(Y = y | E = e)```
///
/// `MapInferenceEngine`s are stateful and must take the evidence `e` as an argument to whatever
/// construction mechanism they employ.
trait MapInferenceEngine {

    /// Infer the most probable assignment `Y = y` given the 
    fn infer(&self, variables: &HashSet<Variable>) -> Result<Assignment>;

}
