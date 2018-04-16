///! Definition of the factor module
///!
///! A `Factor` represents a calculation over a subset of `Variable`s

use std::box::Box;
use ops::FactorLike;
use variable::Variable;

pub enum Factor {
    Identity,
    TableFactor(),
    TreeFactor(),
    RuleFactor(),
    CustomFactor(Box<FactorLike>)
}

impl FactorLike for Factor {

}
