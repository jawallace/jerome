//! Defines the interface to inference engines

use factor::Factor;
use variable::{Assignment, Variable};
use super::Result;

use std::collections::HashSet;

mod importance_sampling;
mod mcmc;
mod variable_elimination;

pub use self::importance_sampling::ImportanceSamplingEngine;
pub use self::mcmc::McmcEngine;
pub use self::variable_elimination::VariableEliminationEngine;


/// A `ConditionalInferenceEngine` is capable of answering Conditional Probability Queries of the form:
///     ```P(Y | E = e)``` 
///
/// `ConditionalInferenceEngine`s are stateful and must take the evidence `e` as an argument to whatever
/// construction mechanism they employ.
pub trait ConditionalInferenceEngine {

    /// Infer the joint distribution ```P(variables | evidence)```
    fn infer(&mut self, variables: &HashSet<Variable>) -> Result<Factor>;

}


/// A `MapInferenceEngine` is cable of answering Maximum a posteiori queries:
///     ```MAP(Y | E = e) = argmax_y P(Y = y | E = e)```
///
/// `MapInferenceEngine`s are stateful and must take the evidence `e` as an argument to whatever
/// construction mechanism they employ.
pub trait MapInferenceEngine {

    /// Infer the most probable assignment `Y = y` given the 
    fn infer(&self, variables: &HashSet<Variable>) -> Result<Assignment>;

}


#[cfg(test)]
/// Tests for the inference engines in this module. Tests are hoisted here to avoid duplication.
/// Any tests specific to the inference engine are held within that submodule's tests module.
///
/// Example derived from Koller & Friedman's student example. Koller & Friedman do not offer an 
/// example of the results of the exact inference on the student (or extended-student) example. 
///
/// However, example 6d of [1] provides the results of exact (via variable elimination) and 
/// approximate (via particle methods) inference of P(I | D=0, L=1, S=0) on a modified version
/// of the K&F Student example. We use that result here to test our implementation.
///
/// [1] https://www.uni-oldenburg.de/en/lcs/probabilistic-programming/webchurch-and-openbugs/
mod tests {
    use super::*;
    use model::directed::{DirectedModel, DirectedModelBuilder};
    use init::Initialization;
    use samplers::{GibbsSampler, LikelihoodWeightedSampler};

    /// Utility function to build the student inference example
    fn build_student_example() -> (Variable, DirectedModel, Assignment) {
        let d = Variable::binary();
        let i = Variable::binary();
        let g = Variable::binary();
        let s = Variable::binary();
        let l = Variable::binary();

        let cpd_g = Factor::cpd(
            g, 
            vec![i, d], 
            array![[[0.3, 0.7], [0.05, 0.95]], 
                   [[0.9, 0.1], [0.5, 0.5]]].into_dyn()
        ).unwrap();

        let cpd_s = Factor::cpd(s, vec![i], array![[0.95, 0.05], [0.2, 0.8]].into_dyn()).unwrap();
        let cpd_l = Factor::cpd(l, vec![g], array![[0.9, 0.1], [0.4, 0.6]].into_dyn()).unwrap();

        let builder = DirectedModelBuilder::new();
        let model = builder.with_variable(&d, HashSet::new(), Initialization::Binomial(0.6))
                           .with_variable(&i, HashSet::new(), Initialization::Binomial(0.7))
                           .with_variable(&g, vec![i, d].into_iter().collect(), Initialization::Table(cpd_g))
                           .with_variable(&s, vec![i].into_iter().collect(), Initialization::Table(cpd_s))
                           .with_variable(&l, vec![g].into_iter().collect(), Initialization::Table(cpd_l))
                           .build()
                           .unwrap();

        let mut evidence = Assignment::new();
        evidence.set(&d, 0);
        evidence.set(&l, 1);
        evidence.set(&s, 0);

        (i, model, evidence)
    }
    
    /// Utility method to test the actual inference task
    fn test_inference(i: Variable, engine: &mut ConditionalInferenceEngine, precision: f64) {
        let f = engine.infer(&vec![i].into_iter().collect());

        assert!(! f.is_err());

        let f = f.unwrap();
        assert_eq!(vec![i], f.scope());
        let mut assn = Assignment::new();
        assn.set(&i, 1);

        let expected = 0.02919708;
        println!("actual = {}", f.value(&assn).unwrap());
        assert!((f.value(&assn).unwrap() - expected).abs() < precision);
    }

    #[test]
    /// Test variable elimination
    fn variable_elimination() {
        let (i, model, evidence) = build_student_example();

        // note that this implicitly tests for_undirected as well!
        let mut engine = VariableEliminationEngine::for_directed(&model, &evidence);

        // the result should be the same on subsequent iterations
        for _ in 0..10 {
            test_inference(i, &mut engine, 0.00000001);
        }
    }
    
    #[test]
    /// Test importance sampling
    fn importance_sampling() {
        let (i, model, evidence) = build_student_example();

        // note that this implicitly tests for_undirected as well!
        let mut sampler = LikelihoodWeightedSampler::new(&model, &evidence);
        let mut engine = ImportanceSamplingEngine::new(&mut sampler, 2000);

        // the result should be the same on subsequent iterations
        for _ in 0..10 {
            test_inference(i, &mut engine, 0.01);
        }
    }
    
    #[test]
    /// Test importance sampling
    fn mcmc() {
        let (i, model, evidence) = build_student_example();

        // note that this implicitly tests for_undirected as well!
        let mut sampler = GibbsSampler::for_directed(&model, &evidence);
        let mut engine = McmcEngine::new(&mut sampler, 10000, 2000);

        // the result should be the same on subsequent iterations
        for _ in 0..10 {
            test_inference(i, &mut engine, 0.01);
        }
    }

}
