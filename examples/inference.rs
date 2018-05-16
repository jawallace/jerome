//! Provides an example of how to use Jerome to perform inference on a Bayesian Network.
//!
//! Jeffrey Wallace
//! EN.605.425 Probabilistic Graphical Models

extern crate jerome;
#[macro_use]
extern crate ndarray;

use jerome as j;
use j::ConditionalInferenceEngine;
use std::collections::HashSet;

fn main() -> j::Result<()> {
    let difficulty = j::Variable::binary();
    let intelligence = j::Variable::binary();
    let grade = j::Variable::discrete(3);
    let sat = j::Variable::binary();
    let letter = j::Variable::binary();

    let scope = StudentVariables(difficulty, intelligence, grade, sat, letter);

    /////////////////////////////////////////////////////
    // Step 1: Build Model
    let model = build_model(scope)?;

    /////////////////////////////////////////////////////
    // Step 2: Compile some evidence
    let mut evidence = j::Assignment::new();
    evidence.set(&difficulty, 0);
    evidence.set(&letter, 1);
    evidence.set(&sat, 0);

    /////////////////////////////////////////////////////
    // Step 3: Build an inference engine
    
    // uncomment for variable elimination
    let mut engine = j::VariableEliminationEngine::for_directed(&model, &evidence);

    // uncomment for importance sampling
    // let mut sampler = j::LikelihoodWeightedSampler::new(&model, &evidence);
    // let mut engine = j::ImportanceSamplingEngine::new(&mut sampler, 2000);
    
    // uncomment for MCMC - gibbs sampling
    // let mut sampler = j::GibbsSampler::for_directed(&model, &evidence);
    // let burnin = 10_000;
    // let samples = 2_000;
    // let mut engine = j::McmcEngine::new(&mut sampler, burnin, samples);


    /////////////////////////////////////////////////////
    // Step 4: Run a Conditional Query

    let scope = vec![intelligence];
    let query = scope.iter().cloned().collect();
    let p = engine.infer(&query)?;

    for (i, assignment) in j::all_assignments(&scope).enumerate() {
        println!("P(I = {} | D = 0, S = 0, L = 1) = {:.4}", i, p.value(&assignment).unwrap());
    }

    Ok(())
}

struct StudentVariables(j::Variable, j::Variable, j::Variable, j::Variable, j::Variable);

fn build_model(vars: StudentVariables) -> j::Result<j::DirectedModel> {
    let StudentVariables(d, i, g, s, l) = vars;

    ///////////////////////////////////////////////////
    // Step 2: Build CPTs for variables with parents
    let cpt_g = j::Factor::cpd(
        g, 
        vec![i, d], 
        array![
            [[0.3, 0.4, 0.3], [0.05, 0.25, 0.7]],
            [[0.9, 0.08, 0.02], [0.5, 0.3, 0.2]]
        ].into_dyn()
    )?;

    let cpt_s = j::Factor::cpd(
        s,
        vec![i],
        array![
            [0.95, 0.05],
            [0.2, 0.8]
        ].into_dyn()
    )?;

    let cpt_l = j::Factor::cpd(
        l,
        vec![g],
        array![
            [0.1, 0.9],
            [0.4, 0.6],
            [0.99, 0.01]
        ].into_dyn()
    )?;

    ///////////////////////////////////////////////////
    // Step 2: Build the Model
    let mut builder = j::DirectedModelBuilder::new();
    builder = builder.with_named_variable(&d, "D", HashSet::new(), j::Initialization::Binomial(0.6));
    builder = builder.with_named_variable(&i, "I", HashSet::new(), j::Initialization::Binomial(0.7));
    builder = builder.with_named_variable(
        &g, "G", vec![d, i].into_iter().collect(), j::Initialization::Table(cpt_g)
    );
    builder = builder.with_named_variable(
        &s, "S", vec![i].into_iter().collect(), j::Initialization::Table(cpt_s)
    );
    builder = builder.with_named_variable(
        &l, "L", vec![g].into_iter().collect(), j::Initialization::Table(cpt_l)
    );
    
    builder.build()
}


