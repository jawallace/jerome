//! Provides an example of how to use Jerome to estimate inference on a Bayesian Network.
//!
//! Jeffrey Wallace
//! EN.605.425 Probabilistic Graphical Models

extern crate jerome;
#[macro_use]
extern crate ndarray;

use jerome as j;
use j::{Estimator, IndependentSampler, Model};
use std::collections::HashSet;

fn main() -> j::Result<()> {
    let difficulty = j::Variable::binary();
    let intelligence = j::Variable::binary();
    let grade = j::Variable::discrete(3);
    let sat = j::Variable::binary();
    let letter = j::Variable::binary();

    let scope = StudentVariables(difficulty, intelligence, grade, sat, letter);

    ////////////////////////////////////////////////////////////////////////////
    // Step 1:  Build Truth and Target Models
    // 
    // Note:    the target model is initialized with incorrect parameters
    let truth = build_model(scope.clone(), ModelType::Truth)?;
    let target = build_model(scope.clone(), ModelType::Target)?;

    ////////////////////////////////////////////////////////////////////////////
    // Step 2:  Build dataset from truth
    // 
    // Note:    this will generate samples from the truth distribution using
    //          forward sampling 
    let sampler = j::ForwardSampler::new(&truth);
    let num_samples = 10_000;
    let dataset: Vec<j::Assignment> = (0..num_samples).map(|_| sampler.ind_sample()).collect();
    
    ////////////////////////////////////////////////////////////////////////////
    // Step 3:  Build Estimator for the parameters 
    // 
    // Note:    this will sample from trutha
    let mut estimator = j::ModelMLEstimator::new(&target)?;
   
    ////////////////////////////////////////////////////////////////////////////
    // Step 4:  Estimate the model's parameters
    let estimated_model = estimator.estimate(dataset.iter())?;
    
    let scope = vec![intelligence, difficulty, grade, sat, letter];

    let mut acc_truth = 0.0;
    let mut acc_prior = 0.0;
    let mut acc_posterior = 0.0;
    println!("                                        | Truth         | Before        | Estimated");
    println!("-----------------------------------------------------------------------------------------");
    for assignment in j::all_assignments(&scope) {
        let p_truth     = truth.probability(&assignment)?;
        let p_prior     = target.probability(&assignment)?;
        let p_posterior = estimated_model.probability(&assignment)?;

        println!(
            "P(I = {}, D = {}, G = {}, S = {}, L = {})\t|\t{:.4}\t|\t{:.4}\t|\t{:.4}", 
            assignment.get(&intelligence).unwrap(),
            assignment.get(&difficulty).unwrap(),
            assignment.get(&grade).unwrap(),
            assignment.get(&sat).unwrap(),
            assignment.get(&letter).unwrap(),
            p_truth,
            p_prior,
            p_posterior
        );

        acc_truth += p_truth;
        acc_prior += p_prior;
        acc_posterior += p_posterior;
    }

    println!("-----------------------------------------------------------------------------------------");
    println!(
        "TOTAL:                                 \t|\t{:.4}\t|\t{:.4}\t|\t{:.4}", 
        acc_truth,
        acc_prior,
        acc_posterior
    );

    Ok(())
}

enum ModelType {
    Truth,
    Target
}

#[derive(Clone)]
struct StudentVariables(j::Variable, j::Variable, j::Variable, j::Variable, j::Variable);

fn build_model(vars: StudentVariables, mtype: ModelType) -> j::Result<j::DirectedModel> {
    let StudentVariables(d, i, g, s, l) = vars;

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

    let mut builder = j::DirectedModelBuilder::new();

    match mtype {
        ModelType::Truth => {
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
        },
        ModelType::Target => {
            builder = builder.with_named_variable(&d, "D", HashSet::new(), j::Initialization::Uniform);
            builder = builder.with_named_variable(&i, "I", HashSet::new(), j::Initialization::Uniform);
            builder = builder.with_named_variable(
                &g, "G", vec![d, i].into_iter().collect(), j::Initialization::Uniform
            );
            builder = builder.with_named_variable(
                &s, "S", vec![i].into_iter().collect(), j::Initialization::Uniform
            );
            builder = builder.with_named_variable(
                &l, "L", vec![g].into_iter().collect(), j::Initialization::Uniform
            );
        }
    }

    builder.build()
}


