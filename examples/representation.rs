//! Provides an example of how to use Jerome to represent Bayesian Networks.
//!
//! This example is taken from Koller & Friedman Exercise 3.4
//!
//! Jeffrey Wallace
//! EN.605.425 Probabilistic Graphical Models

extern crate jerome;
#[macro_use]
extern crate ndarray;

use jerome as j;
use j::Model;
use std::collections::HashSet;

fn main() -> j::Result<()> {

    ///////////////////////////////////////////////////
    // Step 1: Define variables

    let difficulty = j::Variable::binary();
    let intelligence = j::Variable::binary();
    let grade = j::Variable::discrete(3);
    let sat = j::Variable::binary();
    let letter = j::Variable::binary();

    ///////////////////////////////////////////////////
    // Step 2: Build CPTs for variables with parents
    let cpt_g = j::Factor::cpd(
        grade, 
        vec![intelligence, difficulty], 
        array![
            [[0.3, 0.4, 0.3], [0.05, 0.25, 0.7]],
            [[0.9, 0.08, 0.02], [0.5, 0.3, 0.2]]
        ].into_dyn()
    )?;

    let cpt_s = j::Factor::cpd(
        sat,
        vec![intelligence],
        array![
            [0.95, 0.05],
            [0.2, 0.8]
        ].into_dyn()
    )?;

    let cpt_l = j::Factor::cpd(
        letter,
        vec![grade],
        array![
            [0.1, 0.9],
            [0.4, 0.6],
            [0.99, 0.01]
        ].into_dyn()
    )?;

    ///////////////////////////////////////////////////
    // Step 2: Build the Model
    let mut builder = j::DirectedModelBuilder::new();
    builder = builder.with_named_variable(&difficulty, "D", HashSet::new(), j::Initialization::Binomial(0.6));
    builder = builder.with_named_variable(&intelligence, "I", HashSet::new(), j::Initialization::Binomial(0.7));
    builder = builder.with_named_variable(
        &grade, "G", vec![difficulty, intelligence].into_iter().collect(), j::Initialization::Table(cpt_g)
    );
    builder = builder.with_named_variable(
        &sat, "S", vec![intelligence].into_iter().collect(), j::Initialization::Table(cpt_s)
    );
    builder = builder.with_named_variable(
        &letter, "L", vec![grade].into_iter().collect(), j::Initialization::Table(cpt_l)
    );
    
    let model = builder.build()?;

    ///////////////////////////////////////////////////
    // Step 3: Determine Probability of Assignments
    let scope = vec![intelligence, difficulty, grade, sat, letter];

    let mut acc = 0.0;
    for assignment in j::all_assignments(&scope) {
        let p = model.probability(&assignment)?;

        println!(
            "P(I = {}, D = {}, G = {}, S = {}, L = {}) = {:.4}", 
            assignment.get(&intelligence).unwrap(),
            assignment.get(&difficulty).unwrap(),
            assignment.get(&grade).unwrap(),
            assignment.get(&sat).unwrap(),
            assignment.get(&letter).unwrap(),
            p
        );

        acc += p;
    }

    println!("---------------------------------------------");
    println!("TOTAL:                                 {:.4}", acc);

    Ok(())
}
