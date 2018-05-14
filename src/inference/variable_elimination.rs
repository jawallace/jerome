//! Defines a `ConditionalInferenceEngine` that uses exact inference by variable elimination to
//! answer conditoinal inference queries.
//!
//! Implementation of Koller & Friedman Algorithm 9.1 - Sum-Product-VE

use factor::Factor;
use model::Model;
use model::directed::DirectedModel;
use model::undirected::UndirectedModel;
use super::ConditionalInferenceEngine;
use util::{JeromeError, Result};
use variable::{Assignment, Variable};

use std::collections::HashSet;
use std::collections::HashMap;

pub struct VariableEliminationEngine {
    
    /// the `UndirectedModel` (a 'bag of factors'), conditioned on the provided evidence, to use
    /// for the inference task
    model: UndirectedModel,

    /// precomputed preferred elimination order based on max-cardinality heuristic
    order: Vec<Variable>

}


impl VariableEliminationEngine {

    pub fn for_directed(model: &DirectedModel, evidence: &Assignment) -> Self {
        VariableEliminationEngine::for_undirected(
            &UndirectedModel::from(model),
            evidence
        )
    }

    pub fn for_undirected(model: &UndirectedModel, evidence: &Assignment) -> Self {
        // reduce the provided model with the evidence - this is the model we will use for variable
        // elimination
        let reduced = model.condition(evidence);
        // precompute the preferred elimination order using the max-cardinatlity heuristic.
        let order = max_cardinality_elimination_order(&reduced);

        VariableEliminationEngine {
            model: reduced,
            order: order
        }
    }

}

/// Compute the preferred elimination order by the max-cardinality heuristic
fn max_cardinality_elimination_order(model: &UndirectedModel) -> Vec<Variable> {
    // since we do not explictly hold the graph structure, we need to determine the neighbors of
    // each variable.
    let mut neighbors: HashMap<Variable, HashSet<Variable>> = model.variables()
                                                                   .iter()
                                                                   .map(|v| (*v, HashSet::new()))
                                                                   .collect();

    for f in model.factors().iter() {
        let scope = f.scope();
        for i in 0..(scope.len() - 1) {
            let vi = scope[i];
            for j in i..scope.len() {
                let vj = scope[j];
                neighbors.get_mut(&vi).unwrap().insert(vj);
                neighbors.get_mut(&vj).unwrap().insert(vi);
            }
        }
    }

    // set of marked variables
    let mut marked = HashSet::new();
    // the (reverse) elimination order
    let mut elimination = Vec::new();
    // the variables in the model
    let vars: Vec<Variable> = model.variables().into_iter().collect();

    // for |vars| iterations
    for _ in 0..vars.len() {
        let mut idx = None;

        // loop over all variables
        for (vidx, v) in vars.iter().enumerate() {
            // if we have already marked this variable, it is already in the elimination order so
            // we don't process it again
            if marked.contains(v) {
                continue;
            }

            // otherwise, count the number of marked neighbors
            let ct = neighbors[v].iter().filter(|&n| marked.contains(n)).count();

            // if there are more neighbors, update the max index to this variable's index
            if let Some((_, max)) = idx {
                if ct > max {
                    idx = Some((vidx, ct));
                }
            } else {
                idx = Some((vidx, ct));
            }
        }

        // invariant: this will *always* be Some
        // add the selected variable to the elimination order and marked variable list
        if let Some((i, _)) = idx {
            elimination.push(vars[i]);
            marked.insert(vars[i]);
        } else {
            panic!("This should be unreachable");
        }
    }

    // we need to reverse the elimination order before returning
    elimination.reverse();
    elimination
}


impl ConditionalInferenceEngine for VariableEliminationEngine {

    fn infer(&mut self, variables: &HashSet<Variable>) -> Result<Factor> {
        // check input arguments
        if variables.iter().any(|v| ! self.model.variables().contains(v)) {
            // a variable requested is not found in the (reduced) model
            return Err(JeromeError::InvalidScope);
        }

        let mut phis = self.model.factors().clone();
        for &var in self.order.iter() {
            if variables.contains(&var) {
                // we are computing P(var | e), so do not eliminate the variable
                continue;
            }

            // Otherwise, time to get rid of var
            let (phi_1prime, phi_2prime): (Vec<Factor>, Vec<Factor>) = phis
                                           .into_iter()
                                           .partition(|f| f.scope().contains(&var));

            // product step - multiply factors with var
            // Safe to unwrap the result of product here. 
            // We know the inputs are correct, so it will not fail.
            let psi = phi_1prime.into_iter()
                                .fold(Factor::Identity, |acc, phi| acc.product(&phi).unwrap());
                          
            // sum step - marginalize psi over var
            let tau = psi.marginalize(var);

            phis = phi_2prime;
            phis.push(tau);
        }

        // multiply together remaining phis
        let phi_star = phis.into_iter()
                           .fold(Factor::Identity, |acc, phi| acc.product(&phi).unwrap());

        // now we have an unnormalized distribution. We need the partition function to return a
        // conditional probability.
        return Ok(phi_star.normalize());
    }

}


#[cfg(test)]
mod tests {
    use super::*;
    use model::directed::DirectedModelBuilder;
    use init::Initialization;
    
    /// This example is taken from Koller & Friedman Example 9.3 and Figure 9.11.
    /// Verified by manual selection because the order is not deterministic. The annotation below can
    /// be uncommented to run this test - it will fail (by design), but will print out the
    /// calculated order, which can be verified against the graph constructed below.
    // #[test]
    #[allow(dead_code)]
    fn max_cardinality() {
        let c = Variable::binary();
        let d = Variable::binary();
        let i = Variable::binary();
        let g = Variable::binary();
        let s = Variable::binary();
        let l = Variable::binary();
        let j = Variable::binary();
        let h = Variable::binary();

        let builder = DirectedModelBuilder::new();
        let model = builder.with_variable(&c, HashSet::new(), Initialization::Uniform)
                           .with_variable(&d, vec![c].into_iter().collect(), Initialization::Uniform)
                           .with_variable(&i, HashSet::new(), Initialization::Uniform)
                           .with_variable(&g, vec![d, i].into_iter().collect(), Initialization::Uniform)
                           .with_variable(&s, vec![i].into_iter().collect(), Initialization::Uniform)
                           .with_variable(&l, vec![g].into_iter().collect(), Initialization::Uniform)
                           .with_variable(&j, vec![l, s].into_iter().collect(), Initialization::Uniform)
                           .with_variable(&h, vec![g, j].into_iter().collect(), Initialization::Uniform)
                           .build()
                           .unwrap();

        let model = UndirectedModel::from(&model);
        let order = max_cardinality_elimination_order(&model);

        println!("{:?}", order);

        // by failing on purpose, the print out above will be displayed
        assert!(false); 
    }

}
