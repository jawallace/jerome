//! Defines an `UndirectedModel` which is a Markovian model that represents the factorization of a
//! probability distribution P

use factor::Factor;
use init::Initialization;
use util::{Result, JeromeError};
use variable::{all_assignments, Assignment, Variable};
use super::Model;

use bidir_map::BidirMap;

use std::collections::HashSet;

/// Represents a Markovian Network - an Undirected Probabilistic Graphical Model.
///
/// # Representation
/// Although a Markovian Network typically is represented as a graph, this representation does not
/// explicitly define the graph structure. Instead, it uses a logical view of a Markovian Network
/// as a collection of `Factor`s to present the semantics of operations over a Markovian Network. 
/// Although there are connections between these `Factor`s, they are not explicitly defined. 
pub struct UndirectedModel {

    /// The `Factor`s that comprise the `UndirectedModel`
    factors: Vec<Factor>,

    /// The `Variable`s that comprise the `UndirectedModel` and their names.
    variables: BidirMap<Variable, String>,

    /// The partition function of the Gibbs Distribution.
    partition: f64

}


impl UndirectedModel {
    
    /// Get the partition function of the `Factor`.
    pub fn partition(&self) -> f64 {
        self.partition
    }
}


impl Model for UndirectedModel {

    type Model_Type = UndirectedModel;
    
    /// Lookup a `Variable` in the `DirectedModel` based on the name
    fn lookup_variable(&self, name: &str) -> Option<&Variable> {
        self.variables.get_by_second(&String::from(name))
    }

    /// Lookup a `Variable`'s name in the `DirectedModel`.
    fn lookup_name(&self, var: &Variable) -> Option<&String> {
        self.variables.get_by_first(var)
    }

    /// Get all `Variable`s in the model.
    fn variables(&self) -> HashSet<Variable> {
        self.variables.first_col().cloned().collect()
    }

    /// Get the number of `Variable`s in the the `DirectedModel`
    fn num_variables(&self) -> usize {
        self.variables.first_col().count()
    }

    /// Condition the `Model` given the evidence.
    ///
    /// # Args
    /// * `evidence`: a partial `Assignment` of the `Variable`s in this `Model`.
    ///
    /// # Returns:
    /// a new `Model` with scope ```self.vars() - evidence.keys()``` that represents the
    /// conditional distribution ```P(self.scope() - evidence.keys() | evidence.keys())```
    fn condition(&self, evidence: &Assignment) -> Self {
        let factors = self.factors.iter().map(|ref f| f.reduce(evidence)).collect();
        let variables: BidirMap<Variable, String> = self.variables
                                                        .iter()
                                                        .filter(|(v, _)| evidence.get(v).is_none())
                                                        .map(|(&v, n)| (v, n.clone()))
                                                        .collect();

        let scope = variables.first_col().cloned().collect();
        let partition = compute_partition(&scope, &factors);

        UndirectedModel { factors, variables, partition }
    }


    /// Determine the probability of a full `Assignment` to the `Variable`s in the `Model`.
    ///
    /// Specifically, this computes ```P(zeta)```, where ```zeta``` is a full assignment.
    ///
    /// # Args
    /// * `assignment`: a full `Assignment` to the `Model`
    ///
    /// # Returns
    /// the probability of the `Assignment` given the `Model`
    fn probability(&self, assignment: &Assignment) -> Result<f64> {
        // for every factor in the graph
        self.factors.iter()
                    // get the value of the assignment
                    .map(|ref f| f.value(assignment))
                    // and multiply those probabilities together
                    // but if there are any errors, just return the error
                    .fold(Ok(1.0), |acc, val| acc.and_then(|p| val.map(|v| p * v)))
                    // and finally normalize by the partition function
                    .map(|v| v / self.partition)
    }

}


/// Utility function to compute the partition function given a set of `Factor`s.
fn compute_partition(scope: &Vec<Variable>, factors: &Vec<Factor>) -> f64 {
    let assn_val = |a: Assignment| -> f64 {
        factors.iter().map(|f| f.value(&a).unwrap()).product()
    };
    all_assignments(&scope).map(assn_val).sum()
}


/// An implementation of the [builder pattern] for creating a `UndirectedModel`.
///
/// [builder pattern]: https://en.wikipedia.org/wiki/Builder_pattern
pub struct UndirectedModelBuilder {

    /// The `Factor`s added to the `UndirectedModel`
    factors: Vec<Factor>,

    /// The name <-> variable mapping
    names: BidirMap<Variable, String>,

    /// The error state of the builder, if any
    err: Option<JeromeError>

}

impl UndirectedModelBuilder {

    /// Construct a new `UndirectedModelBuilder`
    pub fn new() -> UndirectedModelBuilder {
        UndirectedModelBuilder {
            factors: Vec::new(),
            names: BidirMap::new(),
            err: None
        }
    }


    /// Declare the name for a `Variable` in this `UndirectedModel`.
    ///
    /// This is optional; `Variable`s added to the `UndirectedModelBuilder` via `with_factor` that
    /// do not have a corresponding name will be assigned a default name.
    pub fn with_named_variable(mut self, var: &Variable, name: &str) -> Self {
        self.names.insert(*var, String::from(name));
        self
    }


    /// Add a `Factor` to the `UndirectedModel`.
    ///
    /// # Arguments
    /// * `scope`: the `Variable`s in the scope of the `Factor`
    /// * `init`: the desired method of initializing the `Factor`
    pub fn with_factor(mut self, scope: HashSet<Variable>, init: Initialization) -> Self {
        if self.err.is_some() {
            return self; 
        }

        match init.build_factor(scope) {
            Ok(f) => { 
                self.factors.push(f)
            },
            Err(e) => {
                self.err = Some(e);
            }
        };

        self
    }


    /// Build the `UndirectedModel`, ensuring consistency of the `Factor`s and `Variable`s
    ///
    /// # Errors
    /// * `JeromeError::InvalidScope` if there is a mismatch between the `Variable`s defined by
    /// calls to `with_named_variable` and `with_factor`
    pub fn build(mut self) -> Result<UndirectedModel> {
        if self.err.is_some() {
            return Err(self.err.unwrap());
        }

        // make sure there are no variables defined but not used in a factor
        for v in self.names.first_col() {
            if ! self.factors.iter().any(|f| f.scope().contains(v)) {
                return Err(JeromeError::InvalidScope);
            }
        }

        // for any unnamed variable in a factor, give it a name
        for ref f in self.factors.iter() {
            for v in f.scope().iter() {
                if ! self.names.contains_first_key(v) {
                    self.names.insert(*v, v.to_string());
                }
            }
        }

        // compute the partition function
        let vars: Vec<Variable> = self.names.first_col().cloned().collect();  

        Ok(UndirectedModel { 
            factors: self.factors.clone(),
            variables: self.names.clone(),
            partition: compute_partition(&vars, &self.factors)
        })
    }

}

#[cfg(test)]
mod tests {

    #[cfg(test)]
    use super::*;

    #[test]
    /// Tests the implementation of `UndirectedModel` using the Misconception example from Koller &
    /// Friedman Section 4.1
    fn misconception() {
        ///////////////////////////////////////////////////////////////////////////////////////////
        // TEST BUILDING THE MODEL
        
        // variables
        let a = Variable::binary();
        let b = Variable::binary();
        let c = Variable::binary();
        let d = Variable::binary();

        // factors
        let ab = Factor::new(vec![a, b], array![[30.0, 5.0], [1.0, 10.0]].into_dyn()).unwrap();
        let bc = Factor::new(vec![b, c], array![[100.0, 1.0], [1.0, 100.0]].into_dyn()).unwrap();
        let cd = Factor::new(vec![c, d], array![[1.0, 100.0], [100.0, 1.0]].into_dyn()).unwrap();
        let da = Factor::new(vec![d, a], array![[100.0, 1.0], [1.0, 100.0]].into_dyn()).unwrap();

        // build model
        let builder = UndirectedModelBuilder::new();
        let model = builder.with_named_variable(&a, "A")
                           .with_named_variable(&b, "B")
                           .with_named_variable(&c, "C")
                           .with_named_variable(&d, "D")
                           .with_factor(vec![a, b].into_iter().collect(), Initialization::Table(ab))
                           .with_factor(vec![b, c].into_iter().collect(), Initialization::Table(bc))
                           .with_factor(vec![c, d].into_iter().collect(), Initialization::Table(cd))
                           .with_factor(vec![d, a].into_iter().collect(), Initialization::Table(da))
                           .build();

        assert!(! model.is_err());

        let model = model.unwrap();
        assert_eq!(7_201_840.0, model.partition());
        
        ///////////////////////////////////////////////////////////////////////////////////////////
        // TEST PROBABILITIES
        let mut assn = Assignment::new();
        assn.set(&a, 0);
        assn.set(&b, 0);
        assn.set(&c, 0);
        assn.set(&d, 0);
        assert!((0.04 - model.probability(&assn).unwrap()).abs() < 0.005);
        
        let mut assn = Assignment::new();
        assn.set(&a, 0);
        assn.set(&b, 1);
        assn.set(&c, 1);
        assn.set(&d, 0);
        assert!((0.69 - model.probability(&assn).unwrap()).abs() < 0.005);
        
        let mut assn = Assignment::new();
        assn.set(&a, 1);
        assn.set(&b, 0);
        assn.set(&c, 0);
        assn.set(&d, 1);
        assert!((0.14 - model.probability(&assn).unwrap()).abs() < 0.005);
        
        let mut assn = Assignment::new();
        assn.set(&a, 1);
        assn.set(&b, 1);
        assn.set(&c, 0);
        assn.set(&d, 1);
        assert!((0.014 - model.probability(&assn).unwrap()).abs() < 0.0005);

        // test incomplete assignment
        let mut assn = Assignment::new();
        assn.set(&a, 1);
        assn.set(&b, 1);
        assn.set(&c, 0);
        assert!(model.probability(&assn).is_err());
        
        ///////////////////////////////////////////////////////////////////////////////////////////
        // TEST CONDITIONING
        let mut evidence = Assignment::new();
        evidence.set(&a, 0);
        evidence.set(&c, 1);
        let new_model = model.condition(&evidence);
        assert_eq!(2, new_model.num_variables());
        assert!(new_model.lookup_name(&b).is_some());
        assert!(new_model.lookup_name(&d).is_some());

        // test probability of new model
        let mut assn = Assignment::new();
        assn.set(&b, 0);
        assn.set(&d, 0);
        assert!((0.057 - new_model.probability(&assn).unwrap()) < 0.0005);
    }
}

