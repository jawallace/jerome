//! Defines a `DirectedModel`, which is a Bayesian model that represents the factorization of 
//! a probability distribution P

use factor::Factor;
use init::Initialization;
use util::{Result, JeromeError};
use variable::{Assignment, Variable};
use super::Model;

use bidir_map::BidirMap;
use indexmap::IndexMap;

use std::collections::HashSet;


/// Represents a Bayesian Network - a Directed Probabilistic Graphical Model.
///
/// # Representation
/// The network is represented as a Directed Acyclic Graph (DAG). A traditional graph data
/// structure is not used for the simple representation of a `DirectedModel`; instead, the
/// Conditional Probability Distribution (CPD) of each `Variable` implicitly defines the edges of
/// the graph. The `Variable`s are held in their topological order to faciliate efficient
/// computations over the graph.
pub struct DirectedModel {

    /// The `Variable`s comprising the scope of the `DirectedModel` and their associated CPDs. Note
    /// that the `Factor` associated with a `Variable` ```X``` has scope ```X U Pa(X)````, where
    /// ```Pa(X)``` are the parent's of ```X```. Therefore, in the DAG represented by this map,
    /// there are edges ```P -> X forall P in X.scope() where P != X```
    graph: IndexMap<Variable, Factor>,

    /// The user-defined names of each `Variable`. This is a two way lookup ```(`Variable`->Name)```
    /// and ```(Name->`Variable`)```
    names: BidirMap<Variable, String>

}

impl DirectedModel {
   
    /// Get the `Factor` for the given variable in this model.
    pub fn cpd(&self, v: &Variable) -> Option<&Factor> {
        self.graph.get(v)
    }

    /// Get a topological order of the `DirectedModel`
    pub fn topological_order(&self) -> Vec<Variable> {
        self.graph.keys().cloned().collect() 
    }
}

impl Model for DirectedModel {

    type Model_Type = DirectedModel;
    
    /// Lookup a `Variable` in the `DirectedModel` based on the name
    fn lookup_variable(&self, name: &str) -> Option<&Variable> {
        self.names.get_by_second(&String::from(name))
    }

    /// Lookup a `Variable`'s name in the `DirectedModel`.
    fn lookup_name(&self, var: &Variable) -> Option<&String> {
        self.names.get_by_first(var)
    }

    /// Get all `Variable`s in the model.
    fn variables(&self) -> HashSet<Variable> {
        self.graph.keys().map(|&v| v).collect()
    }

    /// Get the number of `Variable`s in the the `DirectedModel`
    fn num_variables(&self) -> usize {
        self.graph.len()
    }

    /// Condition the `DirectedModel` given the evidence.
    fn condition(&self, evidence: &Assignment) -> Self {
        let mut builder = DirectedModelBuilder::new();

        // For each variable in the graph
        for (var, ref cpt) in self.graph.iter() {
            if let None = evidence.get(var) {
                // if the variable is *not* in the evidence, then it belongs in the new graph with
                // a CPT reduced by the evidence
                let new_cpt = cpt.reduce(evidence);
                let parents: HashSet<Variable> = new_cpt.scope().into_iter().filter(|v| v != var).collect();
                // safe to unwrap, we *know* var is in this model
                let name = self.lookup_name(var).unwrap();
                
                builder = builder.with_named_variable(var, name.as_str(), parents, Initialization::Table(new_cpt));
            }
        }

        builder.build().unwrap()
    }

    /// Determine the probability of a full `Assignment` to the `Variable`s in the `DirectedModel`.
    fn probability(&self, assignment: &Assignment) -> Result<f64> {
        // for every variable in the graph
        self.graph.values()
                  // get the probability of the assignment
                  .map(|ref cpt| cpt.value(assignment)) 
                  // and multiply those probability by the chain rule
                  // but if there are any errors, just return the error
                  .fold(Ok(1.0), |acc, val| acc.and_then(|p| val.map(|v| p * v)))
    }
}


/// An implementation of the [builder pattern] for creating a `DirectedModel`.
///
/// At the moment, models must be assembled in topological order. A more flexible API, while
/// desireable, would introduce a lot of complexity I don't currently have time for.
///
/// [builder pattern]: https://en.wikipedia.org/wiki/Builder_pattern
pub struct DirectedModelBuilder {

    /// The `Variable`s and their associated CPDs
    factors: IndexMap<Variable, Factor>,

    /// The names of each `Variable`
    names: BidirMap<Variable, String>,

    /// The error state of the builder
    err: Option<JeromeError>

}


impl DirectedModelBuilder {

    /// Construct a new `DirectedModelBuilder` representing an empty `DirectedModel`
    pub fn new() -> Self {
        DirectedModelBuilder { 
            factors: IndexMap::new(),
            names: BidirMap::new(),
            err: None
        }
    }


    /// Add an anonymous `Variable` to the `DirectedModel`.
    ///
    /// # Args
    /// * `var`: the variable to add to the model
    /// * `parents`: the parent variables. The parents must already be in the model.
    /// * `init`: the initialization mechanism for the CPD of `var` in the model.
    pub fn with_variable(
        self, 
        var: &Variable, 
        parents: HashSet<Variable>, 
        init: Initialization,
    ) -> Self {
        self.add_variable(var, var.to_string(), parents, init)
    }


    /// Add a named `Variable` to the `DirectedModel`.
    ///
    /// # Args
    /// * `var`: the variable to add to the model
    /// * `name`: the name for the variable. 
    /// * `parents`: the parent variables. The parents must already be in the model.
    /// * `init`: the initialization mechanism for the CPD of `var` in the model.
    pub fn with_named_variable(
        self, 
        var: &Variable, 
        name: &str,
        parents: HashSet<Variable>, 
        init: Initialization,
    ) -> Self {
        self.add_variable(var, String::from(name), parents, init)
    }


    /// Complete building the model.
    ///
    /// # Returns
    /// the `DirectedModel`, or an error if one was generated during the building process
    ///
    /// # Postcondition
    /// This call consumes the `DirectedModelBuilder`
    pub fn build(self) -> Result<DirectedModel> {
        if let Some(e) = self.err {
            Err(e)
        } else {
            Ok(self.to_model())
        }
    }

    /// Internal function that constructs the model
    fn to_model(self) -> DirectedModel {
        DirectedModel { graph: self.factors, names: self.names }
    }

    /// Internal function that acutally does the variable addition to the model
    fn add_variable(
        mut self, 
        var: &Variable, 
        name: String,
        parents: HashSet<Variable>, 
        init: Initialization,
    ) -> Self {
        ///////////////////////////////////////////////////////////////////////
        // 1) if we are in an error state, do nothing
        if self.err.is_some() {
            return self;
        }

        ///////////////////////////////////////////////////////////////////////
        // 2) Check for error conditions
        if parents.iter().any(|v| ! self.factors.contains_key(v)) {
            self.err = Some(JeromeError::MissingParent);
            return self;
        }

        if self.factors.contains_key(var) {
            self.err = Some(JeromeError::DuplicateVariable);
        }

        ///////////////////////////////////////////////////////////////////////
        // 3) Build the factor based on the initialization
        let factor = init.build_cpd(*var, parents.clone());
        
        if let Err(e) = factor {
            self.err = Some(e);
            return self;
        }

        let factor = factor.unwrap();

        ///////////////////////////////////////////////////////////////////////
        // 4) Add to current model
        self.factors.insert(*var, factor);
        self.names.insert(*var, name);

        self
    }
}

#[cfg(test)]
mod tests {

    #[cfg(test)]
    use super::*;

    #[test]
    fn build_empty() {
        let b = DirectedModelBuilder::new();
        let model = b.build();

        assert!(! model.is_err());

        let model = model.unwrap();
        assert_eq!(model.num_variables(), 0);
        assert!(model.variables().is_empty());
    }


    #[test]
    /// Tests building a model with a single binary variable
    fn build_simple() {
        let v = Variable::binary();
        let b = DirectedModelBuilder::new();
        let model = b.with_variable(&v, HashSet::new(), Initialization::Uniform).build().unwrap();
   
        let vars = model.variables();
        assert_eq!(1, vars.len());
        assert!(vars.contains(&v));
        let name = model.lookup_name(&v).unwrap();
        let v2 = model.lookup_variable(name.as_str()).unwrap();
        assert_eq!(&v, v2);

        let f = model.cpd(&v).unwrap();
        assert!(! f.is_identity());
        assert!(f.is_cpd());
        assert_eq!(vec![v], f.scope());
        let mut a = Assignment::new(); 
        a.set(&v, 0);
        assert_eq!(0.5, f.value(&a).unwrap());
        let mut a = Assignment::new();
        a.set(&v, 1);
        assert_eq!(0.5, f.value(&a).unwrap());
    }
    
    
    #[test]
    /// Tests building a model with a single, named binary variable
    fn build_named_simple() {
        let v = Variable::binary();
        let b = DirectedModelBuilder::new();
        let model = b.with_named_variable(&v, "foo", HashSet::new(), Initialization::Uniform)
                     .build()
                     .unwrap();
   
        let vars = model.variables();
        assert_eq!(1, vars.len());
        assert!(vars.contains(&v));
        let name = model.lookup_name(&v).unwrap();
        assert_eq!(name, "foo");
        let v2 = model.lookup_variable(name.as_str()).unwrap();
        assert_eq!(&v, v2);

        let f = model.cpd(&v).unwrap();
        assert!(! f.is_identity());
        assert!(f.is_cpd());
        assert_eq!(vec![v], f.scope());
        let mut a = Assignment::new(); 
        a.set(&v, 0);
        assert_eq!(0.5, f.value(&a).unwrap());
        let mut a = Assignment::new();
        a.set(&v, 1);
        assert_eq!(0.5, f.value(&a).unwrap());
    }

    
    #[test]
    /// Tests building a model with a single binary variable
    /// Example taken from Koller & Friedman Section 3.1.2
    fn intelligence() {
        let intelligence = Variable::binary();
        let sat = Variable::binary();

        let sfactor = Factor::cpd(sat, vec![intelligence], array![[0.95, 0.05], [0.2, 0.8]].into_dyn()).unwrap();

        ///////////////////////////////////////////////////////////////////////////////////////////
        // TEST BUILDING
        let b = DirectedModelBuilder::new();
        let model = b.with_named_variable(
                        &intelligence, 
                        "I", 
                        HashSet::new(), 
                        Initialization::Multinomial(&[0.7, 0.3])
                     ).with_named_variable(
                         &sat, 
                         "S", 
                         vec![intelligence].into_iter().collect(), 
                         Initialization::Table(sfactor)
                     ).build().unwrap();


        assert_eq!("I", model.lookup_name(&intelligence).unwrap());
        assert_eq!(&intelligence, model.lookup_variable("I").unwrap());
        assert_eq!("S", model.lookup_name(&sat).unwrap());
        assert_eq!(&sat, model.lookup_variable("S").unwrap());
        assert_eq!(2, model.num_variables());        

        ///////////////////////////////////////////////////////////////////////////////////////////
        // TEST GETTING PROBABILITY OF ASSIGNMENT
        let mut a = Assignment::new();
        a.set(&intelligence, 0);
        a.set(&sat, 0);
        assert_eq!(model.probability(&a).unwrap(), 0.7 * 0.95);
        
        let mut a = Assignment::new();
        a.set(&intelligence, 0);
        a.set(&sat, 1);
        assert_eq!(model.probability(&a).unwrap(), 0.7 * 0.05);
        
        let mut a = Assignment::new();
        a.set(&intelligence, 1);
        a.set(&sat, 0);
        assert_eq!(model.probability(&a).unwrap(), 0.3 * 0.2);
        
        let mut a = Assignment::new();
        a.set(&intelligence, 1);
        a.set(&sat, 1);
        assert_eq!(model.probability(&a).unwrap(), 0.3 * 0.8);

        // test partial assignment
        let mut a = Assignment::new();
        a.set(&intelligence, 1);
        assert!(model.probability(&a).is_err());

        ///////////////////////////////////////////////////////////////////////////////////////////
        // TEST CONDITIONING A DIRECTED MODEL
        let mut evidence = Assignment::new();
        evidence.set(&intelligence, 0);
        let new_model = model.condition(&evidence);

        // make sure structure looks right
        assert_eq!(1, new_model.num_variables());
        assert_eq!("S", new_model.lookup_name(&sat).unwrap().as_str());
        assert_eq!(&sat, new_model.lookup_variable("S").unwrap());

        // now check the probability
        for i in 0..2 {
            let mut a = Assignment::new();
            a.set(&sat, i);

            let expected = if i == 0 { 0.95 } else { 0.05 };
            assert_eq!(expected, new_model.probability(&a).unwrap());
        }
    }
}
