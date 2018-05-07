//! Defines a `Model`, which is a Bayesian (directed) or Markovian (undirected) graphical model
//! representing the factorization of a probability distribution P.

use factor::Factor;
use init::Initialization;
use util::{Result, JeromeError};
use variable::{Assignment, Variable};

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

    fn empty() -> Self {
        let graph = IndexMap::new();
        let names = BidirMap::new();

        DirectedModel { graph, names }
    }

    fn new(graph: IndexMap<Variable, Factor>, names: BidirMap<Variable, String>) -> Self {
        DirectedModel { graph, names }
    }

    /// Get the `Variable`s in this `DirectedModel`.
    ///
    /// # Returns
    /// the `Variable`s that comprise the `DirectedModel`
    pub fn vars(&self) -> Vec<&Variable> {
        self.graph.keys().collect() 
    }

    /// Condition the `DirectedModel` given the evidence.
    ///
    /// # Args
    /// evidence: a partial `Assignment` of the `Variable`s in this `DirectedModel`.
    ///
    /// # Returns:
    /// a new `DirectedModel` with scope ```self.vars() - evidence.keys()``` that represents the
    /// conditional distribution ```P(self.scope() - evidence.keys() | evidence.keys())```
    pub fn condition(&self, _evidence: &Assignment) -> Self {
        DirectedModel::empty()
    }

    /// Determine the probability of a full `Assignment` to the `Variable`s in the `DirectedModel`.
    ///
    /// Specifically, this computes ```P(zeta)```, where ```zeta``` is a full assignment.
    ///
    /// # Args
    /// assignment: a full `Assignment` to the `DirectedModel`
    ///
    /// # Returns
    /// the probability of the `Assignment` given the `DirectedModel`
    pub fn probability(&self, _assignment: &Assignment) -> Result<f64> {
        Ok(0.0)
    }

}



/// An implementation of the &[builder pattern] for creating a `DirectedModel`.
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


    pub fn with_variable(
        &mut self, 
        var: &Variable, 
        parents: HashSet<Variable>, 
        init: Initialization,
    ) -> &mut Self {
        self.add_variable(var, var.to_string(), parents, init)
    }


    /// Add a `Variable` ```var``` to the `DirectedModel`.
    ///
    /// # Args
    /// var: the variable to add to the model
    /// name: an optional name for the variable. If a name is not provided, one will be generated.
    /// parents: the parent variables of ```var``` in the model. The parents must already be in the
    /// init: the initialization mechanism for the CPD of ```var``` in the model.
    ///
    /// # Returns
    /// the builder object
    pub fn with_named_variable(
        &mut self, 
        var: &Variable, 
        name: String,
        parents: HashSet<Variable>, 
        init: Initialization,
    ) -> &mut Self {
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

    fn to_model(self) -> DirectedModel {
        DirectedModel { graph: self.factors, names: self.names }
    }

    /// Internal function that acutally does the variable addition to the model
    fn add_variable(
        &mut self, 
        var: &Variable, 
        name: String,
        parents: HashSet<Variable>, 
        init: Initialization,
    ) -> &mut Self {
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
        let mut scope = parents.clone();
        scope.insert(*var);

        let factor = init.build_factor(scope);
        
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

