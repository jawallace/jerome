//! Defines a `Model`, which is a Bayesian (directed) or Markovian (undirected) graphical model
//! representing the factorization of a probability distribution P.

use super::{Result, JeromeError};
use super::variable::{Assignment, Variable};
use super::factor::Factor;

use bidir_map::BidirMap;
use indexmap::IndexMap;
use ndarray::prelude as nd;
use ndarray_rand::RandomExt;
use rand::distributions::Range;

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

/// Defines possible ways to initialize a `Variable`s CPD.
pub enum Initialization<'a> {
    /// A uniform distribution over all possibilities
    Uniform,

    /// Randomly initialize the weights of the CPD. 
    Random,

    /// Initialize the CPD as a Binomial distribution with parameter ```p```.
    /// Note that this `Initialization` is valid only to a `Variable` with no parents.
    Binomial(f64),
    
    /// Initialize the CPD as a Multinomial distribution with parameters ```p_0, p_1...```.
    /// Note that this `Initialization` is valid only to a `Variable` with no parents.
    Multinomial(&'a [f64]),

    /// User defined CPD
    Table(Factor)
}


impl<'a> Initialization<'a> {
   
    /// Construct a factor, initialized based on ```self```
    pub fn build_factor(self, scope: HashSet<Variable>) -> Result<Factor> {
        ///////////////////////////////////////////////////////////////////////////////
        // Trivial cases
       
        if scope.is_empty() {
            return Err(JeromeError::InvalidScope);
        }

        // if this is a user defined factor, it just needs to be verified and returned
        if let Initialization::Table(f) = self {
            if ! f.is_cpd() {
                return Err(JeromeError::InvalidInitialization);
            } else if f.scope().iter().all(|v| scope.contains(v)) {
                return Ok(f);
            } else {
                return Err(JeromeError::InvalidScope);
            }
        }

        ///////////////////////////////////////////////////////////////////////////////
        // Check for errors
        if scope.len() == 1 {
            let var = scope.iter().next().unwrap();

            match self {

                // A binomial distribution on a non-binary variable
                Initialization::Binomial(_) if var.cardinality() != 2 => {
                    return Err(JeromeError::InvalidInitialization);
                },

                // A multinomial distribution with an incorrect number of parameters
                Initialization::Multinomial(ps) if ps.len() != var.cardinality() => {
                    return Err(JeromeError::InvalidInitialization);
                },

                _ => ()
            }
        } else {
            match self {

                // A binomial/multinomial on a non-unit scope
                Initialization::Binomial(_) | Initialization::Multinomial(_) => {
                    return Err(JeromeError::InvalidInitialization);
                },

                _ => ()
            }
        }

        ///////////////////////////////////////////////////////////////////////////////
        // now, build factor
        let shape: Vec<usize> = scope.iter().map(|v| v.cardinality()).collect();

        let tbl = match self {
            Initialization::Uniform => {
                // normalizing constant is just the number of elements
                let z: usize = shape.iter().product();
                let val = 1. / (z as f64);
                nd::Array::from_elem(shape, val).into_dyn()
            },
            Initialization::Random => {
                let mut tbl = nd::Array::random(shape, Range::new(1.0, 100.0));
                let z = tbl.scalar_sum();
                tbl.mapv_inplace(|e| e / z);
                tbl.into_dyn()
            },
            Initialization::Binomial(p) => {
                array![p, (1.0 - p)].into_dyn()
            },
            Initialization::Multinomial(p) => {
                nd::Array::from_iter(p.iter().map(|&x| x)).into_dyn()
            },
            Initialization::Table(_) => panic!("unreachable")
        };

        Factor::new(scope.into_iter().collect(), tbl, true)
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

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::variable::all_assignments;
    use std;

    #[test]
    fn random_init() {
        let a = Variable::binary();
        let b = Variable::discrete(10);
        let c = Variable::discrete(3);

        let init = Initialization::Random;

        let scope: HashSet<Variable> = vec![a, b, c].into_iter().collect();
        let factor = init.build_factor(scope.clone());
        assert!(! factor.is_err());
        
        let factor = factor.unwrap();
        assert!(factor.is_cpd());
        assert!(! factor.is_identity());
        let fscope: HashSet<Variable> = factor.scope().into_iter().collect();
        assert_eq!(scope, fscope);

        let sum: f64 = all_assignments(&factor.scope()).map(|a| factor.value(&a).unwrap()).sum();
        assert!(
            (1.0 - sum) < 0.001
        );
    }

    #[test]
    fn uniform_init() {
        let a = Variable::binary();
        let b = Variable::discrete(10);
        let c = Variable::discrete(3);

        let init = Initialization::Uniform;

        let scope: HashSet<Variable> = vec![a, b, c].into_iter().collect();
        let factor = init.build_factor(scope.clone());
        assert!(! factor.is_err());
        
        let factor = factor.unwrap();
        assert!(factor.is_cpd());
        assert!(! factor.is_identity());
        let fscope: HashSet<Variable> = factor.scope().into_iter().collect();
        assert_eq!(scope, fscope);

        let expected = 1.0 / ((a.cardinality() * b.cardinality() * c.cardinality()) as f64);
        for assn in all_assignments(&factor.scope()) {
            assert!(
                (expected - factor.value(&assn).unwrap()).abs() < std::f64::EPSILON
            );
        }
    }

    #[test]
    fn binomial_init() {
        let a = Variable::binary();

        let init = Initialization::Binomial(0.25);

        let mut scope = HashSet::new();
        scope.insert(a);

        let factor = init.build_factor(scope.clone());
        assert!(! factor.is_err());
        
        let factor = factor.unwrap();
        assert!(factor.is_cpd());
        assert!(! factor.is_identity());
        let fscope: HashSet<Variable> = factor.scope().into_iter().collect();
        assert_eq!(scope, fscope);

        let mut assn = Assignment::new();
        assn.set(&a, 0);
        assert!(
            (0.25 - factor.value(&assn).unwrap()).abs() < std::f64::EPSILON
        );
        
        let mut assn = Assignment::new();
        assn.set(&a, 1);
        assert!(
            (0.75 - factor.value(&assn).unwrap()).abs() < std::f64::EPSILON
        );
    }

    #[test]
    fn multinomial_init() {
        let a = Variable::discrete(3);

        let init = Initialization::Multinomial(&[ 0.1, 0.7, 0.2 ]);

        let mut scope = HashSet::new();
        scope.insert(a);

        let factor = init.build_factor(scope.clone());
        assert!(! factor.is_err());
        
        let factor = factor.unwrap();
        assert!(factor.is_cpd());
        assert!(! factor.is_identity());
        let fscope: HashSet<Variable> = factor.scope().into_iter().collect();
        assert_eq!(scope, fscope);

        let mut assn = Assignment::new();
        assn.set(&a, 0);
        assert!(
            (0.1 - factor.value(&assn).unwrap()).abs() < std::f64::EPSILON
        );
        
        let mut assn = Assignment::new();
        assn.set(&a, 1);
        assert!(
            (0.7 - factor.value(&assn).unwrap()).abs() < std::f64::EPSILON
        );
        
        let mut assn = Assignment::new();
        assn.set(&a, 2);
        assert!(
            (0.2 - factor.value(&assn).unwrap()).abs() < std::f64::EPSILON
        );
    }

    #[test]
    fn factor_init() {
        let a = Variable::discrete(3);
        let b = Variable::binary();

        let tbl = array![[0.1, 0.2], [0.3, 0.1], [0.2, 0.1]].into_dyn();
        let f = Factor::new(vec![a, b], tbl.clone(), true).unwrap();

        let init = Initialization::Table(f);

        let mut scope = HashSet::new();
        scope.insert(a);
        scope.insert(b);

        let factor = init.build_factor(scope.clone());
        assert!(! factor.is_err());
        
        let factor = factor.unwrap();
        assert!(factor.is_cpd());
        assert!(! factor.is_identity());
        let fscope: HashSet<Variable> = factor.scope().into_iter().collect();
        assert_eq!(scope, fscope);

        for (x, y) in (0..3).zip(0..2) {
            let idx = [x, y];
            let mut assn = Assignment::new();
            assn.set(&a, x);
            assn.set(&b, y);

            let expected = tbl[nd::IxDyn(&idx)];

            assert!(
                (expected - factor.value(&assn).unwrap()).abs() < std::f64::EPSILON
            );
        }
    }
}
