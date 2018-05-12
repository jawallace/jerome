//! Module containing initialization routines for the parameters of a model.

use factor::Factor;
use util::{JeromeError, Result};
use variable::Variable;

use ndarray::prelude as nd;
use ndarray_rand::RandomExt;
use rand::distributions::Range;

use std::collections::HashSet;

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
 
    /// Construct a CPD, initialized based on ```self```
    ///
    /// # Args
    /// * `scope`: a set of `Variable`s over which to build the `Factor`
    ///
    /// # Returns
    /// a `Factor`, initialized according to ```self```. The order of the `Variable`s in the
    /// resulting `Factor` is undefined.
    pub fn build_cpd(self, var: Variable, parents: HashSet<Variable>) -> Result<Factor> {
        ///////////////////////////////////////////////////////////////////////////////
        // Trivial cases

        // if this is a user defined factor, it just needs to be verified and returned
        if let Initialization::Table(f) = self {
            if ! f.is_cpd() {
                return Err(JeromeError::InvalidInitialization);
            } 
            
            let s = f.scope();
            if parents.iter().all(|v| s.contains(v)) && s.len() == parents.len() + 1 && s.contains(&var) {
                return Ok(f);
            } else {
                return Err(JeromeError::InvalidScope);
            }
        }
        
        ///////////////////////////////////////////////////////////////////////////////
        // Check for errors
        if parents.len() == 0 {

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
        // now, build CPD
        let mut shape: Vec<usize> = parents.iter().map(|v| v.cardinality()).collect();
        shape.push(var.cardinality());

        let tbl = match self {
            Initialization::Uniform => {
                // normalizing constant is just the number of elements
                let val = 1. / (var.cardinality() as f64);
                nd::Array::from_elem(shape, val).into_dyn()
            },
            Initialization::Random => {
                let ax = nd::Axis(shape.len() - 1);
                let mut tbl = nd::Array::random(shape, Range::new(1.0, 100.0));
                let z = tbl.sum_axis(ax);
                (tbl / z).into_dyn()
            },
            Initialization::Binomial(p) => {
                array![p, (1.0 - p)].into_dyn()
            },
            Initialization::Multinomial(p) => {
                nd::Array::from_iter(p.iter().map(|&x| x)).into_dyn()
            },
            Initialization::Table(_) => panic!("unreachable")
        };

        Factor::cpd(var, parents.into_iter().collect(), tbl)
    }

    /// Construct a factor, initialized based on ```self```
    ///
    /// # Args
    /// * `scope`: a set of `Variable`s over which to build the `Factor`
    ///
    /// # Returns
    /// a `Factor`, initialized according to ```self```. The order of the `Variable`s in the
    /// resulting `Factor` is undefined.
    pub fn build_factor(self, scope: HashSet<Variable>) -> Result<Factor> {
        ///////////////////////////////////////////////////////////////////////////////
        // Trivial cases
       
        if scope.is_empty() {
            return Err(JeromeError::InvalidScope);
        }

        // if this is a user defined factor, it just needs to be verified and returned
        if let Initialization::Table(f) = self {
            let s = f.scope();
            if s.iter().all(|v| scope.contains(v)) && s.len() == scope.len() {
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

        Factor::new(scope.into_iter().collect(), tbl)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use variable::{all_assignments, Assignment};
    use std;


    #[test]
    fn empty_scope() {
        let scope = HashSet::new();
        assert!(Initialization::Uniform.build_factor(scope.clone()).is_err());
        assert!(Initialization::Random.build_factor(scope.clone()).is_err());
        assert!(Initialization::Binomial(0.5).build_factor(scope.clone()).is_err());
        assert!(Initialization::Multinomial(&[0.333, 0.333, 0.333]).build_factor(scope.clone()).is_err());
    }


    #[test]
    fn invalid_scope_subset() {
        let a = Variable::discrete(3);
        let b = Variable::binary();

        let tbl = array![[0.1, 0.2], [0.3, 0.1], [0.2, 0.1]].into_dyn();
        let f = Factor::new(vec![a, b], tbl.clone()).unwrap();

        let init = Initialization::Table(f);

        let mut scope = HashSet::new();
        scope.insert(a);

        assert!(init.build_factor(scope).is_err());
    }
   

    #[test]
    fn invalid_scope_superset() {
        let a = Variable::discrete(3);
        let b = Variable::binary();
        let c = Variable::binary();

        let tbl = array![[0.1, 0.2], [0.3, 0.1], [0.2, 0.1]].into_dyn();
        let f = Factor::new(vec![a, b], tbl.clone()).unwrap();

        let init = Initialization::Table(f);

        let mut scope = HashSet::new();
        scope.insert(a);
        scope.insert(b);
        scope.insert(c);

        assert!(init.build_factor(scope).is_err());
    }


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
        let f = Factor::new(vec![a, b], tbl.clone()).unwrap();

        let init = Initialization::Table(f);

        let mut scope = HashSet::new();
        scope.insert(a);
        scope.insert(b);

        let factor = init.build_factor(scope.clone());
        assert!(! factor.is_err());
        
        let factor = factor.unwrap();
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
