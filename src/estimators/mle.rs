/// Defines `Estimator`s that use Maximum Likelihood Estimation to estimate the value of parameters
/// given a dataset.

use factor::{Factor, Table};
use init::Initialization;
use model::directed::{DirectedModel, DirectedModelBuilder};
use model::Model;
use super::Estimator;
use variable::{Assignment, Variable};
use util::{JeromeError, Result};

use ndarray::prelude as nd;

/// Defines the `LocalMLEstimator`, a Maximum Likelihood `Estimator` for the Conditional Probability 
/// Distribution of a single variable in a Bayesian framework.
///
/// Implementation of the MLE Parameter Estimation scheme for conditional probability distributions
/// described in Koller & Friedman Section 17.2
pub struct LocalMLEstimator {

    /// The scope of the estimated `Factor`
    scope: Vec<Variable>,

    /// The current CPT for the estimated `Factor`
    table: Table

}


impl LocalMLEstimator {

    /// Construct an ML estimator for the given factor
    pub fn new(factor: &Factor) -> Result<Self> {
        if factor.is_identity() || ! factor.is_cpd() {
            return Err(JeromeError::NotACPD); 
        }

        let scope = factor.scope();
        let shape: Vec<usize> = scope.iter().map(|v| v.cardinality()).collect();
        let table = Table::zeros(shape);

        Ok(LocalMLEstimator { scope, table })
    }

}


impl<'a> Estimator<'a, Factor> for LocalMLEstimator {

    fn estimate(&mut self, dataset: impl Iterator<Item = &'a Assignment>) -> Result<Factor> {
        // each call to estimate must be independent, so first let's zero the table.
        self.table *= 0.0;

        // count the number of instances of each configuration, using self.table as an accumulator
        for sample in dataset {
            let idx: Vec<Option<&usize>> = self.scope.iter().map(|v| sample.get(v)).collect();
            if idx.iter().any(|i| i.is_none()) {
                return Err(JeromeError::IncompleteAssignment);
            }

            let idx: Vec<usize> = idx.iter().map(|i| i.unwrap()).cloned().collect();

            self.table[nd::IxDyn(&idx)] += 1.0;
        }

        // now, we estimate each parameter by using the sufficient statistics (see K&F Eq. 17.5):
        //                  M[u, x]     <-- each value in the table
        //      theta x|u = -------
        //                   M[x]       <-- sum along last axis of table
        // where u is an assignment to self.scope[:-1] and x is an assignment to self.scope[-1]
        let mut m_u = self.table.sum_axis(nd::Axis(self.scope.len() - 1));
        if m_u.iter().any(|&ct| ct == 0.0) {
            return Err(JeromeError::DivideByZero);
        }

        if m_u.ndim() > 0 {
            let ct = m_u.shape()[0];
            m_u = m_u.into_shape((ct, 1)).unwrap().into_dyn();
        }

        let new_table = self.table.clone() / m_u;

        Factor::cpd(
            self.scope[self.scope.len() - 1], 
            self.scope.iter().cloned().take(self.scope.len() - 1).collect(),
            new_table
        )
    }
}


/// A Maximium Likelihood estimator for a `DirectedModel`
///
/// Based on the decomposability of the likelihood function, each CPD can be estimated separately
/// and therefore the `ModelMLEstimator` is really just a 'bag-o-`LocalMLEstimator`s'
pub struct ModelMLEstimator<'a> {

    /// The model for which to estimate the parameters
    model: &'a DirectedModel,

    /// The `Estimator` for each local CPD
    estimators: Vec<LocalMLEstimator>

}


impl<'a> ModelMLEstimator<'a> {
    
    pub fn new(model: &'a DirectedModel) -> Result<Self> {
        let estimators: Vec<Result<LocalMLEstimator>> = model.topological_order()
                                                             .iter()
                                                             .map(|v| model.cpd(&v).unwrap())
                                                             .map(|f| LocalMLEstimator::new(&f))
                                                             .collect();

        if let Some(ref r) = estimators.iter().find(|r| r.is_err()) {
            return match r {
                Err(e) => Err(e.clone()),
                _ => panic!("unreachable")
            };
        } 
        
        Ok(ModelMLEstimator { 
            model, 
            estimators: estimators.into_iter().map(|r| r.unwrap()).collect()
        })
    }

}


impl<'a> Estimator<'a, DirectedModel> for ModelMLEstimator<'a> {

    fn estimate(&mut self, dataset: impl Iterator<Item = &'a Assignment>) -> Result<DirectedModel> {
        let data: Vec<Assignment> = dataset.cloned().collect();

        let new_factors: Vec<Result<Factor>> = self.estimators
                                                   .iter_mut()
                                                   .map(|e| e.estimate(data.iter()))
                                                   .collect();

        let mut builder = DirectedModelBuilder::new();

        for (v, r) in self.model.topological_order().iter().zip(new_factors.into_iter()) {
            if let Err(e) = r {
                return Err(e);
            }

            let f = r.unwrap();
            let scope = f.scope();
            let num_vars = scope.len();

            builder = builder.with_named_variable(
                &v,
                self.model.lookup_name(&v).unwrap(),
                scope.into_iter().take(num_vars - 1).collect(),
                Initialization::Table(f)
            );
        }

        builder.build()
    }

}


#[cfg(test)]
mod tests {

    use super::*;
    use std::iter::repeat;
    use variable::all_assignments;
    use std::collections::HashSet;

    #[test]
    /// Test MLE of a single, binary variable (a weighted coin)
    fn coin_toss() {
        let c = Variable::binary();
        let f = Factor::cpd(c, Vec::new(), array![ 0.5, 0.5 ].into_dyn()).unwrap();

        let mut a = Assignment::new();
        a.set(&c, 0);
        let zeros = repeat(a).take(30);

        let mut a = Assignment::new();
        a.set(&c, 1);
        let ones = repeat(a).take(70);

        let mut estimator = LocalMLEstimator::new(&f).unwrap();
       
        let dataset: Vec<Assignment> = zeros.chain(ones).collect();
        let factor = estimator.estimate(dataset.iter()).unwrap();

        let mut a = Assignment::new();
        a.set(&c, 0);
        assert_eq!(0.3, factor.value(&a).unwrap());

        let mut a = Assignment::new();
        a.set(&c, 1);
        assert_eq!(0.7, factor.value(&a).unwrap());
    }
   

    #[test]
    /// Test X (binary) -> Y (binary) factor
    ///
    /// CPT:
    ///    | y0 | y1 
    /// ---+----+-----
    /// x0 | .2 | .8
    /// ---+----------
    /// x1 | .9 | .1
    fn one_parent() {
        let x = Variable::binary();
        let y = Variable::binary();

        let f = Factor::cpd(y, vec![x], array![[0.5, 0.5], [0.5, 0.5]].into_dyn()).unwrap();

        let mut a = Assignment::new();
        a.set(&x, 0);
        a.set(&y, 0);
        let zz = repeat(a).take(20);

        let mut a = Assignment::new();
        a.set(&x, 0);
        a.set(&y, 1);
        let zo = repeat(a).take(80);
        
        let mut a = Assignment::new();
        a.set(&x, 1);
        a.set(&y, 0);
        let oz = repeat(a).take(9);
        
        let mut a = Assignment::new();
        a.set(&x, 1);
        a.set(&y, 1);
        let oo = repeat(a).take(1);

        let mut estimator = LocalMLEstimator::new(&f).unwrap();
       
        let dataset: Vec<Assignment> = zz.chain(zo).chain(oz).chain(oo).collect();
        let factor = estimator.estimate(dataset.iter());
        assert!(factor.is_ok());
        let factor = factor.unwrap();
        assert!(factor.is_cpd());

        let vars = vec![x, y];
        let actual = all_assignments(&vars).map(|a| factor.value(&a).unwrap());
        let expected = vec![0.2, 0.8, 0.9, 0.1];
        assert!(expected.iter().zip(actual).all(|(e, a)| *e == a));
    }
    
    #[test]
    /// Test X (binary) -> Y (multinomial) factor
    ///
    /// CPT:
    ///    | y0 | y1 | y2
    /// ---+----+----+----
    /// x0 | .2 | .5 | .3
    /// ---+---------+----
    /// x1 | .7 | .1 | .1
    fn one_parent_binary_discrete() {
        let x = Variable::binary();
        let y = Variable::discrete(3);

        let f = Factor::cpd(y, vec![x], array![[0.1, 0.8, 0.1], [0.1, 0.8, 0.1]].into_dyn()).unwrap();

        let mut a = Assignment::new();
        a.set(&x, 0);
        a.set(&y, 0);
        let zz = repeat(a).take(20);

        let mut a = Assignment::new();
        a.set(&x, 0);
        a.set(&y, 1);
        let zo = repeat(a).take(50);
        
        let mut a = Assignment::new();
        a.set(&x, 0);
        a.set(&y, 2);
        let zt = repeat(a).take(30);
        
        let mut a = Assignment::new();
        a.set(&x, 1);
        a.set(&y, 0);
        let oz = repeat(a).take(8);
        
        let mut a = Assignment::new();
        a.set(&x, 1);
        a.set(&y, 1);
        let oo = repeat(a).take(1);
        
        let mut a = Assignment::new();
        a.set(&x, 1);
        a.set(&y, 2);
        let ot = repeat(a).take(1);

        let mut estimator = LocalMLEstimator::new(&f).unwrap();
       
        let dataset: Vec<Assignment> = zz.chain(zo)
                                         .chain(oz)
                                         .chain(oo)
                                         .chain(zt)
                                         .chain(ot)
                                         .collect();

        let factor = estimator.estimate(dataset.iter());
        assert!(factor.is_ok());
        let factor = factor.unwrap();
        assert!(factor.is_cpd());

        let vars = vec![x, y];
        let actual = all_assignments(&vars).map(|a| factor.value(&a).unwrap());
        let expected = vec![0.2, 0.5, 0.3, 0.8, 0.1, 0.1];
        assert!(expected.iter().zip(actual).all(|(e, a)| *e == a));
    }
    
    #[test]
    /// Test X (multinomial) -> Y (binomial) factor
    ///
    /// CPT:
    ///    | y0 | y1 
    /// ---+----+----
    /// x0 | .8 | .2 
    /// ---+---------
    /// x1 | .5 | .5 
    /// ---+----+----
    /// x2 | .3 | .7
    fn one_parent_discrete_binary() {
        let x = Variable::discrete(3);
        let y = Variable::binary();

        let f = Factor::cpd(y, vec![x], array![[0.1, 0.9], [0.1, 0.9], [0.1, 0.9]].into_dyn()).unwrap();

        let mut a = Assignment::new();
        a.set(&x, 0);
        a.set(&y, 0);
        let zz = repeat(a).take(80);

        let mut a = Assignment::new();
        a.set(&x, 0);
        a.set(&y, 1);
        let zo = repeat(a).take(20);
        
        let mut a = Assignment::new();
        a.set(&x, 1);
        a.set(&y, 0);
        let oz = repeat(a).take(500);
        
        let mut a = Assignment::new();
        a.set(&x, 1);
        a.set(&y, 1);
        let oo = repeat(a).take(500);
        
        let mut a = Assignment::new();
        a.set(&x, 2);
        a.set(&y, 0);
        let tz = repeat(a).take(3);
        
        let mut a = Assignment::new();
        a.set(&x, 2);
        a.set(&y, 1);
        let to = repeat(a).take(7);

        let mut estimator = LocalMLEstimator::new(&f).unwrap();
       
        let dataset: Vec<Assignment> = zz.chain(zo)
                                         .chain(oz)
                                         .chain(oo)
                                         .chain(tz)
                                         .chain(to)
                                         .collect();

        let factor = estimator.estimate(dataset.iter());
        assert!(factor.is_ok());
        let factor = factor.unwrap();
        assert!(factor.is_cpd());

        let vars = vec![x, y];
        let actual = all_assignments(&vars).map(|a| factor.value(&a).unwrap());
        let expected = vec![0.8, 0.2, 0.5, 0.5, 0.3, 0.7];
        assert!(expected.iter().zip(actual).all(|(e, a)| *e == a));
    }


    #[test]
    /// Test X (binomial) -> Y (binomial) model
    ///
    /// CPT X:
    /// x0 | .3
    /// x1 | .7
    ///
    /// CPT Y:
    ///    | y0 | y1 
    /// ---+----+----
    /// x0 | .8 | .2 
    /// ---+---------
    /// x1 | .5 | .5 
    ///
    /// Assuming 1000 samples:
    ///     x0: 300
    ///         y0: 300 * .8 = 240
    ///         y1: 300 * .2 = 60
    ///     x1: 700
    ///         y0: 700 * .5 = 350
    ///         y1: 350 * .5 = 350
    fn one_parent_model() {
        let x = Variable::binary();
        let y = Variable::binary();

        let builder = DirectedModelBuilder::new();
        let model = builder.with_named_variable(&x, "X", HashSet::new(), Initialization::Binomial(0.5))
                           .with_named_variable(&y, "Y", vec![x].into_iter().collect(), Initialization::Uniform)
                           .build()
                           .unwrap();

        let mut a = Assignment::new();
        a.set(&x, 0);
        a.set(&y, 0);
        let zz = repeat(a).take(240);

        let mut a = Assignment::new();
        a.set(&x, 0);
        a.set(&y, 1);
        let zo = repeat(a).take(60);
        
        let mut a = Assignment::new();
        a.set(&x, 1);
        a.set(&y, 0);
        let oz = repeat(a).take(350);
        
        let mut a = Assignment::new();
        a.set(&x, 1);
        a.set(&y, 1);
        let oo = repeat(a).take(350);
        
        let dataset: Vec<Assignment> = zz.chain(zo)
                                         .chain(oz)
                                         .chain(oo)
                                         .collect();
        
        let mut estimator = ModelMLEstimator::new(&model).unwrap();

        let new_model = estimator.estimate(dataset.iter());
        assert!(new_model.is_ok());
        let new_model = new_model.unwrap();

        assert_eq!(vec![x, y], new_model.topological_order());
        assert_eq!("X", new_model.lookup_name(&x).unwrap());
        assert_eq!("Y", new_model.lookup_name(&y).unwrap());

        let vars = vec![x, y];
        let actual = all_assignments(&vars).map(|a| new_model.probability(&a).unwrap());
        let expected = vec![0.3 * 0.8, 0.3 * 0.2, 0.7 * 0.5, 0.7 * 0.5];
        assert!(expected.iter().zip(actual).all(|(e, a)| *e == a));
    }
}

