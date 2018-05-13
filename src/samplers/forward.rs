//! Defines a simple forward sampler for Bayesian `Model`s
//!
//! Implementation of Koller & Friedman Algorithm 12.1 (pp 489)

use model::directed::DirectedModel;
use variable::Assignment;
use super::{IndependentSampler, Sampler};

/// A simple, stateless `Sampler` for Bayesian Models
pub struct ForwardSampler<'a> {
    
    /// The `DirectedModel` to sample
    model: &'a DirectedModel
}


impl<'a> ForwardSampler<'a> {
    
    pub fn new(model: &'a DirectedModel) -> Self {
        ForwardSampler { model }
    }

    fn get_sample(&self) -> Assignment {
        let mut a = Assignment::new();

        for ref var in self.model.topological_order().iter() {
            let cpd = self.model.cpd(var).unwrap();

            // this cannot fail, because we iterate in topological order so each variable will get
            // a full assignment (minus itself) thus satistfying the contract of sample_cpd
            let v_assignment = cpd.sample_cpd(&a).unwrap();
            a.set(&var, v_assignment);
        }

        a
    }
}

impl<'a> Sampler for ForwardSampler<'a> {

    fn sample(&mut self) -> Assignment {
        self.get_sample()
    }

}


impl<'a> IndependentSampler for ForwardSampler<'a> {

    fn ind_sample(&self) -> Assignment {
        self.get_sample()
    }

}


#[cfg(test)]
mod tests {

    use super::*; 
    use model::directed::DirectedModelBuilder;
    use factor::Factor;
    use init::Initialization;
    use variable::Variable;

    use std::collections::HashSet;

    #[test]
    fn sample() {
        let intelligence = Variable::binary();
        let sat = Variable::binary();

        let sfactor = Factor::cpd(sat, vec![intelligence], array![[0.95, 0.05], [0.2, 0.8]].into_dyn()).unwrap();

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

        let mut sampler = ForwardSampler::new(&model);

        for _ in 0..100 {
            let a = sampler.sample();
            
            assert!(a.get(&intelligence).is_some());
            assert!(*a.get(&intelligence).unwrap() <= 1);
            assert!(a.get(&sat).is_some());
            assert!(*a.get(&sat).unwrap() <= 1);
        }

        for _ in 0..100 {
            let a = sampler.ind_sample();
            
            assert!(a.get(&intelligence).is_some());
            assert!(*a.get(&intelligence).unwrap() <= 1);
            assert!(a.get(&sat).is_some());
            assert!(*a.get(&sat).unwrap() <= 1);
        }
    }

}
