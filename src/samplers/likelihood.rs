//! Defines a `Sampler` for likelihood weighted particle generation for `DirectedModel`s.  
//! 
//! Koller & Friedman Algorithm 12.2 (pp 493)

use super::{WeightedSampler, IndependentWeightedSampler, WeightedSample};
use variable::Assignment;
use model::directed::DirectedModel;


/// A simple `Sampler` for Bayesian Models that uses likelihood weighted sampling to draw full
/// assignments from the `DirectedModel` given evidence.
pub struct LikelihoodWeightedSampler<'a> {

    /// The model from which to sample
    model: &'a DirectedModel,

    /// The evidence on which to condition
    evidence: &'a Assignment

}


impl<'a> LikelihoodWeightedSampler<'a> {

    pub fn new(model: &'a DirectedModel, evidence: &'a Assignment) -> Self {
        LikelihoodWeightedSampler { model, evidence }
    }


    fn get_sample(&self) -> WeightedSample {
        let mut a = Assignment::new();
        let mut w = 1.0;

        for ref var in self.model.topological_order().iter() {
            let cpd = self.model.cpd(var).unwrap();

            if let Some(&val) = self.evidence.get(var) {
                a.set(&var, val);
                // update the weight by P(var | Pa(var))
                w = w * cpd.value(&a).unwrap();
            } else {
                // this cannot fail, because we iterate in topological order so each variable will get
                // a full assignment (minus itself) thus satistfying the contract of sample_cpd
                let v_assignment = cpd.sample_cpd(&a).unwrap();
                a.set(&var, v_assignment);
            }
        }

        WeightedSample(a, w)
    }

}


impl<'a> WeightedSampler for LikelihoodWeightedSampler<'a> {

    fn weighted_sample(&mut self) -> WeightedSample {
        self.get_sample()
    }

}


impl<'a> IndependentWeightedSampler for LikelihoodWeightedSampler<'a> {
    
    fn ind_weighted_sample(&self) -> WeightedSample {
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


    /// Example taken from Koller & Friedman Figure 12.1 and Example 2.3
    #[test]
    fn sample() {
        let d = Variable::binary();
        let i = Variable::binary();
        let g = Variable::discrete(3);
        let s = Variable::binary();
        let l = Variable::binary();

        let cpd_g = Factor::cpd(g, vec![i, d], array![[ [0.3, 0.4, 0.3], [0.05, 0.25, 0.7] ],
                                                      [ [0.9, 0.08, 0.02], [0.5, 0.3, 0.2] ]].into_dyn()
        ).unwrap();

        let cpd_l = Factor::cpd(l, vec![g], array![ [0.1, 0.9], [0.4, 0.6], [0.99, 0.01] ].into_dyn()).unwrap();
        let cpd_s = Factor::cpd(s, vec![i], array![ [0.95, 0.05], [0.2, 0.8] ].into_dyn()).unwrap();

        let model = DirectedModelBuilder::new().with_variable(
                                                    &d, 
                                                    vec![].into_iter().collect(), 
                                                    Initialization::Binomial(0.6)
                                                ).with_variable(
                                                    &i, 
                                                    vec![].into_iter().collect(), 
                                                    Initialization::Binomial(0.7)
                                                ).with_variable(
                                                    &g, 
                                                    vec![i,d].into_iter().collect(), 
                                                    Initialization::Table(cpd_g)
                                                ).with_variable(
                                                    &s, 
                                                    vec![i].into_iter().collect(), 
                                                    Initialization::Table(cpd_s)
                                                ).with_variable(
                                                    &l, 
                                                    vec![g].into_iter().collect(), 
                                                    Initialization::Table(cpd_l)
                                                ).build().unwrap();


        let mut evidence = Assignment::new();
        evidence.set(&l, 0);
        evidence.set(&s, 1);

        let mut sampler = LikelihoodWeightedSampler::new(&model, &evidence);
        for _ in 0..100 {
            let WeightedSample(particle, weight) = sampler.weighted_sample();

            assert!(particle.get(&d).is_some());
            assert!(*particle.get(&d).unwrap() < 2);
            assert!(particle.get(&i).is_some());
            assert!(*particle.get(&i).unwrap() < 2);
            assert!(particle.get(&g).is_some());
            assert!(*particle.get(&g).unwrap() < 3);

            assert!(particle.get(&s).is_some());
            assert_eq!(*particle.get(&s).unwrap(), 1);
            assert!(particle.get(&l).is_some());
            assert_eq!(*particle.get(&l).unwrap(), 0);

            assert!(weight >= 0.0);
            assert!(weight <= 1.0);
        }
        
        for _ in 0..100 {
            let WeightedSample(particle, weight) = sampler.ind_weighted_sample();

            assert!(particle.get(&d).is_some());
            assert!(*particle.get(&d).unwrap() < 2);
            assert!(particle.get(&i).is_some());
            assert!(*particle.get(&i).unwrap() < 2);
            assert!(particle.get(&g).is_some());
            assert!(*particle.get(&g).unwrap() < 3);

            assert!(particle.get(&s).is_some());
            assert_eq!(*particle.get(&s).unwrap(), 1);
            assert!(particle.get(&l).is_some());
            assert_eq!(*particle.get(&l).unwrap(), 0);

            assert!(weight >= 0.0);
            assert!(weight <= 1.0);
        }

        // verify the weight in example 12.3
        loop {
            let WeightedSample(particle, weight) = sampler.weighted_sample();

            let dval = *particle.get(&d).unwrap();
            let ival = *particle.get(&i).unwrap();
            let gval = *particle.get(&g).unwrap();

            if dval == 1 && ival == 0 && gval == 1 {
                assert!((weight - 0.02).abs() < 0.001);
                break;
            }
        }
    }
}


