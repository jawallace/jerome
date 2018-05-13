//! Defines a Gibbs `Sampler`.
//!
//! Implementation of Koller & Friedman Algorithm 12.4

use factor::Factor;
use model::Model;
use model::directed::DirectedModel;
use model::undirected::UndirectedModel;
use super::{IndependentWeightedSampler, LikelihoodWeightedSampler, Sampler, WeightedSample};
use variable::{all_assignments, Assignment, Variable};

use rand;
use rand::distributions::{Range, IndependentSample};

use std::collections::HashSet;

pub struct GibbsSampler {

    factors: Vec<Factor>,

    variables: HashSet<Variable>,

    sample: Assignment

}


impl GibbsSampler {

    /// Construct a new `GibbsSampler` from the `DirectedModel`
    ///
    /// Initializes the assignment from a `LikelihoodWeightedSampler`
    pub fn for_directed(model: &DirectedModel, evidence: &Assignment) -> Self {
        let factors = model.topological_order().iter().map(|v| model.cpd(v).unwrap()).cloned().collect();
        let variables = model.topological_order().iter().filter(|v| evidence.get(v).is_none()).cloned().collect();

        // use likelihood sampling to draw an initial sample from the mutilated Bayesian network
        let t0_sampler = LikelihoodWeightedSampler::new(model, evidence);
        let WeightedSample(sample, _) = t0_sampler.ind_weighted_sample();

        GibbsSampler { factors, variables, sample }
    }


    /// Construct a new `GibbsSampler` from the `UndirectedModel`
    ///
    /// Initializes the assignment by independently setting each `Variable` by drawing uniformly
    /// from it's values;
    ///
    /// TODO: is this actually valid? Is there a 'mutilated' network approach for undirected models
    /// like there is for directed?
    pub fn for_undirected(model: &UndirectedModel, evidence: &Assignment) -> Self {
        let factors = model.factors().clone();
        let variables: HashSet<Variable> = model.variables()
                                                .iter()
                                                .filter(|v| evidence.get(v).is_none())
                                                .cloned()
                                                .collect();

        let mut sample = Assignment::new();
        let mut rng = rand::thread_rng();
        for ref v in model.variables().iter() {
            if let Some(&val) = evidence.get(v) {
                sample.set(v, val);
            } else {
                let between = Range::new(0, v.cardinality());
                sample.set(v, between.ind_sample(&mut rng));
            }
        }

        GibbsSampler { factors, variables, sample }
    }

}

impl Sampler for GibbsSampler {

    fn sample(&mut self) -> Assignment {
        let mut rng = rand::thread_rng();
        let between = Range::new(0.0, 1.0);

        // for each variable in the sample set
        for &v in self.variables.iter() {

            //////////////////////////////////////////////////////////////
            // 1) unset v from the current sample
            self.sample.unset(&v);

            //////////////////////////////////////////////////////////////
            // 2) compute P(v | variables - {v})
            let factors_with_v: Vec<Factor> = self.factors
                                                  .iter()
                                                  .filter(|f| f.scope().contains(&v))
                                                  .map(|f| f.reduce(&self.sample))
                                                  .collect();

            let f = factors_with_v.iter()
                                  .fold(Ok(Factor::Identity), |acc, f1| acc.and_then(|f2| f1.product(&f2)))
                                  .unwrap();

            let scope = vec![v];
            let vals: Vec<f64> = all_assignments(&scope).map(|a| f.value(&a).unwrap()).collect();
            let sum: f64 = vals.iter().sum();
            let p: Vec<f64> = vals.iter().map(|val| val / sum).collect();

            //////////////////////////////////////////////////////////////
            // 3) Sample v from P and set v to the value in the sample
            let draw = between.ind_sample(&mut rng);
            let mut upper = 0.0;
            for i in 0..p.len() {
                upper = upper + p[i];
                if draw < upper {
                    self.sample.set(&v, i);     
                    break;
                }
            }
        }

        // return the sample
        self.sample.clone()
    }

}


#[cfg(test)]
mod tests {

    use super::*; 
    use model::directed::DirectedModelBuilder;
    use model::undirected::UndirectedModelBuilder;
    use factor::Factor;
    use init::Initialization;
    use variable::Variable;


    /// Example taken from Koller & Friedman Figure 12.1 and Example 2.3
    #[test]
    fn directed_student() {
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

        // need a better way to test this
        let mut sampler = GibbsSampler::for_directed(&model, &evidence);
        for _ in 0..100 {
            let particle = sampler.sample();

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
        }
    }
   

    /// Undirected equivalent of example taken from Koller & Friedman Figure 12.1 and Example 2.3
    #[test]
    fn undirected_student() {
        let d = Variable::binary();
        let i = Variable::binary();
        let g = Variable::discrete(3);
        let s = Variable::binary();
        let l = Variable::binary();

        let cpd_g = Factor::new(vec![i, d, g], array![[ [0.3, 0.4, 0.3], [0.05, 0.25, 0.7] ],
                                                      [ [0.9, 0.08, 0.02], [0.5, 0.3, 0.2] ]].into_dyn()
        ).unwrap();

        let cpd_l = Factor::new(vec![g, l], array![ [0.1, 0.9], [0.4, 0.6], [0.99, 0.01] ].into_dyn()).unwrap();
        let cpd_s = Factor::new(vec![i, s], array![ [0.95, 0.05], [0.2, 0.8] ].into_dyn()).unwrap();

        let model = UndirectedModelBuilder::new().with_factor(
                                                    vec![d].into_iter().collect(), 
                                                    Initialization::Binomial(0.6)
                                                 ).with_factor(
                                                    vec![i].into_iter().collect(),
                                                    Initialization::Binomial(0.7)
                                                 ).with_factor(
                                                    vec![i,d,g].into_iter().collect(), 
                                                    Initialization::Table(cpd_g)
                                                 ).with_factor(
                                                    vec![i,s].into_iter().collect(), 
                                                    Initialization::Table(cpd_s)
                                                 ).with_factor(
                                                    vec![g,l].into_iter().collect(), 
                                                    Initialization::Table(cpd_l)
                                                 ).build().unwrap();


        let mut evidence = Assignment::new();
        evidence.set(&l, 0);
        evidence.set(&s, 1);

        // need a better way to test this
        let mut sampler = GibbsSampler::for_undirected(&model, &evidence);
        for _ in 0..100 {
            let particle = sampler.sample();

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
        }
    }

    /// test to verify correctness of sampling method
    /// Constructs a directed network as follows:
    ///     A  B
    ///     \  /
    ///      C
    /// And uses C as evidence. This way, A and B are independent given the evidence and we can
    /// verify that the sampling is consistent with the distribution of A and B
    #[test]
    fn directed_test() {
        let a = Variable::binary();
        let b = Variable::binary();
        let c = Variable::binary();

        let model = DirectedModelBuilder::new().with_variable(
                                                    &a, 
                                                    vec![].into_iter().collect(), 
                                                    Initialization::Binomial(0.75)
                                                ).with_variable(
                                                    &b, 
                                                    vec![].into_iter().collect(), 
                                                    Initialization::Binomial(0.5)
                                                ).with_variable(
                                                    &c,
                                                    vec![a, b].into_iter().collect(),
                                                    Initialization::Uniform
                                                ).build().unwrap();

        let mut evidence = Assignment::new();
        evidence.set(&c, 0);

        let mut sampler = GibbsSampler::for_directed(&model, &evidence);

        let mut a_ct = 0;
        let mut b_ct = 0;
        let iters = 500;

        for _ in 0..iters {
            let particle = sampler.sample();

            a_ct = a_ct + *particle.get(&a).unwrap();
            b_ct = b_ct + *particle.get(&b).unwrap();
        }

        let p_a = 1.0 - (a_ct as f64) / (iters as f64);
        let p_b = 1.0 - (b_ct as f64) / (iters as f64);
        println!("A Statistics = {}", p_a);
        println!("B Statistics = {}", p_b);

        // generous error tolerance here. we will still probably fail this test occasionally.
        // should introduce seeding to ensure repeatability
        assert!((p_a - 0.75).abs() < 0.05);
        assert!((p_b - 0.5).abs() < 0.05);
    }
}
