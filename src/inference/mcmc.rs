//! Defines a `ConditionalInferenceEngine` for Markov-Chain Monte-Carlo methods.
//!
//! Implementation of MCMC Inference for Conditional Queries described in Koller & Friedman
//! 12.3.5.2

use factor::{Factor, Table};
use samplers::Sampler;
use super::ConditionalInferenceEngine;
use util::{JeromeError, Result};
use variable::Variable;

use ndarray::prelude as nd;
use std::collections::HashSet;

/// A `ConditionalInferenceEngine` for Bayesian or Markovian models. This is intended for
/// Markov-Chain Monte-Carlo `Sampler`s.
pub struct McmcEngine<'a, S: 'a + Sampler> {

    /// The weighted sampler for the given `DirectedModel`
    sampler: &'a mut S,

    /// The number of samples to use
    samples: usize

}

impl<'a, S: Sampler> McmcEngine<'a, S> {

    pub fn new(sampler: &'a mut S, burnin: usize, samples: usize) -> Self {
        // let the sampler burnin
        for _ in 0..burnin {
            let _ = sampler.sample(); 
        }

        McmcEngine { sampler, samples }
    }

}

impl<'a, S: 'a + Sampler> ConditionalInferenceEngine for McmcEngine<'a, S> {

    fn infer(&mut self, variables: &HashSet<Variable>) -> Result<Factor> {
        // initialize the factor table. We must assign an order to variables.
        let scope: Vec<Variable> = variables.iter().cloned().collect();
        let shape: Vec<usize> = variables.iter().map(|v| v.cardinality()).collect();
        let mut table = Table::zeros(shape);

        // sample away...
        for i in 0..self.samples {
            let a = self.sampler.sample();

            let idx: Vec<Option<&usize>> = scope.iter().map(|v| a.get(v)).collect();

            // on the first iteration, verify the assignment matches the scope
            if i == 0 {
                if idx.iter().any(|v| v.is_none()) {
                    return Err(JeromeError::InvalidScope);
                }
            }

            let idx: Vec<usize> = idx.iter().map(|v| v.unwrap()).cloned().collect();

            table[nd::IxDyn(&idx)] += 1.0;
        }

        let factor = Factor::new(scope, table)?;
        Ok(factor.normalize())
    }

}

