//! Defines an importance-sampling `InferenceEngine` for approximate inference using particle-based
//! approximations.
//!
//! Implementation of Importance Sampling via Normalized Likelihood Weighting, described in Koller & 
//! Friedman 12.2.3.3

use factor::{Factor, Table};
use samplers::{WeightedSampler, WeightedSample};
use super::ConditionalInferenceEngine;
use util::{JeromeError, Result};
use variable::Variable;

use ndarray::prelude as nd;
use std::collections::HashSet;

/// An `InferenceEngine` for Bayesian Models using Importance Sampling
pub struct ImportanceSamplingEngine<'a, S: 'a + WeightedSampler> {

    /// The weighted sampler for the given `DirectedModel`
    sampler: &'a mut S,

    /// The number of samples to use
    samples: usize
}

impl<'a, S: WeightedSampler> ImportanceSamplingEngine<'a, S> {

    pub fn new(sampler: &'a mut S, samples: usize) -> Self {
        ImportanceSamplingEngine { sampler, samples }
    }

}

impl<'a, S: 'a + WeightedSampler> ConditionalInferenceEngine for ImportanceSamplingEngine<'a, S> {

    fn infer(&mut self, variables: &HashSet<Variable>) -> Result<Factor> {
        // initialize the factor table. We must assign an order to variables.
        let scope: Vec<Variable> = variables.iter().cloned().collect();
        let shape: Vec<usize> = variables.iter().map(|v| v.cardinality()).collect();
        let mut table = Table::zeros(shape);

        // sample away...
        for i in 0..self.samples {
            let WeightedSample(a, w) = self.sampler.weighted_sample();

            let idx: Vec<Option<&usize>> = scope.iter().map(|v| a.get(v)).collect();

            // on the first iteration, verify the assignment matches the scope
            if i == 0 {
                if idx.iter().any(|v| v.is_none()) {
                    return Err(JeromeError::InvalidScope);
                }
            }

            let idx: Vec<usize> = idx.iter().map(|v| v.unwrap()).cloned().collect();

            table[nd::IxDyn(&idx)] += w;
        }

        let factor = Factor::new(scope, table)?;
        Ok(factor.normalize())
    }

}

