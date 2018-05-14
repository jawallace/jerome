//! Defines an importance-sampling `InferenceEngine` for approximate inference using particle-based
//! approximations.
//!
//! Implementation of Importance Sampling via Normalized Likelihood Weighting, described in Koller & 
//! Friedman 12.2.3.3

use factor::{Factor, Table};
use model::Model;
use model::directed::DirectedModel;
use samplers::{IndependentWeightedSampler, LikelihoodWeightedSampler, WeightedSample};
use super::ConditionalInferenceEngine;
use util::{JeromeError, Result};
use variable::{Assignment, Variable};

use ndarray::prelude as nd;
use std::collections::HashSet;

/// An `InferenceEngine` for Bayesian Models using Importance Sampling
pub struct ImportanceSamplingEngine<'a> {
  
    /// The model over which to perform inference
    model: &'a DirectedModel,

    /// The Likelihood-Weighted sampler for the given `DirectedModel`
    sampler: LikelihoodWeightedSampler<'a>,

    /// The number of samples to use
    samples: usize
}

impl<'a> ImportanceSamplingEngine<'a> {

    pub fn new(model: &'a DirectedModel, evidence: &'a Assignment, samples: usize) -> Self {
        ImportanceSamplingEngine {
            model,
            sampler: LikelihoodWeightedSampler::new(model, evidence),
            samples
        }
    }

}

impl<'a> ConditionalInferenceEngine for ImportanceSamplingEngine<'a> {

    fn infer(&self, variables: &HashSet<Variable>) -> Result<Factor> {
        // check input arguments
        if variables.iter().any(|v| ! self.model.variables().contains(v)) {
            // a variable requested is not found in the (reduced) model
            return Err(JeromeError::InvalidScope);
        }

        // initialize the factor table. We must assign an order to variables.
        let scope: Vec<Variable> = variables.iter().cloned().collect();
        let shape: Vec<usize> = variables.iter().map(|v| v.cardinality()).collect();
        let mut table = Table::zeros(shape);

        // sample away...
        // TODO improve termination criteria
        for _ in 0..self.samples {
            let WeightedSample(a, w) = self.sampler.ind_weighted_sample();

            let idx: Vec<usize> = scope.iter().map(|v| a.get(v).unwrap()).cloned().collect();
            table[nd::IxDyn(&idx)] += w;
        }

        let factor = Factor::new(scope, table)?;
        Ok(factor.normalize())
    }

}

