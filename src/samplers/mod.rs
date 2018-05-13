//! Defines the `Sampler` trait - an object that can randomly sample from a `Model`.

use variable::Assignment;

pub mod forward;
pub mod likelihood;

pub use self::forward::ForwardSampler;
pub use self::likelihood::LikelihoodWeightedSampler;

pub trait Sampler {
  
    /// Sample from the associated `Model`.
    fn sample(&mut self) -> Assignment;

}


pub trait IndependentSampler {
  
    /// Sample from the associated `Model`.
    fn ind_sample(&self) -> Assignment;

}


/// A sample (a full assignment) and the associated weight.
/// Used for likelihood weighting.
pub struct WeightedSample(Assignment, f64);

pub trait WeightedSampler {
    
    fn weighted_sample(&mut self) -> WeightedSample;

}

pub trait IndepdendentWeightedSampler {

    fn ind_weighted_sample(&self) -> WeightedSample;

}

