//! Defines the `Sampler` trait - an object that can randomly sample from a `Model`.

use variable::Assignment;

pub mod forward;
pub use self::forward::ForwardSampler;

pub trait Sampler {
  
    /// Sample from the associated `Model`.
    fn sample(&mut self) -> Assignment;

}


pub trait IndependentSampler {
  
    /// Sample from the associated `Model`.
    fn ind_sample(&self) -> Assignment;

}

