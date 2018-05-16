//! Defines an `Estimator`, which is used to estimate parameters of a `Model` from a dataset.

use variable::Assignment;
use util::Result;

mod mle;
pub use self::mle::LocalMLEstimator;
pub use self::mle::ModelMLEstimator;

/// A trait that represents the ability to estimate the parameters of some model (be it a `Model`
/// or just a local CPD.
pub trait Estimator<'a, T> {

    /// Estimate the value of the parameters from the given dataset
    fn estimate(&mut self, dataset: impl Iterator<Item = &'a Assignment>) -> Result<T>;

}
