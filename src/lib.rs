extern crate bidir_map;
extern crate indexmap;

// ugly, but seems necessary atm: https://github.com/rust-lang/rust/issues/44342
#[cfg(test)]
#[macro_use]
extern crate itertools;
#[cfg(not(test))]
extern crate itertools;

#[macro_use]
extern crate ndarray;
extern crate ndarray_rand;
extern crate rand;

pub mod variable;
pub mod factor;
pub mod model;
pub mod init;
pub mod util;
pub use util::{Result, JeromeError};

