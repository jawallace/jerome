extern crate bidir_map;
extern crate indexmap;
#[macro_use]
extern crate itertools;
#[macro_use]
extern crate ndarray;
extern crate ndarray_rand;
extern crate rand;

pub mod variable;
pub mod factor;
pub mod model;
pub mod util;
pub use util::{Result, JeromeError};

