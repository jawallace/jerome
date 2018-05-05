#[macro_use]
extern crate ndarray;

#[macro_use] 
extern crate itertools;

pub mod variable;
pub mod factor;
pub mod util;
pub use util::{Result, JeromeError};

