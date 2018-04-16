//! Defines the `Error` type for the jerome library

use std::error;
use std::fmt;
use std::result;

pub type Result<T> = result::Result<T, Error>;

#[derive(Debug)]
pub enum Error {

    /// Represents an incomplete assignment where a complete assignment was required.
    /// The value in the tuple is the names of the variables that were missing from the assignment.
    IncompleteAssignment(Vec<String>),

    /// A general error with the given description
    General(String, Option<&Error>),

    /// An unknown error condition
    Unknown

}

impl error::Error for Error {

    fn description(&self) -> &str {
        match *self {
            IncompleteAssignment(names) {
                format!("Missing assignments to the following required Variables: {:?}", names)
            },
            General(err, _) => err,
            Unknown => "An unknown error occured"
        }
    }

    fn cause(&self) -> Option<&error::Error> {
        match *self {
            General(_, err) => err,
            _ => None,
        }
    }

}

impl fmt::Display for Error {

    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, self.description())
    }

}

