//! Defines the `Error` type for the jerome library

use std::error::Error;
use std::fmt;
use std::result;

pub type Result<T> = result::Result<T, JeromeError>;

#[derive(Debug)]
pub enum JeromeError {

    /// Represents an incomplete assignment where a complete assignment was required.
    /// The value in the tuple is the names of the variables that were missing from the assignment.
    IncompleteAssignment,

    /// Represents an error where a certain constraint on a scope was not satisfied
    InvalidScope,

    /// Exactly what it sounds like
    DivideByZero,

    /// Represents an error where there was a parent variable expected, but not found
    MissingParent,

    /// Represents a variable that was present multiple times in a situation where it should only
    /// have been present once
    DuplicateVariable,

    /// Represents an attempt to initialize a variable with an incompatible Initialization
    InvalidInitialization,

    /// A general error with the given description
    General(String),

    /// An unknown error condition
    Unknown

}

impl Error for JeromeError {

    fn description(&self) -> &str {
        match self {
            &JeromeError::IncompleteAssignment => "Missing assignments to the required Variables",
            &JeromeError::InvalidScope => "Provided scope did not satisfy constraints",
            &JeromeError::DivideByZero => "Encountered division by zero",
            &JeromeError::MissingParent => "Missing a parent from the model",
            &JeromeError::DuplicateVariable => "A variable was encountered twice",
            &JeromeError::InvalidInitialization => "The user requested an invalid initialization",
            &JeromeError::General(ref err) => err.as_str(),
            &JeromeError::Unknown => "An unknown error occured"
        }
    }

    fn cause(&self) -> Option<&Error> {
        None
    }

}

impl fmt::Display for JeromeError {

    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.description())
    }

}

