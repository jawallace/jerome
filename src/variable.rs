//! Definition of the variable module
//!
//! A `Variable` represents a random variable in a Probabilistic Graphic Models.

use std::collections::HashMap;
use super::JeromeError;

/// A `Variable` in a Probablistic Graphical Model. A `Variable` is a discrete quantity that can take
/// on a fixed number of potential values. These values are represented by simple integers,
/// starting at 0.
///
/// # Notes
/// A `Variable` is lightweight and designed to be copyable for ease of use. 
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct Variable {

    /// The unique identifier of the `Variable`
    id: usize,

    /// The cardinality of the `Variable` - i.e. how many possible values it can take on
    cardinality: usize

}

static mut variable_id: usize = 0;

impl Variable {

    /// Construct a new `Variable` with a unique identifier
    ///
    /// # Notes
    /// This is *not* thread-safe
    unsafe fn new(cardinality: usize) -> Self {
        let id = variable_id;
        variable_id += 1;
        
        Variable { id, cardinality }
    }

    /// Construct a new binary `Variable`.
    pub fn binary() -> Self {
        unsafe {
            Variable::new(2)
        }
    }

    /// Construct a new discrete `Variable` with a certain number of values
    pub fn discrete(cardinality: usize) -> Self {
        unsafe {
            Variable::new(cardinality)
        }
    }

    /// Get the 'cardinality' of the variable's domain. i.e. - how many values are valid assignments the 
    /// variable may take 
    pub fn cardinality(&self) -> usize {
        self.cardinality
    }

}

/// Represents an assignment of one or more `Variable`s to values.
pub struct Assignment {
    assignments: HashMap<Variable, usize>
}

impl Assignment {

    /// Construct a new, empty assignment.
    pub fn new() -> Self {
        Assignment { assignments: HashMap::new() }
    }

    /// Add an assignment to a variable.
    ///
    ///
    pub fn set(&mut self, v: Variable, value: usize) -> Option<JeromeError> {
        if self.assignments.contains_key(&v) {
            return Some(JeromeError::General(format!("Already contains an assignment to {:?}", v)));
        } 

        if value < v.cardinality() {
            self.assignments.insert(v, value);
            return None;
        }

        Some(
            JeromeError::General(
                format!("Error - cannot assign variable with cardinality {} a value of {}", v.cardinality(), value)
            )
        )
    }

    pub fn get(&self, v: Variable) -> Option<&usize> {
        self.assignments.get(&v)
    }

}

// Unit Tests for the Variable struct.
#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn binary() {
        let v = Variable::binary();
        assert_eq!(v.cardinality(), 2);
        
        let v2 = v;
        assert_eq!(v, v2);

        let v3 = Variable::binary();
        assert_ne!(v, v3);
        assert_ne!(v2, v3);
    }

    #[test]
    fn discrete() {
        let v = Variable::discrete(10);
        assert_eq!(v.cardinality(), 10);
        
        let v2 = v;
        assert_eq!(v, v2);

        let v3 = Variable::discrete(10);
        assert_ne!(v, v3);
        assert_ne!(v2, v3);
    }

    #[test]
    fn assignment() {
        let v = Variable::binary();
        let v2 = Variable::discrete(10);

        let mut assn = Assignment::new();
        if let None = assn.set(v, 2) {
            panic!("Did not fail when attempting to add an out of range value");
        }

        if let Some(_) = assn.set(v, 1) {
            panic!("Failed to add a value in range");
        }

        if let None = assn.set(v, 0) {
            panic!("Did not fail when adding a duplicate assignment");
        }

        match assn.get(v) {
            Some(&val) => assert_eq!(1, val),
            None => panic!("Returned incorrect value")
        };

        if let Some(_) = assn.get(v2) {
            panic!("Did not fail when attempting to retrieve an un-added variable");
        }

        if let None = assn.set(v2, 25) {
            panic!("Did not fail when attempting to add an out of range value");
        }

        if let Some(_) = assn.set(v2, 5) {
            panic!("Failed to add a value in range");
        }
    }
}
