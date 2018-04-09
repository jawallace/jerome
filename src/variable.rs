//! Definition of the variable module
//!
//! A `Variable` represents a random variable in a Probabilistic Graphic Models.

pub trait Observeable<T> {

    fn observe(&mut self, val: T);

    fn assignment(&self) -> Option<T>;

}

#[derive(Clone, Debug)]
pub struct BinaryVariable {
    /// The name of the `Variable`
    name: String,
    assignment: Option<bool>
}

impl Observeable<bool> for BinaryVariable {
    
    fn observe(&mut self, val: bool) {
        self.assignment = Some(val);
    }

    fn assignment(&self) -> Option<bool> {
        self.assignment
    }

}

#[derive(Clone, Debug)]
pub struct DiscreteVariable {
    /// The name of the `Variable`
    name: String,
    count: u32,
    assignment: Option<u32>
}

impl Observeable<u32> for DiscreteVariable {
    
    fn observe(&mut self, val: u32) {
        if val >= self.count {
            panic!("Invalid value ({}) for DiscreteVariable with count ({})", val, self.count);    
        }

        self.assignment = Some(val);
    }

    fn assignment(&self) -> Option<u32> {
        self.assignment
    }

}

#[derive(Clone, Debug)]
pub struct EnumeratedVariable {
    /// The name of the `Variable`
    name: String,
    set: Vec<String>,
    assignment: Option<usize>
}

impl Observeable<String> for EnumeratedVariable {
    
    fn observe(&mut self, val: String) {
        if let Some(idx) = self.set.iter().position(|v| *v == val) {
            self.assignment = Some(idx);
        } else {
            panic!("Invalid value ({}) for EnumeratedVariable with values ({:?})", val, self.set);    
        }
    }

    fn assignment(&self) -> Option<String> {
        match self.assignment {
            Some(idx) => Some(self.set[idx].clone()),
            _ => None
        }
    }
}

#[derive(Clone, Debug)]
pub struct ContinuousVariable {
    /// The name of the `Variable`
    name: String,
    assignment: Option<f64>
}

impl Observeable<f64> for ContinuousVariable {
    
    fn observe(&mut self, val: f64) {
        self.assignment = Some(val);
    }
   

    fn assignment(&self) -> Option<f64> {
        self.assignment
    }

}


/// A `Domain` defines the domain of a random variable, the range of values over which it is
/// defined.
#[derive(Clone, Debug)]
pub enum Variable {

    /// A binary variable - can take on the values `true` and `false`
    Binary(BinaryVariable),

    /// A discrete variable with integer tags. The value passed as an argument defines the number
    /// of values. The corresponding tags are in the range `0...val`
    Discrete(DiscreteVariable),

    /// An enumerated set of named values. This is a discrete variable, just with more
    /// understandable tags
    Enumerated(EnumeratedVariable),

    /// A continuous variable. The allowed values are all real numbers. The underlying value is
    /// represented by a floating point value
    Continuous(ContinuousVariable)
}

impl Variable {

    /// Construct a new `Variable` with a domain of `Domain::Binary`
    pub fn new_binary(name: &str) -> Variable {
        Variable::Binary(BinaryVariable { 
            name: String::from(name),
            assignment: None
        })
    }

    /// Construct a new `Variable` with a domain of `Domain::Discrete`
    pub fn new_discrete(name: &str, count: u32) -> Variable {
        Variable::Discrete(DiscreteVariable { 
            name: String::from(name), 
            count: count,
            assignment: None
        })
    }

    /// Construct a new `Variable` with a domain of `Domain::Enumerated`
    pub fn new_enumerated(name: &str, values: &Vec<&str>) -> Variable {
        Variable::Enumerated(EnumeratedVariable {
            name: String::from(name),
            set: values.into_iter().map(|s| String::from(*s)).collect(),
            assignment: None
        })
    }

    /// Construct a new `Variable` with a domain of `Domain::Continuous`
    pub fn new_continuous(name: &str) -> Variable {
        Variable::Continuous(ContinuousVariable {
            name: String::from(name),
            assignment: None
        })
    }

    /// Get the name of the `Variable`
    pub fn name(&self) -> &str {
        match *self {
            Variable::Binary(ref bv) => &bv.name,
            Variable::Discrete(ref dv) => &dv.name,
            Variable::Enumerated(ref ev) => &ev.name,
            Variable::Continuous(ref cv) => &cv.name
        }
    }

    /// Check if this `Variable` is discrete
    pub fn is_discrete(&self) -> bool {
        match *self {
            Variable::Continuous(_) => true,
            _ => false
        }
    }
}

// Unit Tests for the Variable struct.
#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn binary() {
        let mut var = Variable::new_binary("Foo");
        assert_eq!(var.name(), "Foo");

        if let Variable::Binary(ref mut bin) = var {
            assert_eq!(bin.assignment(), None);
            bin.observe(true);
            if let Some(b) = bin.assignment() {
                assert!(b);
            } else {
                panic!("None after observing");
            }
        } else {
            panic!("Wrong variable type");
        }
    }

    #[test]
    fn discrete() {
        let mut var = Variable::new_discrete("Foo", 10);
        assert_eq!(var.name(), "Foo");
        
        if let Variable::Discrete(ref mut dis) = var {
            assert_eq!(dis.assignment(), None);
            dis.observe(0);
            if let Some(d) = dis.assignment() {
                assert_eq!(d, 0);
            } else {
                panic!("None after observing");
            }
        } else {
            panic!("Wrong variable type");
        }
    }

    #[test]
    #[should_panic]
    fn discrete_observe_err() {
        let mut var = Variable::new_discrete("Foo", 10);

        if let Variable::Discrete(ref mut dis) = var {
            dis.observe(25);
        }
    }

    #[test]
    fn enumerated() {
        let values = vec!["Probabilistic", "Graphical", "Models"];
        let mut var = Variable::new_enumerated("Foo", &values);
        assert_eq!(var.name(), "Foo");
        
        if let Variable::Enumerated(ref mut en) = var {
            assert_eq!(en.assignment(), None);
            en.observe(String::from("Probabilistic"));
            if let Some(v) = en.assignment() {
                assert_eq!(v, "Probabilistic");
            } else {
                panic!("None after observing");
            }
        } else {
            panic!("Wrong variable type");
        }
    }
    
    #[test]
    #[should_panic]
    fn enumerated_observe_err() {
        let values = vec!["Probabilistic", "Graphical", "Models"];
        let mut var = Variable::new_enumerated("Foo", &values);

        if let Variable::Enumerated(ref mut en) = var {
            en.observe(String::from("FooBar"));
        }
    }

    #[test]
    fn continuous() {
        let mut var = Variable::new_continuous("Foo");
        assert_eq!(var.name(), "Foo");
        
        if let Variable::Continuous(ref mut cv) = var {
            assert_eq!(cv.assignment(), None);
            cv.observe(1.05);
            if let Some(v) = cv.assignment() {
                assert!((1.05 - v).abs() < 0.00005);
            } else {
                panic!("None after observing");
            }
        } else {
            panic!("Wrong variable type");
        }
    }

}

