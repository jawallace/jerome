# jerome
A general probabilistic graphical models framework for Rust.

In the same vein as [edward](http://edwardlib.org) (which is named after an influential statistician) `jerome` is named for [Jerome
Cornfield](https://en.wikipedia.org/wiki/Jerome_Cornfield), a pioneer of Bayesian statistics that was the first to link a cause-effect relationship
between smoking and lung cancer and laid the groundwork for ["much of modern epidemiology"](https://blogs.sas.com/content/iml/2013/03/18/biography-of-jerome-cornfield.html).

# Getting Started
To use `jerome`, you will need Rust 1.26+. Instructions for installing Rust are
[here](https://blog.rust-lang.org/2018/05/10/Rust-1.26.html). 

The fastest way to start using `jerome` is to take a look at the examples in `examples/`. These examples can be run via
cargo, Rust's build tool:

    # an example of using jerome to represent Bayesian Networks
    cargo run --example representation
    # an example of using jerome to run inference on a Bayesian network
    cargo run --example inference
    # an example of using jerome to estimate parameters for a Bayesian network
    cargo run --example estimation

For example running the [estimation](examples/estimation.rs) example looks like this:

![jerome](jerome.png)

`jerome` comes with a comprehensive test suite. Tests can be run from cargo:

    cargo test

Note that, at the moment, the tests that use random number generation are not seeded, and therefore occasionally fail
if, for example, samples drawn from a model for inference yield a probability slightly outside of the defined limits.

Finally, HTML documentation for `jerome` and it's dependencies can be generated via cargo's `doc` command:

    cargo doc
    firefox target/doc/jerome/index.html

# Goals
The ultimate goal for `jerome` is to provide an efficient framework for Bayesian inference in Rust.

For the short term (i.e. for the purposes of this being a class project) the goal is to demonstrate the three 'pillars'
of probabilistic graphical models for **directed (Bayesian) models**:
* representation
* inference
* learning

# Tasks
Below is a rough outline of what has been accomplished so far. Please note that I was able to accomplish the goals I set
forth in the project outline (those are in bold below). The other tasks are things that I was hoping to get to and plan
to address in the future to end up with a useful framework.

## Representation
- [x] **Represent discrete random variables**
- [ ] Represent continous random variables
- [x] **Represent directed models**
- [x] Represent undirected models

## Inference
- [x] **Exact inference for directed models** (Variable Elimination)
- [ ] Exact inference for undirected models
- [x] **Approximate inference for directed models** (Importance Sampling, MCMC Methods)
- [x] Approximate inference for undirected models (MCMC Methods)

## Learning
- [x] **Maximum Likelihood parameter estimation for directed models**
- [ ] Bayesian parameter estimation for directed models

