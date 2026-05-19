use crate::model::State;

pub trait Harness {
    fn normalize(&self, raw_input: &str) -> Result<State, anyhow::Error>;
}

pub struct DefaultHarness;

impl Harness for DefaultHarness {
    fn normalize(&self, _raw_input: &str) -> Result<State, anyhow::Error> {
        // Phase 2 implementation will go here
        Ok(State {
            entities: std::collections::HashMap::new(),
        })
    }
}
