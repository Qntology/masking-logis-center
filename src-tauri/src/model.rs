use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct State {
    pub entities: HashMap<String, Entity>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Entity {
    pub info: Info,
    pub status: String,
    pub components: Vec<Component>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Info {
    pub name: String,
    pub metadata: HashMap<String, String>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Component {
    pub id: String,
    pub component_type: String,
    pub status: String,
    pub details: Option<String>,
    pub reason: Option<String>,
}

impl State {
    pub fn to_yaml(&self) -> Result<String, serde_yaml::Error> {
        serde_yaml::to_string(self)
    }

    pub fn from_yaml(yaml: &str) -> Result<Self, serde_yaml::Error> {
        serde_yaml::from_str(yaml)
    }
}
