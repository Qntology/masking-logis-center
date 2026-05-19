use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, Clone, Copy, PartialEq)]
pub enum Domain {
    Commerce,
    Logistics,
    Trade,
}

impl Domain {
    pub fn as_str(&self) -> &'static str {
        match self {
            Domain::Commerce => "COMMERCE",
            Domain::Logistics => "LOGISTICS",
            Domain::Trade => "TRADE",
        }
    }

    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_uppercase().as_str() {
            "COMMERCE" => Some(Domain::Commerce),
            "LOGISTICS" => Some(Domain::Logistics),
            "TRADE" => Some(Domain::Trade),
            _ => None,
        }
    }
}
