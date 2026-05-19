use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, Clone, Copy, PartialEq)]
pub enum Domain {
    Commerce,
    Logistics,
    Trade,
    Other,
}

impl Domain {
    pub fn as_str(&self) -> &'static str {
        match self {
            Domain::Commerce => "COMMERCE",
            Domain::Logistics => "LOGISTICS",
            Domain::Trade => "TRADE",
            Domain::Other => "OTHER",
        }
    }

    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_uppercase().as_str() {
            "COMMERCE" => Some(Domain::Commerce),
            "LOGISTICS" => Some(Domain::Logistics),
            "TRADE" => Some(Domain::Trade),
            "OTHER" => Some(Domain::Other),
            _ => Some(Domain::Other), // 매칭 실패 시 기본값을 Other로 안전하게 처리
        }
    }
}
