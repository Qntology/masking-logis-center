use crate::model::{State, Entity, Info, Component};
use crate::domain::Domain;
use scraper::{Html, Selector, Node};
use std::collections::HashMap;

pub trait Harness {
    fn normalize(&self, raw_input: &str) -> Result<State, anyhow::Error>;
}

pub struct DefaultHarness;

impl DefaultHarness {
    fn clean_html(&self, html: &str) -> String {
        let document = Html::parse_document(html);
        let mut cleaned_html = String::new();

        self.process_node(document.root_element().clone(), &mut cleaned_html);
        cleaned_html
    }

    fn process_node(&self, element_ref: scraper::ElementRef, output: &mut String) {
        let element = element_ref.value();
        let tag = element.name();
        
        // Skip forbidden tags
        if matches!(tag, "script" | "style" | "link" | "noscript" | "iframe" | "svg" | "meta" | "br" | "hr" | "source") {
            return;
        }

        // Handle <select> logic
        if tag == "select" {
            output.push_str("<select>");
            for child in element_ref.children() {
                if let Node::Element(child_el) = child.value() {
                    if child_el.name() == "option" {
                        let is_selected = child_el.attr("selected").is_some();
                        if is_selected {
                            output.push_str("<option selected>");
                            for grandchild in child.children() {
                                if let Node::Text(text) = grandchild.value() {
                                    output.push_str(text);
                                }
                            }
                            output.push_str("</option>");
                        }
                    }
                }
            }
            output.push_str("</select>");
            return;
        }

        // Generic tag processing (strip attributes)
        if tag == "input" {
            let mut attrs = Vec::new();
            for attr in &["value", "selected", "checked"] {
                if let Some(val) = element.attr(attr) {
                    attrs.push(format!("{}=\"{}\"", attr, val));
                }
            }
            if attrs.is_empty() {
                output.push_str(&format!("<{}>", tag));
            } else {
                output.push_str(&format!("<{} {}>", tag, attrs.join(" ")));
            }
        } else {
            output.push_str(&format!("<{}>", tag));
        }

        for child in element_ref.children() {
            match child.value() {
                Node::Element(_) => {
                    if let Some(child_ref) = scraper::ElementRef::wrap(child) {
                        self.process_node(child_ref, output);
                    }
                }
                Node::Text(text) => {
                    output.push_str(text);
                }
                _ => {}
            }
        }
        output.push_str(&format!("</{}>", tag));
    }
}

impl Harness for DefaultHarness {
    fn normalize(&self, raw_input: &str) -> Result<State, anyhow::Error> {
        let cleaned_html = self.clean_html(raw_input);
        
        let document = Html::parse_fragment(&cleaned_html);
        let input_selector = Selector::parse("input").unwrap();
        
        let mut components = Vec::new();
        for (i, element) in document.select(&input_selector).enumerate() {
            components.push(Component {
                id: format!("input_{}", i),
                component_type: "input".to_string(),
                status: "detected".to_string(),
                details: element.attr("value").map(|v| v.to_string()),
                reason: None,
            });
        }
        
        let mut entities = HashMap::new();
        entities.insert("page_root".to_string(), Entity {
            domain: Domain::Commerce, // Default
            info: Info {
                name: "root".to_string(),
                metadata: HashMap::new(),
            },
            status: "active".to_string(),
            components,
        });
        
        Ok(State { entities })
    }
}
