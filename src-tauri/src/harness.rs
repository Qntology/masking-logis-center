use crate::model::{State, Entity, Info, Component};
use scraper::{Html, Selector, Node, ElementRef};
use std::collections::HashMap;

pub trait Harness {
    fn normalize(&self, raw_input: &str) -> Result<State, anyhow::Error>;
}

pub struct DefaultHarness;

impl DefaultHarness {
    fn clean_html(&self, html: &str) -> String {
        let document = Html::parse_document(html);
        let mut cleaned_html = String::new();

        self.process_node(&document.tree, document.root_element(), &mut cleaned_html);
        cleaned_html
    }

    fn process_node(&self, tree: &scraper::node::Arena<scraper::Node>, node_id: scraper::node::NodeId, output: &mut String) {
        let node = &tree[node_id];
        match node {
            scraper::Node::Element(element) => {
                let tag = element.name();
                
                // Skip forbidden tags
                if matches!(tag, "script" | "style" | "link" | "noscript" | "iframe" | "svg" | "meta" | "br" | "hr" | "source") {
                    return;
                }

                // Handle <select> logic
                if tag == "select" {
                    output.push_str("<select>");
                    for child in tree.children(node_id) {
                        let child_node = &tree[child];
                        if let Node::Element(child_el) = child_node {
                            if child_el.name() == "option" {
                                let is_selected = child_el.attr("selected").is_some();
                                if is_selected {
                                    output.push_str("<option selected>");
                                    self.process_children(tree, child, output);
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
                        if element.attr(attr).is_some() {
                            attrs.push(format!("{}=\"{}\"", attr, element.attr(attr).unwrap()));
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

                self.process_children(tree, node_id, output);
                output.push_str(&format!("</{}>", tag));
            }
            scraper::Node::Text(text) => {
                output.push_str(text);
            }
            _ => {}
        }
    }

    fn process_children(&self, tree: &scraper::node::Arena<scraper::Node>, node_id: scraper::node::NodeId, output: &mut String) {
        for child in tree.children(node_id) {
            self.process_node(tree, child, output);
        }
    }
}

impl Harness for DefaultHarness {
    fn normalize(&self, raw_input: &str) -> Result<State, anyhow::Error> {
        let cleaned_html = self.clean_html(raw_input);
        
        // Example: Map inputs to components
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
