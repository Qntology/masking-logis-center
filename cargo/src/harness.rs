use crate::model::{State, Entity, Info, Component};
use crate::domain::Domain;
use scraper::{Html, Selector, Node};
use std::collections::HashMap;

pub trait Harness {
    fn normalize(&self, raw_input: &str) -> Result<State, anyhow::Error>;
}

pub struct DefaultHarness;

impl DefaultHarness {
    /// 🚀 모든 태그와 속성을 제거하고 중첩을 평탄화하여 순수 텍스트만 추출합니다.
    pub fn clean_html(&self, html: &str) -> String {
        // 🚀 만약 입력값이 이미 태그가 없는 순수 텍스트라면 파싱 과정을 건너뛰고 그대로 반환합니다.
        if !html.contains('<') || !html.contains('>') {
            return html.trim().to_string();
        }

        let document = Html::parse_document(html);
        let mut cleaned_text = String::new();

        // 루트 요소부터 재귀적으로 텍스트만 수집합니다.
        self.process_node_as_pug(document.root_element().clone(), &mut cleaned_text);
        
        // 결과가 비어있을 경우 원본에서 태그만 강제로 제거한 값을 반환하는 폴백 로직을 적용합니다.
        if cleaned_text.trim().is_empty() {
            return html.split('<')
                .filter_map(|s| s.split_once('>').map(|(_, t)| t.trim()))
                .filter(|t| !t.is_empty())
                .collect::<Vec<_>>()
                .join("\n");
        }

        // 연속된 줄바꿈이나 공백을 정리하여 결과물을 깔끔하게 만듭니다.
        cleaned_text.lines()
            .map(|line| line.trim())
            .filter(|line| !line.is_empty())
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// 🚀 태그 중첩을 무시하고 오직 의미 있는 내용만 추출하는 PUG 스타일 재귀 함수입니다.
    fn process_node_as_pug(&self, element_ref: scraper::ElementRef, output: &mut String) {
        let element = element_ref.value();
        let tag = element.name();
        
        // 1. 데이터적 가치가 없는 메타/리소스 태그들은 자식 노드까지 완전히 무시합니다.
        // 🚀 [버그 픽스] header, footer, nav는 실제 텍스트가 많이 포함되므로 무시 목록에서 제거하여 누락을 방지합니다.
        if matches!(tag, "script" | "style" | "link" | "noscript" | "iframe" | "svg" | "meta" | "head" | "path") {
            return;
        }

        // 2. 특정 요소(input, select, img)의 속성 및 값 추출 로직
        if tag == "input" {
            if let Some(val) = element.attr("value") {
                output.push_str(val);
                output.push('\n');
            }
            return;
        }

        if tag == "select" {
            for child in element_ref.children() {
                if let Node::Element(child_el) = child.value() {
                    // 선택된 옵션의 텍스트만 추출
                    if child_el.name() == "option" && child_el.attr("selected").is_some() {
                        if let Some(child_ref) = scraper::ElementRef::wrap(child) {
                            output.push_str(&child_ref.text().collect::<Vec<_>>().concat());
                            output.push('\n');
                        }
                    }
                }
            }
            return;
        }

        // 🚀 img 태그 처리: src 값을 추출하되 base64(data:) 형식은 제외
        if tag == "img" {
            if let Some(src) = element.attr("src") {
                if !src.starts_with("data:") {
                    output.push_str("[Image: ");
                    output.push_str(src);
                    output.push_str("]\n");
                }
            }
            return;
        }

        // 3. 일반적인 태그(div, span, p 등)는 무시하고 자식 노드로 파고들어 텍스트를 찾습니다.
        for child in element_ref.children() {
            match child.value() {
                Node::Element(_) => {
                    if let Some(child_ref) = scraper::ElementRef::wrap(child) {
                        self.process_node_as_pug(child_ref, output);
                    }
                }
                Node::Text(text) => {
                    // 순수 텍스트 노드인 경우 내용만 추가합니다.
                    let t = text.trim();
                    if !t.is_empty() {
                        output.push_str(t);
                        output.push('\n'); // 평탄화를 위해 줄바꿈 삽입
                    }
                }
                _ => {}
            }
        }
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
