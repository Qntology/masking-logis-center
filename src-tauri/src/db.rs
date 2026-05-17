use lancedb::connect;
use lancedb::index::scalar::{FtsIndexBuilder, FullTextSearchQuery};
use lancedb::index::Index;
use lancedb::query::QueryBase;
use lancedb::query::ExecutableQuery;
use serde::{Deserialize, Serialize};
use lance_tokenizer::Language;

#[derive(Serialize, Deserialize, Debug)]
pub struct CommerceRecord {
    pub id: String,
    pub host: String,
    pub url: String,
    pub domain: String,
    pub context: String,
    pub status: String,
    pub track: String,
    pub version: i32,
    pub created_at: i64,
    pub updated_at: i64,
}
pub async fn get_or_create_table() -> Result<lancedb::Table, lancedb::Error> {
    let db = connect("data/commerce-db").execute().await?;

    match db.open_table("commerce_records").execute().await {
        Ok(table) => Ok(table),
        Err(_) => {
            // 빈 테이블을 명시적 스키마로 생성
            let table = db.create_table("commerce_records", arrow_array::RecordBatch::new_empty(
                std::sync::Arc::new(arrow_schema::Schema::new(vec![
                    arrow_schema::Field::new("id", arrow_schema::DataType::Utf8, false),
                    arrow_schema::Field::new("host", arrow_schema::DataType::Utf8, false),
                    arrow_schema::Field::new("url", arrow_schema::DataType::Utf8, false),
                    arrow_schema::Field::new("domain", arrow_schema::DataType::Utf8, false),
                    arrow_schema::Field::new("context", arrow_schema::DataType::Utf8, false),
                    arrow_schema::Field::new("status", arrow_schema::DataType::Utf8, false),
                    arrow_schema::Field::new("track", arrow_schema::DataType::Utf8, false),
                    arrow_schema::Field::new("version", arrow_schema::DataType::Int32, false),
                    arrow_schema::Field::new("created_at", arrow_schema::DataType::Int64, false),
                    arrow_schema::Field::new("updated_at", arrow_schema::DataType::Int64, false),
                ]))
            )).execute().await?;

            let fts_builder = FtsIndexBuilder::new("ngram".to_string(), Language::English)
                .ngram_min_length(2)
                .ngram_max_length(5)
                .ngram_prefix_only(false);

            table.create_index(&["context"], Index::FTS(fts_builder)).execute().await?;
            Ok(table)
        }
    }
}

pub async fn save_records(_records: Vec<CommerceRecord>) -> Result<(), lancedb::Error> {
    Ok(())
}

pub async fn search_context(query: &str) -> Result<Vec<CommerceRecord>, lancedb::Error> {
    let table = get_or_create_table().await?;
    let _results = table
        .query()
        .full_text_search(FullTextSearchQuery::new(query.to_string()))
        .execute()
        .await?;
    Ok(vec![])
}
