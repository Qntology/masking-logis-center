use lancedb::connect;
use lancedb::index::scalar::{FtsIndexBuilder, FullTextSearchQuery};
use lancedb::index::Index;
use lancedb::query::QueryBase;
use lancedb::query::ExecutableQuery;
use serde::{Deserialize, Serialize};
use lance_tokenizer::Language;
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use std::sync::Arc;

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

#[allow(dead_code)]
pub async fn get_or_create_table() -> Result<lancedb::Table, lancedb::Error> {
    let db = connect("data/commerce-db").execute().await?;

    match db.open_table("commerce_records").execute().await {
        Ok(table) => Ok(table),
        Err(_) => {
            let schema = Arc::new(Schema::new(vec![
                Field::new("id", DataType::Utf8, false),
                Field::new("host", DataType::Utf8, false),
                Field::new("url", DataType::Utf8, false),
                Field::new("domain", DataType::Utf8, false),
                Field::new("context", DataType::Utf8, false),
                Field::new("status", DataType::Utf8, false),
                Field::new("track", DataType::Utf8, false),
                Field::new("version", DataType::Int32, false),
                Field::new("created_at", DataType::Int64, false),
                Field::new("updated_at", DataType::Int64, false),
            ]));
            let empty_batch = RecordBatch::new_empty(schema);
            let table = db.create_table("commerce_records", empty_batch).execute().await?;

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

#[allow(dead_code)]
pub async fn search_context(query: &str) -> Result<Vec<CommerceRecord>, lancedb::Error> {
    let table = get_or_create_table().await?;
    let _results = table
        .query()
        .full_text_search(FullTextSearchQuery::new(query.to_string()))
        .execute()
        .await?;
    Ok(vec![])
}
