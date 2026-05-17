use lancedb::connect;
use lancedb::index::scalar::{FtsIndexBuilder, FullTextSearchQuery};
use lancedb::index::Index;
use lancedb::query::QueryBase;
use lancedb::query::ExecutableQuery;
use serde::{Deserialize, Serialize};
use lance_tokenizer::Language;
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use arrow::array::{ArrayRef, StringArray, Int32Array, Int64Array, Float32Array, Float32Builder, FixedSizeListArray, FixedSizeListBuilder};
use futures::StreamExt;
use std::sync::Arc;

#[derive(Serialize, Deserialize, Debug)]
pub struct CommerceRecord {
    pub id: String,
    pub host: String,
    pub url: String,
    pub title: String,
    pub domain: String,
    pub context: String,
    pub status: String,
    pub track: String,
    pub version: i32,
    pub created_at: i64,
    pub updated_at: i64,
    #[serde(default)]
    pub vector: Vec<f32>,
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
                Field::new("title", DataType::Utf8, false),
                Field::new("domain", DataType::Utf8, false),
                Field::new("context", DataType::Utf8, false),
                Field::new("status", DataType::Utf8, false),
                Field::new("track", DataType::Utf8, false),
                Field::new("version", DataType::Int32, false),
                Field::new("created_at", DataType::Int64, false),
                Field::new("updated_at", DataType::Int64, false),
                Field::new("vector", DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), 768), true),
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

pub async fn save_records(records: Vec<CommerceRecord>) -> Result<(), lancedb::Error> {
    let table = get_or_create_table().await?;
    
    // 동일한 ID를 가진 DRAFT 상태의 레코드가 있다면 삭제하여 덮어쓰기(Upsert) 환경 구성
    for record in &records {
        if record.status == "DRAFT" {
            let expr = format!("id = '{}' AND status = 'DRAFT'", record.id);
            let _ = table.delete(&expr).await;
        }
    }

    if records.is_empty() {
        return Ok(());
    }

    // Arrow Array로 변환하여 실질적 DB 삽입 준비 (title 포함)
    let id_array = Arc::new(StringArray::from(records.iter().map(|r| r.id.as_str()).collect::<Vec<&str>>())) as ArrayRef;
    let host_array = Arc::new(StringArray::from(records.iter().map(|r| r.host.as_str()).collect::<Vec<&str>>())) as ArrayRef;
    let url_array = Arc::new(StringArray::from(records.iter().map(|r| r.url.as_str()).collect::<Vec<&str>>())) as ArrayRef;
    let title_array = Arc::new(StringArray::from(records.iter().map(|r| r.title.as_str()).collect::<Vec<&str>>())) as ArrayRef;
    let domain_array = Arc::new(StringArray::from(records.iter().map(|r| r.domain.as_str()).collect::<Vec<&str>>())) as ArrayRef;
    let context_array = Arc::new(StringArray::from(records.iter().map(|r| r.context.as_str()).collect::<Vec<&str>>())) as ArrayRef;
    let status_array = Arc::new(StringArray::from(records.iter().map(|r| r.status.as_str()).collect::<Vec<&str>>())) as ArrayRef;
    let track_array = Arc::new(StringArray::from(records.iter().map(|r| r.track.as_str()).collect::<Vec<&str>>())) as ArrayRef;
    let version_array = Arc::new(Int32Array::from(records.iter().map(|r| r.version).collect::<Vec<i32>>())) as ArrayRef;
    let created_at_array = Arc::new(Int64Array::from(records.iter().map(|r| r.created_at).collect::<Vec<i64>>())) as ArrayRef;
    let updated_at_array = Arc::new(Int64Array::from(records.iter().map(|r| r.updated_at).collect::<Vec<i64>>())) as ArrayRef;

    // 768 차원의 벡터 배열을 생성합니다. 빈 벡터일 경우 0.0으로 채웁니다.
    let mut vector_builder = FixedSizeListBuilder::new(Float32Builder::new(), 768);
    for record in &records {
        let vec_data = if record.vector.len() == 768 {
            record.vector.clone()
        } else {
            vec![0.0; 768]
        };
        vector_builder.values().append_slice(&vec_data);
        vector_builder.append(true);
    }
    let vector_array = Arc::new(vector_builder.finish()) as ArrayRef;

    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Utf8, false),
        Field::new("host", DataType::Utf8, false),
        Field::new("url", DataType::Utf8, false),
        Field::new("title", DataType::Utf8, false),
        Field::new("domain", DataType::Utf8, false),
        Field::new("context", DataType::Utf8, false),
        Field::new("status", DataType::Utf8, false),
        Field::new("track", DataType::Utf8, false),
        Field::new("version", DataType::Int32, false),
        Field::new("created_at", DataType::Int64, false),
        Field::new("updated_at", DataType::Int64, false),
        Field::new("vector", DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), 768), true),
    ]));

    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![id_array, host_array, url_array, title_array, domain_array, context_array, status_array, track_array, version_array, created_at_array, updated_at_array, vector_array]
    ).map_err(|e| lancedb::Error::Runtime { message: e.to_string() })?;

    // RecordBatch 인스턴스를 단일 요소 Vec으로 감싸서 직접 전달하여 Scannable 트레이트 조건을 만족시킵니다.
    table.add(vec![batch]).execute().await?;
    
    println!("[LanceDB] 성공적으로 저장되었습니다! 저장된 데이터 수: {}", records.len());
    
    Ok(())
}

pub async fn fetch_drafts() -> Result<Vec<CommerceRecord>, lancedb::Error> {
    let table = get_or_create_table().await?;
    
    // lancedb 0.29.0 버전에 맞게 조건부 필터링 메서드를 only_if로 변경합니다.
    let mut stream = table.query().only_if("status = 'DRAFT'").execute().await?;
    let mut results = Vec::new();

    while let Some(batch_result) = stream.next().await {
        let batch = batch_result.map_err(|e| lancedb::Error::Runtime { message: e.to_string() })?;
        
        let ids = batch.column(0).as_any().downcast_ref::<StringArray>().unwrap();
        let hosts = batch.column(1).as_any().downcast_ref::<StringArray>().unwrap();
        let urls = batch.column(2).as_any().downcast_ref::<StringArray>().unwrap();
        let titles = batch.column(3).as_any().downcast_ref::<StringArray>().unwrap();
        let domains = batch.column(4).as_any().downcast_ref::<StringArray>().unwrap();
        let contexts = batch.column(5).as_any().downcast_ref::<StringArray>().unwrap();
        let statuses = batch.column(6).as_any().downcast_ref::<StringArray>().unwrap();
        let tracks = batch.column(7).as_any().downcast_ref::<StringArray>().unwrap();
        let versions = batch.column(8).as_any().downcast_ref::<Int32Array>().unwrap();
        let created_ats = batch.column(9).as_any().downcast_ref::<Int64Array>().unwrap();
        let updated_ats = batch.column(10).as_any().downcast_ref::<Int64Array>().unwrap();

        // 벡터 데이터 추출 (인덱스 11번)
        let vectors_list = batch.column(11).as_any().downcast_ref::<FixedSizeListArray>().unwrap();
        let vectors_values = vectors_list.values().as_any().downcast_ref::<Float32Array>().unwrap();

        for i in 0..batch.num_rows() {
            // 각 행(row)에 해당하는 768 차원 벡터 추출
            let start_idx = i * 768;
            let end_idx = start_idx + 768;
            let mut vector_data = Vec::with_capacity(768);
            for j in start_idx..end_idx {
                vector_data.push(vectors_values.value(j));
            }

            results.push(CommerceRecord {
                id: ids.value(i).to_string(),
                host: hosts.value(i).to_string(),
                url: urls.value(i).to_string(),
                title: titles.value(i).to_string(),
                domain: domains.value(i).to_string(),
                context: contexts.value(i).to_string(),
                status: statuses.value(i).to_string(),
                track: tracks.value(i).to_string(),
                version: versions.value(i),
                created_at: created_ats.value(i),
                updated_at: updated_ats.value(i),
                vector: vector_data,
            });
        }
    }
    
    Ok(results)
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
