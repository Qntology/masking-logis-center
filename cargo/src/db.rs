use lancedb::connect;
use lancedb::index::scalar::{FtsIndexBuilder, FullTextSearchQuery};
use lancedb::index::Index;
use lancedb::query::{ExecutableQuery, QueryBase};
use serde::{Deserialize, Serialize};
use lance_tokenizer::Language;
use arrow::datatypes::{DataType, Field, Schema, Int32Type, Int64Type, Float32Type};
use arrow::record_batch::RecordBatch;
use arrow::array::{ArrayRef, StringArray, Int32Array, Int64Array, Float32Builder, FixedSizeListBuilder, AsArray};
use std::sync::Arc;
use futures::StreamExt;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct CommerceRecord {
    pub id: String,
    pub host: String,
    pub url: String,
    pub title: String,
    pub domain: String,
    pub context: String,
    #[serde(default)]
    pub masking: String,
    pub status: String,
    pub track: String,
    pub version: i32,
    pub created_at: i64,
    pub updated_at: i64,
    #[serde(default)]
    pub vector: Vec<f32>,
}

pub async fn get_or_create_table() -> Result<lancedb::Table, lancedb::Error> {
    // LanceDB 저장 폴더가 없으면 os error 3이 발생할 수 있으므로 상위 디렉토리를 미리 생성해줍니다.
    let _ = std::fs::create_dir_all("data/db");
    let db = connect("data/db").execute().await?;
    let table_name = "terminal"; // Use v3 to ensure schema compatibility

    match db.open_table(table_name).execute().await {
        Ok(table) => Ok(table),
        Err(_) => {
            let schema = Arc::new(Schema::new(vec![
                Field::new("id", DataType::Utf8, false),
                Field::new("host", DataType::Utf8, false),
                Field::new("url", DataType::Utf8, false),
                Field::new("title", DataType::Utf8, false),
                Field::new("domain", DataType::Utf8, false),
                Field::new("context", DataType::Utf8, false),
                Field::new("masking", DataType::Utf8, false),
                Field::new("status", DataType::Utf8, false),
                Field::new("track", DataType::Utf8, false),
                Field::new("version", DataType::Int32, false),
                Field::new("created_at", DataType::Int64, false),
                Field::new("updated_at", DataType::Int64, false),
                Field::new("vector", DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), 768), true),
            ]));
            let empty_batch = RecordBatch::new_empty(schema);
            let table = db.create_table(table_name, empty_batch).execute().await?;

            let fts_builder = FtsIndexBuilder::new("ngram".to_string(), Language::English)
                .ngram_min_length(2)
                .ngram_max_length(5)
                .ngram_prefix_only(false);

            table.create_index(&["masking"], Index::FTS(fts_builder)).execute().await?;
            Ok(table)
        }
    }
}

pub async fn save_records(records: Vec<CommerceRecord>, categorizer: Option<&crate::categorizer::Categorizer>) -> Result<(), lancedb::Error> {
    let table = get_or_create_table().await?;

    let mut records = records;
    for record in &mut records {
        // DRAFT 상태이거나 아직 마스킹 로직을 타지 않아 비어있는 경우 context 텍스트를 기본값으로 채워줍니다.
        // (실제 정규 마스킹 처리는 push_data 시점 백엔드에서 수행됨)
        if record.masking.is_empty() {
            record.masking = record.context.clone();
        }
        
        // Domain Categorization
        if let Some(cat) = categorizer {
            if let Ok(json_res) = cat.preprocess_web(&record.context).await {
                if let Some(domain_str) = json_res.get("domain").and_then(|v| v.as_str()) {
                    record.domain = domain_str.to_uppercase();
                }
            }
        }
    }
    
    // 상태와 무관하게 동일한 ID가 존재하면 덮어쓰기 위해 선제 삭제합니다.
    for record in &records {
        let expr = format!("id = '{}'", record.id);
        let _ = table.delete(&expr).await;
    }

    if records.is_empty() {
        return Ok(());
    }

    let id_array = Arc::new(StringArray::from(records.iter().map(|r| r.id.as_str()).collect::<Vec<&str>>())) as ArrayRef;
    let host_array = Arc::new(StringArray::from(records.iter().map(|r| r.host.as_str()).collect::<Vec<&str>>())) as ArrayRef;
    let url_array = Arc::new(StringArray::from(records.iter().map(|r| r.url.as_str()).collect::<Vec<&str>>())) as ArrayRef;
    let title_array = Arc::new(StringArray::from(records.iter().map(|r| r.title.as_str()).collect::<Vec<&str>>())) as ArrayRef;
    let domain_array = Arc::new(StringArray::from(records.iter().map(|r| r.domain.as_str()).collect::<Vec<&str>>())) as ArrayRef;
    let context_array = Arc::new(StringArray::from(records.iter().map(|r| r.context.as_str()).collect::<Vec<&str>>())) as ArrayRef;
    let masking_array = Arc::new(StringArray::from(records.iter().map(|r| r.masking.as_str()).collect::<Vec<&str>>())) as ArrayRef;
    let status_array = Arc::new(StringArray::from(records.iter().map(|r| r.status.as_str()).collect::<Vec<&str>>())) as ArrayRef;
    let track_array = Arc::new(StringArray::from(records.iter().map(|r| r.track.as_str()).collect::<Vec<&str>>())) as ArrayRef;
    let version_array = Arc::new(Int32Array::from(records.iter().map(|r| r.version).collect::<Vec<i32>>())) as ArrayRef;
    let created_at_array = Arc::new(Int64Array::from(records.iter().map(|r| r.created_at).collect::<Vec<i64>>())) as ArrayRef;
    let updated_at_array = Arc::new(Int64Array::from(records.iter().map(|r| r.updated_at).collect::<Vec<i64>>())) as ArrayRef;

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
        Field::new("masking", DataType::Utf8, false),
        Field::new("status", DataType::Utf8, false),
        Field::new("track", DataType::Utf8, false),
        Field::new("version", DataType::Int32, false),
        Field::new("created_at", DataType::Int64, false),
        Field::new("updated_at", DataType::Int64, false),
        Field::new("vector", DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), 768), true),
    ]));

    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![id_array, host_array, url_array, title_array, domain_array, context_array, masking_array, status_array, track_array, version_array, created_at_array, updated_at_array, vector_array]
    ).map_err(|e| lancedb::Error::Runtime { message: e.to_string() })?;

    table.add(vec![batch]).execute().await?;
    
    println!("[LanceDB] 성공적으로 저장되었습니다! 저장된 데이터 수: {}", records.len());
    
    Ok(())
}

pub async fn fetch_drafts() -> Result<Vec<CommerceRecord>, lancedb::Error> {
    let table = get_or_create_table().await?;
    // PUSHED 상태의 아이템도 UI 목록에 유지하기 위해 함께 불러옵니다.
    let mut stream = table.query().only_if("status IN ('DRAFT', 'PUSHED')").execute().await?;
    let mut results = Vec::new();

    while let Some(batch_result) = stream.next().await {
        let batch = batch_result.map_err(|e| lancedb::Error::Runtime { message: e.to_string() })?;
        extract_from_batch(&batch, &mut results)?;
    }
    
    Ok(results)
}

#[allow(dead_code)]
pub async fn search_context(query: &str, domain_filter: Option<&str>) -> Result<Vec<CommerceRecord>, lancedb::Error> {
    let table = get_or_create_table().await?;
    let mut search_query = table.query()
        .full_text_search(FullTextSearchQuery::new(query.to_string()));
    
    if let Some(domain) = domain_filter {
        let filter = format!("domain = '{}'", domain.to_uppercase());
        search_query = search_query.only_if(filter);
    }
    
    let mut stream = search_query.execute().await?;
    let mut results = Vec::new();

    while let Some(batch_result) = stream.next().await {
        let batch = batch_result.map_err(|e| lancedb::Error::Runtime { message: e.to_string() })?;
        extract_from_batch(&batch, &mut results)?;
    }
    
    Ok(results)
}

fn extract_from_batch(batch: &RecordBatch, results: &mut Vec<CommerceRecord>) -> Result<(), lancedb::Error> {
    let ids = batch.column(0).as_string::<i32>();
    let hosts = batch.column(1).as_string::<i32>();
    let urls = batch.column(2).as_string::<i32>();
    let titles = batch.column(3).as_string::<i32>();
    let domains = batch.column(4).as_string::<i32>();
    let contexts = batch.column(5).as_string::<i32>();
    let maskings = batch.column(6).as_string::<i32>();
    let statuses = batch.column(7).as_string::<i32>();
    let tracks = batch.column(8).as_string::<i32>();
    let versions = batch.column(9).as_primitive::<Int32Type>();
    let created_ats = batch.column(10).as_primitive::<Int64Type>();
    let updated_ats = batch.column(11).as_primitive::<Int64Type>();
    let vectors_list = batch.column(12).as_fixed_size_list();
    let vectors_values = vectors_list.values().as_primitive::<Float32Type>();

    for i in 0..batch.num_rows() {
        let mut vector_data = Vec::with_capacity(768);
        for j in 0..768 {
            vector_data.push(vectors_values.value(i * 768 + j));
        }

        results.push(CommerceRecord {
            id: ids.value(i).to_string(),
            host: hosts.value(i).to_string(),
            url: urls.value(i).to_string(),
            title: titles.value(i).to_string(),
            domain: domains.value(i).to_string(),
            context: contexts.value(i).to_string(),
            masking: maskings.value(i).to_string(),
            status: statuses.value(i).to_string(),
            track: tracks.value(i).to_string(),
            version: versions.value(i),
            created_at: created_ats.value(i),
            updated_at: updated_ats.value(i),
            vector: vector_data,
        });
    }
    Ok(())
}
