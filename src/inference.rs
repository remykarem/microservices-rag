//! inference.rs
//!
//! Performs semantic search over your Qdrant collection.
//!
//! fn inference(query: &str, k: u8) -> Vec<Document>
//!   1. Embeds the query text via your embedding server
//!   2. Queries Qdrant's /points/search endpoint
//!   3. Returns the top-k payloads decoded as Documents
//!
//! Assumes:
//! - Same model + vector size as your indexer
//! - `Document` is identical to what you indexed

use crate::ingest::rust_parser::Document;
use crate::transform::embedder_client::EmbedderClient;
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::json;

const QDRANT_URL: &str = "http://localhost:6333";
const EMBED_MODEL: &str = "text-embedding-embeddinggemma-300m"; // must match what you used to index
const VECTOR_SIZE: usize = 768;
const COLLECTION: &str = "thing"; // or pass dynamically

pub async fn inference(query: &str, k: u8) -> Result<Vec<QdrantPoint>> {
    // 1. embed query
    let embedder = EmbedderClient::new(EMBED_MODEL, Some(VECTOR_SIZE))?;
    let vec = embedder.embed_text(query).await?;

    // 2. search Qdrant
    let url = format!("{QDRANT_URL}/collections/{COLLECTION}/points/query");
    // let body = QdrantSearchRequest {
    //     vector: vec,
    //     limit: k as usize,
    // };
    let body = json!(
    { "query": { "recommend":{
                "positive": [vec]
            }
        }
    });

    let resp = reqwest::Client::new()
        .post(&url)
        .json(&body)
        .send()
        .await
        .context("failed to send Qdrant search request")?;

    if !resp.status().is_success() {
        let status = resp.status();
        let text = resp.text().await.unwrap_or_default();
        anyhow::bail!("Qdrant search failed: {status} {text}");
    }

    let result: QdrantSearchResponse = resp.json().await?;
    let docs = result
        .result
        .points
        .into_iter()
        .collect();

    Ok(docs)
}

#[derive(Debug, Serialize)]
struct QdrantSearchRequest {
    vector: Vec<f32>,
    limit: usize,
}

#[derive(Debug, Deserialize)]
struct QdrantSearchResponse {
    result: QdrantPoints,
}

#[derive(Debug, Deserialize)]
struct QdrantPoints {
    points: Vec<QdrantPoint>,
}

#[derive(Debug, Deserialize)]
pub struct QdrantPoint {
    id: String,
    score: f32,
}
