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

use crate::client::embedder_client::EmbedderClient;
use crate::client::llm_client::ask_llm;
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

const QDRANT_URL: &str = "http://localhost:6333";
const EMBED_MODEL: &str = "text-embedding-embeddinggemma-300m"; // must match what you used to index
const VECTOR_SIZE: usize = 768;
const COLLECTION: &str = "gobiz"; // or pass dynamically

pub async fn rag(query: &str, repo: &str) -> Result<Vec<QdrantPoint>> {
    let repo = repo.trim();

    // 1. embed query
    let embedder = EmbedderClient::new(EMBED_MODEL, Some(VECTOR_SIZE))?;
    let vec = embedder.embed_text(query.trim()).await?;

    // 2. search Qdrant
    let url = format!("{QDRANT_URL}/collections/{COLLECTION}/points/query");
    let body = if repo == "*" {
        json!({
            "query": {
                "recommend": {
                    "positive": [vec]
                }
            },
            "filter": {
                "must": {
                    "key": "repo",
                    "match": {
                        "value": repo
                    },
                }
            },
            "with_payload": ["code", "repo"],
            "limit": 3,
        })
    } else {
        json!({
            "query": {
                "recommend": {
                    "positive": [vec]
                }
            },
            "with_payload": ["code", "repo"],
            "limit": 3,
        })
    };

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

    // Get response
    let docs = result.result.points.into_iter().collect();

    // "Augment" response with natural language
    let prompt = format!(
        r#"so i embedded the following query to an embedding model:

{query}

and got the following result:

------------start result------------
{docs:#?}
------------end result------------

so... can you give me a response to my query?
    "#
    );
    ask_llm(
        "http://localhost:1234/v1/chat/completions",
        "lm-studio",            // LM Studio typically uses this as API key
        "qwen/qwen3-coder-30b", // Your model name
        &prompt,
    )
    .await;

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
    payload: Value,
}
