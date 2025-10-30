//! embedder_client.rs
//!
//! Minimal, robust client for calling an OpenAI-compatible embeddings endpoint
//! (e.g., LM Studio running locally). Provides single and batch embedding.
//!
//! Assumptions:
//! - Server endpoint is hardcoded to localhost (LM Studio style).
//! - Model name is configurable.
//! - Single-threaded; no parallelization.
//! - One embedding per document (no chunking).
//!
//! Example:
//! ```ignore
//! use crate::embedder_client::EmbedderClient;
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!   let client = EmbedderClient::new("text-embedding-3-small", None)?;
//!   let vec = client.embed_text("hello world").await?;
//!   println!("dim = {}", vec.len());
//!
//!   Ok(())
//! }
//! ```

use std::time::Duration;

use reqwest::StatusCode;
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Default base URL for LM Studio (OpenAI-compatible) embeddings API.
const DEFAULT_EMBED_BASE_URL: &str = "http://localhost:1234/v1";
/// Default per-request timeout.
const DEFAULT_TIMEOUT_SECS: u64 = 60;

/// High-level client for embedding text via an OpenAI-compatible server.
pub struct EmbedderClient {
    http: reqwest::Client,
    /// Base URL to the API (e.g., http://localhost:1234/v1).
    base_url: String,
    /// Embedding model name (e.g., "text-embedding-3-small").
    model: String,
    /// Optional expected vector dimension; if set, responses are validated.
    expected_dim: Option<usize>,
}

#[derive(Error, Debug)]
pub enum EmbedError {
    #[error("http error: {0}")]
    Http(#[from] reqwest::Error),

    #[error("server returned {status}: {body}")]
    Status { status: StatusCode, body: String },

    #[error("empty embedding response")]
    EmptyResponse,

    #[error("embedding count mismatch: sent {sent}, got {got}")]
    CountMismatch { sent: usize, got: usize },

    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimMismatch { expected: usize, got: usize },

    #[error("serialization error: {0}")]
    Serde(#[from] serde_json::Error),
}

impl EmbedderClient {
    /// Create a new client.
    ///
    /// - `model`: embedding model name (configurable).
    /// - `expected_dim`: if Some(d), will validate that returned vectors have length `d`.
    /// - Endpoint is hardcoded to `DEFAULT_EMBED_BASE_URL` as requested.
    pub fn new<S: Into<String>>(model: S, expected_dim: Option<usize>) -> Result<Self, EmbedError> {
        let http = reqwest::Client::builder()
            .timeout(Duration::from_secs(DEFAULT_TIMEOUT_SECS))
            .build()?;

        Ok(Self {
            http,
            base_url: DEFAULT_EMBED_BASE_URL.to_string(),
            model: model.into(),
            expected_dim,
        })
    }

    /// Override the timeout
    pub fn with_timeout_secs(mut self, secs: u64) -> Result<Self, EmbedError> {
        self.http = reqwest::Client::builder()
            .timeout(Duration::from_secs(secs))
            .build()?;
        Ok(self)
    }

    /// Returns the embeddings URL (POST).
    fn embeddings_url(&self) -> String {
        format!("{}/embeddings", self.base_url)
    }

    /// Embed a single text string.
    pub async fn embed_text(&self, text: &str) -> Result<Vec<f32>, EmbedError> {
        let mut out = self.embed_texts_raw(&[text]).await?;
        let first = out.pop().ok_or(EmbedError::EmptyResponse)?.embedding;

        if let Some(expected) = self.expected_dim {
            if first.len() != expected {
                return Err(EmbedError::DimMismatch {
                    expected,
                    got: first.len(),
                });
            }
        }
        Ok(first)
    }

    /// Embed multiple texts in a single request.
    ///
    /// Returns embeddings in the same order as the inputs.
    pub async fn embed_texts<T: AsRef<str>>(
        &self,
        texts: &[T],
    ) -> Result<Vec<Vec<f32>>, EmbedError> {
        let n = texts.len();
        if n == 0 {
            return Ok(Vec::new());
        }

        let data = self.embed_texts_raw(texts).await?;
        if data.len() != n {
            return Err(EmbedError::CountMismatch {
                sent: n,
                got: data.len(),
            });
        }

        // Sort by index to preserve order (some servers already do this).
        let mut pairs: Vec<(usize, Vec<f32>)> = data
            .into_iter()
            .map(|d| {
                let idx = d.index.unwrap_or(0);
                let emb = d.embedding;
                (idx, emb)
            })
            .collect();
        pairs.sort_by_key(|(i, _)| *i);

        let mut result = Vec::with_capacity(n);
        for (_, emb) in pairs {
            if let Some(expected) = self.expected_dim {
                if emb.len() != expected {
                    return Err(EmbedError::DimMismatch {
                        expected,
                        got: emb.len(),
                    });
                }
            }
            result.push(emb);
        }
        Ok(result)
    }

    /// Low-level call that performs the HTTP request and returns raw data entries.
    async fn embed_texts_raw<T: AsRef<str>>(
        &self,
        texts: &[T],
    ) -> Result<Vec<EmbeddingDatum>, EmbedError> {
        // Build input as array to leverage batch endpoint behavior
        let input: Vec<String> = texts.iter().map(|t| t.as_ref().to_string()).collect();

        let req = EmbeddingsRequest {
            model: self.model.clone(),
            input: EmbeddingInput::Array(input),
            // Encoding format defaults to floats for embeddings; keep explicit if needed:
            // encoding_format: Some("float".to_string()),
        };

        let url = self.embeddings_url();
        let resp = self.http.post(&url).json(&req).send().await?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(EmbedError::Status { status, body });
        }

        let parsed: EmbeddingsResponse = resp.json().await?;
        if parsed.data.is_empty() {
            return Err(EmbedError::EmptyResponse);
        }
        Ok(parsed.data)
    }
}

/// Request payload compatible with OpenAI-style embeddings API.
/// LM Studio mirrors this shape.
#[derive(Debug, Serialize)]
struct EmbeddingsRequest {
    model: String,
    // #[serde(flatten)]
    input: EmbeddingInput,
    // Uncomment if your server requires it:
    // encoding_format: Option<String>,
}

#[derive(Debug, Serialize)]
#[serde(untagged)]
enum EmbeddingInput {
    Single { input: String },
    Array(Vec<String>),
}

/// Response payload (subset) for embeddings.
#[derive(Debug, Deserialize)]
struct EmbeddingsResponse {
    data: Vec<EmbeddingDatum>,
    // model, usage, etc. are omitted but can be added if needed
}

#[derive(Debug, Deserialize)]
struct EmbeddingDatum {
    #[serde(default)]
    index: Option<usize>,
    #[serde(default)]
    embedding: Vec<f32>,
}
