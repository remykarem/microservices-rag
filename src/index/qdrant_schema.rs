//! qdrant_schema.rs
//!
//! Minimal schema manager for Qdrant collections:
//! - ensure_collection(): create if missing, else validate vector params
//!
//! Assumes Qdrant is reachable at a base URL.

use std::cmp::PartialEq;
use std::time::Duration;

use reqwest::StatusCode;
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum Distance {
    Cosine,
    Dot,
    Euclid,
}

#[derive(Error, Debug)]
pub enum SchemaError {
    #[error("http: {0}")]
    Http(#[from] reqwest::Error),

    #[error("server returned {status}: {body}")]
    Status { status: StatusCode, body: String },

    #[error("collection '{0}' exists with incompatible vector params (size/distance)")]
    IncompatibleCollection(String),

    #[error("serde: {0}")]
    Serde(#[from] serde_json::Error),
}

#[derive(Clone)]
pub struct QdrantSchema {
    http: reqwest::Client,
    base_url: String,
}

impl QdrantSchema {
    pub fn new<S: Into<String>>(base_url: S) -> Result<Self, SchemaError> {
        let http = reqwest::Client::builder()
            .timeout(Duration::from_secs(30))
            .build()?;
        Ok(Self {
            http,
            base_url: base_url.into(),
        })
    }

    fn collections_url(&self) -> String {
        format!("{}/collections", self.base_url)
    }

    fn collection_url(&self, name: &str) -> String {
        format!("{}/collections/{}", self.base_url, name)
    }

    /// Ensure a collection exists with the desired vectors config.
    /// - Create if missing
    /// - Validate (size, distance) if exists
    pub async fn ensure_collection(
        &self,
        name: &str,
        vector_size: usize,
        distance: Distance,
    ) -> Result<(), SchemaError> {
        match self.get_collection(name).await {
            Ok(Some(info)) => {
                // Validate vector params
                if !info.matches(vector_size, distance) {
                    return Err(SchemaError::IncompatibleCollection(name.to_string()));
                }
                Ok(())
            }
            Ok(None) => {
                // Create collection
                self.create_collection(name, vector_size, distance).await
            }
            Err(err) => Err(err),
        }
    }

    async fn get_collection(&self, name: &str) -> Result<Option<CollectionInfo>, SchemaError> {
        let url = self.collection_url(name);
        let resp = self.http.get(url).send().await?;

        match resp.status() {
            s if s.is_success() => {
                let body: GetCollectionResponse = resp.json().await?;
                Ok(Some(CollectionInfo::from_get_response(body)))
            }
            StatusCode::NOT_FOUND => Ok(None),
            status => {
                let body = resp.text().await.unwrap_or_default();
                Err(SchemaError::Status { status, body })
            }
        }
    }

    async fn create_collection(
        &self,
        name: &str,
        vector_size: usize,
        distance: Distance,
    ) -> Result<(), SchemaError> {
        let url = self.collection_url(name);
        let body = CreateCollectionRequest {
            vectors: VectorsConfig::Single(VectorParams {
                size: vector_size,
                distance,
            }),
        };
        println!("{:?}", serde_json::to_string(&body)?);

        let resp = self.http.put(url).json(&body).send().await?;
        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(SchemaError::Status {
                status,
                body: text,
            });
        }
        Ok(())
    }
}

/// ---
/// Minimal request/response models (subset of Qdrant API)
/// ---

#[derive(Debug, Serialize)]
struct CreateCollectionRequest {
    vectors: VectorsConfig,
}

#[derive(Debug, Serialize)]
#[serde(untagged)]
enum VectorsConfig {
    Single(VectorParams),
    // Named(HashMap<String, VectorParams>),
}

#[derive(Debug, Serialize, Deserialize, Clone)]
struct VectorParams {
    size: usize,
    distance: Distance,
}

#[derive(Debug, Deserialize)]
struct GetCollectionResponse {
    // status/ time omitted
    result: Option<CollectionResult>,
}

#[derive(Debug, Deserialize)]
struct CollectionResult {
    config: Option<CollectionConfig>,
}

#[derive(Debug, Deserialize)]
struct CollectionConfig {
    params: Option<CollectionParams>,
}

#[derive(Debug, Deserialize)]
struct CollectionParams {
    vectors: Option<VectorsConfigRead>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum VectorsConfigRead {
    Single(VectorParams),
    // Named(HashMap<String, VectorParams>),
}

/// Extracted/normalized info used for validation.
struct CollectionInfo {
    vectors: Option<VectorParams>,
}

impl CollectionInfo {
    fn from_get_response(resp: GetCollectionResponse) -> Self {
        let vectors = resp
            .result
            .and_then(|r| r.config)
            .and_then(|c| c.params)
            .and_then(|p| match p.vectors {
                Some(VectorsConfigRead::Single(v)) => Some(v),
                _ => None, // treat multi-vector as incompatible for now
            });

        Self { vectors }
    }

    fn matches(&self, size: usize, distance: Distance) -> bool {
        match &self.vectors {
            Some(v) => v.size == size && v.distance == distance,
            None => false,
        }
    }
}
