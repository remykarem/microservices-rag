pub mod client;
mod index;
mod indexing;
mod inference;
pub mod ingest;
mod terminal;
pub mod transform;

use anyhow::{Context, Result};
use std::fmt::{Display, Formatter};
// -------- hardcoded config (per your requirements) --------

const QDRANT_URL: &str = "http://localhost:6333";
const EMBED_BASE_MODEL: &str = "text-embedding-embeddinggemma-300m"; // change as needed
const VECTOR_SIZE: usize = 768; // must match model

const INCLUDE_FILENAME_DOC: bool = true;
const EMBED_BATCH: usize = 64;
const UPSERT_BATCH: usize = 64;
const UPSERT_RETRIES: usize = 3;

// daemon-ish loop controls
const RESCAN_INTERVAL_SECS: u64 = 600; // 10 minutes (safe default)

// ----------------------------------------------------------

#[derive(Debug, Clone)]
enum Mode {
    Index,
    Query,
    Thing,
}

impl Display for Mode {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    terminal::terminal().await
}
