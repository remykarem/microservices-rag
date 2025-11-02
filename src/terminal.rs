use inquire::{Select, Text};

use crate::client::llm_client::ask_llm;
use crate::indexing;
use crate::inference::rag;
use anyhow::Result;
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

pub async fn terminal() -> Result<(), Box<dyn std::error::Error>> {
    // First choice: Index or Query
    let mode = Select::new("Select operation:", vec![Mode::Query, Mode::Index]).prompt()?;

    match mode {
        Mode::Index => {
            let path = Text::new("Enter path to index:").prompt()?;
            indexing::index(&path).await?;
        }
        Mode::Query => {
            let prompt = Text::new("Enter your query:").prompt()?;
            let repo = Select::new("Enter repository name:", vec!["*"]).prompt()?;
            let docs = rag(&prompt, repo).await?;
            println!("{:#?}", docs);
        }
        Mode::Thing => {
            let prompt = Text::new("Prompt:").prompt()?;
            ask_llm(
                "http://localhost:1234/v1/chat/completions",
                "lm-studio",            // LM Studio typically uses this as API key
                "qwen/qwen3-coder-30b", // Your model name
                &prompt,
            )
            .await;
        }
    }

    Ok(())
}
