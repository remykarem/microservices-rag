//! doc_normalizer.rs
//!
//! Cleans, enriches, and canonicalizes parsed Documents before embedding/indexing.
//!
//! Responsibilities:
//! - Normalize whitespace / dedent code blocks
//! - Truncate overly large code fields (optional safeguard)
//! - Compute stable `hash_source` (content hash)
//! - Attach `timestamp_indexed`
//!
//! Does not mutate `repo`, `file_path`, or line ranges.

use crate::index::id_generator;
use crate::ingest::rust_parser::Document;
use chrono::{DateTime, Utc};

pub struct DocNormalizer {
    pub max_code_chars: usize, // safeguard (e.g., 100_000)
}

impl Default for DocNormalizer {
    fn default() -> Self {
        Self {
            max_code_chars: 100_000,
        }
    }
}

impl DocNormalizer {
    pub fn normalize(&self, doc: Document) -> NormalizedDoc {
        // Clean code (trim, dedent, and cap length)
        let code = normalize_code(&doc.code, self.max_code_chars);
        let doc_comment = doc.doc_comment.as_ref().map(|s| s.trim().to_string());
        let signature = doc.signature.as_ref().map(|s| s.trim().to_string());

        let hash_source =
            id_generator::content_hash(signature.as_deref(), doc_comment.as_deref(), &code);

        let timestamp_indexed = Utc::now();

        NormalizedDoc {
            repo: doc.repo,
            file_path: doc.file_path,
            symbol_name: doc.symbol_name,
            kind: doc.kind.as_str().to_string(),
            signature,
            doc_comment,
            code,
            parent_type: doc.parent_type,
            line_start: doc.line_start,
            line_end: doc.line_end,
            hash_source,
            timestamp_indexed,
        }
    }
}

/// Canonical normalized Document ready for embedding and upsert.
#[derive(Debug, Clone)]
pub struct NormalizedDoc {
    pub repo: String,
    pub file_path: String,
    pub symbol_name: String,
    pub kind: String,
    pub signature: Option<String>,
    pub doc_comment: Option<String>,
    pub code: String,
    pub parent_type: Option<String>,
    pub line_start: u32,
    pub line_end: u32,
    pub hash_source: String,
    pub timestamp_indexed: DateTime<Utc>,
}

/// ---- helpers ----
fn normalize_code(src: &str, max_chars: usize) -> String {
    let mut s = src.trim().replace("\r\n", "\n");
    if s.len() > max_chars {
        s.truncate(max_chars);
    }
    // basic dedent (remove common leading whitespace)
    if let Some(first_nonempty) = s.lines().find(|l| !l.trim().is_empty()) {
        let indent = first_nonempty
            .chars()
            .take_while(|c| c.is_whitespace())
            .count();
        if indent > 0 {
            s = s
                .lines()
                .map(|l| {
                    if l.len() >= indent {
                        l[indent..].to_string()
                    } else {
                        l.to_string()
                    }
                })
                .collect::<Vec<_>>()
                .join("\n");
        }
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ingest::rust_parser::{Document, DocumentKind};

    #[test]
    fn normalizes_code_and_hash() {
        let doc = Document {
            repo: "r".into(),
            file_path: "src/lib.rs".into(),
            symbol_name: "foo".into(),
            kind: DocumentKind::Function,
            signature: Some("fn foo()".into()),
            doc_comment: Some("/// docs".into()),
            code: "    fn foo() {}".into(),
            parent_type: None,
            line_start: 1,
            line_end: 2,
        };
        let norm = DocNormalizer::default().normalize(doc);
        assert!(norm.hash_source.len() > 10);
        assert!(norm.code.starts_with("fn foo"));
    }
}
