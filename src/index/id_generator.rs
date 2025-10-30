//! id_generator.rs
//!
//! Deterministic ID and content-hash helpers for Qdrant points.
//! - Point IDs: UUIDv5 derived from a canonical document key
//! - Content hash: SHA-256 hex for future incremental update logic

use sha2::{Digest, Sha256};
use uuid::Uuid;

/// Canonicalizes the "document identity" that should remain stable across runs.
///
/// Recommended inputs:
/// - repo (collection name)
/// - file_path (relative path from repo root)
/// - symbol_name (e.g., fn name, struct name, or filename for type=filename)
/// - kind (function|method|struct|enum|trait|filename)
pub fn canonical_document_key(
    repo: &str,
    file_path: &str,
    symbol_name: &str,
    kind: &str,
) -> String {
    format!(
        "repo={}|path={}|symbol={}|type={}",
        repo.trim(),
        file_path.trim(),
        symbol_name.trim(),
        kind.trim().to_lowercase()
    )
}

/// Stable, deterministic UUIDv5 point ID derived from the canonical key.
///
/// We use the URL namespace to avoid inventing a custom one; for determinism,
/// only the name (canonical key) matters.
pub fn deterministic_point_id(
    repo: &str,
    file_path: &str,
    symbol_name: &str,
    kind: &str,
) -> String {
    let key = canonical_document_key(repo, file_path, symbol_name, kind);
    let id = Uuid::new_v5(&Uuid::NAMESPACE_URL, key.as_bytes());
    id.to_string()
}

/// Content hash capturing the current *contents* of a document. This can be used
/// later to detect changes and skip re-embedding.
///
/// A practical canonicalization: signature + doc_comment + code
pub fn content_hash(signature: Option<&str>, doc_comment: Option<&str>, code: &str) -> String {
    let mut hasher = Sha256::new();
    if let Some(sig) = signature {
        hasher.update(b"SIG\0");
        hasher.update(sig.as_bytes());
    }
    if let Some(doc) = doc_comment {
        hasher.update(b"DOC\0");
        hasher.update(doc.as_bytes());
    }
    hasher.update(b"CODE\0");
    hasher.update(code.as_bytes());
    let bytes = hasher.finalize();
    hex::encode(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_ids_are_stable() {
        let a = deterministic_point_id("repoA", "src/lib.rs", "foo", "function");
        let b = deterministic_point_id("repoA", "src/lib.rs", "foo", "function");
        assert_eq!(a, b);
    }

    #[test]
    fn ids_change_when_key_changes() {
        let a = deterministic_point_id("repoA", "src/lib.rs", "foo", "function");
        let b = deterministic_point_id("repoA", "src/lib.rs", "bar", "function");
        assert_ne!(a, b);
    }

    #[test]
    fn content_hash_changes_with_code() {
        let h1 = content_hash(Some("fn foo()"), Some("/// docs"), "let x = 1;");
        let h2 = content_hash(Some("fn foo()"), Some("/// docs"), "let x = 2;");
        assert_ne!(h1, h2);
    }
}
