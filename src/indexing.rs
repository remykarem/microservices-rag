// src/indexing
//! Wires everything together into a minimal daemon:
//! - ensure qdrant collection (per repo)
//! - scan repo (respects .gitignore)
//! - parse rust → Documents
//! - normalize → NormalizedDoc
//! - build canonical text
//! - embed → vectors (LM Studio / OpenAI-compatible endpoint)
//! - upsert → Qdrant
//!
//! Run: `cargo run -- /path/to/repo`
//! (If no path provided, defaults to current directory.)
//!
//! This loop is idempotent. We re-index on each cycle (static update behavior).

use crate::client::embedder_client::EmbedderClient;
use crate::client::qdrant_client::{PointWrite, QdrantClient};
use crate::index::id_generator::deterministic_point_id;
use crate::index::qdrant_schema::{Distance, QdrantSchema};
use crate::ingest::repo_scanner::ProjectScanner;
use crate::ingest::rust_parser::{CodeParser, ParseLanguage};
use crate::transform::doc_normalizer::{DocNormalizer, NormalizedDoc};
use crate::{
    EMBED_BASE_MODEL, EMBED_BATCH, INCLUDE_FILENAME_DOC, QDRANT_URL, UPSERT_BATCH, UPSERT_RETRIES,
    VECTOR_SIZE,
};
use anyhow::{Context, Result};
use indicatif::{ProgressBar, ProgressIterator, ProgressStyle};
use serde_json::json;
use std::env;
use std::path::{Path, PathBuf};
use std::time::Duration;

const DISTANCE: Distance = Distance::Cosine;

pub async fn index(p: &str) -> Result<()> {
    let root = PathBuf::from(p);
    let collection = repo_name(&root)?;

    eprintln!(
        "Indexing repo:\n  root: {}\n  collection: {}\n",
        root.display(),
        collection
    );

    // clients
    let schema = QdrantSchema::new(QDRANT_URL)?;
    let qdrant = QdrantClient::new()?;
    let embedder = EmbedderClient::new(EMBED_BASE_MODEL, Some(VECTOR_SIZE))?;

    tick_once(&schema, &qdrant, &embedder, &root, &collection).await;

    Ok(())
}

async fn tick_once(
    schema: &QdrantSchema,
    qdrant: &QdrantClient,
    embedder: &EmbedderClient,
    root: &Path,
    collection: &str,
) {
    // 1) ensure collection
    if let Err(e) = schema
        .ensure_collection(collection, VECTOR_SIZE, DISTANCE)
        .await
    {
        eprintln!("[ensure_collection] error: {e:#}");
        return;
    }

    // 2) scan repo (.gitignore-aware)
    let scanner = ProjectScanner::new();
    let repo_roots = match scanner.scan_project(root) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("[scan_project] error: {e:#}");
            return;
        }
    };

    for repo_root in repo_roots {
        let files = match scanner.scan_repo(&repo_root) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("[scan_repo] error: {e:#}");
                return;
            }
        };
        if files.is_empty() {
            eprintln!("[scan_repo] no files found");
            continue;
        }
        eprintln!("[1/4] Scan repo {repo_root:?}; {} files", files.len());

        // 3) parse → documents
        let mut rust_parser = match CodeParser::new(ParseLanguage::Rust) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("[parser] init error: {e:#}");
                return;
            }
        };
        let mut kotlin_parser = match CodeParser::new(ParseLanguage::Kotlin) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("[parser] init error: {e:#}");
                return;
            }
        };
        let mut ts_parser = match CodeParser::new(ParseLanguage::TypeScript) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("[parser] init error: {e:#}");
                return;
            }
        };
        let mut js_parser = match CodeParser::new(ParseLanguage::JavaScript) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("[parser] init error: {e:#}");
                return;
            }
        };

        let mut all_docs = Vec::new();
        for f in files.into_iter().progress() {
            if f.file_path.ends_with("rs") {
                match rust_parser.parse_file(&f.repo, &f.file_path, &f.source, INCLUDE_FILENAME_DOC)
                {
                    Ok(mut docs) => all_docs.append(&mut docs),
                    Err(e) => eprintln!("[parse_file] {}: {e:#}", f.file_path),
                }
            } else if f.file_path.ends_with("kt") {
                match kotlin_parser.parse_file(
                    &f.repo,
                    &f.file_path,
                    &f.source,
                    INCLUDE_FILENAME_DOC,
                ) {
                    Ok(mut docs) => all_docs.append(&mut docs),
                    Err(e) => eprintln!("[parse_file] {}: {e:#}", f.file_path),
                }
            } else if f.file_path.ends_with("ts") {
                match kotlin_parser.parse_file(
                    &f.repo,
                    &f.file_path,
                    &f.source,
                    INCLUDE_FILENAME_DOC,
                ) {
                    Ok(mut docs) => all_docs.append(&mut docs),
                    Err(e) => eprintln!("[parse_file] {}: {e:#}", f.file_path),
                }
            } else if f.file_path.ends_with("js") {
                match kotlin_parser.parse_file(
                    &f.repo,
                    &f.file_path,
                    &f.source,
                    INCLUDE_FILENAME_DOC,
                ) {
                    Ok(mut docs) => all_docs.append(&mut docs),
                    Err(e) => eprintln!("[parse_file] {}: {e:#}", f.file_path),
                }
            }
        }
        if all_docs.is_empty() {
            eprintln!("[parse] yielded 0 documents");
            return;
        }
        eprintln!("[2/4] Normalising {} documents", all_docs.len());

        // 4) normalize
        let normalizer = DocNormalizer::default();
        let norm_docs: Vec<NormalizedDoc> = all_docs
            .into_iter()
            .map(|d| normalizer.normalize(d))
            .collect();
        eprintln!("[3/4] Embedding {} docs", norm_docs.len());

        let pb = ProgressBar::new_spinner();
        pb.set_style(ProgressStyle::with_template("{prefix} {spinner} {wide_msg}").unwrap());
        pb.set_prefix("[3/4]");
        pb.set_message("Embedding and upserting...");
        pb.enable_steady_tick(Duration::from_millis(100));

        // 5) embed + 6) upsert (batched)
        if let Err(e) = embed_and_upsert(collection, &norm_docs, embedder, qdrant).await {
            eprintln!("[index] error: {e:#}");
            return;
        }

        pb.finish_with_message("Done!");

        eprintln!(
            "[index] done: upserted {} documents into '{}'",
            norm_docs.len(),
            collection
        );
    }
}

async fn embed_and_upsert(
    collection: &str,
    docs: &[NormalizedDoc],
    embedder: &EmbedderClient,
    qdrant: &QdrantClient,
) -> Result<()> {
    let mut start = 0usize;
    while start < docs.len() {
        let end = (start + EMBED_BATCH).min(docs.len());
        let batch = &docs[start..end];

        // build canonical texts
        let inputs: Vec<String> = batch.iter().map(build_embedding_input).collect();

        // embed (single call per batch)
        let vectors = embedder
            .embed_texts(&inputs)
            .await
            .with_context(|| format!("embed_texts failed on range [{start}..{end})"))?;

        // map to Qdrant points
        let points: Vec<PointWrite> = batch
            .iter()
            .zip(vectors.into_iter())
            .map(|(d, vec)| {
                let id = deterministic_point_id(&d.repo, &d.file_path, &d.symbol_name, &d.kind);
                let payload = json!({
                    "repo": d.repo,
                    "file_path": d.file_path,
                    "symbol_name": d.symbol_name,
                    "type": d.kind, // "function" | "method" | ...
                    "code": d.code,
                    "line_start": d.line_start,
                    "line_end": d.line_end,
                    "parent_type": d.parent_type,
                    "signature": d.signature,
                    "doc_comment": d.doc_comment,
                    "hash_source": d.hash_source,
                    "timestamp_indexed": d.timestamp_indexed.timestamp(),
                });
                PointWrite {
                    id,
                    vector: vec,
                    payload,
                }
            })
            .collect();

        eprintln!("[4/4] Upserting [{start}..{end}");

        // upsert (may split further if points > UPSERT_BATCH)
        qdrant
            .upsert_points_batched(collection, points, UPSERT_BATCH, UPSERT_RETRIES)
            .await
            .with_context(|| format!("upsert_points_batched failed on range [{start}..{end})"))?;

        start = end;
    }
    Ok(())
}

/// Canonical embedding template (consistent with earlier design).
fn build_embedding_input(d: &NormalizedDoc) -> String {
    let parent = d.parent_type.as_deref().unwrap_or("");
    let signature = d.signature.as_deref().unwrap_or("");
    let docs = d.doc_comment.as_deref().unwrap_or("");

    format!(
        "repo: {repo}\npath: {path}\ntype: {kind}\nsymbol: {sym}\nparent: {parent}\nlines: {ls}-{le}\n\n[DOC]\n{doc}\n\n[SIGNATURE]\n{sig}\n\n[CODE]\n{code}",
        repo = d.repo,
        path = d.file_path,
        kind = d.kind,
        sym = d.symbol_name,
        parent = parent,
        ls = d.line_start,
        le = d.line_end,
        doc = docs,
        sig = signature,
        code = d.code
    )
}

pub fn query_repo_from_args() -> Result<(String, String)> {
    let mut args = env::args().skip(1);
    let query = args.next().unwrap();
    let repo = args.next().unwrap();

    Ok((query, repo))
}

fn repo_name(root: &PathBuf) -> Result<String> {
    let name = root
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("repo")
        .to_string();

    // Qdrant collection name must be URL/slug friendly; normalize lightly:
    let norm = name
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() {
                c.to_ascii_lowercase()
            } else {
                '-'
            }
        })
        .collect::<String>();
    Ok(norm.trim_matches('-').to_string())
}
