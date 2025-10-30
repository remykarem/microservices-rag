//! rust_parser.rs
//!
//! Parses a Rust source file into "Document" units using tree-sitter-rust.
//! Documents include:
//! - filename (the entire file as one document)
//! - structs, enums, traits
//! - free functions
//! - methods (functions inside `impl` that have a `self` receiver)
//!
//! For each document we record:
//! - repo, file_path
//! - kind: function | method | struct | enum | trait | filename
//! - symbol_name (e.g., function/struct name, or file basename for filename docs)
//! - signature (best-effort; header without body for items with bodies)
//! - doc_comment (leading `///` or `//!` lines grouped)
//! - code (full code snippet for that node; for filename, full file text)
//! - parent_type (for methods, the impl target type string)
//! - line_start, line_end (1-based inclusive)
//!
//! Notes:
//! - Doc comments are collected heuristically by scanning contiguous `///` / `//!`
//!   lines directly above the item (stopping at blank/non-comment code).
//! - Method detection: a function inside an `impl_item` with a `self_parameter`
//!   in its parameter list.
//! - Parent type: we extract the full `impl <...> <Target> for <Trait>? {` header
//!   slice between `impl` and the `{`, then normalize whitespace.
//!
//! This file only depends on tree-sitter and serde/thiserror; it does not perform I/O.

use serde::{Deserialize, Serialize};
use std::path::Path;
use thiserror::Error;
use tree_sitter::{Node, Parser, Point, Range};

#[derive(Debug, Error)]
pub enum RustParserError {
    #[error("tree-sitter parse failed")]
    ParseFailed,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DocumentKind {
    Function,
    Method,
    Struct,
    Enum,
    Trait,
    Filename,
}

impl DocumentKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            DocumentKind::Function => "function",
            DocumentKind::Method => "method",
            DocumentKind::Struct => "struct",
            DocumentKind::Enum => "enum",
            DocumentKind::Trait => "trait",
            DocumentKind::Filename => "filename",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Document {
    pub repo: String,
    pub file_path: String,
    pub symbol_name: String,
    pub kind: DocumentKind,
    pub signature: Option<String>,
    pub doc_comment: Option<String>,
    pub code: String,
    pub parent_type: Option<String>,
    pub line_start: u32,
    pub line_end: u32,
}

/// Primary parser type
pub struct RustParser {
    parser: Parser,
}

pub enum Language {
    Rust,
    Kotlin,
}

impl RustParser {
    pub fn new(language: Language) -> Result<Self, RustParserError> {
        let mut parser = Parser::new();

        let lang = match language {
            Language::Rust => tree_sitter_rust::LANGUAGE,
            Language::Kotlin => tree_sitter_kotlin::LANGUAGE,
        };
        parser
            .set_language(&lang.into())
            .map_err(|_| RustParserError::ParseFailed)?;
        Ok(Self { parser })
    }

    /// Parse a single Rust file into Documents.
    ///
    /// - `repo`: repo/collection name
    /// - `file_path`: relative path from repo root
    /// - `source`: full file contents
    /// - `include_filename_doc`: whether to include a top-level "filename" document
    pub fn parse_file(
        &mut self,
        repo: &str,
        file_path: &str,
        source: &str,
        include_filename_doc: bool,
    ) -> Result<Vec<Document>, RustParserError> {
        let tree = self
            .parser
            .parse(source, None)
            .ok_or(RustParserError::ParseFailed)?;
        let root = tree.root_node();

        let mut out = Vec::new();

        if include_filename_doc {
            out.push(self.build_filename_document(repo, file_path, source));
        }

        // Walk the tree and collect items of interest.
        let _ = root.walk();
        let mut stack: Vec<Node> = vec![root];

        while let Some(node) = stack.pop() {
            // Push children to stack (manual DFS)
            for i in (0..node.child_count()).rev() {
                if let Some(ch) = node.child(i) {
                    stack.push(ch);
                }
            }

            match node.kind() {
                // Structs, Enums, Traits
                "struct_item" | "enum_item" | "trait_item" => {
                    if let Some(doc) = self.extract_named_item(repo, file_path, source, node) {
                        out.push(doc);
                    }
                }

                // Function items
                "function_item" => {
                    let parent = node.parent();
                    let is_method = parent
                        .as_ref()
                        .map(|p| p.kind() == "impl_item")
                        .unwrap_or(false)
                        && has_self_parameter(node);

                    if is_method {
                        if let Some(doc) =
                            self.extract_method_item(repo, file_path, source, node, parent.unwrap())
                        {
                            out.push(doc);
                        }
                    } else {
                        if let Some(doc) = self.extract_function_item(repo, file_path, source, node)
                        {
                            out.push(doc);
                        }
                    }
                }

                _ => {}
            }
        }

        Ok(out)
    }

    fn build_filename_document(&self, repo: &str, file_path: &str, source: &str) -> Document {
        let symbol_name = Path::new(file_path)
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or(file_path)
            .to_string();
        let total_lines = 1 + byte_count(source.as_bytes(), b'\n') as u32;

        Document {
            repo: repo.to_string(),
            file_path: file_path.to_string(),
            symbol_name,
            kind: DocumentKind::Filename,
            signature: None,
            doc_comment: None,
            code: source.to_string(),
            parent_type: None,
            line_start: 1,
            line_end: total_lines,
        }
    }

    fn extract_named_item(
        &self,
        repo: &str,
        file_path: &str,
        source: &str,
        node: Node,
    ) -> Option<Document> {
        // Determine kind and name child
        let (kind, name_child_kind) = match node.kind() {
            "struct_item" => (DocumentKind::Struct, "type_identifier"),
            "enum_item" => (DocumentKind::Enum, "type_identifier"),
            "trait_item" => (DocumentKind::Trait, "type_identifier"),
            _ => return None,
        };

        let name = child_text_by_kind(source, node, name_child_kind)?;
        let (line_start, line_end) = lines_of(&node);
        let signature = extract_item_signature(source, node);
        let doc_comment = leading_doc_comment_block_above(source, node);
        let code = slice_source(source, node.byte_range());

        Some(Document {
            repo: repo.to_string(),
            file_path: file_path.to_string(),
            symbol_name: name,
            kind,
            signature,
            doc_comment,
            code,
            parent_type: None,
            line_start,
            line_end,
        })
    }

    fn extract_function_item(
        &self,
        repo: &str,
        file_path: &str,
        source: &str,
        node: Node,
    ) -> Option<Document> {
        // Free function
        let name = child_text_by_kind(source, node, "identifier")?;
        let (line_start, line_end) = lines_of(&node);
        let signature = extract_item_signature(source, node);
        let doc_comment = leading_doc_comment_block_above(source, node);
        let code = slice_source(source, node.byte_range());

        Some(Document {
            repo: repo.to_string(),
            file_path: file_path.to_string(),
            symbol_name: name,
            kind: DocumentKind::Function,
            signature,
            doc_comment,
            code,
            parent_type: None,
            line_start,
            line_end,
        })
    }

    fn extract_method_item(
        &self,
        repo: &str,
        file_path: &str,
        source: &str,
        func_node: Node,
        impl_node: Node,
    ) -> Option<Document> {
        let name = child_text_by_kind(source, func_node, "identifier")?;
        let (line_start, line_end) = lines_of(&func_node);
        let signature = extract_item_signature(source, func_node);
        let doc_comment = leading_doc_comment_block_above(source, func_node);
        let code = slice_source(source, func_node.byte_range());

        let parent_type = extract_impl_target_type(source, impl_node);

        Some(Document {
            repo: repo.to_string(),
            file_path: file_path.to_string(),
            symbol_name: name,
            kind: DocumentKind::Method,
            signature,
            doc_comment,
            code,
            parent_type,
            line_start,
            line_end,
        })
    }
}

/// ---- helpers ----
fn lines_of(node: &Node) -> (u32, u32) {
    let Range {
        start_point: Point { row: sr, .. },
        end_point: Point { row: er, .. },
        ..
    } = node.range();
    (sr as u32 + 1, er as u32 + 1)
}

fn slice_source(source: &str, range: std::ops::Range<usize>) -> String {
    source.get(range).map(|s| s.to_string()).unwrap_or_default()
}

/// Returns the text of a child node with the given kind, if it exists.
fn child_text_by_kind(source: &str, node: Node, kind: &str) -> Option<String> {
    for i in 0..node.child_count() {
        if let Some(ch) = node.child(i) {
            if ch.kind() == kind {
                return Some(slice_source(source, ch.byte_range()));
            }
        }
    }
    None
}

/// Detects if a function_item has a self receiver parameter.
fn has_self_parameter(func_node: Node) -> bool {
    // Look for a "parameters" child that contains a "self_parameter" descendant.
    for i in 0..func_node.child_count() {
        if let Some(ch) = func_node.child(i) {
            if ch.kind() == "parameters" {
                // descend to find "self_parameter"
                let mut stack = vec![ch];
                while let Some(n) = stack.pop() {
                    if n.kind() == "self_parameter" {
                        return true;
                    }
                    for j in 0..n.child_count() {
                        if let Some(c) = n.child(j) {
                            stack.push(c);
                        }
                    }
                }
            }
        }
    }
    false
}

/// Try to extract a best-effort signature for an item:
/// - For function_item: from start up to (but excluding) the body block `{` if present; else whole node
/// - For struct/enum/trait: from start to the `{` or `;`, whichever comes first
fn extract_item_signature(source: &str, node: Node) -> Option<String> {
    // If there's a body block, cut before it
    let mut cutoff = node.end_byte();
    for i in 0..node.child_count() {
        if let Some(ch) = node.child(i) {
            match ch.kind() {
                // function body block
                "block" => {
                    cutoff = ch.start_byte();
                    break;
                }
                // item body braces (struct/enum/trait)
                "field_declaration_list" | "enum_variant_list" | "declaration_list" => {
                    cutoff = ch.start_byte();
                    break;
                }
                _ => {}
            }
        }
    }

    // Also handle a trailing semicolon items (e.g., trait fns w/o body)
    // If the node has a ';' token as last child, include up to it.
    // Otherwise, use cutoff as determined above.
    let text_full = slice_source(source, node.byte_range());
    if cutoff < node.end_byte() {
        let head = &source[node.start_byte()..cutoff];
        return Some(head.trim().to_string());
    }

    // Try to find the first ';' relative to node start.
    if let Some(idx) = text_full.find(';') {
        let head = &text_full[..=idx];
        return Some(head.trim().to_string());
    }

    Some(text_full.trim().to_string())
}

/// Extracts the target type string from an `impl_item`.
/// Heuristic: take the slice between the `impl` keyword and the opening `{`,
/// then normalize whitespace. This captures patterns like:
/// - `impl Foo { ... }`
/// - `impl<T> Foo<T> { ... }`
/// - `impl Trait for Type { ... }` (we still return the full header piece)
fn extract_impl_target_type(source: &str, impl_node: Node) -> Option<String> {
    let text = slice_source(source, impl_node.byte_range());
    // Find "impl" and the first "{" after it.
    let impl_pos = text.find("impl")?;
    let brace_pos = text.find('{')?;
    if brace_pos <= impl_pos + 4 {
        return None;
    }
    let mid = &text[impl_pos + 4..brace_pos]; // skip "impl"
    let norm = normalize_ws(mid);
    Some(norm)
}

/// Collects leading doc comment lines directly above the node:
/// consecutive lines starting with "///" or "//!", stopping at the first
/// non-comment, non-blank line.
fn leading_doc_comment_block_above(source: &str, node: Node) -> Option<String> {
    let start_byte = node.start_byte();
    let prefix = &source[..start_byte];
    let mut lines: Vec<&str> = prefix.split_inclusive('\n').collect();
    if lines.is_empty() {
        return None;
    }
    // Drop the trailing partial line if not ending with '\n'
    if !prefix.ends_with('\n') {
        if let Some(last) = lines.pop() {
            // last is a partial prefix of the line where node begins; ignore
            let _ = last;
        }
    }

    // Walk upward accumulating contiguous `///` or `//!` lines (trim trailing '\n')
    let mut collected_rev: Vec<String> = Vec::new();
    let mut seen_any = false;

    for raw in lines.iter().rev() {
        let line = raw.trim_end_matches('\n');
        let trimmed = line.trim_start();

        if trimmed.starts_with("///") || trimmed.starts_with("//!") {
            // Strip exactly the marker and one optional space for readability
            let content = trimmed
                .trim_start_matches("///")
                .trim_start_matches("//!")
                .trim_start()
                .to_string();
            collected_rev.push(content);
            seen_any = true;
            continue;
        }

        // also consider block doc comments immediately above (rare in practice)
        if trimmed.starts_with("/**") || trimmed.starts_with("/*!") {
            // naive: collect until the matching "*/" going upward isn't trivial,
            // so we include only this line as a header hint.
            collected_rev.push(trimmed.to_string());
            seen_any = true;
            continue;
        }

        // blank lines between doc comments are allowed and included
        if seen_any && trimmed.is_empty() {
            collected_rev.push(String::new());
            continue;
        }

        // hit non-comment code or a gap—stop
        if !seen_any {
            // no doc block found
            return None;
        } else {
            break;
        }
    }

    if collected_rev.is_empty() {
        None
    } else {
        collected_rev.reverse();
        let joined = collected_rev.join("\n").trim_end().to_string();
        Some(joined)
    }
}

fn normalize_ws(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn byte_count(haystack: &[u8], needle: u8) -> usize {
    haystack.iter().filter(|&&b| b == needle).count()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cmp::PartialEq;

    impl PartialEq for DocumentKind {
        fn eq(&self, other: &Self) -> bool {
            todo!()
        }
    }

    #[test]
    fn test_leading_doc_comment() {
        let src = r#"
/// Top doc
/// more
fn foo() {}
"#;
        let mut p = RustParser::new(Language::Rust).unwrap();
        let docs = p
            .parse_file("repo", "src/lib.rs", src, false)
            .expect("parse");
        let f = docs
            .iter()
            .find(|d| d.kind == DocumentKind::Function)
            .unwrap();
        assert_eq!(f.symbol_name, "foo");
        assert_eq!(f.doc_comment.as_deref(), Some("Top doc\nmore"));
        assert!(f.signature.as_ref().unwrap().starts_with("fn foo"));
    }

    #[test]
    fn test_method_detection_and_parent() {
        let src = r#"
struct A;

impl A {
    /// doc
    fn bar(&self, x: i32) -> i32 { x + 1 }
}

fn baz() {}
"#;
        let mut p = RustParser::new(Language::Rust).unwrap();
        let docs = p.parse_file("repo", "a.rs", src, false).expect("parse");

        let meth = docs
            .iter()
            .find(|d| d.kind == DocumentKind::Method)
            .unwrap();
        assert_eq!(meth.symbol_name, "bar");
        assert!(meth.parent_type.as_ref().unwrap().contains("A"));
        let fun = docs.iter().find(|d| d.symbol_name == "baz").unwrap();
        assert_eq!(fun.kind, DocumentKind::Function);
    }

    #[test]
    fn test_filename_document() {
        let src = "fn x() {}\n";
        let mut p = RustParser::new(Language::Rust).unwrap();
        let docs = p.parse_file("r", "src/main.rs", src, true).expect("ok");
        let file = docs
            .iter()
            .find(|d| matches!(d.kind, DocumentKind::Filename))
            .unwrap();
        assert_eq!(file.symbol_name, "main.rs");
        assert_eq!(file.line_start, 1);
        assert_eq!(file.line_end, 2);
        assert_eq!(file.code, src);
    }
}
