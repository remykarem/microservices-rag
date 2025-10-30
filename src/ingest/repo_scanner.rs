//! repo_scanner.rs
//!
//! Recursively walks a repository and yields `.rs` source files,
//! respecting `.gitignore`, `.ignore`, and standard ignore patterns.
//!
//! Each repo corresponds to one Qdrant collection.
//!
//! Responsibilities:
//! - Detect repo name (basename of root path)
//! - Use `ignore::WalkBuilder` to honor .gitignore/.ignore
//! - Yield `(repo_name, relative_path, source_code)` for each valid file

use ignore::{DirEntry, WalkBuilder};
use std::fs;
use std::path::{Path, PathBuf};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ProjectScannerError {
    #[error("I/O: {0}")]
    Io(#[from] std::io::Error),

    #[error("Invalid UTF-8 in {0}")]
    InvalidUtf8(String),
}

#[derive(Debug, Clone)]
pub struct FileEntry {
    pub repo: String,
    pub file_path: String, // relative to repo root
    pub source: String,
}

pub struct ProjectScanner;

impl ProjectScanner {
    pub fn new() -> Self {
        Self
    }

    /// Scans the project and returns all valid .rs files in nested vectors.
    pub fn scan_project(&self, root: &Path) -> Result<Vec<PathBuf>, ProjectScannerError> {
        let mut directories = Vec::new();

        if root.is_dir() {
            for entry in fs::read_dir(root).expect("Failed to read directory") {
                let entry = entry.expect("Failed to get directory entry");
                let path = entry.path();

                if path.is_dir() {
                    directories.push(path);
                }
            }
        }

        Ok(directories)
    }

    /// Scans the repository and returns all Rust source files (.rs),
    /// honoring `.gitignore` and `.ignore` rules automatically.
    pub fn scan_repo(&self, root: &Path) -> Result<Vec<FileEntry>, ProjectScannerError> {
        let repo_name = root
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown")
            .to_string();

        let mut out = Vec::new();

        let walker = WalkBuilder::new(root)
            .hidden(false) // allow hidden dirs; ignore crate will filter per .gitignore
            .follow_links(false)
            .git_ignore(true)
            .git_exclude(true)
            .git_global(true)
            .ignore(true)
            .build();

        for result in walker {
            let entry = match result {
                Ok(e) => e,
                Err(_) => continue,
            };

            if should_include(&entry) {
                let rel_path = entry.path().strip_prefix(root).unwrap_or(entry.path());
                let rel_str = rel_path.to_string_lossy().to_string();
                let content = fs::read_to_string(entry.path())
                    .map_err(|e| ProjectScannerError::Io(e))
                    .and_then(|s| {
                        if !s.is_char_boundary(s.len()) {
                            Err(ProjectScannerError::InvalidUtf8(rel_str.clone()))
                        } else {
                            Ok(s)
                        }
                    })?;

                out.push(FileEntry {
                    repo: repo_name.clone(),
                    file_path: rel_str,
                    source: content,
                });
            }
        }

        Ok(out)
    }
}

/// Check whether a file should be included in scanning.
fn should_include(entry: &DirEntry) -> bool {
    let path = entry.path();
    if !entry.file_type().map(|t| t.is_file()).unwrap_or(false) {
        return false;
    }
    match path.extension().and_then(|x| x.to_str()) {
        Some("rs") | Some("kt") => true,
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn respects_gitignore() {
        let tmpdir = tempfile::tempdir().unwrap();
        let repo = tmpdir.path();
        // Create a .gitignore file
        fs::write(repo.join(".gitignore"), "ignored.rs\n").unwrap();

        // Create files
        fs::write(repo.join("main.rs"), "fn main() {}").unwrap();
        fs::write(repo.join("ignored.rs"), "fn ignored() {}").unwrap();

        let scanner = ProjectScanner::new();
        let files = scanner.scan_repo(repo).unwrap();

        let filenames: Vec<_> = files.iter().map(|f| &f.file_path).collect();
        assert!(filenames.contains(&&"main.rs".to_string()));
        assert!(!filenames.contains(&&"ignored.rs".to_string()));
    }
}
