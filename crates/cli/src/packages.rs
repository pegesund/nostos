//! Package management for Nostos extensions.
//!
//! Handles fetching, building, and loading extensions from GitHub repositories.
//! Extensions are specified in `nostos.toml` files.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::fs;

/// Package configuration from nostos.toml
#[derive(Debug, Clone)]
pub struct PackageConfig {
    pub name: String,
    pub version: String,
    pub extensions: HashMap<String, ExtensionDep>,
}

/// An extension dependency
#[derive(Debug, Clone)]
pub struct ExtensionDep {
    pub git: String,
    pub branch: Option<String>,
    pub tag: Option<String>,
    pub version: Option<String>,
}

/// Extension build result with library and optional module path
#[derive(Debug, Clone)]
pub struct ExtensionResult {
    pub name: String,
    pub library_path: PathBuf,
    pub module_dir: PathBuf,  // Directory containing .nos wrapper files
}

/// Error type for package operations
#[derive(Debug)]
pub enum PackageError {
    IoError(std::io::Error),
    TomlError(String),
    GitError(String),
    BuildError(String),
    NotFound(String),
}

impl std::fmt::Display for PackageError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PackageError::IoError(e) => write!(f, "IO error: {}", e),
            PackageError::TomlError(s) => write!(f, "TOML error: {}", s),
            PackageError::GitError(s) => write!(f, "Git error: {}", s),
            PackageError::BuildError(s) => write!(f, "Build error: {}", s),
            PackageError::NotFound(s) => write!(f, "Not found: {}", s),
        }
    }
}

impl From<std::io::Error> for PackageError {
    fn from(e: std::io::Error) -> Self {
        PackageError::IoError(e)
    }
}

/// Get the cache directory for extensions
pub fn extensions_cache_dir() -> PathBuf {
    let home = dirs::home_dir().unwrap_or_else(|| PathBuf::from("."));
    home.join(".nostos").join("extensions")
}

/// Parse a nostos.toml file
pub fn parse_config(path: &Path) -> Result<PackageConfig, PackageError> {
    let content = fs::read_to_string(path)?;
    parse_toml(&content)
}

/// Parse TOML content into PackageConfig
fn parse_toml(content: &str) -> Result<PackageConfig, PackageError> {
    let mut config = PackageConfig {
        name: String::new(),
        version: "0.1.0".to_string(),
        extensions: HashMap::new(),
    };

    let mut current_section = "";
    let mut current_ext_name = String::new();
    let mut current_ext = ExtensionDep {
        git: String::new(),
        branch: None,
        tag: None,
        version: None,
    };

    for line in content.lines() {
        let line = line.trim();

        // Skip empty lines and comments
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        // Section headers
        if line.starts_with('[') && line.ends_with(']') {
            // Save previous extension if any
            if !current_ext_name.is_empty() && !current_ext.git.is_empty() {
                config.extensions.insert(current_ext_name.clone(), current_ext.clone());
                current_ext_name.clear();
                current_ext = ExtensionDep {
                    git: String::new(),
                    branch: None,
                    tag: None,
                    version: None,
                };
            }

            let section = &line[1..line.len()-1];
            if section == "package" || section == "extensions" {
                current_section = section;
            } else if section.starts_with("extensions.") {
                current_section = "extension";
                current_ext_name = section.strip_prefix("extensions.").unwrap().to_string();
            }
            continue;
        }

        // Key-value pairs
        if let Some((key, value)) = line.split_once('=') {
            let key = key.trim();
            let value = value.trim().trim_matches('"');

            match current_section {
                "package" => {
                    match key {
                        "name" => config.name = value.to_string(),
                        "version" => config.version = value.to_string(),
                        _ => {}
                    }
                }
                "extension" => {
                    match key {
                        "git" => current_ext.git = value.to_string(),
                        "branch" => current_ext.branch = Some(value.to_string()),
                        "tag" => current_ext.tag = Some(value.to_string()),
                        "version" => current_ext.version = Some(value.to_string()),
                        _ => {}
                    }
                }
                "extensions" => {
                    // Inline table: glam = { git = "..." }
                    if let Some(inline) = parse_inline_table(value) {
                        config.extensions.insert(key.to_string(), inline);
                    }
                }
                _ => {}
            }
        }
    }

    // Save last extension
    if !current_ext_name.is_empty() && !current_ext.git.is_empty() {
        config.extensions.insert(current_ext_name, current_ext);
    }

    Ok(config)
}

/// Parse inline table like { git = "...", branch = "..." }
fn parse_inline_table(s: &str) -> Option<ExtensionDep> {
    let s = s.trim();
    if !s.starts_with('{') || !s.ends_with('}') {
        return None;
    }
    let inner = &s[1..s.len()-1];

    let mut ext = ExtensionDep {
        git: String::new(),
        branch: None,
        tag: None,
        version: None,
    };

    for part in inner.split(',') {
        if let Some((key, value)) = part.split_once('=') {
            let key = key.trim();
            let value = value.trim().trim_matches('"');
            match key {
                "git" => ext.git = value.to_string(),
                "branch" => ext.branch = Some(value.to_string()),
                "tag" => ext.tag = Some(value.to_string()),
                "version" => ext.version = Some(value.to_string()),
                _ => {}
            }
        }
    }

    if ext.git.is_empty() {
        None
    } else {
        Some(ext)
    }
}

/// Fetch an extension from GitHub
pub fn fetch_extension(name: &str, dep: &ExtensionDep) -> Result<PathBuf, PackageError> {
    let cache_dir = extensions_cache_dir();
    fs::create_dir_all(&cache_dir)?;

    // Create a unique directory name based on repo URL
    let repo_name = dep.git
        .trim_end_matches(".git")
        .rsplit('/')
        .next()
        .unwrap_or(name);

    let ext_dir = cache_dir.join(repo_name);

    if ext_dir.exists() {
        // Update existing repo
        eprintln!("Updating extension {}...", name);
        let status = Command::new("git")
            .args(["pull", "--ff-only"])
            .current_dir(&ext_dir)
            .status()?;

        if !status.success() {
            // If pull fails, try a fresh clone
            fs::remove_dir_all(&ext_dir)?;
            clone_repo(dep, &ext_dir)?;
        }
    } else {
        // Clone new repo
        eprintln!("Fetching extension {}...", name);
        clone_repo(dep, &ext_dir)?;
    }

    Ok(ext_dir)
}

fn clone_repo(dep: &ExtensionDep, target: &Path) -> Result<(), PackageError> {
    let mut args = vec!["clone", "--depth", "1"];

    if let Some(ref branch) = dep.branch {
        args.push("--branch");
        args.push(branch);
    } else if let Some(ref tag) = dep.tag {
        args.push("--branch");
        args.push(tag);
    }

    args.push(&dep.git);
    args.push(target.to_str().unwrap());

    let status = Command::new("git")
        .args(&args)
        .status()?;

    if !status.success() {
        return Err(PackageError::GitError(format!(
            "Failed to clone {} to {:?}",
            dep.git, target
        )));
    }

    Ok(())
}

/// Build an extension (cargo build --release)
pub fn build_extension(ext_dir: &Path) -> Result<PathBuf, PackageError> {
    eprintln!("Building extension in {:?}...", ext_dir);

    let status = Command::new("cargo")
        .args(["build", "--release"])
        .current_dir(ext_dir)
        .status()?;

    if !status.success() {
        return Err(PackageError::BuildError(format!(
            "Failed to build extension in {:?}",
            ext_dir
        )));
    }

    // Find the .so/.dylib file
    let release_dir = ext_dir.join("target").join("release");

    // Look for .so (Linux) or .dylib (macOS)
    for entry in fs::read_dir(&release_dir)? {
        let entry = entry?;
        let path = entry.path();
        if let Some(ext) = path.extension() {
            if ext == "so" || ext == "dylib" {
                let name = path.file_name().unwrap().to_str().unwrap();
                // Skip deps directory libraries
                if name.starts_with("lib") && !name.contains('-') {
                    return Ok(path);
                }
            }
        }
    }

    Err(PackageError::NotFound(format!(
        "No .so/.dylib found in {:?}",
        release_dir
    )))
}

/// Fetch and build all extensions from a config
pub fn fetch_and_build_all(config: &PackageConfig) -> Result<Vec<ExtensionResult>, PackageError> {
    let mut results = Vec::new();

    for (name, dep) in &config.extensions {
        let ext_dir = fetch_extension(name, dep)?;
        let lib_path = build_extension(&ext_dir)?;
        eprintln!("Built extension {}: {:?}", name, lib_path);
        results.push(ExtensionResult {
            name: name.clone(),
            library_path: lib_path,
            module_dir: ext_dir,  // The extension dir contains .nos wrapper files
        });
    }

    Ok(results)
}

/// Look for nostos.toml in the given directory or its parents
pub fn find_config(start_dir: &Path) -> Option<PathBuf> {
    let mut current = start_dir.to_path_buf();

    loop {
        let config_path = current.join("nostos.toml");
        if config_path.exists() {
            return Some(config_path);
        }

        if !current.pop() {
            break;
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_inline_table() {
        let result = parse_inline_table(r#"{ git = "https://github.com/foo/bar", branch = "main" }"#);
        assert!(result.is_some());
        let ext = result.unwrap();
        assert_eq!(ext.git, "https://github.com/foo/bar");
        assert_eq!(ext.branch, Some("main".to_string()));
    }

    #[test]
    fn test_parse_toml() {
        let content = r#"
[package]
name = "my-project"
version = "0.1.0"

[extensions]
glam = { git = "https://github.com/pegesund/nostos-glam" }

[extensions.kafka]
git = "https://github.com/example/nostos-kafka"
branch = "main"
"#;
        let config = parse_toml(content).unwrap();
        assert_eq!(config.name, "my-project");
        assert_eq!(config.extensions.len(), 2);
        assert!(config.extensions.contains_key("glam"));
        assert!(config.extensions.contains_key("kafka"));
    }
}
