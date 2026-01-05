//! Project configuration (nostos.toml)

use serde::{Deserialize, Serialize};
use std::path::Path;
use std::fs;

/// Project configuration from nostos.toml
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectConfig {
    pub project: ProjectInfo,
    /// Binary entry points (optional)
    #[serde(default, rename = "bin")]
    pub bins: Vec<BinEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectInfo {
    pub name: String,
    #[serde(default = "default_version")]
    pub version: String,
}

/// A binary entry point definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BinEntry {
    /// Name of the binary (used with --bin flag)
    pub name: String,
    /// Entry point: module.function (e.g., "server.main" or "main.main")
    pub entry: String,
    /// Whether this is the default entry point
    #[serde(default)]
    pub default: bool,
}

fn default_version() -> String {
    "0.1.0".to_string()
}

impl ProjectConfig {
    /// Create a new project config with the given name
    pub fn new(name: &str) -> Self {
        Self {
            project: ProjectInfo {
                name: name.to_string(),
                version: default_version(),
            },
            bins: Vec::new(),
        }
    }

    /// Load from a nostos.toml file
    pub fn load(path: &Path) -> Result<Self, String> {
        let content = fs::read_to_string(path)
            .map_err(|e| format!("Failed to read {}: {}", path.display(), e))?;
        toml::from_str(&content)
            .map_err(|e| format!("Failed to parse {}: {}", path.display(), e))
    }

    /// Save to a nostos.toml file
    pub fn save(&self, path: &Path) -> Result<(), String> {
        let content = toml::to_string_pretty(self)
            .map_err(|e| format!("Failed to serialize config: {}", e))?;
        fs::write(path, content)
            .map_err(|e| format!("Failed to write {}: {}", path.display(), e))
    }

    /// Create default config and save it
    pub fn create_default(project_root: &Path) -> Result<Self, String> {
        let name = project_root
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("nostos-project");

        let config = Self::new(name);
        let config_path = project_root.join("nostos.toml");
        config.save(&config_path)?;
        Ok(config)
    }

    /// Get a binary entry by name
    pub fn get_bin(&self, name: &str) -> Option<&BinEntry> {
        self.bins.iter().find(|b| b.name == name)
    }

    /// Get the default binary entry (marked with default = true)
    pub fn get_default_bin(&self) -> Option<&BinEntry> {
        self.bins.iter().find(|b| b.default)
    }

    /// Check if project has any binary entries defined
    pub fn has_bins(&self) -> bool {
        !self.bins.is_empty()
    }

    /// Get list of available binary names
    pub fn bin_names(&self) -> Vec<&str> {
        self.bins.iter().map(|b| b.name.as_str()).collect()
    }
}

impl Default for ProjectConfig {
    fn default() -> Self {
        Self::new("nostos-project")
    }
}
