//! Project configuration (nostos.toml)

use serde::{Deserialize, Serialize};
use std::path::Path;
use std::fs;

/// Project configuration from nostos.toml
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectConfig {
    pub project: ProjectInfo,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectInfo {
    pub name: String,
    #[serde(default = "default_version")]
    pub version: String,
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
}

impl Default for ProjectConfig {
    fn default() -> Self {
        Self::new("nostos-project")
    }
}
