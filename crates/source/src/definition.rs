//! Definition types for source management

use seahash::hash;

/// Kind of definition
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DefKind {
    Function,
    Type,
    Trait,
    Variable,
}

impl DefKind {
    /// Check if this kind can have overloads
    pub fn can_overload(&self) -> bool {
        matches!(self, DefKind::Function)
    }

    /// Determine kind from name (convention: uppercase = type/trait)
    pub fn from_name(name: &str) -> DefKind {
        if name.starts_with("var ") {
            DefKind::Variable
        } else if name.chars().next().map(|c| c.is_uppercase()).unwrap_or(false) {
            // Could be Type or Trait, need to check content
            DefKind::Type
        } else {
            DefKind::Function
        }
    }
}

/// A single definition (one function overload, one type, etc.)
#[derive(Debug, Clone)]
pub struct Definition {
    /// Full source text
    pub source: String,

    /// Content hash for change detection
    pub content_hash: u64,

    /// Signature for functions (e.g., "Int, Int" for add(x: Int, y: Int))
    /// Used to distinguish overloads
    pub signature: Option<String>,
}

impl Definition {
    /// Create a new definition from source
    pub fn new(source: String) -> Self {
        let content_hash = hash(source.as_bytes());
        Self {
            source,
            content_hash,
            signature: None,
        }
    }

    /// Create with explicit signature
    pub fn with_signature(source: String, signature: String) -> Self {
        let content_hash = hash(source.as_bytes());
        Self {
            source,
            content_hash,
            signature: Some(signature),
        }
    }

    /// Update source and recompute hash
    pub fn update(&mut self, new_source: String) -> bool {
        let new_hash = hash(new_source.as_bytes());
        if new_hash != self.content_hash {
            self.source = new_source;
            self.content_hash = new_hash;
            true
        } else {
            false
        }
    }

    /// Check if content matches
    pub fn matches_hash(&self, other_hash: u64) -> bool {
        self.content_hash == other_hash
    }
}

/// Groups overloaded functions or holds a single definition
#[derive(Debug, Clone)]
pub struct DefinitionGroup {
    /// Base name (e.g., "add")
    pub name: String,

    /// Kind of definition
    pub kind: DefKind,

    /// All overloads (for functions) or single definition
    pub overloads: Vec<Definition>,

    /// Changed since last git commit
    pub git_dirty: bool,
}

impl DefinitionGroup {
    /// Create a new group with a single definition
    pub fn new(name: String, kind: DefKind, source: String) -> Self {
        Self {
            name,
            kind,
            overloads: vec![Definition::new(source)],
            git_dirty: true,
        }
    }

    /// Create from multiple sources (for overloaded functions)
    pub fn from_sources(name: String, kind: DefKind, sources: Vec<String>) -> Self {
        Self {
            name,
            kind,
            overloads: sources.into_iter().map(Definition::new).collect(),
            git_dirty: true,
        }
    }

    /// Add an overload
    pub fn add_overload(&mut self, source: String) {
        self.overloads.push(Definition::new(source));
        self.git_dirty = true;
    }

    /// Update all overloads from combined source
    /// Returns true if anything changed
    pub fn update_from_source(&mut self, combined_source: &str) -> bool {
        let new_overloads: Vec<String> = combined_source
            .split("\n\n")
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect();

        // Check if anything changed
        if new_overloads.len() != self.overloads.len() {
            self.overloads = new_overloads.into_iter().map(Definition::new).collect();
            self.git_dirty = true;
            return true;
        }

        let mut changed = false;
        for (i, new_source) in new_overloads.into_iter().enumerate() {
            if i < self.overloads.len() {
                if self.overloads[i].update(new_source) {
                    changed = true;
                }
            } else {
                self.overloads.push(Definition::new(new_source));
                changed = true;
            }
        }

        if changed {
            self.git_dirty = true;
        }
        changed
    }

    /// Get combined source (all overloads separated by blank lines)
    pub fn combined_source(&self) -> String {
        self.overloads
            .iter()
            .map(|d| d.source.as_str())
            .collect::<Vec<_>>()
            .join("\n\n")
    }

    /// Mark as synced to git
    pub fn mark_git_clean(&mut self) {
        self.git_dirty = false;
    }

    /// Check if needs git commit
    pub fn needs_git_commit(&self) -> bool {
        self.git_dirty
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_definition_hash() {
        let def1 = Definition::new("add(x: Int) = x + 1".to_string());
        let def2 = Definition::new("add(x: Int) = x + 1".to_string());
        let def3 = Definition::new("add(x: Int) = x + 2".to_string());

        assert_eq!(def1.content_hash, def2.content_hash);
        assert_ne!(def1.content_hash, def3.content_hash);
    }

    #[test]
    fn test_definition_update() {
        let mut def = Definition::new("add(x: Int) = x + 1".to_string());
        let original_hash = def.content_hash;

        // Same content - no change
        assert!(!def.update("add(x: Int) = x + 1".to_string()));
        assert_eq!(def.content_hash, original_hash);

        // Different content - changed
        assert!(def.update("add(x: Int) = x + 2".to_string()));
        assert_ne!(def.content_hash, original_hash);
    }

    #[test]
    fn test_group_combined_source() {
        let group = DefinitionGroup::from_sources(
            "add".to_string(),
            DefKind::Function,
            vec![
                "add(x: Int, y: Int) = x + y".to_string(),
                "add(x: String, y: String) = x ++ y".to_string(),
            ],
        );

        let combined = group.combined_source();
        assert!(combined.contains("add(x: Int, y: Int)"));
        assert!(combined.contains("add(x: String, y: String)"));
        assert!(combined.contains("\n\n"));
    }
}
