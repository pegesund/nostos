//! Module representation for source management

use std::collections::HashMap;
use crate::definition::{DefKind, DefinitionGroup};

/// Path to a module, e.g., ["utils", "math"] for utils.math
pub type ModulePath = Vec<String>;

/// Format module path as string (e.g., "utils.math")
pub fn module_path_to_string(path: &ModulePath) -> String {
    path.join(".")
}

/// Parse module path from string
pub fn module_path_from_string(s: &str) -> ModulePath {
    s.split('.').map(String::from).collect()
}

/// A module containing definitions
#[derive(Debug, Clone)]
pub struct Module {
    /// Path like ["utils", "math"]
    pub path: ModulePath,

    /// Import statements for this module
    pub imports: Vec<String>,

    /// All definitions in this module, keyed by base name
    definitions: HashMap<String, DefinitionGroup>,

    /// Needs to be written to main file
    pub dirty: bool,
}

impl Module {
    /// Create a new empty module
    pub fn new(path: ModulePath) -> Self {
        Self {
            path,
            imports: Vec::new(),
            definitions: HashMap::new(),
            dirty: false,
        }
    }

    /// Get the module name (last component of path)
    pub fn name(&self) -> &str {
        self.path.last().map(|s| s.as_str()).unwrap_or("main")
    }

    /// Get the full path as a string
    pub fn path_string(&self) -> String {
        module_path_to_string(&self.path)
    }

    /// Add a definition to this module
    pub fn add_definition(&mut self, group: DefinitionGroup) {
        self.definitions.insert(group.name.clone(), group);
        self.dirty = true;
    }

    /// Get a definition by name
    pub fn get_definition(&self, name: &str) -> Option<&DefinitionGroup> {
        self.definitions.get(name)
    }

    /// Get a mutable definition by name
    pub fn get_definition_mut(&mut self, name: &str) -> Option<&mut DefinitionGroup> {
        self.definitions.get_mut(name)
    }

    /// Remove a definition
    pub fn remove_definition(&mut self, name: &str) -> Option<DefinitionGroup> {
        let removed = self.definitions.remove(name);
        if removed.is_some() {
            self.dirty = true;
        }
        removed
    }

    /// Check if a definition exists
    pub fn has_definition(&self, name: &str) -> bool {
        self.definitions.contains_key(name)
    }

    /// Get all definition names
    pub fn definition_names(&self) -> impl Iterator<Item = &str> {
        self.definitions.keys().map(|s| s.as_str())
    }

    /// Get all definitions
    pub fn definitions(&self) -> impl Iterator<Item = &DefinitionGroup> {
        self.definitions.values()
    }

    /// Get definitions that need git commit
    pub fn dirty_definitions(&self) -> impl Iterator<Item = &DefinitionGroup> {
        self.definitions.values().filter(|d| d.needs_git_commit())
    }

    /// Mark all definitions as git clean
    pub fn mark_all_git_clean(&mut self) {
        for def in self.definitions.values_mut() {
            def.mark_git_clean();
        }
    }

    /// Set imports
    pub fn set_imports(&mut self, imports: Vec<String>) {
        self.imports = imports;
        self.dirty = true;
    }

    /// Generate module file content
    pub fn generate_file_content(&self) -> String {
        let mut content = String::new();

        // Imports first
        for import in &self.imports {
            content.push_str(import);
            content.push('\n');
        }
        if !self.imports.is_empty() {
            content.push('\n');
        }

        // Collect definitions by kind
        let mut types: Vec<&DefinitionGroup> = Vec::new();
        let mut traits: Vec<&DefinitionGroup> = Vec::new();
        let mut functions: Vec<&DefinitionGroup> = Vec::new();
        let mut variables: Vec<&DefinitionGroup> = Vec::new();

        for def in self.definitions.values() {
            match def.kind {
                DefKind::Type => types.push(def),
                DefKind::Trait => traits.push(def),
                DefKind::Function => functions.push(def),
                DefKind::Variable => variables.push(def),
            }
        }

        // Sort each category by name
        types.sort_by_key(|d| &d.name);
        traits.sort_by_key(|d| &d.name);
        functions.sort_by_key(|d| &d.name);
        variables.sort_by_key(|d| &d.name);

        // Types
        for def in types {
            content.push_str(&def.combined_source());
            content.push_str("\n\n");
        }

        // Traits
        for def in traits {
            content.push_str(&def.combined_source());
            content.push_str("\n\n");
        }

        // Functions
        for def in functions {
            content.push_str(&def.combined_source());
            content.push_str("\n\n");
        }

        // Variables
        for def in variables {
            content.push_str(&def.combined_source());
            content.push_str("\n\n");
        }

        // Trim trailing whitespace
        content.trim_end().to_string() + "\n"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_module_path() {
        let path = vec!["utils".to_string(), "math".to_string()];
        assert_eq!(module_path_to_string(&path), "utils.math");

        let parsed = module_path_from_string("utils.math");
        assert_eq!(parsed, path);
    }

    #[test]
    fn test_module_generation() {
        let mut module = Module::new(vec!["utils".to_string(), "math".to_string()]);
        module.set_imports(vec!["use core.Show".to_string()]);

        module.add_definition(DefinitionGroup::new(
            "Point".to_string(),
            DefKind::Type,
            "type Point = { x: Int, y: Int }".to_string(),
        ));

        module.add_definition(DefinitionGroup::from_sources(
            "add".to_string(),
            DefKind::Function,
            vec![
                "add(x: Int, y: Int) = x + y".to_string(),
                "add(x: Float, y: Float) = x + y".to_string(),
            ],
        ));

        let content = module.generate_file_content();
        assert!(content.starts_with("use core.Show"));
        assert!(content.contains("type Point"));
        assert!(content.contains("add(x: Int"));
        assert!(content.contains("add(x: Float"));
    }
}
