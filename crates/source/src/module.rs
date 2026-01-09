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
#[allow(dead_code)]
pub fn module_path_from_string(s: &str) -> ModulePath {
    s.split('.').map(String::from).collect()
}

/// A group of definitions that should be displayed together
pub type DefinitionGrouping = Vec<String>;

/// A module containing definitions
#[derive(Debug, Clone)]
pub struct Module {
    /// Path like ["utils", "math"]
    pub path: ModulePath,

    /// Import statements for this module
    pub imports: Vec<String>,

    /// Use statements for this module (e.g., "use nalgebra.*")
    pub use_stmts: Vec<String>,

    /// All definitions in this module, keyed by base name
    definitions: HashMap<String, DefinitionGroup>,

    /// Groups of definitions that should be kept together
    /// Each inner Vec is a list of definition names in display order
    pub groups: Vec<DefinitionGrouping>,

    /// Needs to be written to main file
    pub dirty: bool,
}

impl Module {
    /// Create a new empty module
    pub fn new(path: ModulePath) -> Self {
        Self {
            path,
            imports: Vec::new(),
            use_stmts: Vec::new(),
            definitions: HashMap::new(),
            groups: Vec::new(),
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

    /// Clear all definitions
    pub fn clear_definitions(&mut self) {
        self.definitions.clear();
        self.dirty = true;
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

    /// Set use statements
    pub fn set_use_stmts(&mut self, use_stmts: Vec<String>) {
        self.use_stmts = use_stmts;
        self.dirty = true;
    }

    /// Set groups
    pub fn set_groups(&mut self, groups: Vec<DefinitionGrouping>) {
        self.groups = groups;
        self.dirty = true;
    }

    /// Add a group
    pub fn add_group(&mut self, group: DefinitionGrouping) {
        self.groups.push(group);
        self.dirty = true;
    }

    /// Get the group containing a definition, if any
    pub fn get_group_for(&self, name: &str) -> Option<&DefinitionGrouping> {
        self.groups.iter().find(|g| g.contains(&name.to_string()))
    }

    /// Get all definition names in the same group as the given name
    /// Returns just the name itself if not in a group
    pub fn get_grouped_names(&self, name: &str) -> Vec<String> {
        if let Some(group) = self.get_group_for(name) {
            group.clone()
        } else {
            vec![name.to_string()]
        }
    }

    /// Validate that no definition appears in multiple groups
    /// Returns list of duplicates if any found
    pub fn validate_groups(&self) -> Result<(), Vec<String>> {
        use std::collections::HashSet;
        let mut seen: HashSet<&str> = HashSet::new();
        let mut duplicates: Vec<String> = Vec::new();

        for group in &self.groups {
            for name in group {
                if seen.contains(name.as_str()) {
                    duplicates.push(name.clone());
                } else {
                    seen.insert(name);
                }
            }
        }

        if duplicates.is_empty() {
            Ok(())
        } else {
            Err(duplicates)
        }
    }

    /// Generate module file content with group ordering
    pub fn generate_file_content(&self) -> String {
        let mut content = String::new();

        // Together directives first (as comments)
        for group in &self.groups {
            if !group.is_empty() {
                content.push_str("# together ");
                content.push_str(&group.join(" "));
                content.push('\n');
            }
        }
        if !self.groups.is_empty() {
            content.push('\n');
        }

        // Imports
        for import in &self.imports {
            content.push_str(import);
            content.push('\n');
        }
        if !self.imports.is_empty() {
            content.push('\n');
        }

        // Build a set of definitions that are in groups
        use std::collections::HashSet;
        let grouped_defs: HashSet<&str> = self.groups
            .iter()
            .flatten()
            .map(|s| s.as_str())
            .collect();

        // Collect ungrouped definitions by kind
        let mut types: Vec<&DefinitionGroup> = Vec::new();
        let mut traits: Vec<&DefinitionGroup> = Vec::new();
        let mut functions: Vec<&DefinitionGroup> = Vec::new();
        let mut variables: Vec<&DefinitionGroup> = Vec::new();

        for def in self.definitions.values() {
            if grouped_defs.contains(def.name.as_str()) {
                continue; // Will be handled with groups
            }
            match def.kind {
                DefKind::Type => types.push(def),
                DefKind::Trait => traits.push(def),
                DefKind::Function => functions.push(def),
                DefKind::Variable => variables.push(def),
            }
        }

        // Sort ungrouped definitions
        types.sort_by_key(|d| &d.name);
        traits.sort_by_key(|d| &d.name);
        functions.sort_by_key(|d| &d.name);
        variables.sort_by_key(|d| &d.name);

        // Collect groups sorted by first member
        let mut sorted_groups: Vec<&DefinitionGrouping> = self.groups.iter().collect();
        sorted_groups.sort_by_key(|g| g.first().map(|s| s.as_str()).unwrap_or(""));

        // Merge ungrouped and grouped, keeping category order (types, traits, functions, variables)
        // Groups are placed where their first member would appear alphabetically

        // Types (ungrouped)
        for def in &types {
            content.push_str(&def.combined_source());
            content.push_str("\n\n");
        }

        // Traits (ungrouped)
        for def in &traits {
            content.push_str(&def.combined_source());
            content.push_str("\n\n");
        }

        // Functions and groups (interleaved by first-member alphabetical order)
        let mut func_idx = 0;
        let mut group_idx = 0;

        // Filter groups to only those containing functions (or mixed)
        let func_groups: Vec<&&DefinitionGrouping> = sorted_groups.iter()
            .filter(|g| g.iter().any(|n| {
                self.definitions.get(n).map(|d| d.kind == DefKind::Function).unwrap_or(false)
            }))
            .collect();

        loop {
            let func_name = functions.get(func_idx).map(|f| f.name.as_str());
            let group_first = func_groups.get(group_idx).and_then(|g| g.first()).map(|s| s.as_str());

            match (func_name, group_first) {
                (Some(f), Some(g)) => {
                    if f <= g {
                        // Output ungrouped function
                        content.push_str(&functions[func_idx].combined_source());
                        content.push_str("\n\n");
                        func_idx += 1;
                    } else {
                        // Output group
                        self.write_group_content(&mut content, func_groups[group_idx]);
                        group_idx += 1;
                    }
                }
                (Some(_), None) => {
                    content.push_str(&functions[func_idx].combined_source());
                    content.push_str("\n\n");
                    func_idx += 1;
                }
                (None, Some(_)) => {
                    self.write_group_content(&mut content, func_groups[group_idx]);
                    group_idx += 1;
                }
                (None, None) => break,
            }
        }

        // Variables (ungrouped)
        for def in &variables {
            content.push_str(&def.combined_source());
            content.push_str("\n\n");
        }

        // Trim trailing whitespace
        content.trim_end().to_string() + "\n"
    }

    /// Write a group's content to the output
    fn write_group_content(&self, content: &mut String, group: &DefinitionGrouping) {
        for (i, name) in group.iter().enumerate() {
            if let Some(def) = self.definitions.get(name) {
                content.push_str(&def.combined_source());
                if i < group.len() - 1 {
                    content.push_str("\n\n");
                }
            }
        }
        content.push_str("\n\n");
    }

    /// Get combined source for a group (for editing)
    /// Returns None if the definition is not in a group (use get_definition instead)
    pub fn get_group_source(&self, name: &str) -> Option<String> {
        // Only return grouped source if actually in a group
        let group = self.get_group_for(name)?;

        let mut sources: Vec<String> = Vec::new();

        // Include together directive
        sources.push(format!("# together {}", group.join(" ")));
        sources.push(String::new()); // blank line

        for n in group {
            if let Some(def) = self.definitions.get(n) {
                sources.push(def.combined_source());
            }
        }

        if sources.len() <= 2 {
            // Only has directive and blank line, no actual definitions
            None
        } else {
            Some(sources.join("\n\n"))
        }
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
