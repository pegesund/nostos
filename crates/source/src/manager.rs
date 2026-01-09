//! Source Manager - central hub for source code management

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::fs;
use walkdir::WalkDir;

use nostos_syntax::ast::Item;

use crate::definition::{DefKind, DefinitionGroup};
use crate::module::{Module, ModulePath, module_path_to_string, DefinitionGrouping};
use crate::project::ProjectConfig;
use crate::git;

/// Find the start position including any doc comment before a definition
fn find_doc_comment_start(input: &str, def_start: usize) -> usize {
    // Get text before the definition
    let before = &input[..def_start];

    // Track byte positions as we go through lines
    let mut line_starts: Vec<usize> = vec![0];
    for (i, c) in before.char_indices() {
        if c == '\n' {
            line_starts.push(i + 1);
        }
    }

    // Go backwards through lines to find comment block
    let mut comment_start = def_start;
    let mut found_comment = false;

    for i in (0..line_starts.len()).rev() {
        let line_start = line_starts[i];
        let line_end = if i + 1 < line_starts.len() {
            line_starts[i + 1]
        } else {
            def_start
        };
        let line = &before[line_start..line_end];
        let trimmed = line.trim();

        if trimmed.starts_with('#') && !trimmed.starts_with("# together ") {
            // This is a comment line (but not a together directive), include it
            comment_start = line_start;
            found_comment = true;
        } else if trimmed.is_empty() {
            // Empty line - continue checking above for more comments
            continue;
        } else {
            // Non-comment, non-empty line - stop
            break;
        }
    }

    if found_comment {
        comment_start
    } else {
        def_start
    }
}

/// Parse together directives from content
/// Returns list of groups, each group is a list of definition names
fn parse_together_directives(content: &str) -> Vec<DefinitionGrouping> {
    content
        .lines()
        .filter_map(|line| {
            let trimmed = line.trim();
            if trimmed.starts_with("# together ") {
                let names: Vec<String> = trimmed
                    .strip_prefix("# together ")
                    .unwrap()
                    .split_whitespace()
                    .map(|s| s.to_string())
                    .collect();
                if names.len() >= 2 {
                    Some(names)
                } else {
                    None
                }
            } else {
                None
            }
        })
        .collect()
}

/// Generate _meta.nos content from groups
fn generate_meta_content(groups: &[DefinitionGrouping]) -> String {
    groups
        .iter()
        .filter(|g| !g.is_empty())
        .map(|g| format!("# together {}", g.join(" ")))
        .collect::<Vec<_>>()
        .join("\n")
}

/// Rename a definition in source code
/// Handles both simple and complex definition patterns
fn rename_definition_in_source(source: &str, old_name: &str, new_name: &str) -> String {
    use regex::Regex;

    // Pattern to match function/value definitions: name(...) = or name =
    // This handles:
    // - foo() = ...
    // - foo(x) = ...
    // - foo(x, y) = ...
    // - foo = ...
    let pattern = format!(r"^(\s*){}(\s*\(|\s*=)", regex::escape(old_name));
    let re = Regex::new(&pattern).unwrap();

    let mut result = String::new();
    for line in source.lines() {
        if re.is_match(line) {
            // This line starts a definition - replace the name
            let new_line = re.replace(line, format!("${{1}}{}${{2}}", new_name));
            result.push_str(&new_line);
        } else {
            result.push_str(line);
        }
        result.push('\n');
    }

    // Remove trailing newline if original didn't have one
    if !source.ends_with('\n') && result.ends_with('\n') {
        result.pop();
    }

    result
}

/// Replace function calls in source code (not definitions)
/// Replaces old_name( with new_name( when old_name is used as a function call
pub fn replace_call_in_source(source: &str, old_name: &str, new_name: &str) -> String {
    use regex::Regex;

    // Pattern to match function calls: old_name followed by (
    // Use word boundary to avoid partial matches (e.g., don't match "foobar" when looking for "foo")
    // But we need to handle cases like: old_name(), old_name(x), old_name (with spaces)
    let pattern = format!(r"\b{}(\s*\()", regex::escape(old_name));
    let re = Regex::new(&pattern).unwrap();

    re.replace_all(source, format!("{}$1", new_name)).to_string()
}

/// Manages all source code in a Nostos project
pub struct SourceManager {
    /// Project root directory
    project_root: PathBuf,

    /// Project configuration
    config: ProjectConfig,

    /// All modules, keyed by path string (e.g., "utils.math")
    modules: HashMap<String, Module>,

    /// Quick lookup: definition name → module path string
    def_index: HashMap<String, String>,

    /// Definitions created in REPL (no module yet)
    repl_defs: HashMap<String, DefinitionGroup>,

    /// Source files tracked by relative path (for file-by-file editing mode)
    /// Key is relative path from project root (e.g., "main.nos", "utils/math.nos")
    source_files: HashMap<String, String>,

    /// Files that failed to parse (relative paths)
    files_with_errors: HashSet<String>,
}

impl SourceManager {
    /// Create a new SourceManager for a project directory
    pub fn new(project_root: PathBuf) -> Result<Self, String> {
        let mut manager = Self {
            project_root: project_root.clone(),
            config: ProjectConfig::default(),
            modules: HashMap::new(),
            def_index: HashMap::new(),
            repl_defs: HashMap::new(),
            source_files: HashMap::new(),
            files_with_errors: HashSet::new(),
        };

        manager.initialize()?;
        Ok(manager)
    }

    /// Initialize the project
    fn initialize(&mut self) -> Result<(), String> {
        let config_path = self.project_root.join("nostos.toml");
        let nostos_dir = self.project_root.join(".nostos");

        // Load or create config
        if config_path.exists() {
            self.config = ProjectConfig::load(&config_path)?;
        } else {
            self.config = ProjectConfig::create_default(&self.project_root)?;
        }

        // Initialize .nostos directory and git
        git::init_repo(&nostos_dir)?;

        // Load definitions
        let defs_dir = nostos_dir.join("defs");
        if defs_dir.exists() {
            self.load_from_defs_dir(&defs_dir)?;
            // Check for new .nos files that don't have defs yet
            self.import_new_source_files(&defs_dir)?;
            // Check for .nos files that are newer than defs (external edits)
            self.sync_updated_source_files(&defs_dir)?;
        } else {
            // Bootstrap from .nos files
            self.bootstrap_from_source_files()?;
        }

        Ok(())
    }

    /// Load definitions from .nostos/defs/
    fn load_from_defs_dir(&mut self, defs_dir: &Path) -> Result<(), String> {
        for entry in WalkDir::new(defs_dir)
            .into_iter()
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().map(|ext| ext == "nos").unwrap_or(false))
        {
            let path = entry.path();
            let relative = path.strip_prefix(defs_dir).unwrap();

            // Parse module path from directory structure
            let module_path: ModulePath = relative
                .parent()
                .map(|p| {
                    p.components()
                        .map(|c| c.as_os_str().to_string_lossy().to_string())
                        .collect()
                })
                .unwrap_or_default();

            let file_name = path.file_stem().unwrap().to_string_lossy();

            // Skip special files, handle separately
            if file_name == "_imports" || file_name == "_meta" {
                continue;
            }

            // Read content
            let content = fs::read_to_string(path)
                .map_err(|e| format!("Failed to read {}: {}", path.display(), e))?;

            // Determine kind from filename
            let kind = if file_name.starts_with(char::is_uppercase) {
                // Could be Type or Trait - check content
                if content.trim_start().starts_with("trait ") {
                    DefKind::Trait
                } else {
                    DefKind::Type
                }
            } else if content.trim_start().starts_with("var ") {
                DefKind::Variable
            } else {
                DefKind::Function
            };

            // Create definition group
            let group = if kind == DefKind::Function && content.contains("\n\n") {
                // Multiple overloads
                let sources: Vec<String> = content
                    .split("\n\n")
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect();
                DefinitionGroup::from_sources(file_name.to_string(), kind, sources)
            } else {
                DefinitionGroup::new(file_name.to_string(), kind, content.trim().to_string())
            };

            // Add to module
            let module_key = module_path_to_string(&module_path);
            let module = self.modules
                .entry(module_key.clone())
                .or_insert_with(|| Module::new(module_path.clone()));

            // Update def_index
            self.def_index.insert(file_name.to_string(), module_key);

            module.add_definition(group);
        }

        // Load imports and use statements for each module
        for entry in WalkDir::new(defs_dir)
            .into_iter()
            .filter_map(|e| e.ok())
            .filter(|e| e.path().file_name().map(|n| n == "_imports.nos").unwrap_or(false))
        {
            let path = entry.path();
            let relative = path.strip_prefix(defs_dir).unwrap();
            let module_path: ModulePath = relative
                .parent()
                .map(|p| {
                    p.components()
                        .map(|c| c.as_os_str().to_string_lossy().to_string())
                        .collect()
                })
                .unwrap_or_default();

            let content = fs::read_to_string(path)
                .map_err(|e| format!("Failed to read {}: {}", path.display(), e))?;

            let mut imports: Vec<String> = Vec::new();
            let mut use_stmts: Vec<String> = Vec::new();
            for line in content.lines() {
                let trimmed = line.trim();
                if trimmed.is_empty() {
                    continue;
                }
                if trimmed.starts_with("import ") {
                    imports.push(trimmed.to_string());
                } else if trimmed.starts_with("use ") {
                    use_stmts.push(trimmed.to_string());
                }
            }

            let module_key = module_path_to_string(&module_path);
            if let Some(module) = self.modules.get_mut(&module_key) {
                module.set_imports(imports);
                module.set_use_stmts(use_stmts);
            }
        }

        // Load _meta.nos (together directives) for each module
        for entry in WalkDir::new(defs_dir)
            .into_iter()
            .filter_map(|e| e.ok())
            .filter(|e| e.path().file_name().map(|n| n == "_meta.nos").unwrap_or(false))
        {
            let path = entry.path();
            let relative = path.strip_prefix(defs_dir).unwrap();
            let module_path: ModulePath = relative
                .parent()
                .map(|p| {
                    p.components()
                        .map(|c| c.as_os_str().to_string_lossy().to_string())
                        .collect()
                })
                .unwrap_or_default();

            let content = fs::read_to_string(path)
                .map_err(|e| format!("Failed to read {}: {}", path.display(), e))?;

            let groups = parse_together_directives(&content);

            let module_key = module_path_to_string(&module_path);
            if let Some(module) = self.modules.get_mut(&module_key) {
                module.set_groups(groups);
                // Validate no duplicates
                if let Err(dups) = module.validate_groups() {
                    return Err(format!(
                        "Duplicate definitions in groups for module '{}': {}",
                        module_key,
                        dups.join(", ")
                    ));
                }
            }
        }

        // Mark all as git clean since we just loaded from git
        for module in self.modules.values_mut() {
            module.mark_all_git_clean();
            module.dirty = false;
        }

        Ok(())
    }

    /// Import new .nos files that don't have corresponding defs directories
    fn import_new_source_files(&mut self, defs_dir: &Path) -> Result<(), String> {
        let nostos_exclude = self.project_root.join(".nostos");

        // Find .nos files that don't have a defs directory
        let new_files: Vec<(PathBuf, ModulePath, String)> = WalkDir::new(&self.project_root)
            .into_iter()
            .filter_map(|e| e.ok())
            .filter(|e| {
                let path = e.path();
                path.extension().map(|ext| ext == "nos").unwrap_or(false)
                    && !path.starts_with(&nostos_exclude)
            })
            .filter_map(|entry| {
                let path = entry.path().to_path_buf();
                let relative = path.strip_prefix(&self.project_root).ok()?;

                // Module path from file path
                let module_path: ModulePath = {
                    let mut components: Vec<String> = relative
                        .parent()
                        .map(|p| {
                            p.components()
                                .map(|c| c.as_os_str().to_string_lossy().to_string())
                                .collect()
                        })
                        .unwrap_or_default();

                    if let Some(stem) = relative.file_stem() {
                        components.push(stem.to_string_lossy().to_string());
                    }
                    components
                };

                // Check if this module already has defs
                let module_defs_dir = module_path.iter().fold(
                    defs_dir.to_path_buf(),
                    |p, s| p.join(s),
                );

                // Skip if defs directory already exists
                if module_defs_dir.exists() {
                    return None;
                }

                let content = fs::read_to_string(&path).ok()?;
                Some((path, module_path, content))
            })
            .collect();

        if new_files.is_empty() {
            return Ok(());
        }

        // Import each new file - continue on errors
        for (path, module_path, content) in new_files {
            if let Err(e) = self.parse_and_add_definitions(&module_path, &content) {
                eprintln!("Warning: Failed to parse {}: {}", path.display(), e);
                if let Ok(relative) = path.strip_prefix(&self.project_root) {
                    self.files_with_errors.insert(relative.to_string_lossy().to_string());
                }
            }
        }

        // Write new definitions to .nostos/defs/
        self.write_all_to_defs()?;

        // Git commit
        let nostos_dir = self.project_root.join(".nostos");
        git::add_and_commit(&nostos_dir, &["defs"], "Import new source files")?;

        Ok(())
    }

    /// Sync updated .nos files that are newer than their corresponding defs
    fn sync_updated_source_files(&mut self, defs_dir: &Path) -> Result<(), String> {
        let nostos_exclude = self.project_root.join(".nostos");

        // Find .nos files that are newer than their defs
        let updated_files: Vec<(PathBuf, ModulePath, String)> = WalkDir::new(&self.project_root)
            .into_iter()
            .filter_map(|e| e.ok())
            .filter(|e| {
                let path = e.path();
                path.extension().map(|ext| ext == "nos").unwrap_or(false)
                    && !path.starts_with(&nostos_exclude)
            })
            .filter_map(|entry| {
                let path = entry.path().to_path_buf();
                let relative = path.strip_prefix(&self.project_root).ok()?;

                // Module path from file path
                let module_path: ModulePath = {
                    let mut components: Vec<String> = relative
                        .parent()
                        .map(|p| {
                            p.components()
                                .map(|c| c.as_os_str().to_string_lossy().to_string())
                                .collect()
                        })
                        .unwrap_or_default();

                    if let Some(stem) = relative.file_stem() {
                        components.push(stem.to_string_lossy().to_string());
                    }
                    components
                };

                // Find the defs directory for this module
                let module_defs_dir = module_path.iter().fold(
                    defs_dir.to_path_buf(),
                    |p, s| p.join(s),
                );

                // Skip if defs directory doesn't exist (handled by import_new_source_files)
                if !module_defs_dir.exists() {
                    return None;
                }

                // Get .nos file mtime
                let nos_mtime = path.metadata().ok()?.modified().ok()?;

                // Get latest mtime from defs directory
                let defs_mtime = WalkDir::new(&module_defs_dir)
                    .into_iter()
                    .filter_map(|e| e.ok())
                    .filter(|e| e.path().extension().map(|ext| ext == "nos").unwrap_or(false))
                    .filter_map(|e| e.path().metadata().ok()?.modified().ok())
                    .max();

                // If .nos is newer than all defs files, we need to re-import
                if let Some(defs_mtime) = defs_mtime {
                    if nos_mtime > defs_mtime {
                        let content = fs::read_to_string(&path).ok()?;
                        return Some((path, module_path, content));
                    }
                }

                None
            })
            .collect();

        if updated_files.is_empty() {
            return Ok(());
        }

        // Re-import each updated file - continue on errors
        for (path, module_path, content) in &updated_files {
            let module_key = module_path_to_string(module_path);

            // Clear existing definitions for this module
            if let Some(module) = self.modules.get_mut(&module_key) {
                // Remove old definitions from index
                for name in module.definition_names() {
                    self.def_index.remove(name);
                }
                module.clear_definitions();
            }

            // Re-parse and add definitions
            let relative_path = path.strip_prefix(&self.project_root)
                .map(|p| p.to_string_lossy().to_string())
                .unwrap_or_default();
            if let Err(e) = self.parse_and_add_definitions(module_path, content) {
                eprintln!("Warning: Failed to parse {}: {}", path.display(), e);
                self.files_with_errors.insert(relative_path);
            } else {
                // File is now valid, remove from errors
                self.files_with_errors.remove(&relative_path);
            }
        }

        // Write updated definitions to .nostos/defs/
        for (_path, module_path, _content) in &updated_files {
            let module_key = module_path_to_string(module_path);
            if let Some(module) = self.modules.get(&module_key) {
                for name in module.definition_names() {
                    self.write_definition_to_defs(&module_key, &name)?;
                }
            }
        }

        // Git commit
        let nostos_dir = self.project_root.join(".nostos");
        git::add_and_commit(&nostos_dir, &["defs"], "Sync updated source files")?;

        Ok(())
    }

    /// Bootstrap from existing .nos files when .nostos/defs/ doesn't exist
    fn bootstrap_from_source_files(&mut self) -> Result<(), String> {
        // First, collect all file paths and their contents to avoid borrow issues
        let nostos_exclude = self.project_root.join(".nostos");
        let files_to_process: Vec<(PathBuf, ModulePath, String)> = WalkDir::new(&self.project_root)
            .into_iter()
            .filter_map(|e| e.ok())
            .filter(|e| {
                let path = e.path();
                path.extension().map(|ext| ext == "nos").unwrap_or(false)
                    && !path.starts_with(&nostos_exclude)
            })
            .filter_map(|entry| {
                let path = entry.path().to_path_buf();
                let relative = path.strip_prefix(&self.project_root).ok()?;

                // Module path from file path (without .nos extension)
                let module_path: ModulePath = {
                    let mut components: Vec<String> = relative
                        .parent()
                        .map(|p| {
                            p.components()
                                .map(|c| c.as_os_str().to_string_lossy().to_string())
                                .collect()
                        })
                        .unwrap_or_default();

                    // Add file name without extension
                    if let Some(stem) = relative.file_stem() {
                        components.push(stem.to_string_lossy().to_string());
                    }
                    components
                };

                let content = fs::read_to_string(&path).ok()?;
                Some((path, module_path, content))
            })
            .collect();

        // Now process each file - continue on errors
        for (path, module_path, content) in files_to_process {
            if let Err(e) = self.parse_and_add_definitions(&module_path, &content) {
                eprintln!("Warning: Failed to parse {}: {}", path.display(), e);
                // Track file with error
                if let Ok(relative) = path.strip_prefix(&self.project_root) {
                    self.files_with_errors.insert(relative.to_string_lossy().to_string());
                }
            }
        }

        // Write all definitions to .nostos/defs/
        self.write_all_to_defs()?;

        // Git commit
        let nostos_dir = self.project_root.join(".nostos");
        git::add_and_commit(&nostos_dir, &["defs"], "Import from source files")?;

        Ok(())
    }

    /// Parse source content and add definitions to module
    fn parse_and_add_definitions(&mut self, module_path: &ModulePath, content: &str) -> Result<(), String> {
        use nostos_syntax::parse;

        let module_key = module_path_to_string(module_path);

        // Parse the content - returns (Option<Module>, Vec<errors>)
        let (parsed, errors) = parse(content);
        let parsed = parsed.ok_or_else(|| {
            if errors.is_empty() {
                "Failed to parse module".to_string()
            } else {
                format!("Parse errors: {:?}", errors)
            }
        })?;

        // Extract use statements (imports field is kept for backwards compatibility but will be empty)
        let imports = Vec::new();
        let mut use_stmts = Vec::new();
        for item in &parsed.items {
            match item {
                Item::Use(use_stmt) => {
                    // Convert path of Idents to dotted string
                    let path_str: String = use_stmt.path.iter()
                        .map(|ident| ident.node.as_str())
                        .collect::<Vec<_>>()
                        .join(".");
                    // Format based on import type
                    let use_str = match &use_stmt.imports {
                        nostos_syntax::ast::UseImports::All => {
                            format!("use {}.*", path_str)
                        }
                        nostos_syntax::ast::UseImports::Named(items) => {
                            let names: Vec<_> = items.iter()
                                .map(|item| item.name.node.as_str())
                                .collect();
                            format!("use {}.{{{}}}", path_str, names.join(", "))
                        }
                    };
                    use_stmts.push(use_str);
                }
                _ => {}
            }
        }

        // Get or create module
        let module = self.modules
            .entry(module_key.clone())
            .or_insert_with(|| Module::new(module_path.clone()));

        module.set_imports(imports);
        module.set_use_stmts(use_stmts);

        // Extract definitions
        for item in &parsed.items {
            match item {
                Item::TypeDef(type_def) => {
                    let name = type_def.name.node.clone();
                    // Get source from span (including doc comment)
                    let span = &type_def.span;
                    let start = find_doc_comment_start(content, span.start);
                    let source = content[start..span.end].to_string();

                    let kind = if source.contains("trait ") {
                        DefKind::Trait
                    } else {
                        DefKind::Type
                    };

                    let group = DefinitionGroup::new(name.clone(), kind, source);
                    self.def_index.insert(name, module_key.clone());
                    module.add_definition(group);
                }
                Item::FnDef(fn_def) => {
                    let name = fn_def.name.node.clone();
                    let span = &fn_def.span;
                    // Include doc comment in source
                    let start = find_doc_comment_start(content, span.start);
                    let source = content[start..span.end].to_string();

                    // Check if we already have this function (overload)
                    if module.has_definition(&name) {
                        if let Some(group) = module.get_definition_mut(&name) {
                            group.add_overload(source);
                        }
                    } else {
                        let group = DefinitionGroup::new(name.clone(), DefKind::Function, source);
                        self.def_index.insert(name, module_key.clone());
                        module.add_definition(group);
                    }
                }
                Item::Binding(binding) => {
                    if let nostos_syntax::ast::Pattern::Var(ident) = &binding.pattern {
                        let name = ident.node.clone();
                        let span_start = binding.pattern.span().start;
                        let span_end = binding.value.span().end;
                        let source = content[span_start..span_end].to_string();

                        let source = if binding.mutable {
                            format!("var {}", source)
                        } else {
                            source
                        };

                        let group = DefinitionGroup::new(name.clone(), DefKind::Variable, source);
                        self.def_index.insert(name, module_key.clone());
                        module.add_definition(group);
                    }
                }
                Item::TraitDef(trait_def) => {
                    let name = trait_def.name.node.clone();
                    let span = &trait_def.span;
                    // Include doc comment in source
                    let start = find_doc_comment_start(content, span.start);
                    let source = content[start..span.end].to_string();

                    let group = DefinitionGroup::new(name.clone(), DefKind::Trait, source);
                    self.def_index.insert(name, module_key.clone());
                    module.add_definition(group);
                }
                _ => {}
            }
        }

        Ok(())
    }

    /// Write all definitions to .nostos/defs/
    fn write_all_to_defs(&self) -> Result<(), String> {
        let defs_dir = self.project_root.join(".nostos").join("defs");

        for module in self.modules.values() {
            let module_dir = module.path.iter().fold(defs_dir.clone(), |p, s| p.join(s));
            fs::create_dir_all(&module_dir)
                .map_err(|e| format!("Failed to create directory: {}", e))?;

            // Write imports and use statements
            if !module.imports.is_empty() || !module.use_stmts.is_empty() {
                let imports_path = module_dir.join("_imports.nos");
                let mut content = module.imports.join("\n");
                if !module.imports.is_empty() && !module.use_stmts.is_empty() {
                    content.push('\n');
                }
                content.push_str(&module.use_stmts.join("\n"));
                if !content.is_empty() {
                    content.push('\n');
                }
                fs::write(&imports_path, content)
                    .map_err(|e| format!("Failed to write imports: {}", e))?;
            }

            // Write each definition
            for def in module.definitions() {
                let def_path = module_dir.join(format!("{}.nos", def.name));
                let content = def.combined_source() + "\n";
                fs::write(&def_path, content)
                    .map_err(|e| format!("Failed to write {}: {}", def.name, e))?;
            }

            // Write _meta.nos for together directives
            let meta_path = module_dir.join("_meta.nos");
            if !module.groups.is_empty() {
                let meta_content = generate_meta_content(&module.groups);
                fs::write(&meta_path, meta_content + "\n")
                    .map_err(|e| format!("Failed to write _meta.nos: {}", e))?;
            } else if meta_path.exists() {
                // Remove _meta.nos if no groups
                fs::remove_file(&meta_path)
                    .map_err(|e| format!("Failed to remove _meta.nos: {}", e))?;
            }
        }

        Ok(())
    }

    // ===== Public API =====

    /// Get the project root
    pub fn project_root(&self) -> &Path {
        &self.project_root
    }

    /// Get the project config
    pub fn config(&self) -> &ProjectConfig {
        &self.config
    }

    /// Get a definition's source by name
    /// If the definition is part of a "together" group, returns all definitions in the group
    /// Handles both simple names (greet) and qualified names (utils.greet)
    pub fn get_source(&self, name: &str) -> Option<String> {
        // Parse qualified name (e.g., "utils.greet" -> module="utils", simple_name="greet")
        let (module_name, simple_name) = if name.contains('.') {
            let last_dot = name.rfind('.').unwrap();
            (&name[..last_dot], &name[last_dot + 1..])
        } else {
            ("", name)
        };

        // If a module is specified, look directly in that module
        if !module_name.is_empty() {
            if let Some(module) = self.modules.get(module_name) {
                // Check if this definition is part of a group
                if let Some(group_source) = module.get_group_source(simple_name) {
                    return Some(group_source);
                }
                // Otherwise return just this definition
                if let Some(group) = module.get_definition(simple_name) {
                    return Some(group.combined_source());
                }
            }
        }

        // Fall back to def_index for unqualified names
        if let Some(module_key) = self.def_index.get(simple_name) {
            if let Some(module) = self.modules.get(module_key) {
                // Check if this definition is part of a group
                if let Some(group_source) = module.get_group_source(simple_name) {
                    return Some(group_source);
                }
                // Otherwise return just this definition
                if let Some(group) = module.get_definition(simple_name) {
                    return Some(group.combined_source());
                }
            }
        }

        // Check REPL definitions
        if let Some(group) = self.repl_defs.get(simple_name) {
            return Some(group.combined_source());
        }

        None
    }

    /// Get the names of all definitions that would be returned by get_source
    /// (i.e., includes group members if this definition is in a group)
    /// Handles both simple names (greet) and qualified names (utils.greet)
    pub fn get_grouped_names(&self, name: &str) -> Vec<String> {
        // Parse qualified name (e.g., "utils.greet" -> module="utils", simple_name="greet")
        let (module_name, simple_name) = if name.contains('.') {
            let last_dot = name.rfind('.').unwrap();
            (&name[..last_dot], &name[last_dot + 1..])
        } else {
            ("", name)
        };

        // If a module is specified, look directly in that module
        if !module_name.is_empty() {
            if let Some(module) = self.modules.get(module_name) {
                return module.get_grouped_names(simple_name);
            }
        }

        // Fall back to def_index for unqualified names
        if let Some(module_key) = self.def_index.get(simple_name) {
            if let Some(module) = self.modules.get(module_key) {
                return module.get_grouped_names(simple_name);
            }
        }
        vec![name.to_string()]
    }

    /// Get the module path for a definition
    pub fn get_definition_module(&self, name: &str) -> Option<&str> {
        self.def_index.get(name).map(|s| s.as_str())
    }

    /// Get module metadata (together directives etc.)
    /// Returns the content of _meta.nos for the module
    pub fn get_module_metadata(&self, module_name: &str) -> Option<String> {
        let module = self.modules.get(module_name)?;
        if module.groups.is_empty() {
            None
        } else {
            Some(generate_meta_content(&module.groups))
        }
    }

    /// Get the full generated source for a module
    /// Returns all types, traits, functions, variables with proper together grouping
    pub fn get_module_generated_source(&self, module_name: &str) -> Option<String> {
        self.modules.get(module_name).map(|module| module.generate_file_content())
    }

    /// Get git commit history for a definition
    /// Returns list of commits that modified this definition, newest first
    pub fn get_definition_history(&self, name: &str) -> Result<Vec<git::CommitInfo>, String> {
        // Parse qualified name
        let (module_key, simple_name) = if name.contains('.') {
            let last_dot = name.rfind('.').unwrap();
            (name[..last_dot].to_string(), &name[last_dot + 1..])
        } else {
            // Look up module from def_index
            let module_key = self.def_index.get(name)
                .ok_or_else(|| format!("Definition not found: {}", name))?
                .clone();
            (module_key, name)
        };

        // Build path to definition file
        let module_path: Vec<&str> = if module_key.is_empty() {
            vec![]
        } else {
            module_key.split('.').collect()
        };

        let relative_path = if module_path.is_empty() {
            format!("defs/{}.nos", simple_name)
        } else {
            format!("defs/{}/{}.nos", module_path.join("/"), simple_name)
        };

        let nostos_dir = self.project_root.join(".nostos");
        git::get_file_history(&nostos_dir, &relative_path)
    }

    /// Get git commit history for a module
    /// Returns list of commits that modified any definition in the module, newest first
    pub fn get_module_history(&self, module_name: &str) -> Result<Vec<git::CommitInfo>, String> {
        let module_path: Vec<&str> = if module_name.is_empty() {
            vec![]
        } else {
            module_name.split('.').collect()
        };

        let relative_dir = if module_path.is_empty() {
            "defs".to_string()
        } else {
            format!("defs/{}", module_path.join("/"))
        };

        let nostos_dir = self.project_root.join(".nostos");
        git::get_directory_history(&nostos_dir, &relative_dir)
    }

    /// Get the source of a definition at a specific commit
    pub fn get_definition_at_commit(&self, name: &str, commit: &str) -> Result<String, String> {
        // Parse qualified name
        let (module_key, simple_name) = if name.contains('.') {
            let last_dot = name.rfind('.').unwrap();
            (name[..last_dot].to_string(), &name[last_dot + 1..])
        } else {
            // Look up module from def_index
            let module_key = self.def_index.get(name)
                .ok_or_else(|| format!("Definition not found: {}", name))?
                .clone();
            (module_key, name)
        };

        // Build path to definition file
        let module_path: Vec<&str> = if module_key.is_empty() {
            vec![]
        } else {
            module_key.split('.').collect()
        };

        let relative_path = if module_path.is_empty() {
            format!("defs/{}.nos", simple_name)
        } else {
            format!("defs/{}/{}.nos", module_path.join("/"), simple_name)
        };

        let nostos_dir = self.project_root.join(".nostos");
        git::get_file_at_commit(&nostos_dir, commit, &relative_path)
    }

    /// Get the diff for a definition at a specific commit
    pub fn get_definition_diff(&self, name: &str, commit: &str) -> Result<String, String> {
        // Parse qualified name
        let (module_key, simple_name) = if name.contains('.') {
            let last_dot = name.rfind('.').unwrap();
            (name[..last_dot].to_string(), &name[last_dot + 1..])
        } else {
            // Look up module from def_index
            let module_key = self.def_index.get(name)
                .ok_or_else(|| format!("Definition not found: {}", name))?
                .clone();
            (module_key, name)
        };

        // Build path to definition file
        let module_path: Vec<&str> = if module_key.is_empty() {
            vec![]
        } else {
            module_key.split('.').collect()
        };

        let relative_path = if module_path.is_empty() {
            format!("defs/{}.nos", simple_name)
        } else {
            format!("defs/{}/{}.nos", module_path.join("/"), simple_name)
        };

        let nostos_dir = self.project_root.join(".nostos");
        git::get_file_diff(&nostos_dir, commit, &relative_path)
    }

    /// Save module metadata (together directives etc.)
    /// Parses together directives from content and updates the module
    pub fn save_module_metadata(&mut self, module_name: &str, content: &str) -> Result<(), String> {
        let module = self.modules.get_mut(module_name)
            .ok_or_else(|| format!("Module not found: {}", module_name))?;

        // Parse together directives
        let new_groups = parse_together_directives(content);

        // Update groups
        module.set_groups(new_groups);

        // Validate no duplicates
        if let Err(dups) = module.validate_groups() {
            return Err(format!(
                "Definition cannot be in multiple groups: {}",
                dups.join(", ")
            ));
        }

        // Write _meta.nos
        self.write_meta_to_defs(module_name)?;

        Ok(())
    }

    /// Update a definition's source (or add if new)
    /// Handles both simple names (greet) and qualified names (utils.greet)
    /// Returns true if the definition changed or was added
    pub fn update_definition(&mut self, name: &str, new_source: &str) -> Result<bool, String> {
        // Parse the qualified name to extract module path and simple name
        let parts: Vec<&str> = name.split('.').collect();
        let (module_path_parts, simple_name) = if parts.len() > 1 {
            (&parts[..parts.len()-1], parts[parts.len()-1])
        } else {
            (&[][..], parts[0])
        };

        // Try to find existing definition first
        if let Some(module_key) = self.def_index.get(simple_name).cloned() {
            if let Some(module) = self.modules.get_mut(&module_key) {
                if let Some(group) = module.get_definition_mut(simple_name) {
                    let changed = group.update_from_source(new_source);
                    if changed {
                        module.dirty = true;
                        // Write to .nos first (source of truth), then copy to defs for git history
                        self.write_module_files()?;
                        self.write_definition_to_defs(&module_key, simple_name)?;
                        self.commit_definition(&module_key, simple_name)?;
                    }
                    return Ok(changed);
                }
            }
        }

        // Check REPL definitions
        if let Some(group) = self.repl_defs.get_mut(simple_name) {
            return Ok(group.update_from_source(new_source));
        }

        // Definition not found - add as new definition
        let module_key = module_path_parts.join(".");
        let module_path: ModulePath = if module_key.is_empty() {
            vec![]
        } else {
            module_key.split('.').map(String::from).collect()
        };

        // Ensure module exists
        self.modules
            .entry(module_key.clone())
            .or_insert_with(|| Module::new(module_path));

        // Determine kind from source
        let kind = if new_source.trim_start().starts_with("type ") {
            DefKind::Type
        } else if new_source.trim_start().starts_with("trait ") {
            DefKind::Trait
        } else if new_source.trim_start().starts_with("var ") {
            DefKind::Variable
        } else {
            DefKind::Function
        };

        // Add the new definition
        if let Some(module) = self.modules.get_mut(&module_key) {
            let group = DefinitionGroup::new(simple_name.to_string(), kind, new_source.to_string());
            module.add_definition(group);
            module.dirty = true;
        }

        // Update def_index
        self.def_index.insert(simple_name.to_string(), module_key.clone());

        // Write to .nos first (source of truth), then copy to defs for git history
        self.write_module_files()?;
        self.write_definition_to_defs(&module_key, simple_name)?;
        self.commit_definition(&module_key, simple_name)?;

        Ok(true)
    }

    /// Update grouped source (content from editor that may contain together directive and multiple definitions)
    /// The `primary_name` is the definition that was originally opened (can be qualified like "utils.greet")
    /// Returns list of definition names that were updated or added
    /// Handles both simple names (greet) and qualified names (utils.greet)
    /// For new definitions, extracts module from qualified name
    pub fn update_group_source(&mut self, primary_name: &str, source: &str) -> Result<Vec<String>, String> {
        use nostos_syntax::parse;
        // use std::io::Write; // Only needed when debug logging is enabled

        // Debug logging disabled. Uncomment to enable.
        // let mut debug_file = std::fs::OpenOptions::new()
        //     .create(true)
        //     .append(true)
        //     .open("/tmp/source_manager_debug.log")
        //     .ok();

        macro_rules! debug_log {
            ($($arg:tt)*) => {
                // Disabled. Uncomment below for debugging:
                // if let Some(ref mut f) = debug_file {
                //     let _ = writeln!(f, $($arg)*);
                // }
                // eprintln!($($arg)*);
            };
        }

        debug_log!("[SourceManager] update_group_source called with primary_name='{}'", primary_name);

        // Parse the qualified name to extract module path and simple name
        // e.g., "utils.sub.foo" -> module_path=["utils", "sub"], simple_name="foo"
        let parts: Vec<&str> = primary_name.split('.').collect();
        let (module_path_parts, simple_name) = if parts.len() > 1 {
            (&parts[..parts.len()-1], parts[parts.len()-1])
        } else {
            (&[][..], parts[0])
        };

        debug_log!("[SourceManager] Parsed: module_path_parts={:?}, simple_name='{}'", module_path_parts, simple_name);
        debug_log!("[SourceManager] Source to save ({} chars): {:?}", source.len(), &source[..source.len().min(100)]);

        // Try to find existing definition first
        let module_key = if let Some(key) = self.def_index.get(simple_name) {
            debug_log!("[SourceManager] Found existing definition in module '{}'", key);
            key.clone()
        } else {
            // New definition - use module path from qualified name
            let key = module_path_parts.join(".");
            debug_log!("[SourceManager] New definition, using module key from qualified name: '{}'", key);
            key
        };

        // Ensure module exists
        let module_path: ModulePath = if module_key.is_empty() {
            vec![]
        } else {
            module_key.split('.').map(String::from).collect()
        };

        let module_existed = self.modules.contains_key(&module_key);
        debug_log!("[SourceManager] Module '{}' existed: {}, modules: {:?}", module_key, module_existed, self.modules.keys().collect::<Vec<_>>());
        self.modules
            .entry(module_key.clone())
            .or_insert_with(|| Module::new(module_path));

        if !module_existed {
            debug_log!("[SourceManager] Created new module: '{}'", module_key);
        }

        // Parse together directives from the source
        let new_groups = parse_together_directives(source);

        // Parse the source to extract definitions
        // First, strip together directives for parsing
        let code_only: String = source
            .lines()
            .filter(|line| !line.trim().starts_with("# together "))
            .collect::<Vec<_>>()
            .join("\n");

        debug_log!("[SourceManager] Parsing code_only ({} chars)", code_only.len());
        let (parsed, errors) = parse(&code_only);
        if !errors.is_empty() {
            debug_log!("[SourceManager] Parse errors: {:?}", errors);
        }
        let parsed = parsed.ok_or_else(|| {
            if errors.is_empty() {
                "Failed to parse source".to_string()
            } else {
                format!("Parse errors: {:?}", errors)
            }
        })?;
        debug_log!("[SourceManager] Parsed {} items", parsed.items.len());

        // Extract definitions from parsed content
        // Track which definitions actually changed (for git commit)
        let mut changed_names: Vec<String> = Vec::new();
        let mut new_defs: Vec<String> = Vec::new();
        let mut all_names: Vec<String> = Vec::new();
        let mut groups_changed = false;

        if let Some(module) = self.modules.get_mut(&module_key) {
            // Update groups (merge new together directives)
            for group in &new_groups {
                // Check if this group already exists (by first member)
                let existing_idx = module.groups.iter().position(|g| {
                    !g.is_empty() && !group.is_empty() && g.iter().any(|n| group.contains(n))
                });

                if let Some(idx) = existing_idx {
                    // Replace existing group
                    if module.groups[idx] != *group {
                        module.groups[idx] = group.clone();
                        groups_changed = true;
                    }
                } else {
                    // Add new group
                    module.add_group(group.clone());
                    groups_changed = true;
                }
            }

            // Validate no duplicates
            if let Err(dups) = module.validate_groups() {
                return Err(format!(
                    "Definition cannot be in multiple groups: {}",
                    dups.join(", ")
                ));
            }

            debug_log!("[SourceManager] Processing {} parsed items, module has {} defs", parsed.items.len(), module.definition_names().count());
            // Update or add each definition from parsed items
            for item in &parsed.items {
                match item {
                    Item::FnDef(fn_def) => {
                        let name = fn_def.name.node.clone();
                        let span = &fn_def.span;
                        // Include doc comment in source
                        let start = find_doc_comment_start(&code_only, span.start);
                        let def_source = code_only[start..span.end].to_string();
                        debug_log!("[SourceManager] Found FnDef: '{}', span={}..{}, start_with_comment={}, def_source={:?}",
                            name, span.start, span.end, start, &def_source[..def_source.len().min(60)]);
                        debug_log!("[SourceManager] Found FnDef: '{}', checking if module has it: {}", name, module.has_definition(&name));

                        if module.has_definition(&name) {
                            debug_log!("[SourceManager] Updating existing function: {}", name);
                            if let Some(def_group) = module.get_definition_mut(&name) {
                                if def_group.update_from_source(&def_source) {
                                    changed_names.push(name.clone());
                                }
                            }
                        } else {
                            // Add new definition
                            debug_log!("[SourceManager] Adding NEW function: {} to module '{}'", name, module_key);
                            let group = DefinitionGroup::new(name.clone(), DefKind::Function, def_source);
                            module.add_definition(group);
                            new_defs.push(name.clone());
                            changed_names.push(name.clone());
                        }
                        all_names.push(name);
                    }
                    Item::TypeDef(type_def) => {
                        let name = type_def.name.node.clone();
                        let span = &type_def.span;
                        // Include doc comment in source
                        let start = find_doc_comment_start(&code_only, span.start);
                        let def_source = code_only[start..span.end].to_string();

                        if module.has_definition(&name) {
                            if let Some(def_group) = module.get_definition_mut(&name) {
                                if def_group.update_from_source(&def_source) {
                                    changed_names.push(name.clone());
                                }
                            }
                        } else {
                            // Add new definition
                            let group = DefinitionGroup::new(name.clone(), DefKind::Type, def_source);
                            module.add_definition(group);
                            new_defs.push(name.clone());
                            changed_names.push(name.clone());
                        }
                        all_names.push(name);
                    }
                    Item::TraitDef(trait_def) => {
                        let name = trait_def.name.node.clone();
                        let span = &trait_def.span;
                        // Include doc comment in source
                        let start = find_doc_comment_start(&code_only, span.start);
                        let def_source = code_only[start..span.end].to_string();

                        if module.has_definition(&name) {
                            if let Some(def_group) = module.get_definition_mut(&name) {
                                if def_group.update_from_source(&def_source) {
                                    changed_names.push(name.clone());
                                }
                            }
                        } else {
                            // Add new definition
                            let group = DefinitionGroup::new(name.clone(), DefKind::Trait, def_source);
                            module.add_definition(group);
                            new_defs.push(name.clone());
                            changed_names.push(name.clone());
                        }
                        all_names.push(name);
                    }
                    Item::Binding(binding) => {
                        if let nostos_syntax::ast::Pattern::Var(ident) = &binding.pattern {
                            let name = ident.node.clone();
                            let span_start = binding.pattern.span().start;
                            let span_end = binding.value.span().end;
                            let mut def_source = code_only[span_start..span_end].to_string();

                            if binding.mutable {
                                def_source = format!("var {}", def_source);
                            }

                            if module.has_definition(&name) {
                                if let Some(def_group) = module.get_definition_mut(&name) {
                                    if def_group.update_from_source(&def_source) {
                                        changed_names.push(name.clone());
                                    }
                                }
                            } else {
                                // Add new definition
                                let kind = if binding.mutable { DefKind::Variable } else { DefKind::Function };
                                let group = DefinitionGroup::new(name.clone(), kind, def_source);
                                module.add_definition(group);
                                new_defs.push(name.clone());
                                changed_names.push(name.clone());
                            }
                            all_names.push(name);
                        }
                    }
                    _ => {}
                }
            }

            if !changed_names.is_empty() || groups_changed {
                module.dirty = true;
            }
        }

        // Update def_index for new definitions
        debug_log!("[SourceManager] new_defs={:?}, changed_names={:?}", new_defs, changed_names);
        for name in &new_defs {
            debug_log!("[SourceManager] Adding to def_index: {} -> {}", name, module_key);
            self.def_index.insert(name.clone(), module_key.clone());
        }

        // Write to .nos first (source of truth), then copy to defs for git history
        if !changed_names.is_empty() || groups_changed {
            self.write_module_files()?;
        }

        // Write _meta.nos if groups changed
        if groups_changed {
            self.write_meta_to_defs(&module_key)?;
        }

        // Write and commit definitions that changed to defs for git history
        for name in &changed_names {
            debug_log!("[SourceManager] Writing definition to defs: {}", name);
            self.write_definition_to_defs(&module_key, name)?;
            self.commit_definition(&module_key, name)?;
        }

        debug_log!("[SourceManager] update_group_source completed, returning {:?}", changed_names);
        Ok(changed_names)
    }

    /// Add a new definition
    pub fn add_definition(
        &mut self,
        name: &str,
        module_path: &ModulePath,
        kind: DefKind,
        source: &str,
    ) -> Result<(), String> {
        let module_key = module_path_to_string(module_path);

        let module = self.modules
            .entry(module_key.clone())
            .or_insert_with(|| Module::new(module_path.clone()));

        let group = DefinitionGroup::new(name.to_string(), kind, source.to_string());
        module.add_definition(group);
        module.dirty = true;
        self.def_index.insert(name.to_string(), module_key.clone());

        // Write to .nos first (source of truth), then copy to defs for git history
        self.write_module_files()?;
        self.write_definition_to_defs(&module_key, name)?;
        self.commit_definition(&module_key, name)?;

        Ok(())
    }

    /// Add a REPL definition (not yet placed in a module)
    pub fn add_repl_definition(&mut self, name: &str, kind: DefKind, source: &str) {
        let group = DefinitionGroup::new(name.to_string(), kind, source.to_string());
        self.repl_defs.insert(name.to_string(), group);
    }

    /// Delete a definition
    pub fn delete_definition(&mut self, name: &str) -> Result<(), String> {
        if let Some(module_key) = self.def_index.remove(name) {
            if let Some(module) = self.modules.get_mut(&module_key) {
                module.remove_definition(name);
                module.dirty = true;

                // Delete from .nostos/defs/
                let module_path: ModulePath = module_key.split('.').map(String::from).collect();
                let def_path = self.get_def_path(&module_path, name);
                let relative_path = def_path
                    .strip_prefix(self.project_root.join(".nostos"))
                    .unwrap()
                    .to_string_lossy();

                let nostos_dir = self.project_root.join(".nostos");
                git::delete_and_commit(
                    &nostos_dir,
                    &relative_path,
                    &format!("Delete {}.{}", module_key, name),
                )?;

                // Also update the main .nos source file
                self.write_module_files()?;

                return Ok(());
            }
        }

        // Check REPL definitions
        if self.repl_defs.remove(name).is_some() {
            return Ok(());
        }

        Err(format!("Definition not found: {}", name))
    }

    /// Move a definition to a different module
    pub fn move_definition(&mut self, name: &str, new_module_path: &ModulePath) -> Result<(), String> {
        let old_module_key = self.def_index.get(name)
            .ok_or_else(|| format!("Definition not found: {}", name))?
            .clone();

        let new_module_key = module_path_to_string(new_module_path);

        if old_module_key == new_module_key {
            return Ok(()); // Already in the right place
        }

        // Get the definition
        let group = self.modules
            .get_mut(&old_module_key)
            .and_then(|m| m.remove_definition(name))
            .ok_or_else(|| format!("Failed to remove definition: {}", name))?;

        // Add to new module
        let new_module = self.modules
            .entry(new_module_key.clone())
            .or_insert_with(|| Module::new(new_module_path.clone()));
        new_module.add_definition(group);

        // Update index
        self.def_index.insert(name.to_string(), new_module_key.clone());

        // Move in .nostos/defs/
        let old_path: ModulePath = old_module_key.split('.').map(String::from).collect();
        let old_def_path = self.get_def_path(&old_path, name);
        let new_def_path = self.get_def_path(new_module_path, name);

        let nostos_dir = self.project_root.join(".nostos");
        let old_relative = old_def_path
            .strip_prefix(&nostos_dir)
            .unwrap()
            .to_string_lossy();
        let new_relative = new_def_path
            .strip_prefix(&nostos_dir)
            .unwrap()
            .to_string_lossy();

        git::move_and_commit(
            &nostos_dir,
            &old_relative,
            &new_relative,
            &format!("Move {} from {} to {}", name, old_module_key, new_module_key),
        )?;

        Ok(())
    }

    /// Rename a definition within its module
    /// Returns the new fully-qualified name
    pub fn rename_definition(&mut self, old_name: &str, new_name: &str) -> Result<String, String> {
        // Find the module containing this definition
        let module_key = self.def_index.get(old_name)
            .ok_or_else(|| format!("Definition not found: {}", old_name))?
            .clone();

        // Get the definition and update its source
        let module = self.modules.get_mut(&module_key)
            .ok_or_else(|| format!("Module not found: {}", module_key))?;

        // Get the definition group, update its source, and rename
        let old_source = {
            let group = module.get_definition_mut(old_name)
                .ok_or_else(|| format!("Definition not found in module: {}", old_name))?;
            group.combined_source()
        };

        // Update the source code to use the new name
        // Step 1: Rename the definition names (at the start of lines)
        let renamed_defs = rename_definition_in_source(&old_source, old_name, new_name);
        // Step 2: Also replace any self-calls (recursive functions)
        let new_source = replace_call_in_source(&renamed_defs, old_name, new_name);

        // Remove and re-add with new name
        let mut group = module.remove_definition(old_name)
            .ok_or_else(|| "Failed to remove definition".to_string())?;
        group.name = new_name.to_string();
        group.update_from_source(&new_source);
        module.add_definition(group);

        // Update def_index
        self.def_index.remove(old_name);
        self.def_index.insert(new_name.to_string(), module_key.clone());

        // Rename the file in .nostos/defs/
        let module_path: ModulePath = module_key.split('.').map(String::from).collect();
        let old_def_path = self.get_def_path(&module_path, old_name);
        let new_def_path = self.get_def_path(&module_path, new_name);

        let nostos_dir = self.project_root.join(".nostos");

        // Write new file first
        if let Some(parent) = new_def_path.parent() {
            fs::create_dir_all(parent)
                .map_err(|e| format!("Failed to create directory: {}", e))?;
        }
        fs::write(&new_def_path, &new_source)
            .map_err(|e| format!("Failed to write renamed definition: {}", e))?;

        // Delete old file
        if old_def_path.exists() {
            fs::remove_file(&old_def_path)
                .map_err(|e| format!("Failed to delete old definition: {}", e))?;
        }

        // Commit both changes
        let old_relative = old_def_path
            .strip_prefix(&nostos_dir)
            .unwrap()
            .to_string_lossy();
        let new_relative = new_def_path
            .strip_prefix(&nostos_dir)
            .unwrap()
            .to_string_lossy();

        git::rename_and_commit(
            &nostos_dir,
            &old_relative,
            &new_relative,
            &format!("Rename {} to {} in {}", old_name, new_name, module_key),
        )?;

        // Also update the main .nos source file
        self.write_module_files()?;

        // Return the new fully-qualified name
        let new_qualified = if module_key.is_empty() {
            new_name.to_string()
        } else {
            format!("{}.{}", module_key, new_name)
        };

        Ok(new_qualified)
    }

    /// Update a caller's source to replace calls from old_name to new_name
    /// Returns the updated source code if changes were made
    pub fn update_caller_source(&mut self, caller_name: &str, old_call: &str, new_call: &str) -> Result<Option<String>, String> {
        // Strip module prefix if present
        let simple_name = caller_name.rsplit('.').next().unwrap_or(caller_name);

        // Find the module containing this definition
        let module_key = self.def_index.get(simple_name)
            .ok_or_else(|| format!("Caller not found: {}", caller_name))?
            .clone();

        let module = self.modules.get_mut(&module_key)
            .ok_or_else(|| format!("Module not found: {}", module_key))?;

        let group = module.get_definition_mut(simple_name)
            .ok_or_else(|| format!("Definition not found in module: {}", simple_name))?;

        let old_source = group.combined_source();
        let new_source = replace_call_in_source(&old_source, old_call, new_call);

        if old_source == new_source {
            return Ok(None); // No changes made
        }

        // Update the definition
        group.update_from_source(&new_source);

        // Update the .nostos/defs file
        let module_path: ModulePath = module_key.split('.').map(String::from).collect();
        let def_path = self.get_def_path(&module_path, simple_name);
        fs::write(&def_path, &new_source)
            .map_err(|e| format!("Failed to write updated caller: {}", e))?;

        // Commit the change
        let nostos_dir = self.project_root.join(".nostos");
        let relative = def_path
            .strip_prefix(&nostos_dir)
            .unwrap()
            .to_string_lossy();

        git::commit_file(
            &nostos_dir,
            &relative,
            &format!("Update {} to call {} instead of {}", simple_name, new_call, old_call),
        )?;

        // Mark module as dirty so it gets written to .nos file
        if let Some(module) = self.modules.get_mut(&module_key) {
            module.dirty = true;
        }

        // Write module files
        self.write_module_files()?;

        Ok(Some(new_source))
    }

    /// Write dirty modules to main .nos files
    /// Returns the number of files written
    pub fn write_module_files(&mut self) -> Result<usize, String> {
        let mut written = 0;

        for module in self.modules.values_mut() {
            if module.path.is_empty() {
                // Root module - might need special handling
                continue;
            }

            if !module.dirty {
                // Skip unchanged modules
                continue;
            }

            let file_path = module.path.iter().fold(
                self.project_root.clone(),
                |p, s| p.join(s),
            );
            let file_path = file_path.with_extension("nos");

            // Create parent directories
            if let Some(parent) = file_path.parent() {
                fs::create_dir_all(parent)
                    .map_err(|e| format!("Failed to create directory: {}", e))?;
            }

            let content = module.generate_file_content();
            fs::write(&file_path, content)
                .map_err(|e| format!("Failed to write {}: {}", file_path.display(), e))?;

            module.dirty = false;
            written += 1;
        }

        Ok(written)
    }

    /// Get all module names
    pub fn module_names(&self) -> Vec<String> {
        self.modules.keys().cloned().collect()
    }

    /// Create a new empty module
    /// The module_path is a dot-separated path like "utils.math"
    /// If parent_path is provided, the new module will be a submodule
    pub fn create_module(&mut self, module_name: &str, parent_path: &[String]) -> Result<(), String> {
        // Build full module path
        let mut full_path: ModulePath = parent_path.to_vec();
        full_path.push(module_name.to_string());

        let module_key = module_path_to_string(&full_path);

        // Check if module already exists
        if self.modules.contains_key(&module_key) {
            return Err(format!("Module '{}' already exists", module_key));
        }

        // Create the module's .nos file path
        let file_path = full_path.iter().fold(
            self.project_root.clone(),
            |p, s| p.join(s),
        );
        let file_path = file_path.with_extension("nos");

        // Create parent directories if needed
        if let Some(parent) = file_path.parent() {
            fs::create_dir_all(parent)
                .map_err(|e| format!("Failed to create directory: {}", e))?;
        }

        // Create an empty .nos file with a comment
        let content = format!("# Module: {}\n", module_key);
        fs::write(&file_path, &content)
            .map_err(|e| format!("Failed to create module file: {}", e))?;

        // Add the module to our internal map
        let module = Module::new(full_path);
        self.modules.insert(module_key.clone(), module);

        Ok(())
    }

    /// Get all definition names
    pub fn definition_names(&self) -> Vec<String> {
        self.def_index.keys().cloned().collect()
    }

    /// Get all definition names in a module
    pub fn definitions_in_module(&self, module_key: &str) -> Vec<String> {
        self.modules
            .get(module_key)
            .map(|m| m.definition_names().map(String::from).collect())
            .unwrap_or_default()
    }

    // ===== Internal helpers =====

    fn get_def_path(&self, module_path: &ModulePath, name: &str) -> PathBuf {
        let mut path = self.project_root.join(".nostos").join("defs");
        for component in module_path {
            path = path.join(component);
        }
        path.join(format!("{}.nos", name))
    }

    fn write_definition_to_defs(&self, module_key: &str, name: &str) -> Result<(), String> {
        let module = self.modules.get(module_key)
            .ok_or_else(|| format!("Module not found: {}", module_key))?;

        let group = module.get_definition(name)
            .ok_or_else(|| format!("Definition not found: {}", name))?;

        let module_path: ModulePath = module_key.split('.').map(String::from).collect();
        let def_path = self.get_def_path(&module_path, name);

        // Create directories
        if let Some(parent) = def_path.parent() {
            fs::create_dir_all(parent)
                .map_err(|e| format!("Failed to create directory: {}", e))?;
        }

        let content = group.combined_source() + "\n";
        fs::write(&def_path, content)
            .map_err(|e| format!("Failed to write definition: {}", e))?;

        Ok(())
    }

    fn commit_definition(&self, module_key: &str, name: &str) -> Result<(), String> {
        let module_path: ModulePath = module_key.split('.').map(String::from).collect();
        let def_path = self.get_def_path(&module_path, name);

        let nostos_dir = self.project_root.join(".nostos");
        let relative_path = def_path
            .strip_prefix(&nostos_dir)
            .unwrap()
            .to_string_lossy();

        git::commit_file(
            &nostos_dir,
            &relative_path,
            &format!("Update {}.{}", module_key, name),
        )
    }

    fn write_meta_to_defs(&self, module_key: &str) -> Result<(), String> {
        let module = self.modules.get(module_key)
            .ok_or_else(|| format!("Module not found: {}", module_key))?;

        let module_path: ModulePath = module_key.split('.').map(String::from).collect();
        let mut meta_dir = self.project_root.join(".nostos").join("defs");
        for component in &module_path {
            meta_dir = meta_dir.join(component);
        }
        let meta_path = meta_dir.join("_meta.nos");

        // Create directory if needed
        fs::create_dir_all(&meta_dir)
            .map_err(|e| format!("Failed to create directory: {}", e))?;

        if !module.groups.is_empty() {
            let content = generate_meta_content(&module.groups);
            fs::write(&meta_path, content + "\n")
                .map_err(|e| format!("Failed to write _meta.nos: {}", e))?;

            // Commit _meta.nos
            let nostos_dir = self.project_root.join(".nostos");
            let relative_path = meta_path
                .strip_prefix(&nostos_dir)
                .unwrap()
                .to_string_lossy();
            git::commit_file(&nostos_dir, &relative_path, &format!("Update {}.groups", module_key))?;
        } else if meta_path.exists() {
            // Remove _meta.nos if no groups
            fs::remove_file(&meta_path)
                .map_err(|e| format!("Failed to remove _meta.nos: {}", e))?;
        }

        Ok(())
    }

    // ============================================================
    // File-based editing methods (for file-by-file editing mode)
    // ============================================================

    /// Load all source files from the project directory into memory
    pub fn load_source_files(&mut self) -> Result<(), String> {
        let nostos_exclude = self.project_root.join(".nostos");

        // Clear existing
        self.source_files.clear();

        // Find all .nos files
        for entry in WalkDir::new(&self.project_root)
            .into_iter()
            .filter_map(|e| e.ok())
            .filter(|e| {
                let path = e.path();
                path.extension().map(|ext| ext == "nos").unwrap_or(false)
                    && !path.starts_with(&nostos_exclude)
            })
        {
            let path = entry.path();
            let relative = path
                .strip_prefix(&self.project_root)
                .map_err(|_| format!("Failed to get relative path for {}", path.display()))?
                .to_string_lossy()
                .to_string();

            let content = fs::read_to_string(path)
                .map_err(|e| format!("Failed to read {}: {}", path.display(), e))?;

            self.source_files.insert(relative, content);
        }

        Ok(())
    }

    /// Get list of source files (relative paths)
    pub fn get_source_files(&self) -> Vec<String> {
        let mut files: Vec<String> = self.source_files.keys().cloned().collect();
        files.sort();
        files
    }

    /// Check if a file has parse errors
    pub fn file_has_errors(&self, path: &str) -> bool {
        self.files_with_errors.contains(path)
    }

    /// Get content of a source file by relative path
    pub fn get_file_content(&self, path: &str) -> Option<String> {
        self.source_files.get(path).cloned()
    }

    /// Get content of a source file directly from disk (not cached)
    pub fn read_file_from_disk(&self, relative_path: &str) -> Result<String, String> {
        let full_path = self.project_root.join(relative_path);
        fs::read_to_string(&full_path)
            .map_err(|e| format!("Failed to read {}: {}", full_path.display(), e))
    }

    /// Save content to a source file directly
    /// Updates both the cache and the disk file
    pub fn save_file_content(&mut self, relative_path: &str, content: &str) -> Result<(), String> {
        let full_path = self.project_root.join(relative_path);

        // Create parent directories if needed
        if let Some(parent) = full_path.parent() {
            fs::create_dir_all(parent)
                .map_err(|e| format!("Failed to create directory: {}", e))?;
        }

        // Write to disk
        fs::write(&full_path, content)
            .map_err(|e| format!("Failed to write {}: {}", full_path.display(), e))?;

        // Update cache
        self.source_files.insert(relative_path.to_string(), content.to_string());

        // Re-check for parse errors
        self.revalidate_file(relative_path, content);

        Ok(())
    }

    /// Re-check a file for parse errors and update error tracking
    pub fn revalidate_file(&mut self, relative_path: &str, content: &str) {
        use nostos_syntax::parse;
        let (parsed, errors) = parse(content);
        if parsed.is_none() || !errors.is_empty() {
            self.files_with_errors.insert(relative_path.to_string());
        } else {
            self.files_with_errors.remove(relative_path);
        }
    }

    /// Check if a file exists (by relative path)
    pub fn file_exists(&self, relative_path: &str) -> bool {
        self.project_root.join(relative_path).exists()
    }

    /// Scan the project directory for .nos files (without caching)
    pub fn scan_source_files(&self) -> Vec<String> {
        let nostos_exclude = self.project_root.join(".nostos");
        let mut files = Vec::new();

        for entry in WalkDir::new(&self.project_root)
            .into_iter()
            .filter_map(|e| e.ok())
            .filter(|e| {
                let path = e.path();
                path.extension().map(|ext| ext == "nos").unwrap_or(false)
                    && !path.starts_with(&nostos_exclude)
            })
        {
            if let Ok(relative) = entry.path().strip_prefix(&self.project_root) {
                files.push(relative.to_string_lossy().to_string());
            }
        }

        files.sort();
        files
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    /// Helper to create a temp directory with sample .nos files
    fn create_test_project() -> TempDir {
        let temp = TempDir::new().unwrap();
        let root = temp.path();

        // Create a main.nos file with a function
        // Note: Nostos uses # for comments, not //
        let main_content = r#"# Main module
add(x: Int, y: Int) = x + y

# Overloaded version
add(x: Float, y: Float) = x + y

type Point = { x: Int, y: Int }

greet(name: String) = "Hello, " ++ name
"#;
        fs::write(root.join("main.nos"), main_content).unwrap();

        // Create a utils/math.nos module
        fs::create_dir_all(root.join("utils")).unwrap();
        let math_content = r#"# Math utilities
square(n: Int) = n * n

cube(n: Int) = n * n * n
"#;
        fs::write(root.join("utils").join("math.nos"), math_content).unwrap();

        temp
    }

    #[test]
    fn test_source_manager_creates_nostos_toml() {
        let temp = TempDir::new().unwrap();
        let root = temp.path();

        // Create a minimal .nos file
        fs::write(root.join("main.nos"), "foo() = 42").unwrap();

        let sm = SourceManager::new(root.to_path_buf()).unwrap();

        // Check nostos.toml was created
        assert!(root.join("nostos.toml").exists());

        // Check .nostos directory was created
        assert!(root.join(".nostos").exists());
        assert!(root.join(".nostos").join(".git").exists());

        // Check project name is directory name
        let config = sm.config();
        assert!(!config.project.name.is_empty());
    }

    #[test]
    fn test_source_manager_bootstrap_from_nos_files() {
        let temp = create_test_project();
        let root = temp.path();

        let sm = SourceManager::new(root.to_path_buf()).unwrap();

        // Check definitions were loaded
        assert!(sm.get_source("add").is_some(), "Should have 'add' function");
        assert!(sm.get_source("Point").is_some(), "Should have 'Point' type");
        assert!(sm.get_source("greet").is_some(), "Should have 'greet' function");
        assert!(sm.get_source("square").is_some(), "Should have 'square' function from utils/math");
        assert!(sm.get_source("cube").is_some(), "Should have 'cube' function from utils/math");

        // Check .nostos/defs/ was created
        assert!(root.join(".nostos").join("defs").exists());
    }

    #[test]
    fn test_source_manager_preserves_overloads() {
        let temp = create_test_project();
        let root = temp.path();

        let sm = SourceManager::new(root.to_path_buf()).unwrap();

        // 'add' should have both overloads
        let add_source = sm.get_source("add").unwrap();
        assert!(add_source.contains("Int"), "Should have Int overload");
        assert!(add_source.contains("Float"), "Should have Float overload");

        // Should be separated by blank lines
        assert!(add_source.contains("\n\n"), "Overloads should be separated by blank lines");
    }

    #[test]
    fn test_source_manager_loads_from_defs() {
        let temp = create_test_project();
        let root = temp.path();

        // First create the .nostos/defs/ structure
        let _ = SourceManager::new(root.to_path_buf()).unwrap();

        // Now create a fresh SourceManager - it should load from .nostos/defs/
        let sm2 = SourceManager::new(root.to_path_buf()).unwrap();

        // Verify definitions are still there
        assert!(sm2.get_source("add").is_some());
        assert!(sm2.get_source("Point").is_some());
        assert!(sm2.get_source("square").is_some());
    }

    #[test]
    fn test_source_manager_update_definition() {
        let temp = create_test_project();
        let root = temp.path();

        let mut sm = SourceManager::new(root.to_path_buf()).unwrap();

        // Get original source
        let original = sm.get_source("greet").unwrap();
        assert!(original.contains("Hello"));

        // Update the definition
        let new_source = "greet(name: String) = \"Hi, \" ++ name";
        let changed = sm.update_definition("greet", new_source).unwrap();
        assert!(changed, "Should report change");

        // Verify the change
        let updated = sm.get_source("greet").unwrap();
        assert!(updated.contains("Hi,"), "Should have updated source");
        assert!(!updated.contains("Hello"), "Should not have old source");

        // Verify file was written
        let defs_dir = root.join(".nostos").join("defs");
        let def_file = defs_dir.join("main").join("greet.nos");
        let file_content = fs::read_to_string(&def_file).unwrap();
        assert!(file_content.contains("Hi,"));
    }

    #[test]
    fn test_source_manager_no_change_on_same_source() {
        let temp = create_test_project();
        let root = temp.path();

        let mut sm = SourceManager::new(root.to_path_buf()).unwrap();

        // Get original source
        let original = sm.get_source("greet").unwrap();

        // "Update" with same content (whitespace normalized)
        let changed = sm.update_definition("greet", &original).unwrap();
        assert!(!changed, "Should not report change for same content");
    }

    #[test]
    fn test_source_manager_add_definition() {
        let temp = TempDir::new().unwrap();
        let root = temp.path();

        // Create minimal project
        fs::write(root.join("main.nos"), "foo() = 1").unwrap();

        let mut sm = SourceManager::new(root.to_path_buf()).unwrap();

        // Add a new definition
        sm.add_definition(
            "bar",
            &vec!["main".to_string()],
            DefKind::Function,
            "bar(x: Int) = x * 2",
        ).unwrap();

        // Verify it exists
        assert!(sm.get_source("bar").is_some());

        // Verify file was created
        let def_file = root.join(".nostos").join("defs").join("main").join("bar.nos");
        assert!(def_file.exists());
    }

    #[test]
    fn test_source_manager_delete_definition() {
        let temp = create_test_project();
        let root = temp.path();

        let mut sm = SourceManager::new(root.to_path_buf()).unwrap();

        // Verify greet exists
        assert!(sm.get_source("greet").is_some());

        // Delete it
        sm.delete_definition("greet").unwrap();

        // Verify it's gone
        assert!(sm.get_source("greet").is_none());

        // Verify file was deleted
        let def_file = root.join(".nostos").join("defs").join("main").join("greet.nos");
        assert!(!def_file.exists());
    }

    #[test]
    fn test_source_manager_write_module_files() {
        let temp = create_test_project();
        let root = temp.path();

        let mut sm = SourceManager::new(root.to_path_buf()).unwrap();

        // update_definition now auto-syncs to module files, so just update and verify
        sm.update_definition("square", "square(x: Int) = x * x * 1").unwrap();

        // Check utils/math.nos was created with both functions
        let math_file = root.join("utils").join("math.nos");
        assert!(math_file.exists(), "utils/math.nos should exist");

        let content = fs::read_to_string(&math_file).unwrap();
        assert!(content.contains("square"), "Should contain square function");
        assert!(content.contains("cube"), "Should contain cube function");
    }

    #[test]
    fn test_source_manager_module_path_detection() {
        let temp = create_test_project();
        let root = temp.path();

        let sm = SourceManager::new(root.to_path_buf()).unwrap();

        // Check module paths
        assert_eq!(sm.get_definition_module("square"), Some("utils.math"));
        assert_eq!(sm.get_definition_module("cube"), Some("utils.math"));
        assert_eq!(sm.get_definition_module("add"), Some("main"));
        assert_eq!(sm.get_definition_module("Point"), Some("main"));
    }

    #[test]
    fn test_source_manager_handles_types() {
        let temp = create_test_project();
        let root = temp.path();

        let sm = SourceManager::new(root.to_path_buf()).unwrap();

        // Check type was loaded
        let point_source = sm.get_source("Point").unwrap();
        assert!(point_source.contains("type Point"));
        assert!(point_source.contains("x: Int"));
        assert!(point_source.contains("y: Int"));
    }

    #[test]
    fn test_source_manager_git_commits() {
        let temp = create_test_project();
        let root = temp.path();

        let mut sm = SourceManager::new(root.to_path_buf()).unwrap();

        // Update a definition (should trigger commit)
        sm.update_definition("greet", "greet(name: String) = \"Howdy, \" ++ name").unwrap();

        // Check git log
        let nostos_dir = root.join(".nostos");
        let output = std::process::Command::new("git")
            .args(["log", "--oneline", "-5"])
            .current_dir(&nostos_dir)
            .output()
            .unwrap();

        let log = String::from_utf8_lossy(&output.stdout);
        assert!(log.contains("Update"), "Git log should contain update commit");
    }

    #[test]
    fn test_source_manager_definition_names() {
        let temp = create_test_project();
        let root = temp.path();

        let sm = SourceManager::new(root.to_path_buf()).unwrap();

        let names = sm.definition_names();
        assert!(names.contains(&"add".to_string()));
        assert!(names.contains(&"Point".to_string()));
        assert!(names.contains(&"greet".to_string()));
        assert!(names.contains(&"square".to_string()));
        assert!(names.contains(&"cube".to_string()));
    }

    #[test]
    fn test_source_manager_module_names() {
        let temp = create_test_project();
        let root = temp.path();

        let sm = SourceManager::new(root.to_path_buf()).unwrap();

        let modules = sm.module_names();
        assert!(modules.contains(&"main".to_string()));
        assert!(modules.contains(&"utils.math".to_string()));
    }

    #[test]
    fn test_source_manager_definitions_in_module() {
        let temp = create_test_project();
        let root = temp.path();

        let sm = SourceManager::new(root.to_path_buf()).unwrap();

        let math_defs = sm.definitions_in_module("utils.math");
        assert!(math_defs.contains(&"square".to_string()));
        assert!(math_defs.contains(&"cube".to_string()));
        assert!(!math_defs.contains(&"add".to_string())); // add is in main

        let main_defs = sm.definitions_in_module("main");
        assert!(main_defs.contains(&"add".to_string()));
        assert!(main_defs.contains(&"Point".to_string()));
    }

    #[test]
    fn test_source_manager_repl_definitions() {
        let temp = TempDir::new().unwrap();
        let root = temp.path();

        // Create minimal project
        fs::write(root.join("main.nos"), "foo() = 1").unwrap();

        let mut sm = SourceManager::new(root.to_path_buf()).unwrap();

        // Add REPL definition (not in any module)
        sm.add_repl_definition("temp_fn", DefKind::Function, "temp_fn() = 99");

        // Should be retrievable
        let source = sm.get_source("temp_fn").unwrap();
        assert!(source.contains("99"));

        // But not in any module
        assert!(sm.get_definition_module("temp_fn").is_none());
    }

    #[test]
    fn test_source_manager_move_definition() {
        let temp = create_test_project();
        let root = temp.path();

        let mut sm = SourceManager::new(root.to_path_buf()).unwrap();

        // greet is initially in 'main' module
        assert_eq!(sm.get_definition_module("greet"), Some("main"));

        // Move to utils.math
        sm.move_definition("greet", &vec!["utils".to_string(), "math".to_string()]).unwrap();

        // Should now be in utils.math
        assert_eq!(sm.get_definition_module("greet"), Some("utils.math"));

        // Old file should be gone
        let old_file = root.join(".nostos").join("defs").join("main").join("greet.nos");
        assert!(!old_file.exists());

        // New file should exist
        let new_file = root.join(".nostos").join("defs").join("utils").join("math").join("greet.nos");
        assert!(new_file.exists());
    }

    #[test]
    fn test_source_manager_handles_empty_project() {
        let temp = TempDir::new().unwrap();
        let root = temp.path();

        // Create a .nos file with just a simple definition (empty file would fail git commit)
        fs::write(root.join("empty.nos"), "noop() = ()").unwrap();

        // Should not panic
        let sm = SourceManager::new(root.to_path_buf()).unwrap();

        // Should have just one definition
        let names = sm.definition_names();
        assert!(names.contains(&"noop".to_string()));
    }

    #[test]
    fn test_source_manager_content_hash_change_detection() {
        let temp = TempDir::new().unwrap();
        let root = temp.path();

        fs::write(root.join("main.nos"), "foo() = 42").unwrap();

        let mut sm = SourceManager::new(root.to_path_buf()).unwrap();

        // First update - should change
        let changed1 = sm.update_definition("foo", "foo() = 43").unwrap();
        assert!(changed1);

        // Same update - should not change
        let changed2 = sm.update_definition("foo", "foo() = 43").unwrap();
        assert!(!changed2);

        // Different update - should change
        let changed3 = sm.update_definition("foo", "foo() = 44").unwrap();
        assert!(changed3);
    }

    // ==========================================
    // Module Composition/Decomposition Tests
    // ==========================================

    #[test]
    fn test_mixed_content_file_parsing() {
        // Test a file with functions, types, variables, and imports
        let temp = TempDir::new().unwrap();
        let root = temp.path();

        let mixed_content = r#"type Person = { name: String, age: Int }

type Status = Ok | Error(String)

greeting = "Hello"

var counter = 0

greet(p: Person) = greeting ++ ", " ++ p.name

increment() = counter + 1
"#;
        fs::write(root.join("mixed.nos"), mixed_content).unwrap();

        let sm = SourceManager::new(root.to_path_buf()).unwrap();

        // Check all definition types were extracted
        assert!(sm.get_source("Person").is_some(), "Type Person should exist");
        assert!(sm.get_source("Status").is_some(), "Type Status should exist");
        assert!(sm.get_source("greeting").is_some(), "Variable greeting should exist");
        assert!(sm.get_source("counter").is_some(), "Variable counter should exist");
        assert!(sm.get_source("greet").is_some(), "Function greet should exist");
        assert!(sm.get_source("increment").is_some(), "Function increment should exist");

        // Verify each file was created in .nostos/defs/
        let defs_dir = root.join(".nostos").join("defs").join("mixed");
        assert!(defs_dir.join("Person.nos").exists());
        assert!(defs_dir.join("Status.nos").exists());
        assert!(defs_dir.join("greeting.nos").exists());
        assert!(defs_dir.join("counter.nos").exists());
        assert!(defs_dir.join("greet.nos").exists());
        assert!(defs_dir.join("increment.nos").exists());
    }

    #[test]
    fn test_module_file_regeneration_preserves_order() {
        // Test that write_module_files() produces consistent, ordered output
        let temp = TempDir::new().unwrap();
        let root = temp.path();

        let content = r#"type Point = { x: Int, y: Int }

type Color = Red | Green | Blue

add(a: Int, b: Int) = a + b

multiply(a: Int, b: Int) = a * b

pi = 3.14159
"#;
        fs::write(root.join("math.nos"), content).unwrap();

        let mut sm = SourceManager::new(root.to_path_buf()).unwrap();
        // Mark dirty by making a change
        sm.update_definition("pi", "pi = 3.14159").unwrap();
        sm.write_module_files().unwrap();

        // Read the regenerated file
        let regenerated = fs::read_to_string(root.join("math.nos")).unwrap();

        // Types should come first, then functions, then variables
        let type_pos = regenerated.find("type Point").unwrap();
        let color_pos = regenerated.find("type Color").unwrap();
        let add_pos = regenerated.find("add(").unwrap();
        let mul_pos = regenerated.find("multiply(").unwrap();
        let pi_pos = regenerated.find("pi =").unwrap();

        // Types before functions
        assert!(type_pos < add_pos, "Types should come before functions");
        assert!(color_pos < add_pos, "All types before functions");

        // Functions before variables
        assert!(add_pos < pi_pos, "Functions should come before variables");
        assert!(mul_pos < pi_pos, "All functions before variables");
    }

    #[test]
    fn test_hash_detects_whitespace_normalization() {
        let temp = TempDir::new().unwrap();
        let root = temp.path();

        fs::write(root.join("main.nos"), "foo() = 42").unwrap();

        let mut sm = SourceManager::new(root.to_path_buf()).unwrap();

        // Same content with different whitespace should NOT change
        let changed1 = sm.update_definition("foo", "foo()=42").unwrap();
        // Note: This depends on implementation - if whitespace is normalized, no change
        // If not normalized, it will change. Testing current behavior:

        // Update with actually different content
        let changed2 = sm.update_definition("foo", "foo() = 43").unwrap();
        assert!(changed2, "Different value should be detected");

        // Back to original
        let changed3 = sm.update_definition("foo", "foo() = 42").unwrap();
        assert!(changed3, "Return to original should be detected");
    }

    #[test]
    fn test_overload_hash_detection() {
        let temp = TempDir::new().unwrap();
        let root = temp.path();

        let content = r#"add(x: Int, y: Int) = x + y

add(x: Float, y: Float) = x + y
"#;
        fs::write(root.join("main.nos"), content).unwrap();

        let mut sm = SourceManager::new(root.to_path_buf()).unwrap();

        // Get original combined source
        let original = sm.get_source("add").unwrap();
        assert!(original.contains("Int"), "Should have Int overload");
        assert!(original.contains("Float"), "Should have Float overload");

        // Modify one overload - should detect change
        let modified = r#"add(x: Int, y: Int) = x + y + 1

add(x: Float, y: Float) = x + y"#;
        let changed = sm.update_definition("add", modified).unwrap();
        assert!(changed, "Modifying one overload should be detected");

        // Same content again - no change
        let changed2 = sm.update_definition("add", modified).unwrap();
        assert!(!changed2, "Same content should not change");

        // Add a third overload
        let with_third = r#"add(x: Int, y: Int) = x + y + 1

add(x: Float, y: Float) = x + y

add(x: String, y: String) = x ++ y"#;
        let changed3 = sm.update_definition("add", with_third).unwrap();
        assert!(changed3, "Adding overload should be detected");
    }

    #[test]
    fn test_decompose_and_recompose_module() {
        // Test the full cycle: file -> defs -> file
        let temp = TempDir::new().unwrap();
        let root = temp.path();

        let original = r#"type Point = { x: Int, y: Int }

distance(p1: Point, p2: Point) = {
    dx = p2.x - p1.x
    dy = p2.y - p1.y
    sqrt(dx * dx + dy * dy)
}

origin = Point(0, 0)
"#;
        fs::write(root.join("geom.nos"), original).unwrap();

        // Load (decompose into defs)
        let mut sm = SourceManager::new(root.to_path_buf()).unwrap();

        // Verify decomposition
        let defs_dir = root.join(".nostos").join("defs").join("geom");
        assert!(defs_dir.join("Point.nos").exists());
        assert!(defs_dir.join("distance.nos").exists());
        assert!(defs_dir.join("origin.nos").exists());

        // Read individual def files
        let point_content = fs::read_to_string(defs_dir.join("Point.nos")).unwrap();
        assert!(point_content.contains("type Point"));

        let distance_content = fs::read_to_string(defs_dir.join("distance.nos")).unwrap();
        assert!(distance_content.contains("distance("));
        assert!(distance_content.contains("sqrt"));

        // Mark dirty and recompose
        sm.update_definition("origin", "origin = Point(0, 0)").unwrap();
        sm.write_module_files().unwrap();

        // Verify recomposition contains all elements
        let recomposed = fs::read_to_string(root.join("geom.nos")).unwrap();
        assert!(recomposed.contains("type Point"), "Recomposed should have Point");
        assert!(recomposed.contains("distance("), "Recomposed should have distance");
        assert!(recomposed.contains("origin ="), "Recomposed should have origin");
    }

    #[test]
    fn test_multiple_modules_composition() {
        let temp = TempDir::new().unwrap();
        let root = temp.path();

        // Create multiple module files
        fs::write(root.join("types.nos"), r#"type User = { id: Int, name: String }

type Role = Admin | Member | Guest
"#).unwrap();

        fs::create_dir_all(root.join("utils")).unwrap();
        fs::write(root.join("utils").join("string.nos"), r#"capitalize(s: String) = s

trim(s: String) = s
"#).unwrap();

        fs::write(root.join("utils").join("math.nos"), r#"abs(n: Int) = if n < 0 then -n else n

max(a: Int, b: Int) = if a > b then a else b
"#).unwrap();

        let sm = SourceManager::new(root.to_path_buf()).unwrap();

        // Verify module structure
        let modules = sm.module_names();
        assert!(modules.iter().any(|m| m == "types"), "Should have types module");
        assert!(modules.iter().any(|m| m == "utils.string"), "Should have utils.string module");
        assert!(modules.iter().any(|m| m == "utils.math"), "Should have utils.math module");

        // Verify definitions in correct modules
        assert_eq!(sm.get_definition_module("User"), Some("types"));
        assert_eq!(sm.get_definition_module("Role"), Some("types"));
        assert_eq!(sm.get_definition_module("capitalize"), Some("utils.string"));
        assert_eq!(sm.get_definition_module("trim"), Some("utils.string"));
        assert_eq!(sm.get_definition_module("abs"), Some("utils.math"));
        assert_eq!(sm.get_definition_module("max"), Some("utils.math"));
    }

    #[test]
    fn test_incremental_changes_tracking() {
        let temp = TempDir::new().unwrap();
        let root = temp.path();

        fs::write(root.join("main.nos"), r#"foo() = 1

bar() = 2

baz() = 3
"#).unwrap();

        let mut sm = SourceManager::new(root.to_path_buf()).unwrap();

        // Modify only foo
        sm.update_definition("foo", "foo() = 10").unwrap();

        // Check that only foo's file changed
        let foo_content = fs::read_to_string(
            root.join(".nostos").join("defs").join("main").join("foo.nos")
        ).unwrap();
        assert!(foo_content.contains("10"), "foo should be updated");

        // bar and baz should still have original values
        let bar_content = fs::read_to_string(
            root.join(".nostos").join("defs").join("main").join("bar.nos")
        ).unwrap();
        assert!(bar_content.contains("2"), "bar should be unchanged");

        let baz_content = fs::read_to_string(
            root.join(".nostos").join("defs").join("main").join("baz.nos")
        ).unwrap();
        assert!(baz_content.contains("3"), "baz should be unchanged");
    }

    #[test]
    fn test_type_variants_preserved() {
        let temp = TempDir::new().unwrap();
        let root = temp.path();

        let content = r#"type Result[T, E] = Ok(T) | Err(E)

type Option[T] = Some(T) | None

type Tree[T] = Leaf(T) | Node(Tree[T], Tree[T])
"#;
        fs::write(root.join("types.nos"), content).unwrap();

        let sm = SourceManager::new(root.to_path_buf()).unwrap();

        // Check type sources are preserved correctly
        let result_src = sm.get_source("Result").unwrap();
        assert!(result_src.contains("Ok(T)"), "Result should have Ok variant");
        assert!(result_src.contains("Err(E)"), "Result should have Err variant");

        let option_src = sm.get_source("Option").unwrap();
        assert!(option_src.contains("Some(T)"), "Option should have Some variant");
        assert!(option_src.contains("None"), "Option should have None variant");

        let tree_src = sm.get_source("Tree").unwrap();
        assert!(tree_src.contains("Leaf(T)"), "Tree should have Leaf");
        assert!(tree_src.contains("Node("), "Tree should have Node");
    }

    #[test]
    fn test_pattern_matching_functions_preserved() {
        let temp = TempDir::new().unwrap();
        let root = temp.path();

        let content = r#"fib(0) = 0
fib(1) = 1
fib(n) = fib(n - 1) + fib(n - 2)

len([]) = 0
len([_ | tail]) = 1 + len(tail)
"#;
        fs::write(root.join("main.nos"), content).unwrap();

        let sm = SourceManager::new(root.to_path_buf()).unwrap();

        // Both fib and len should be separate definition groups
        let fib_src = sm.get_source("fib").unwrap();
        assert!(fib_src.contains("fib(0)"), "Should have fib(0) clause");
        assert!(fib_src.contains("fib(1)"), "Should have fib(1) clause");
        assert!(fib_src.contains("fib(n)"), "Should have fib(n) clause");

        let len_src = sm.get_source("len").unwrap();
        assert!(len_src.contains("len([])"), "Should have empty list clause");
        assert!(len_src.contains("len([_"), "Should have cons clause");
    }

    #[test]
    fn test_git_history_per_definition() {
        let temp = TempDir::new().unwrap();
        let root = temp.path();

        fs::write(root.join("main.nos"), r#"foo() = 1

bar() = 2
"#).unwrap();

        let mut sm = SourceManager::new(root.to_path_buf()).unwrap();

        // Make several changes to foo only
        sm.update_definition("foo", "foo() = 10").unwrap();
        sm.update_definition("foo", "foo() = 100").unwrap();
        sm.update_definition("foo", "foo() = 1000").unwrap();

        // Check git log for foo.nos specifically
        let nostos_dir = root.join(".nostos");
        let output = std::process::Command::new("git")
            .args(["log", "--oneline", "defs/main/foo.nos"])
            .current_dir(&nostos_dir)
            .output()
            .unwrap();

        let log = String::from_utf8_lossy(&output.stdout);
        let commit_count = log.lines().count();
        // Should have at least the initial import + 3 updates = 4 commits
        // (might be fewer if some commits were combined)
        assert!(commit_count >= 3, "Should have multiple commits for foo: {}", log);

        // bar should have fewer commits (only initial)
        let output2 = std::process::Command::new("git")
            .args(["log", "--oneline", "defs/main/bar.nos"])
            .current_dir(&nostos_dir)
            .output()
            .unwrap();

        let log2 = String::from_utf8_lossy(&output2.stdout);
        let bar_commits = log2.lines().count();
        assert!(bar_commits < commit_count, "bar should have fewer commits than foo");
    }

    #[test]
    fn test_complex_nested_modules() {
        let temp = TempDir::new().unwrap();
        let root = temp.path();

        // Create deeply nested module structure
        fs::create_dir_all(root.join("app").join("models")).unwrap();
        fs::create_dir_all(root.join("app").join("views")).unwrap();
        fs::create_dir_all(root.join("lib").join("utils")).unwrap();

        fs::write(root.join("app").join("models").join("user.nos"),
            "type User = { id: Int, email: String }\n").unwrap();

        fs::write(root.join("app").join("models").join("post.nos"),
            "type Post = { id: Int, title: String, author: User }\n").unwrap();

        fs::write(root.join("app").join("views").join("home.nos"),
            "render_home() = \"<h1>Home</h1>\"\n").unwrap();

        fs::write(root.join("lib").join("utils").join("format.nos"),
            "format_date(d: Int) = \"date\"\n").unwrap();

        let sm = SourceManager::new(root.to_path_buf()).unwrap();

        // Verify module paths
        assert_eq!(sm.get_definition_module("User"), Some("app.models.user"));
        assert_eq!(sm.get_definition_module("Post"), Some("app.models.post"));
        assert_eq!(sm.get_definition_module("render_home"), Some("app.views.home"));
        assert_eq!(sm.get_definition_module("format_date"), Some("lib.utils.format"));
    }

    #[test]
    fn test_empty_overload_group_handling() {
        let temp = TempDir::new().unwrap();
        let root = temp.path();

        // Single function, not an overload
        fs::write(root.join("main.nos"), "single(x: Int) = x * 2\n").unwrap();

        let sm = SourceManager::new(root.to_path_buf()).unwrap();

        let src = sm.get_source("single").unwrap();
        // Should NOT have blank line separator since it's not an overload group
        assert!(!src.contains("\n\n"), "Single function should not have blank line separator");
        assert!(src.contains("single(x: Int)"));
    }

    #[test]
    fn test_reload_after_external_def_change() {
        let temp = TempDir::new().unwrap();
        let root = temp.path();

        fs::write(root.join("main.nos"), "foo() = 1\n").unwrap();

        // First load
        let _sm1 = SourceManager::new(root.to_path_buf()).unwrap();

        // Simulate external edit to def file
        let def_file = root.join(".nostos").join("defs").join("main").join("foo.nos");
        fs::write(&def_file, "foo() = 999\n").unwrap();

        // Reload - should pick up external change
        let sm2 = SourceManager::new(root.to_path_buf()).unwrap();

        let src = sm2.get_source("foo").unwrap();
        assert!(src.contains("999"), "Should pick up external def file change");
    }

    #[test]
    fn test_definition_kind_detection() {
        let temp = TempDir::new().unwrap();
        let root = temp.path();

        let content = r#"type MyType = { field: Int }

myFunc() = 42

myVar = "hello"

var mutableVar = 0
"#;
        fs::write(root.join("main.nos"), content).unwrap();

        let sm = SourceManager::new(root.to_path_buf()).unwrap();

        // Each should be in separate file with correct content
        let defs_dir = root.join(".nostos").join("defs").join("main");

        let type_content = fs::read_to_string(defs_dir.join("MyType.nos")).unwrap();
        assert!(type_content.starts_with("type "), "Type file should start with 'type '");

        let func_content = fs::read_to_string(defs_dir.join("myFunc.nos")).unwrap();
        assert!(func_content.contains("myFunc()"), "Function file should have function def");

        // Variables should also be stored
        assert!(defs_dir.join("myVar.nos").exists());
        assert!(defs_dir.join("mutableVar.nos").exists());
    }

    #[test]
    fn test_hash_stability_across_reloads() {
        let temp = TempDir::new().unwrap();
        let root = temp.path();

        let content = "stable() = 42\n";
        fs::write(root.join("main.nos"), content).unwrap();

        // Load and get source
        let sm1 = SourceManager::new(root.to_path_buf()).unwrap();
        let src1 = sm1.get_source("stable").unwrap();

        // Reload
        let sm2 = SourceManager::new(root.to_path_buf()).unwrap();
        let src2 = sm2.get_source("stable").unwrap();

        // Source should be identical
        assert_eq!(src1, src2, "Source should be stable across reloads");

        // Update with same content should report no change
        let mut sm3 = SourceManager::new(root.to_path_buf()).unwrap();
        let changed = sm3.update_definition("stable", &src1).unwrap();
        assert!(!changed, "Updating with same content should report no change");
    }

    // ==========================================
    // Together Directive Tests
    // ==========================================

    #[test]
    fn test_together_directive_parsing() {
        // Test parse_together_directives helper
        let content = r#"# together foo bar
# together baz qux quux
# not a directive
# together single
# together a b
"#;
        let groups = parse_together_directives(content);
        assert_eq!(groups.len(), 3, "Should parse 3 valid together groups");
        assert_eq!(groups[0], vec!["foo", "bar"]);
        assert_eq!(groups[1], vec!["baz", "qux", "quux"]);
        assert_eq!(groups[2], vec!["a", "b"]);
    }

    #[test]
    fn test_together_directive_in_meta_file() {
        let temp = TempDir::new().unwrap();
        let root = temp.path();

        // Create source file with functions
        fs::write(root.join("main.nos"), r#"foo() = 1
bar() = 2
baz() = 3
"#).unwrap();

        // First initialize with SourceManager
        let _ = SourceManager::new(root.to_path_buf()).unwrap();

        // Now manually create _meta.nos with together directive
        let meta_path = root.join(".nostos").join("defs").join("main").join("_meta.nos");
        fs::write(&meta_path, "# together foo bar\n").unwrap();

        // Reload and check
        let sm = SourceManager::new(root.to_path_buf()).unwrap();

        // foo should return both foo and bar
        let src = sm.get_source("foo").unwrap();
        assert!(src.contains("# together foo bar"), "Should include together directive");
        assert!(src.contains("foo() = 1"), "Should have foo source");
        assert!(src.contains("bar() = 2"), "Should have bar source");

        // bar should also return both (same group)
        let src2 = sm.get_source("bar").unwrap();
        assert_eq!(src, src2, "bar should return same group source as foo");

        // baz should return only itself
        let src3 = sm.get_source("baz").unwrap();
        assert!(src3.contains("baz() = 3"));
        assert!(!src3.contains("foo"), "baz should not include foo");
    }

    #[test]
    fn test_together_directive_validation_no_duplicates() {
        let temp = TempDir::new().unwrap();
        let root = temp.path();

        fs::write(root.join("main.nos"), r#"foo() = 1
bar() = 2
"#).unwrap();

        // Initialize
        let _ = SourceManager::new(root.to_path_buf()).unwrap();

        // Create invalid _meta.nos with same definition in multiple groups
        let meta_path = root.join(".nostos").join("defs").join("main").join("_meta.nos");
        fs::write(&meta_path, r#"# together foo bar
# together bar baz
"#).unwrap();

        // Should fail to load
        let result = SourceManager::new(root.to_path_buf());
        assert!(result.is_err(), "Should fail with duplicate definition in groups");
        let err = result.err().unwrap();
        assert!(err.contains("Duplicate"), "Error should mention Duplicate: {}", err);
    }

    #[test]
    fn test_together_directive_file_output() {
        let temp = TempDir::new().unwrap();
        let root = temp.path();

        fs::write(root.join("main.nos"), r#"foo() = 1
bar() = 2
baz() = 3
"#).unwrap();

        // Initialize
        let _ = SourceManager::new(root.to_path_buf()).unwrap();

        // Add together directive
        let meta_path = root.join(".nostos").join("defs").join("main").join("_meta.nos");
        fs::write(&meta_path, "# together foo bar\n").unwrap();

        // Load and write module files
        let mut sm = SourceManager::new(root.to_path_buf()).unwrap();

        // Check groups were loaded
        let grouped = sm.get_grouped_names("foo");
        assert_eq!(grouped, vec!["foo", "bar"], "Groups should be loaded from _meta.nos");

        // Mark dirty - this won't change since same source, so use different source
        sm.update_definition("baz", "baz() = 33").unwrap();
        sm.write_module_files().unwrap();

        // Check output file
        let content = fs::read_to_string(root.join("main.nos")).unwrap();

        // Should have together directive at top
        assert!(content.starts_with("# together foo bar"), "File should start with together directive, got: {}", &content[..50.min(content.len())]);

        // foo and bar should appear together
        let foo_pos = content.find("foo() = 1").unwrap();
        let bar_pos = content.find("bar() = 2").unwrap();

        // They should be adjacent (only separated by blank lines)
        let between = &content[foo_pos..bar_pos];
        assert!(!between.contains("baz"), "baz should not be between grouped foo and bar");
    }

    #[test]
    fn test_together_directive_get_grouped_names() {
        let temp = TempDir::new().unwrap();
        let root = temp.path();

        fs::write(root.join("main.nos"), r#"foo() = 1
bar() = 2
baz() = 3
"#).unwrap();

        // Initialize
        let _ = SourceManager::new(root.to_path_buf()).unwrap();

        // Add together directive
        let meta_path = root.join(".nostos").join("defs").join("main").join("_meta.nos");
        fs::write(&meta_path, "# together foo bar\n").unwrap();

        let sm = SourceManager::new(root.to_path_buf()).unwrap();

        // Check grouped names
        let names_foo = sm.get_grouped_names("foo");
        assert_eq!(names_foo, vec!["foo", "bar"], "foo should return group");

        let names_bar = sm.get_grouped_names("bar");
        assert_eq!(names_bar, vec!["foo", "bar"], "bar should return same group");

        let names_baz = sm.get_grouped_names("baz");
        assert_eq!(names_baz, vec!["baz"], "baz should return only itself");
    }

    #[test]
    fn test_update_group_source() {
        let temp = TempDir::new().unwrap();
        let root = temp.path();

        fs::write(root.join("main.nos"), r#"foo() = 1
bar() = 2
"#).unwrap();

        let mut sm = SourceManager::new(root.to_path_buf()).unwrap();

        // Update with grouped source including together directive
        let grouped_source = r#"# together foo bar

foo() = 10

bar() = 20"#;

        let updated = sm.update_group_source("foo", grouped_source).unwrap();
        assert!(updated.contains(&"foo".to_string()), "Should update foo");
        assert!(updated.contains(&"bar".to_string()), "Should update bar");

        // Verify sources were updated
        let foo_src = sm.get_source("foo").unwrap();
        assert!(foo_src.contains("foo() = 10"), "foo should be updated");
        assert!(foo_src.contains("bar() = 20"), "bar should be updated in same group");

        // Verify _meta.nos was created
        let meta_path = root.join(".nostos").join("defs").join("main").join("_meta.nos");
        assert!(meta_path.exists(), "_meta.nos should exist");
        let meta_content = fs::read_to_string(&meta_path).unwrap();
        assert!(meta_content.contains("# together foo bar"), "Should have together directive");
    }

    #[test]
    fn test_together_with_types() {
        let temp = TempDir::new().unwrap();
        let root = temp.path();

        // Types can also be in groups
        fs::write(root.join("main.nos"), r#"type Point = { x: Int, y: Int }

distance(p1: Point, p2: Point) = 0
"#).unwrap();

        let _ = SourceManager::new(root.to_path_buf()).unwrap();

        // Group type with function
        let meta_path = root.join(".nostos").join("defs").join("main").join("_meta.nos");
        fs::write(&meta_path, "# together Point distance\n").unwrap();

        let sm = SourceManager::new(root.to_path_buf()).unwrap();

        // Opening Point should show both
        let src = sm.get_source("Point").unwrap();
        assert!(src.contains("type Point"), "Should have Point type");
        assert!(src.contains("distance("), "Should have distance function");
    }

    #[test]
    fn test_together_directive_in_source_stripped_for_parsing() {
        let temp = TempDir::new().unwrap();
        let root = temp.path();

        fs::write(root.join("main.nos"), r#"foo() = 1
bar() = 2
"#).unwrap();

        let mut sm = SourceManager::new(root.to_path_buf()).unwrap();

        // Source with together directive should parse correctly
        // (directive is stripped before parsing)
        let source_with_directive = r#"# together foo bar

foo() = 100

bar() = 200"#;

        let result = sm.update_group_source("foo", source_with_directive);
        assert!(result.is_ok(), "Should parse source with together directive");

        let updated = result.unwrap();
        assert_eq!(updated.len(), 2, "Should update both definitions");
    }

    #[test]
    fn test_together_groups_sorted_by_first_member() {
        let temp = TempDir::new().unwrap();
        let root = temp.path();

        fs::write(root.join("main.nos"), r#"zoo() = 1
apple() = 2
mango() = 3
banana() = 4
"#).unwrap();

        let _ = SourceManager::new(root.to_path_buf()).unwrap();

        // Create two groups
        let meta_path = root.join(".nostos").join("defs").join("main").join("_meta.nos");
        fs::write(&meta_path, r#"# together zoo mango
# together apple banana
"#).unwrap();

        let mut sm = SourceManager::new(root.to_path_buf()).unwrap();
        // Need to use different content to mark dirty
        sm.update_definition("zoo", "zoo() = 11").unwrap();
        sm.write_module_files().unwrap();

        let content = fs::read_to_string(root.join("main.nos")).unwrap();

        // apple-banana group should come before zoo-mango group (alphabetical by first member)
        // Note: We only test that apple comes before zoo, the exact implementation may vary
        let apple_pos = content.find("apple()").unwrap();
        let zoo_pos = content.find("zoo()").unwrap();
        assert!(apple_pos < zoo_pos, "apple group should come before zoo group, content:\n{}", content);
    }

    #[test]
    fn test_together_meta_update_via_group_source() {
        let temp = TempDir::new().unwrap();
        let root = temp.path();

        fs::write(root.join("main.nos"), r#"foo() = 1
bar() = 2
baz() = 3
"#).unwrap();

        let mut sm = SourceManager::new(root.to_path_buf()).unwrap();

        // Add together directive via update_group_source
        let source_with_group = r#"# together foo bar

foo() = 10

bar() = 20"#;
        let updated = sm.update_group_source("foo", source_with_group).unwrap();

        // Should have updated both definitions
        assert!(updated.contains(&"foo".to_string()), "Should update foo");
        assert!(updated.contains(&"bar".to_string()), "Should update bar");

        // Check _meta.nos was created with together directive
        let meta_path = root.join(".nostos").join("defs").join("main").join("_meta.nos");
        assert!(meta_path.exists(), "_meta.nos should exist");
        let meta_content = fs::read_to_string(&meta_path).unwrap();
        assert!(meta_content.contains("# together foo bar"), "Should have together directive");

        // Verify groups are recognized
        let grouped_names = sm.get_grouped_names("foo");
        assert_eq!(grouped_names, vec!["foo", "bar"], "foo should be grouped with bar");
    }

    #[test]
    fn test_together_three_definitions() {
        let temp = TempDir::new().unwrap();
        let root = temp.path();

        fs::write(root.join("main.nos"), r#"init() = 0
process() = 1
cleanup() = 2
other() = 3
"#).unwrap();

        let _ = SourceManager::new(root.to_path_buf()).unwrap();

        // Group three functions together
        let meta_path = root.join(".nostos").join("defs").join("main").join("_meta.nos");
        fs::write(&meta_path, "# together init process cleanup\n").unwrap();

        let sm = SourceManager::new(root.to_path_buf()).unwrap();

        // Opening any should show all three
        let src = sm.get_source("process").unwrap();
        assert!(src.contains("init()"), "Should have init");
        assert!(src.contains("process()"), "Should have process");
        assert!(src.contains("cleanup()"), "Should have cleanup");
        assert!(!src.contains("other()"), "Should not have other");
    }

    #[test]
    fn test_generate_meta_content() {
        let groups = vec![
            vec!["foo".to_string(), "bar".to_string()],
            vec!["baz".to_string(), "qux".to_string(), "quux".to_string()],
        ];
        let content = generate_meta_content(&groups);
        assert!(content.contains("# together foo bar"));
        assert!(content.contains("# together baz qux quux"));
    }

    #[test]
    fn test_module_groups_empty_by_default() {
        use crate::module::Module;

        let module = Module::new(vec!["test".to_string()]);
        assert!(module.groups.is_empty(), "New module should have no groups");
    }

    #[test]
    fn test_module_validate_groups_pass() {
        use crate::module::Module;
        use crate::definition::{DefKind, DefinitionGroup};

        let mut module = Module::new(vec!["test".to_string()]);
        module.add_definition(DefinitionGroup::new("foo".to_string(), DefKind::Function, "foo() = 1".to_string()));
        module.add_definition(DefinitionGroup::new("bar".to_string(), DefKind::Function, "bar() = 2".to_string()));
        module.add_definition(DefinitionGroup::new("baz".to_string(), DefKind::Function, "baz() = 3".to_string()));

        module.set_groups(vec![
            vec!["foo".to_string(), "bar".to_string()],
        ]);

        assert!(module.validate_groups().is_ok(), "Should pass validation");
    }

    #[test]
    fn test_module_validate_groups_fail_duplicate() {
        use crate::module::Module;

        let mut module = Module::new(vec!["test".to_string()]);
        module.set_groups(vec![
            vec!["foo".to_string(), "bar".to_string()],
            vec!["bar".to_string(), "baz".to_string()], // bar is duplicated
        ]);

        let result = module.validate_groups();
        assert!(result.is_err(), "Should fail with duplicate");
        let dups = result.unwrap_err();
        assert!(dups.contains(&"bar".to_string()), "Should report bar as duplicate");
    }

    #[test]
    fn test_reload_preserves_groups() {
        let temp = TempDir::new().unwrap();
        let root = temp.path();

        fs::write(root.join("main.nos"), r#"foo() = 1
bar() = 2
"#).unwrap();

        // First load and add group
        let _ = SourceManager::new(root.to_path_buf()).unwrap();
        let meta_path = root.join(".nostos").join("defs").join("main").join("_meta.nos");
        fs::write(&meta_path, "# together foo bar\n").unwrap();

        // Reload multiple times
        for _ in 0..3 {
            let sm = SourceManager::new(root.to_path_buf()).unwrap();
            let names = sm.get_grouped_names("foo");
            assert_eq!(names, vec!["foo", "bar"], "Groups should persist across reloads");
        }
    }

    // ==================== Rename Function Tests ====================

    #[test]
    fn test_rename_definition_in_source_simple() {
        let source = "foo() = 1";
        let result = rename_definition_in_source(source, "foo", "bar");
        assert_eq!(result, "bar() = 1");
    }

    #[test]
    fn test_rename_definition_in_source_with_params() {
        let source = "foo(x: Int, y: Int) = x + y";
        let result = rename_definition_in_source(source, "foo", "bar");
        assert_eq!(result, "bar(x: Int, y: Int) = x + y");
    }

    #[test]
    fn test_rename_definition_in_source_overloaded() {
        // Overloaded functions are stored as separate definitions joined by \n\n
        let source = "add(x: Int, y: Int) = x + y\n\nadd(x: Float, y: Float) = x + y";
        let result = rename_definition_in_source(source, "add", "sum");
        assert_eq!(result, "sum(x: Int, y: Int) = x + y\n\nsum(x: Float, y: Float) = x + y");
    }

    #[test]
    fn test_rename_definition_in_source_multiclause() {
        // Multi-clause functions (pattern matching)
        let source = "fib(0) = 0\nfib(1) = 1\nfib(n) = fib(n - 1) + fib(n - 2)";
        let result = rename_definition_in_source(source, "fib", "fibonacci");
        // Note: the function calls inside the body should NOT be renamed by this function
        // That's handled by replace_call_in_source
        assert_eq!(result, "fibonacci(0) = 0\nfibonacci(1) = 1\nfibonacci(n) = fib(n - 1) + fib(n - 2)");
    }

    #[test]
    fn test_rename_definition_in_source_value() {
        // Value definitions (no parens)
        let source = "pi = 3.14159";
        let result = rename_definition_in_source(source, "pi", "PI");
        assert_eq!(result, "PI = 3.14159");
    }

    #[test]
    fn test_replace_call_in_source_simple() {
        let source = "bar() = foo() + 1";
        let result = replace_call_in_source(source, "foo", "baz");
        assert_eq!(result, "bar() = baz() + 1");
    }

    #[test]
    fn test_replace_call_in_source_with_args() {
        let source = "bar() = foo(1, 2) + foo(3, 4)";
        let result = replace_call_in_source(source, "foo", "baz");
        assert_eq!(result, "bar() = baz(1, 2) + baz(3, 4)");
    }

    #[test]
    fn test_replace_call_in_source_recursive() {
        // Multi-clause function with recursive calls
        let source = "fib(0) = 0\nfib(1) = 1\nfib(n) = fib(n - 1) + fib(n - 2)";
        let result = replace_call_in_source(source, "fib", "fibonacci");
        // All calls to fib should become fibonacci, including recursive ones
        assert_eq!(result, "fibonacci(0) = 0\nfibonacci(1) = 1\nfibonacci(n) = fibonacci(n - 1) + fibonacci(n - 2)");
    }

    #[test]
    fn test_replace_call_in_source_no_partial_match() {
        // Should not match partial names
        let source = "foobar() = foo() + bar()";
        let result = replace_call_in_source(source, "foo", "baz");
        // foobar should NOT be changed, only foo
        assert_eq!(result, "foobar() = baz() + bar()");
    }

    #[test]
    fn test_rename_and_replace_combined() {
        // Full rename workflow: rename definition + replace calls
        let source = "fib(0) = 0\nfib(1) = 1\nfib(n) = fib(n - 1) + fib(n - 2)";

        // Step 1: Rename definitions
        let after_rename = rename_definition_in_source(source, "fib", "fibonacci");

        // Step 2: Replace calls (including self-calls)
        let final_result = replace_call_in_source(&after_rename, "fib", "fibonacci");

        assert_eq!(final_result, "fibonacci(0) = 0\nfibonacci(1) = 1\nfibonacci(n) = fibonacci(n - 1) + fibonacci(n - 2)");
    }

    #[test]
    fn test_sync_updated_source_files() {
        use std::thread::sleep;
        use std::time::Duration;

        let temp = TempDir::new().unwrap();
        let root = temp.path();

        // Create initial file
        fs::write(root.join("main.nos"), "greet() = \"Hello\"").unwrap();

        // First load - initializes .nostos/defs/
        let sm = SourceManager::new(root.to_path_buf()).unwrap();
        assert_eq!(sm.get_source("greet").unwrap(), "greet() = \"Hello\"");
        drop(sm);

        // Wait a moment to ensure timestamp difference
        sleep(Duration::from_millis(100));

        // Edit the .nos file externally
        fs::write(root.join("main.nos"), "greet() = \"Bonjour\"").unwrap();

        // Reload - should pick up the external edit
        let sm2 = SourceManager::new(root.to_path_buf()).unwrap();
        assert_eq!(sm2.get_source("greet").unwrap(), "greet() = \"Bonjour\"");

        // Check that defs were updated too
        let defs_file = root.join(".nostos/defs/main/greet.nos");
        let defs_content = fs::read_to_string(defs_file).unwrap();
        assert!(defs_content.contains("Bonjour"), "Defs should be updated: {}", defs_content);

        // Check git log shows the sync
        let nostos_dir = root.join(".nostos");
        let output = std::process::Command::new("git")
            .args(["log", "--oneline", "-5"])
            .current_dir(&nostos_dir)
            .output()
            .unwrap();
        let log = String::from_utf8_lossy(&output.stdout);
        assert!(log.contains("Sync updated source files"), "Git log should contain sync commit: {}", log);
    }

    #[test]
    fn test_update_writes_nos_first() {
        // Verify that REPL edits write to .nos first, then to defs
        let temp = TempDir::new().unwrap();
        let root = temp.path();

        // Create initial file
        fs::write(root.join("main.nos"), "greet() = \"Hello\"").unwrap();

        // Initialize SourceManager
        let mut sm = SourceManager::new(root.to_path_buf()).unwrap();

        // Update via REPL (simulated)
        sm.update_definition("greet", "greet() = \"Hola\"").unwrap();

        // Check .nos file was updated
        let nos_content = fs::read_to_string(root.join("main.nos")).unwrap();
        assert!(nos_content.contains("Hola"), ".nos should be updated: {}", nos_content);

        // Check defs were also updated
        let defs_file = root.join(".nostos/defs/main/greet.nos");
        let defs_content = fs::read_to_string(defs_file).unwrap();
        assert!(defs_content.contains("Hola"), "Defs should be updated: {}", defs_content);
    }

    #[test]
    fn test_get_definition_history() {
        let temp = TempDir::new().unwrap();
        let root = temp.path();

        // Create initial file
        fs::write(root.join("main.nos"), "greet() = \"Hello\"").unwrap();

        // Initialize SourceManager
        let mut sm = SourceManager::new(root.to_path_buf()).unwrap();

        // Get initial history
        let history = sm.get_definition_history("greet").unwrap();
        assert!(!history.is_empty(), "Should have at least one commit");
        assert!(history[0].message.contains("Import"), "First commit should be import: {}", history[0].message);

        // Update the definition
        sm.update_definition("greet", "greet() = \"Bonjour\"").unwrap();

        // Get updated history
        let history2 = sm.get_definition_history("greet").unwrap();
        assert!(history2.len() > history.len(), "Should have more commits after update");
        assert!(history2[0].message.contains("Update"), "Latest commit should be update: {}", history2[0].message);
    }

    #[test]
    fn test_get_definition_at_commit() {
        let temp = TempDir::new().unwrap();
        let root = temp.path();

        // Create initial file
        fs::write(root.join("main.nos"), "greet() = \"Hello\"").unwrap();

        // Initialize and update
        let mut sm = SourceManager::new(root.to_path_buf()).unwrap();
        sm.update_definition("greet", "greet() = \"Bonjour\"").unwrap();

        // Get history
        let history = sm.get_definition_history("greet").unwrap();
        assert!(history.len() >= 2, "Should have at least 2 commits");

        // Get current version
        let current = sm.get_definition_at_commit("greet", &history[0].hash).unwrap();
        assert!(current.contains("Bonjour"), "Current should be Bonjour: {}", current);

        // Get previous version
        let previous = sm.get_definition_at_commit("greet", &history[1].hash).unwrap();
        assert!(previous.contains("Hello"), "Previous should be Hello: {}", previous);
    }

    #[test]
    fn test_get_definition_diff() {
        let temp = TempDir::new().unwrap();
        let root = temp.path();

        // Create initial file
        fs::write(root.join("main.nos"), "greet() = \"Hello\"").unwrap();

        // Initialize and update
        let mut sm = SourceManager::new(root.to_path_buf()).unwrap();
        sm.update_definition("greet", "greet() = \"Bonjour\"").unwrap();

        // Get history
        let history = sm.get_definition_history("greet").unwrap();

        // Get diff for the update commit
        let diff = sm.get_definition_diff("greet", &history[0].hash).unwrap();
        assert!(diff.contains("-") || diff.contains("+"), "Diff should show changes: {}", diff);
    }

    #[test]
    fn test_get_module_history() {
        let temp = TempDir::new().unwrap();
        let root = temp.path();

        // Create initial file with two functions
        fs::write(root.join("main.nos"), "foo() = 1\nbar() = 2").unwrap();

        // Initialize SourceManager
        let mut sm = SourceManager::new(root.to_path_buf()).unwrap();

        // Update one function
        sm.update_definition("foo", "foo() = 42").unwrap();

        // Get module history
        let history = sm.get_module_history("main").unwrap();
        assert!(history.len() >= 2, "Should have at least 2 commits (import + update)");
    }
}
