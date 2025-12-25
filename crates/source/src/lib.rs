//! Source Management for Nostos Projects
//!
//! This crate handles:
//! - Loading projects from directories
//! - Per-definition storage in `.nostos/defs/`
//! - Incremental compilation via content hashing
//! - Generating module files on `:w`

mod manager;
mod project;
mod definition;
mod module;
mod git;

pub use manager::SourceManager;
pub use project::ProjectConfig;
pub use definition::{Definition, DefinitionGroup, DefKind};
pub use module::Module;
pub use git::CommitInfo;
