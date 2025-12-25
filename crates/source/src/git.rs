//! Git integration for .nostos repository

use std::path::Path;
use std::process::Command;

/// Information about a git commit
#[derive(Debug, Clone)]
pub struct CommitInfo {
    /// Full commit hash
    pub hash: String,
    /// Short commit hash (7 chars)
    pub short_hash: String,
    /// Commit message (first line)
    pub message: String,
    /// Commit date (ISO format)
    pub date: String,
    /// Author name
    pub author: String,
}

impl CommitInfo {
    /// Parse a commit info from git log format string
    /// Expected format: "hash|short_hash|date|author|message"
    fn parse(line: &str) -> Option<Self> {
        let parts: Vec<&str> = line.splitn(5, '|').collect();
        if parts.len() < 5 {
            return None;
        }
        Some(CommitInfo {
            hash: parts[0].to_string(),
            short_hash: parts[1].to_string(),
            date: parts[2].to_string(),
            author: parts[3].to_string(),
            message: parts[4].to_string(),
        })
    }
}

/// Initialize git repository if not already initialized
pub fn init_repo(nostos_dir: &Path) -> Result<(), String> {
    let git_dir = nostos_dir.join(".git");
    if git_dir.exists() {
        return Ok(());
    }

    // Create .nostos directory if needed
    std::fs::create_dir_all(nostos_dir)
        .map_err(|e| format!("Failed to create .nostos: {}", e))?;

    // git init
    let output = Command::new("git")
        .args(["init"])
        .current_dir(nostos_dir)
        .output()
        .map_err(|e| format!("Failed to run git init: {}", e))?;

    if !output.status.success() {
        return Err(format!(
            "git init failed: {}",
            String::from_utf8_lossy(&output.stderr)
        ));
    }

    // Create .gitignore
    let gitignore = nostos_dir.join(".gitignore");
    std::fs::write(&gitignore, "*.tmp\n*.swp\n")
        .map_err(|e| format!("Failed to create .gitignore: {}", e))?;

    // Initial commit
    add_and_commit(nostos_dir, &[".gitignore"], "Initialize .nostos repository")?;

    Ok(())
}

/// Stage files and commit
pub fn add_and_commit(nostos_dir: &Path, files: &[&str], message: &str) -> Result<(), String> {
    // git add
    let mut add_cmd = Command::new("git");
    add_cmd.arg("add").current_dir(nostos_dir);
    for file in files {
        add_cmd.arg(file);
    }

    let output = add_cmd
        .output()
        .map_err(|e| format!("Failed to run git add: {}", e))?;

    if !output.status.success() {
        return Err(format!(
            "git add failed: {}",
            String::from_utf8_lossy(&output.stderr)
        ));
    }

    // git commit
    let output = Command::new("git")
        .args(["commit", "-m", message])
        .current_dir(nostos_dir)
        .output()
        .map_err(|e| format!("Failed to run git commit: {}", e))?;

    // Commit can "fail" if there's nothing to commit, which is fine
    // Note: "nothing to commit" message goes to stdout, not stderr
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stdout = String::from_utf8_lossy(&output.stdout);
        if !stderr.contains("nothing to commit") && !stdout.contains("nothing to commit") {
            return Err(format!("git commit failed: {}{}", stderr, stdout));
        }
    }

    Ok(())
}

/// Stage a single file and commit
pub fn commit_file(nostos_dir: &Path, relative_path: &str, message: &str) -> Result<(), String> {
    add_and_commit(nostos_dir, &[relative_path], message)
}

/// Delete a file and commit the deletion
pub fn delete_and_commit(nostos_dir: &Path, relative_path: &str, message: &str) -> Result<(), String> {
    let full_path = nostos_dir.join(relative_path);

    // Remove file
    if full_path.exists() {
        std::fs::remove_file(&full_path)
            .map_err(|e| format!("Failed to delete file: {}", e))?;
    }

    // git add the deletion
    let output = Command::new("git")
        .args(["add", relative_path])
        .current_dir(nostos_dir)
        .output()
        .map_err(|e| format!("Failed to run git add: {}", e))?;

    if !output.status.success() {
        // File might not be tracked, which is fine
        let stderr = String::from_utf8_lossy(&output.stderr);
        if !stderr.contains("did not match any files") {
            return Err(format!("git add failed: {}", stderr));
        }
    }

    // Commit
    let output = Command::new("git")
        .args(["commit", "-m", message])
        .current_dir(nostos_dir)
        .output()
        .map_err(|e| format!("Failed to run git commit: {}", e))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        if !stderr.contains("nothing to commit") {
            return Err(format!("git commit failed: {}", stderr));
        }
    }

    Ok(())
}

/// Move/rename a file and commit
pub fn move_and_commit(
    nostos_dir: &Path,
    from_path: &str,
    to_path: &str,
    message: &str,
) -> Result<(), String> {
    // Ensure destination directory exists
    let to_full = nostos_dir.join(to_path);
    if let Some(parent) = to_full.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| format!("Failed to create directory: {}", e))?;
    }

    // git mv
    let output = Command::new("git")
        .args(["mv", from_path, to_path])
        .current_dir(nostos_dir)
        .output()
        .map_err(|e| format!("Failed to run git mv: {}", e))?;

    if !output.status.success() {
        return Err(format!(
            "git mv failed: {}",
            String::from_utf8_lossy(&output.stderr)
        ));
    }

    // Commit
    let output = Command::new("git")
        .args(["commit", "-m", message])
        .current_dir(nostos_dir)
        .output()
        .map_err(|e| format!("Failed to run git commit: {}", e))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        if !stderr.contains("nothing to commit") {
            return Err(format!("git commit failed: {}", stderr));
        }
    }

    Ok(())
}

/// Rename a file (delete old, add new) and commit
/// Used when the file content also changes (like renaming a definition)
pub fn rename_and_commit(
    nostos_dir: &Path,
    old_path: &str,
    new_path: &str,
    message: &str,
) -> Result<(), String> {
    // Stage the deletion of old file
    // Ignore errors for old path (might not be tracked)
    let _ = Command::new("git")
        .args(["add", old_path])
        .current_dir(nostos_dir)
        .output();

    // Stage the new file
    let output = Command::new("git")
        .args(["add", new_path])
        .current_dir(nostos_dir)
        .output()
        .map_err(|e| format!("Failed to run git add (new): {}", e))?;

    if !output.status.success() {
        return Err(format!(
            "git add failed: {}",
            String::from_utf8_lossy(&output.stderr)
        ));
    }

    // Commit both changes
    let output = Command::new("git")
        .args(["commit", "-m", message])
        .current_dir(nostos_dir)
        .output()
        .map_err(|e| format!("Failed to run git commit: {}", e))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stdout = String::from_utf8_lossy(&output.stdout);
        if !stderr.contains("nothing to commit") && !stdout.contains("nothing to commit") {
            return Err(format!("git commit failed: {}{}", stderr, stdout));
        }
    }

    Ok(())
}

/// Check if repository has uncommitted changes
#[allow(dead_code)]
pub fn has_uncommitted_changes(nostos_dir: &Path) -> bool {
    let output = Command::new("git")
        .args(["status", "--porcelain"])
        .current_dir(nostos_dir)
        .output();

    match output {
        Ok(out) => !out.stdout.is_empty(),
        Err(_) => false,
    }
}

/// Get commit history for a file
/// Returns list of commits that touched the file, newest first
pub fn get_file_history(nostos_dir: &Path, relative_path: &str) -> Result<Vec<CommitInfo>, String> {
    let output = Command::new("git")
        .args([
            "log",
            "--format=%H|%h|%ai|%an|%s",
            "--",
            relative_path,
        ])
        .current_dir(nostos_dir)
        .output()
        .map_err(|e| format!("Failed to run git log: {}", e))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        // Empty history is not an error
        if stderr.contains("does not have any commits") {
            return Ok(Vec::new());
        }
        return Err(format!("git log failed: {}", stderr));
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let commits: Vec<CommitInfo> = stdout
        .lines()
        .filter_map(CommitInfo::parse)
        .collect();

    Ok(commits)
}

/// Get commit history for a directory (all files in it)
/// Returns list of commits that touched any file in the directory, newest first
pub fn get_directory_history(nostos_dir: &Path, relative_dir: &str) -> Result<Vec<CommitInfo>, String> {
    let dir_path = if relative_dir.ends_with('/') {
        relative_dir.to_string()
    } else {
        format!("{}/", relative_dir)
    };

    let output = Command::new("git")
        .args([
            "log",
            "--format=%H|%h|%ai|%an|%s",
            "--",
            &dir_path,
        ])
        .current_dir(nostos_dir)
        .output()
        .map_err(|e| format!("Failed to run git log: {}", e))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        if stderr.contains("does not have any commits") {
            return Ok(Vec::new());
        }
        return Err(format!("git log failed: {}", stderr));
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let commits: Vec<CommitInfo> = stdout
        .lines()
        .filter_map(CommitInfo::parse)
        .collect();

    Ok(commits)
}

/// Get file content at a specific commit
pub fn get_file_at_commit(
    nostos_dir: &Path,
    commit: &str,
    relative_path: &str,
) -> Result<String, String> {
    let file_spec = format!("{}:{}", commit, relative_path);

    let output = Command::new("git")
        .args(["show", &file_spec])
        .current_dir(nostos_dir)
        .output()
        .map_err(|e| format!("Failed to run git show: {}", e))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("git show failed: {}", stderr));
    }

    Ok(String::from_utf8_lossy(&output.stdout).to_string())
}

/// Get the diff for a file at a specific commit
/// Shows what changed in that commit for the given file
pub fn get_file_diff(
    nostos_dir: &Path,
    commit: &str,
    relative_path: &str,
) -> Result<String, String> {
    let output = Command::new("git")
        .args(["show", "--format=", commit, "--", relative_path])
        .current_dir(nostos_dir)
        .output()
        .map_err(|e| format!("Failed to run git show: {}", e))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("git show failed: {}", stderr));
    }

    Ok(String::from_utf8_lossy(&output.stdout).to_string())
}

/// Get diff between two commits for a file
pub fn get_file_diff_between(
    nostos_dir: &Path,
    from_commit: &str,
    to_commit: &str,
    relative_path: &str,
) -> Result<String, String> {
    let output = Command::new("git")
        .args(["diff", from_commit, to_commit, "--", relative_path])
        .current_dir(nostos_dir)
        .output()
        .map_err(|e| format!("Failed to run git diff: {}", e))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("git diff failed: {}", stderr));
    }

    Ok(String::from_utf8_lossy(&output.stdout).to_string())
}
