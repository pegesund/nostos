//! Git integration for .nostos repository

use std::path::Path;
use std::process::Command;

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
