use std::fs;
use std::path::Path;

fn main() {
    println!("cargo:rerun-if-changed=../../stdlib");

    let stdlib_dir = Path::new("../../stdlib");
    let out_dir = std::env::var("OUT_DIR").unwrap();

    let mut entries = Vec::new();
    collect_nos_files(stdlib_dir, stdlib_dir, &mut entries);

    // Sort entries for deterministic builds
    entries.sort_by(|a, b| a.0.cmp(&b.0));

    // Generate embedded stdlib module
    let mut code = String::from("pub const EMBEDDED_STDLIB: &[(&str, &str)] = &[\n");
    for (rel_path, content) in &entries {
        // Escape the content properly
        code.push_str(&format!("    ({:?}, {:?}),\n", rel_path, content));
    }
    code.push_str("];\n");

    // Also embed CORE_MODULES if it exists
    let core_modules_path = stdlib_dir.join("CORE_MODULES");
    if core_modules_path.exists() {
        let content = fs::read_to_string(&core_modules_path).unwrap_or_default();
        code.push_str(&format!("\npub const EMBEDDED_CORE_MODULES: &str = {:?};\n", content));
    } else {
        code.push_str("\npub const EMBEDDED_CORE_MODULES: &str = \"\";\n");
    }

    // Add version for cache invalidation
    code.push_str(&format!("\npub const STDLIB_VERSION: &str = {:?};\n", env!("CARGO_PKG_VERSION")));

    fs::write(Path::new(&out_dir).join("embedded_stdlib.rs"), code).unwrap();
    
    println!("cargo:warning=Embedded {} stdlib files", entries.len());
}

fn collect_nos_files(base: &Path, dir: &Path, entries: &mut Vec<(String, String)>) {
    if let Ok(read_dir) = fs::read_dir(dir) {
        for entry in read_dir.flatten() {
            let path = entry.path();
            if path.is_dir() {
                collect_nos_files(base, &path, entries);
            } else if path.extension().map_or(false, |e| e == "nos") {
                if let Ok(content) = fs::read_to_string(&path) {
                    let rel_path = path.strip_prefix(base).unwrap();
                    entries.push((rel_path.to_string_lossy().to_string(), content));
                }
            }
        }
    }
}
