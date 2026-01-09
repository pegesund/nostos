//! Call graph for tracking function dependencies.
//!
//! The call graph tracks which functions depend on which other functions.
//! This is used to determine what needs to be re-typechecked when a
//! function's type changes.

use std::collections::{HashMap, HashSet, VecDeque};

/// A directed graph tracking function call dependencies.
///
/// For each function, we track:
/// - `dependents`: which functions call this function
/// - `dependencies`: which functions this function calls
#[derive(Debug, Clone, Default)]
pub struct CallGraph {
    /// For each function, which functions call it?
    /// `foo -> {bar, baz}` means bar and baz depend on foo
    dependents: HashMap<String, HashSet<String>>,

    /// For each function, which functions does it call?
    /// `bar -> {foo}` means bar calls foo
    dependencies: HashMap<String, HashSet<String>>,
}

impl CallGraph {
    /// Create a new empty call graph.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add or update a function's dependencies.
    ///
    /// This removes old dependency edges and adds new ones.
    pub fn update(&mut self, name: &str, new_deps: HashSet<String>) {
        // Remove old dependency edges
        if let Some(old_deps) = self.dependencies.get(name) {
            for dep in old_deps.clone() {
                if let Some(dependents) = self.dependents.get_mut(&dep) {
                    dependents.remove(name);
                }
            }
        }

        // Add new dependency edges
        for dep in &new_deps {
            self.dependents
                .entry(dep.clone())
                .or_default()
                .insert(name.to_string());
        }

        // Update dependencies map
        if new_deps.is_empty() {
            self.dependencies.remove(name);
        } else {
            self.dependencies.insert(name.to_string(), new_deps);
        }
    }

    /// Remove a function from the graph entirely.
    pub fn remove(&mut self, name: &str) {
        // Remove from dependencies (what this function calls)
        if let Some(deps) = self.dependencies.remove(name) {
            for dep in deps {
                if let Some(dependents) = self.dependents.get_mut(&dep) {
                    dependents.remove(name);
                }
            }
        }

        // Remove from dependents (what calls this function)
        self.dependents.remove(name);

        // Also remove from all other dependents lists
        for (_, deps) in self.dependents.iter_mut() {
            deps.remove(name);
        }
    }

    /// Get the direct dependents of a function (functions that call it).
    pub fn direct_dependents(&self, name: &str) -> HashSet<String> {
        self.dependents.get(name).cloned().unwrap_or_default()
    }

    /// Get the direct dependencies of a function (functions it calls).
    pub fn direct_dependencies(&self, name: &str) -> HashSet<String> {
        self.dependencies.get(name).cloned().unwrap_or_default()
    }

    /// Get all transitive dependents of a function.
    ///
    /// This returns all functions that depend on `name`, directly or indirectly.
    /// Uses BFS to traverse the dependency graph.
    pub fn transitive_dependents(&self, name: &str) -> HashSet<String> {
        let mut result = HashSet::new();
        let mut queue = VecDeque::new();

        // Start with direct dependents
        if let Some(deps) = self.dependents.get(name) {
            for dep in deps {
                queue.push_back(dep.clone());
            }
        }

        // BFS to find all transitive dependents
        while let Some(current) = queue.pop_front() {
            if result.insert(current.clone()) {
                if let Some(deps) = self.dependents.get(&current) {
                    for dep in deps {
                        if !result.contains(dep) {
                            queue.push_back(dep.clone());
                        }
                    }
                }
            }
        }

        result
    }

    /// Topologically sort a set of function names.
    ///
    /// Returns functions in an order such that if A depends on B,
    /// then B comes before A. This is the order in which functions
    /// should be re-typechecked.
    ///
    /// Only considers edges within the given set.
    pub fn topological_sort(&self, names: &HashSet<String>) -> Vec<String> {
        // Calculate in-degrees (within the given set)
        let mut in_degree: HashMap<String, usize> = HashMap::new();
        let mut queue = VecDeque::new();

        for name in names {
            let deps = self.dependencies.get(name);
            let count = deps
                .map(|d| d.iter().filter(|dep| names.contains(*dep)).count())
                .unwrap_or(0);
            in_degree.insert(name.clone(), count);
            if count == 0 {
                queue.push_back(name.clone());
            }
        }

        // Kahn's algorithm
        let mut result = Vec::new();
        while let Some(name) = queue.pop_front() {
            result.push(name.clone());
            if let Some(dependents) = self.dependents.get(&name) {
                for dep in dependents {
                    if names.contains(dep) {
                        if let Some(count) = in_degree.get_mut(dep) {
                            *count -= 1;
                            if *count == 0 {
                                queue.push_back(dep.clone());
                            }
                        }
                    }
                }
            }
        }

        // If result is shorter than names, there's a cycle
        // For now, just append remaining items (cycle handling can be improved)
        for name in names {
            if !result.contains(name) {
                result.push(name.clone());
            }
        }

        result
    }

    /// Check if a function has any dependents.
    pub fn has_dependents(&self, name: &str) -> bool {
        self.dependents
            .get(name)
            .map(|d| !d.is_empty())
            .unwrap_or(false)
    }

    /// Get all function names in the graph.
    pub fn all_functions(&self) -> HashSet<String> {
        let mut result: HashSet<String> = self.dependencies.keys().cloned().collect();
        for deps in self.dependents.values() {
            result.extend(deps.iter().cloned());
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_graph() {
        let graph = CallGraph::new();
        assert!(graph.direct_dependents("foo").is_empty());
        assert!(graph.direct_dependencies("foo").is_empty());
        assert!(graph.transitive_dependents("foo").is_empty());
    }

    #[test]
    fn test_simple_dependency() {
        let mut graph = CallGraph::new();

        // bar depends on foo
        let mut bar_deps = HashSet::new();
        bar_deps.insert("foo".to_string());
        graph.update("bar", bar_deps);

        assert_eq!(
            graph.direct_dependents("foo"),
            HashSet::from(["bar".to_string()])
        );
        assert_eq!(
            graph.direct_dependencies("bar"),
            HashSet::from(["foo".to_string()])
        );
    }

    #[test]
    fn test_transitive_dependents() {
        let mut graph = CallGraph::new();

        // bar depends on foo
        graph.update("bar", HashSet::from(["foo".to_string()]));
        // baz depends on bar
        graph.update("baz", HashSet::from(["bar".to_string()]));
        // qux depends on baz
        graph.update("qux", HashSet::from(["baz".to_string()]));

        // foo's transitive dependents should be bar, baz, qux
        let deps = graph.transitive_dependents("foo");
        assert_eq!(
            deps,
            HashSet::from(["bar".to_string(), "baz".to_string(), "qux".to_string()])
        );

        // bar's transitive dependents should be baz, qux
        let deps = graph.transitive_dependents("bar");
        assert_eq!(deps, HashSet::from(["baz".to_string(), "qux".to_string()]));
    }

    #[test]
    fn test_topological_sort() {
        let mut graph = CallGraph::new();

        // bar depends on foo
        graph.update("bar", HashSet::from(["foo".to_string()]));
        // baz depends on bar
        graph.update("baz", HashSet::from(["bar".to_string()]));

        let names = HashSet::from(["foo".to_string(), "bar".to_string(), "baz".to_string()]);
        let sorted = graph.topological_sort(&names);

        // foo should come before bar, bar should come before baz
        let foo_pos = sorted.iter().position(|x| x == "foo").unwrap();
        let bar_pos = sorted.iter().position(|x| x == "bar").unwrap();
        let baz_pos = sorted.iter().position(|x| x == "baz").unwrap();

        assert!(foo_pos < bar_pos);
        assert!(bar_pos < baz_pos);
    }

    #[test]
    fn test_update_removes_old_edges() {
        let mut graph = CallGraph::new();

        // bar depends on foo
        graph.update("bar", HashSet::from(["foo".to_string()]));
        assert_eq!(
            graph.direct_dependents("foo"),
            HashSet::from(["bar".to_string()])
        );

        // Now bar depends on baz instead
        graph.update("bar", HashSet::from(["baz".to_string()]));
        assert!(graph.direct_dependents("foo").is_empty());
        assert_eq!(
            graph.direct_dependents("baz"),
            HashSet::from(["bar".to_string()])
        );
    }

    #[test]
    fn test_remove_function() {
        let mut graph = CallGraph::new();

        graph.update("bar", HashSet::from(["foo".to_string()]));
        graph.update("baz", HashSet::from(["bar".to_string()]));

        graph.remove("bar");

        assert!(graph.direct_dependents("foo").is_empty());
        assert!(graph.direct_dependents("bar").is_empty());
        assert!(graph.direct_dependencies("bar").is_empty());
    }

    #[test]
    fn test_diamond_dependency() {
        let mut graph = CallGraph::new();

        // Diamond: d depends on b and c, both depend on a
        //     a
        //    / \
        //   b   c
        //    \ /
        //     d
        graph.update("b", HashSet::from(["a".to_string()]));
        graph.update("c", HashSet::from(["a".to_string()]));
        graph.update("d", HashSet::from(["b".to_string(), "c".to_string()]));

        // a's transitive dependents should be b, c, d
        let deps = graph.transitive_dependents("a");
        assert_eq!(
            deps,
            HashSet::from(["b".to_string(), "c".to_string(), "d".to_string()])
        );
    }

    #[test]
    fn test_multiple_dependencies() {
        let mut graph = CallGraph::new();

        // bar depends on foo and baz
        graph.update(
            "bar",
            HashSet::from(["foo".to_string(), "baz".to_string()]),
        );

        assert_eq!(
            graph.direct_dependents("foo"),
            HashSet::from(["bar".to_string()])
        );
        assert_eq!(
            graph.direct_dependents("baz"),
            HashSet::from(["bar".to_string()])
        );
    }
}
