//! Trait definition and implementation compilation.
//!
//! Contains functions for compiling trait definitions, trait implementations,
//! and helper utilities for trait-related type checking.

use super::*;

impl Compiler {
    /// Compile a trait definition.
    pub(super) fn compile_trait_def(&mut self, def: &TraitDef) -> Result<(), CompileError> {
        // Skip if already registered (idempotent - avoids double registration
        // when pre_register_module_metadata runs before compile_items)
        let check_name = self.qualify_name(&def.name.node);
        if self.trait_defs.contains_key(&check_name) {
            // Even though trait_defs was populated by forward declarations (Pass 1),
            // we still need to store the full AST for default method compilation.
            // Pass 1 doesn't have access to store trait_defs_ast.
            if def.methods.iter().any(|m| m.default_impl.is_some()) {
                if !self.trait_defs_ast.contains_key(&check_name) {
                    self.trait_defs_ast.insert(check_name, def.clone());
                }
            }
            return Ok(());
        }

        // Check that trait method names don't shadow built-in functions
        // Exception: Some traits are designed to integrate with builtins (Show.show, Hash.hash, etc.)
        let trait_name = &def.name.node;
        let is_builtin_trait_integration = |trait_name: &str, method_name: &str| -> bool {
            matches!((trait_name, method_name),
                ("Show", "show") |
                ("Hash", "hash") |
                ("Eq", "eq") |
                ("Copy", "copy") |
                ("Ord", "compare") |
                // Logger trait methods don't conflict because they have different signatures
                ("Logger", "log") |
                ("Logger", "flush")
            )
        };

        for method in &def.methods {
            if is_builtin_trait_integration(trait_name, &method.name.node) {
                continue; // These are intentional integrations
            }
            if let Some((builtin_name, builtin_doc)) = check_builtin_shadowing(&method.name.node) {
                return Err(CompileError::DefinitionError {
                    message: format!(
                        "Trait method '{}' shadows built-in function '{}'. \
                        This would cause conflicts when the method is called.\n\
                        Built-in '{}': {}\n\
                        Consider renaming the trait method to avoid this conflict.",
                        method.name.node, builtin_name, builtin_name, builtin_doc
                    ),
                    span: method.name.span,
                });
            }
        }

        // Qualify trait name with module prefix
        let name = self.qualify_name(&def.name.node);

        let super_traits: Vec<String> = def.super_traits
            .iter()
            .map(|t| t.node.clone())
            .collect();

        // Validate supertraits exist and check for cycles
        for (i, supertrait_ident) in def.super_traits.iter().enumerate() {
            let supertrait_name = &super_traits[i];
            // Try to resolve the supertrait name (qualified, imported, or builtin)
            let qualified_supertrait = self.qualify_name(supertrait_name);
            let imported_supertrait = self.imports.get(supertrait_name).cloned();
            let supertrait_exists = self.trait_defs.contains_key(&qualified_supertrait)
                || self.trait_defs.contains_key(supertrait_name)
                || imported_supertrait.as_ref().map(|n| self.trait_defs.contains_key(n)).unwrap_or(false)
                || self.is_builtin_derivable_trait(supertrait_name);

            if !supertrait_exists {
                return Err(CompileError::UnknownTrait {
                    name: supertrait_name.clone(),
                    span: supertrait_ident.span,
                });
            }

            // Check for cycles: follow the supertrait chain and see if we reach back to this trait
            let resolved_supertrait = if self.trait_defs.contains_key(&qualified_supertrait) {
                qualified_supertrait.clone()
            } else if let Some(ref imported) = imported_supertrait {
                if self.trait_defs.contains_key(imported) {
                    imported.clone()
                } else {
                    supertrait_name.clone()
                }
            } else {
                supertrait_name.clone()
            };

            if let Some(cycle) = self.detect_trait_cycle(&name, &resolved_supertrait, &mut vec![name.clone()]) {
                return Err(CompileError::TraitCycle {
                    cycle,
                    span: supertrait_ident.span,
                });
            }
        }

        let methods: Vec<TraitMethodInfo> = def.methods
            .iter()
            .map(|m| {
                let param_types: Vec<(String, String)> = m.params.iter()
                    .map(|p| {
                        let pname = self.pattern_binding_name(&p.pattern)
                            .unwrap_or_else(|| "_".to_string());
                        let ptype = p.ty.as_ref()
                            .map(|ty| self.type_expr_to_string(ty))
                            .unwrap_or_else(|| "_".to_string());
                        (pname, ptype)
                    })
                    .collect();
                TraitMethodInfo {
                    name: m.name.node.clone(),
                    param_count: m.params.len(),
                    has_default: m.default_impl.is_some(),
                    return_type: m.return_type.as_ref()
                        .map(|ty| {
                            let s = self.type_expr_to_string(ty);
                            // Normalize lowercase "self" to "Self" so downstream
                            // code that does .replace("Self", impl_type) works correctly.
                            if s == "self" { "Self".to_string() } else { s }
                        })
                        .unwrap_or_else(|| "()".to_string()),
                    param_types,
                }
            })
            .collect();

        // Store the full AST if it has any default methods (needed for compile_trait_impl)
        if def.methods.iter().any(|m| m.default_impl.is_some()) {
            self.trait_defs_ast.insert(name.clone(), def.clone());
        }

        self.trait_defs.insert(name.clone(), TraitInfo {
            name,
            visibility: def.visibility,
            super_traits,
            methods,
        });

        Ok(())
    }

    /// Check if a trait is a built-in derivable trait.
    pub(super) fn is_builtin_derivable_trait(&self, name: &str) -> bool {
        matches!(name, "Hash" | "Show" | "Copy" | "Eq" | "Num" | "Ord")
    }

    /// Unified UFCS signature registration for trait methods.
    ///
    /// Registers a trait method's type signature under ALL key variants needed
    /// for method resolution. This ensures both local and cross-module lookups
    /// can find the method, regardless of whether the caller uses a qualified
    /// or unqualified type name.
    ///
    /// Key variants registered:
    /// - `{BareType}.{method}/{arity}` (e.g., "Either.mapLeft/_,_")
    /// - `{Module.BareType}.{method}/{arity}` (e.g., "types.Either.mapLeft/_,_")
    /// - fn_asts entry under bare UFCS key for named argument resolution
    pub(super) fn register_ufcs_method(
        &mut self,
        unqualified_type_name: &str,
        qualified_type_name: &str,
        method_name: &str,
        fn_type: nostos_types::FunctionType,
        method_def: Option<&FnDef>,
    ) {
        // Strip type args from type names (e.g., "Either[Int, String]" -> "Either")
        let bare_unqualified = if let Some(bracket_pos) = unqualified_type_name.find('[') {
            &unqualified_type_name[..bracket_pos]
        } else {
            unqualified_type_name
        };
        let bare_qualified = if let Some(bracket_pos) = qualified_type_name.find('[') {
            &qualified_type_name[..bracket_pos]
        } else {
            qualified_type_name
        };

        // Build arity suffix: "/_,_,..." for N params, "/" for 0 params
        let arity = fn_type.params.len();
        let ufcs_arity_suffix = if arity == 0 {
            "/".to_string()
        } else {
            format!("/{}", vec!["_"; arity].join(","))
        };

        // Register under bare (unqualified) type name
        let bare_ufcs_key = format!("{}.{}", bare_unqualified, method_name);
        let bare_ufcs_fn_name = format!("{}{}", bare_ufcs_key, ufcs_arity_suffix);
        self.trait_method_ufcs_signatures.insert(
            bare_ufcs_fn_name.clone(),
            fn_type.clone(),
        );

        // Register fn_asts entry under UFCS key for named argument resolution.
        // Without this, trait method calls with named args that skip defaults
        // (e.g., s.setup(port: 3000)) fail because param names lookup uses
        // the UFCS key "Server.setup" but fn_asts has "Server.Configurable.setup".
        if let Some(def) = method_def {
            if !self.fn_asts.contains_key(&bare_ufcs_fn_name) {
                self.fn_asts.insert(bare_ufcs_fn_name, def.clone());
            }
        }

        // Register under qualified type name for cross-module lookups.
        // e.g., "shapes.Circle.area/_" so that imported type names can find the method.
        if bare_qualified != bare_unqualified {
            let qualified_ufcs_key = format!("{}.{}", bare_qualified, method_name);
            let qualified_ufcs_fn_name = format!("{}{}", qualified_ufcs_key, ufcs_arity_suffix);
            self.trait_method_ufcs_signatures.insert(
                qualified_ufcs_fn_name.clone(),
                fn_type,
            );

            // Also register fn_asts under qualified key for cross-module named arg resolution.
            // Without this, `app.configure(level: 3)` from another module fails because
            // get_function_param_names can't find param names under the qualified key.
            if let Some(def) = method_def {
                if !self.fn_asts.contains_key(&qualified_ufcs_fn_name) {
                    self.fn_asts.insert(qualified_ufcs_fn_name, def.clone());
                }
            }
        }
    }

    /// Check if a type is a numeric type that implements Num trait.
    pub(super) fn is_numeric_type(type_name: &str) -> bool {
        matches!(type_name,
            "Int" | "Int8" | "Int16" | "Int32" | "Int64" |
            "UInt8" | "UInt16" | "UInt32" | "UInt64" |
            "Float" | "Float32" | "Float64" |
            "BigInt" | "Decimal"
        )
    }

    /// Check if a type is an orderable type that implements Ord trait.
    pub(super) fn is_orderable_type(type_name: &str) -> bool {
        // All numeric types, Char (code point comparison), and String (lexicographic comparison)
        Self::is_numeric_type(type_name) || matches!(type_name, "Char" | "String")
    }

    /// Parse a simple type string back to TypeExpr.
    /// Used to inject return type annotations from trait definitions into implementations.
    pub(super) fn parse_return_type_expr(&self, ty_str: &str) -> Option<TypeExpr> {
        let ty_str = ty_str.trim();
        if ty_str == "()" {
            return Some(TypeExpr::Unit);
        }
        // Handle generic types like "List[Int]", "Option[String]", "Map[String, List[(Int, Int)]]"
        if let Some(bracket_pos) = ty_str.find('[') {
            if ty_str.ends_with(']') {
                let base = &ty_str[..bracket_pos];
                let params_str = &ty_str[bracket_pos + 1..ty_str.len() - 1];
                // Depth-aware split on commas (respects nested parens, brackets, braces)
                let param_strs = Self::split_type_string_at_commas(params_str);
                let params: Vec<TypeExpr> = param_strs.iter()
                    .filter_map(|p| self.parse_return_type_expr(p.trim()))
                    .collect();
                return Some(TypeExpr::Generic(
                    Spanned::new(base.to_string(), Span::default()),
                    params,
                ));
            }
        }
        // Handle tuple types like "(Int, String)", "(Int, (String, Bool))"
        if ty_str.starts_with('(') && ty_str.ends_with(')') {
            let inner = &ty_str[1..ty_str.len() - 1];
            // Check for depth-0 comma to distinguish tuple from grouped expression
            let has_comma = {
                let mut depth = 0;
                let mut found = false;
                for c in inner.chars() {
                    match c {
                        '(' | '[' | '{' => depth += 1,
                        ')' | ']' | '}' => depth -= 1,
                        ',' if depth == 0 => { found = true; break; }
                        _ => {}
                    }
                }
                found
            };
            if has_comma {
                let elem_strs = Self::split_type_string_at_commas(inner);
                let elems: Vec<TypeExpr> = elem_strs.iter()
                    .filter_map(|p| self.parse_return_type_expr(p.trim()))
                    .collect();
                return Some(TypeExpr::Tuple(elems));
            }
            // No comma - parenthesized single type, unwrap
            return self.parse_return_type_expr(inner);
        }
        // Simple named type
        Some(TypeExpr::Name(Spanned::new(ty_str.to_string(), Span::default())))
    }

    /// Split a type string at top-level commas, respecting nested parens, brackets, and braces.
    pub(super) fn split_type_string_at_commas(s: &str) -> Vec<String> {
        let mut result = Vec::new();
        let mut current = String::new();
        let mut depth = 0;
        for c in s.chars() {
            match c {
                '(' | '[' | '{' => {
                    depth += 1;
                    current.push(c);
                }
                ')' | ']' | '}' => {
                    depth -= 1;
                    current.push(c);
                }
                ',' if depth == 0 => {
                    let trimmed = current.trim().to_string();
                    if !trimmed.is_empty() {
                        result.push(trimmed);
                    }
                    current.clear();
                }
                _ => current.push(c),
            }
        }
        let trimmed = current.trim().to_string();
        if !trimmed.is_empty() {
            result.push(trimmed);
        }
        result
    }

    /// Compile a trait implementation.
    /// Pre-register a trait impl's methods without compiling bodies.
    /// This must be called for ALL trait impls before any bodies are compiled,
    /// so that trait methods defined later in the file are visible from earlier impls.
    pub(super) fn pre_register_trait_impl(&mut self, impl_def: &TraitImpl) -> Result<(), CompileError> {
        self.compile_trait_impl_inner(impl_def, true)
    }

    pub(super) fn compile_trait_impl(&mut self, impl_def: &TraitImpl) -> Result<(), CompileError> {
        self.compile_trait_impl_inner(impl_def, false)
    }

    pub(super) fn compile_trait_impl_inner(&mut self, impl_def: &TraitImpl, register_only: bool) -> Result<(), CompileError> {
        // Get the type name from the type expression
        // Use unqualified name for method names (compile_fn_def will add module prefix)
        // Use qualified name for type_traits registration and param_types
        // BUT: builtin types should NOT be qualified (Int, String, etc. are global)
        let unqualified_type_name = self.type_expr_to_string(&impl_def.ty);
        let qualified_type_name = if self.is_builtin_type_name(&unqualified_type_name) {
            unqualified_type_name.clone()
        } else {
            // First try to resolve through imports (e.g., "Box" -> "Containers.Box")
            // This handles the case where a type is imported from a module and a trait
            // is implemented on it outside the module.
            // For specialized types like "Container[Int]", extract the base name "Container"
            // for import lookup, then reattach the type args to the qualified name.
            let (base_name, type_args_suffix) = if let Some(bracket_pos) = unqualified_type_name.find('[') {
                (&unqualified_type_name[..bracket_pos], Some(&unqualified_type_name[bracket_pos..]))
            } else {
                (unqualified_type_name.as_str(), None)
            };
            if let Some(imported) = self.imports.get(base_name) {
                if let Some(suffix) = type_args_suffix {
                    format!("{}{}", imported, suffix)
                } else {
                    imported.clone()
                }
            } else if let Some(imported) = self.imports.get(&unqualified_type_name) {
                imported.clone()
            } else {
                self.qualify_name(&unqualified_type_name)
            }
        };
        // Check if this is a bare (unparameterized) impl on a generic type.
        // E.g., `Box: Showable` where Box has type params [A] - the impl doesn't
        // specify type args, so self should use "_" (wildcard) to avoid TypeArityMismatch.
        // But `Container[Int]: HasValue` specifies the type arg, so self should be
        // "Container[Int]" which has correct arity.
        let impl_specifies_type_args = unqualified_type_name.contains('[');
        let type_def_for_impl = self.type_defs.get(&qualified_type_name)
            .or_else(|| self.type_defs.get(&unqualified_type_name));
        let impl_type_is_generic = !impl_specifies_type_args
            && type_def_for_impl.map(|td| !td.type_params.is_empty()).unwrap_or(false);
        // For generic types without specified type args, build a parameterized type string
        // with lowercase type variables. E.g., Box[A] → "Box[a]", Pair[A,B] → "Pair[a, b]".
        // This preserves the base type name (for field access/trait dispatch) while using
        // fresh type vars (to avoid TypeArityMismatch with concrete instantiations).
        let generic_self_type = if impl_type_is_generic {
            type_def_for_impl.map(|td| {
                let vars: Vec<String> = td.type_params.iter()
                    .map(|p| p.name.node.to_lowercase())
                    .collect();
                format!("{}[{}]", unqualified_type_name, vars.join(", "))
            })
        } else {
            None
        };

        let unqualified_trait_name = impl_def.trait_name.node.clone();
        // Qualify trait name for lookup (trait defined in same module)
        let qualified_trait_name = self.qualify_name(&unqualified_trait_name);

        // Check that the trait exists (unless it's a built-in derivable trait)
        // Try: qualified name, unqualified name, imported name, or builtin
        let imported_trait_name = self.imports.get(&unqualified_trait_name).cloned();
        let trait_exists = self.trait_defs.contains_key(&qualified_trait_name)
            || self.trait_defs.contains_key(&unqualified_trait_name)
            || imported_trait_name.as_ref().map(|n| self.trait_defs.contains_key(n)).unwrap_or(false)
            || self.is_builtin_derivable_trait(&unqualified_trait_name);
        if !trait_exists {
            return Err(CompileError::UnknownTrait {
                name: unqualified_trait_name,
                span: impl_def.trait_name.span,
            });
        }
        // Use the name that exists in trait_defs
        let trait_name = if self.trait_defs.contains_key(&qualified_trait_name) {
            qualified_trait_name
        } else if let Some(ref imported) = imported_trait_name {
            if self.trait_defs.contains_key(imported) {
                imported.clone()
            } else {
                unqualified_trait_name
            }
        } else {
            unqualified_trait_name
        };

        // Check that all supertraits are already implemented for this type
        if let Some(trait_info) = self.trait_defs.get(&trait_name) {
            let super_traits = trait_info.super_traits.clone();
            for supertrait in &super_traits {
                // Check all supertraits (including transitive ones)
                if !self.type_implements_trait_recursive(&qualified_type_name, supertrait) {
                    return Err(CompileError::MissingSupertraitImpl {
                        type_name: unqualified_type_name.clone(),
                        trait_name: trait_name.clone(),
                        supertrait: supertrait.clone(),
                        span: impl_def.trait_name.span,
                    });
                }
            }
        }

        // Record method names for deferred completeness check. Methods may be spread
        // across multiple impl blocks for the same (type, trait) pair.
        {
            let key = (unqualified_type_name.clone(), trait_name.clone());
            let method_names_this_block: HashSet<String> = impl_def.methods.iter()
                .map(|m| m.name.node.clone())
                .collect();
            let entry = self.pending_trait_completeness
                .entry(key)
                .or_insert_with(|| (HashSet::new(), impl_def.trait_name.span));
            entry.0.extend(method_names_this_block);
        }

        // Check that impl method signatures match the trait definition
        if let Some(trait_info) = self.trait_defs.get(&trait_name).cloned() {
            for impl_method in &impl_def.methods {
                let method_name = impl_method.name.node.as_str();
                if let Some(trait_method) = trait_info.methods.iter().find(|m| m.name == method_name) {
                    if let Some(clause) = impl_method.clauses.first() {
                        let impl_param_count = clause.params.len();
                        let trait_param_count = trait_method.param_count;

                        // Check parameter count matches
                        if impl_param_count != trait_param_count {
                            return Err(CompileError::TraitMethodSignatureMismatch {
                                method: method_name.to_string(),
                                ty: unqualified_type_name.clone(),
                                trait_name: trait_name.clone(),
                                detail: format!(
                                    "expected {} parameter(s), found {}",
                                    trait_param_count, impl_param_count
                                ),
                                span: impl_method.name.span,
                            });
                        }

                        // Check parameter types match (where both are explicitly annotated)
                        for (i, impl_param) in clause.params.iter().enumerate() {
                            if let Some(ref impl_ty) = impl_param.ty {
                                if let Some((_, ref trait_ty_str)) = trait_method.param_types.get(i) {
                                    if trait_ty_str != "_" && !trait_ty_str.contains("->") {
                                        let impl_ty_str = self.type_expr_to_string(impl_ty);
                                        // Substitute Self with the implementing type name for comparison
                                        let expected_ty = trait_ty_str.replace("Self", &unqualified_type_name);
                                        if impl_ty_str != expected_ty {
                                            let param_name = trait_method.param_types.get(i)
                                                .map(|(n, _)| n.as_str())
                                                .unwrap_or("?");
                                            return Err(CompileError::TraitMethodSignatureMismatch {
                                                method: method_name.to_string(),
                                                ty: unqualified_type_name.clone(),
                                                trait_name: trait_name.clone(),
                                                detail: format!(
                                                    "parameter `{}` has type `{}`, but trait expects `{}`",
                                                    param_name, impl_ty_str, expected_ty
                                                ),
                                                span: impl_method.name.span,
                                            });
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // Register the trait implementation FIRST so recursive trait method calls work
        // Track which traits this type implements (use qualified type name)
        // Avoid duplicates when pre_register_trait_impl and compile_trait_impl both run
        let traits_vec = self.type_traits
            .entry(qualified_type_name.clone())
            .or_insert_with(Vec::new);
        if !traits_vec.contains(&trait_name) {
            traits_vec.push(trait_name.clone());
        }
        // Also register under base type names so find_trait_method can find the trait:
        // 1. Module-qualified without type args: "types.Either2" (for cross-module lookup)
        // 2. Short base without module prefix: "Either2" (for unqualified lookup)
        let base_for_traits = if let Some(bracket_pos) = qualified_type_name.find('[') {
            &qualified_type_name[..bracket_pos]
        } else {
            qualified_type_name.as_str()
        };
        // Register under module-qualified base (without type args)
        if base_for_traits != qualified_type_name {
            let traits_vec_base = self.type_traits
                .entry(base_for_traits.to_string())
                .or_insert_with(Vec::new);
            if !traits_vec_base.contains(&trait_name) {
                traits_vec_base.push(trait_name.clone());
            }
        }
        // Register under short base (without module prefix or type args)
        let short_base = base_for_traits.rsplit('.').next().unwrap_or(base_for_traits);
        if short_base != qualified_type_name && short_base != base_for_traits {
            let traits_vec2 = self.type_traits
                .entry(short_base.to_string())
                .or_insert_with(Vec::new);
            if !traits_vec2.contains(&trait_name) {
                traits_vec2.push(trait_name.clone());
            }
        }

        // Pre-compute trait param types for each method (clone to avoid borrow issues)
        let trait_method_params: HashMap<String, Vec<(String, String)>> = self.trait_defs.get(&trait_name)
            .map(|ti| ti.methods.iter()
                .map(|m| (m.name.clone(), m.param_types.clone()))
                .collect())
            .unwrap_or_default();

        // FORWARD DECLARATION PASS: Add all trait impl methods to function_indices AND functions
        // before compiling any of them, to enable recursive calls via resolve_function_call
        for method in &impl_def.methods {
            let method_name = method.name.node.clone();
            // Use qualified_type_name so cross-module trait impls register with the
            // correct type prefix (e.g., "types.Shape" not "display.Shape").
            // This ensures find_trait_method can find the function.
            // IMPORTANT: Strip type args from the type name for the function key.
            // find_trait_method always looks up by base name (e.g., "List" not "List[a]"),
            // so the function key must also use the base name. Type args are encoded in
            // the signature suffix (after /) for overload resolution.
            let base_qualified_type = if let Some(bracket_pos) = qualified_type_name.find('[') {
                &qualified_type_name[..bracket_pos]
            } else {
                qualified_type_name.as_str()
            };
            let local_method_name = format!("{}.{}.{}", base_qualified_type, trait_name, method_name);
            let base_name = if base_qualified_type.contains('.') {
                // Type is already fully qualified from another module - use as-is
                local_method_name.clone()
            } else {
                self.qualify_name(&local_method_name)
            };

            // Build signature from parameter types, enriched with trait definition types
            let trait_pts = trait_method_params.get(&method_name);
            let param_types: Vec<String> = method.clauses.first()
                .map(|clause| clause.params.iter().enumerate()
                    .map(|(i, p)| {
                        // First try the impl's own type annotation
                        if let Some(t) = &p.ty {
                            return self.type_expr_to_string(t);
                        }
                        // Fall back to trait definition's type for this parameter
                        if let Some(tpts) = trait_pts {
                            if let Some((pname, trait_ty)) = tpts.get(i) {
                                if pname == "self" && matches!(p.pattern, Pattern::Var(_)) {
                                    // For "self" parameter with simple Var pattern, use the
                                    // implementing type. Skip for Variant patterns like
                                    // Circle(r) - those use pattern matching dispatch instead.
                                    // For generic types, use parameterized form (e.g., "Box[a]")
                                    // to preserve type name for field/trait dispatch while avoiding
                                    // TypeArityMismatch (bare "Box" has 0 args vs Box[Int] has 1).
                                    if let Some(ref gen_ty) = generic_self_type {
                                        return gen_ty.clone();
                                    }
                                    return unqualified_type_name.clone();
                                } else if trait_ty != "_" && !trait_ty.contains("->") {
                                    // Substitute Self with the implementing type.
                                    // For generic types, use parameterized form to avoid
                                    // TypeArityMismatch (e.g., "Pair" → "Pair[a, b]").
                                    let self_sub = if let Some(ref gen_ty) = generic_self_type {
                                        gen_ty.as_str()
                                    } else {
                                        unqualified_type_name.as_str()
                                    };
                                    return trait_ty.replace("Self", self_sub);
                                }
                            }
                        }
                        "_".to_string()
                    })
                    .collect())
                .unwrap_or_default();
            let arity = param_types.len();
            let signature = param_types.join(",");
            let full_name = format!("{}/{}", base_name, signature);

            // Build a signature string for HM inference.
            // This allows cross-module trait methods to be visible to the type inference
            // engine, enabling generic functions to call trait methods on unresolved types.
            let sig_string = {
                let trait_method_info = self.trait_defs.get(&trait_name)
                    .and_then(|ti| ti.methods.iter().find(|m| m.name == method_name));
                let ret_type = trait_method_info
                    .map(|m| {
                        let self_sub = if let Some(ref gen_ty) = generic_self_type {
                            gen_ty.as_str()
                        } else {
                            unqualified_type_name.as_str()
                        };
                        m.return_type.replace("Self", self_sub)
                    })
                    .unwrap_or_else(|| "_".to_string());
                let params_str = param_types.iter()
                    .map(|t| if t == "_" { "a".to_string() } else { t.clone() })
                    .collect::<Vec<_>>()
                    .join(" -> ");
                if params_str.is_empty() {
                    format!("() -> {}", ret_type)
                } else {
                    format!("{} -> {}", params_str, ret_type)
                }
            };

            // Add placeholder to functions for resolve_function_call to find
            if !self.functions.contains_key(&full_name) {
                let placeholder = FunctionValue {
                    name: full_name.clone(),
                    arity,
                    param_names: vec![],
                    code: Arc::new(Chunk::new()),
                    module: if self.module_path.is_empty() { None } else { Some(self.module_path.join(".")) },
                    source_span: None,
                    jit_code: None,
                    call_count: std::sync::atomic::AtomicU32::new(0),
                    debug_symbols: vec![],
                    source_code: None,
                    source_file: None,
                    doc: None,
                    signature: Some(sig_string.clone()),
                    param_types: param_types.clone(),
                    return_type: None,
                    required_params: {
                        // Compute required_params for trait methods with default parameters
                        // so that resolve_function_call can find the method when called
                        // with fewer arguments (e.g., n.add() when add has a default param)
                        if let Some(clause) = method.clauses.first() {
                            let req = clause.params.iter().filter(|p| p.default.is_none()).count();
                            if req < arity { Some(req) } else { None }
                        } else {
                            None
                        }
                    },
                };
                self.functions.insert(full_name.clone(), Arc::new(placeholder));

                // Update functions_by_base index
                let fn_base = full_name.split('/').next().unwrap_or(&full_name);
                self.functions_by_base
                    .entry(fn_base.to_string())
                    .or_insert_with(HashSet::new)
                    .insert(full_name.clone());

                // Update function_prefixes index for O(1) module existence checks
                let parts: Vec<&str> = fn_base.split('.').collect();
                let mut prefix = String::new();
                for part in parts.iter().take(parts.len().saturating_sub(1)) {
                    if !prefix.is_empty() {
                        prefix.push('.');
                    }
                    prefix.push_str(part);
                    self.function_prefixes.insert(format!("{}.", prefix));
                }
            }

            // Add to function_indices if not already there
            if !self.function_indices.contains_key(&full_name) {
                let idx = self.function_list.len() as u16;
                self.function_indices.insert(full_name.clone(), idx);
                self.function_list.push(full_name.clone());
            }

            // Register trait method UFCS signature for HM inference via unified registration.
            // This enables check_pending_method_calls to find trait methods on custom
            // types, which is essential for chained method calls (e.g., x.bimap(...).bimap(...)).
            // NOTE: Cannot use pending_fn_signatures because compile_all clears non-stdlib entries.
            {
                // Build function type with unique var IDs for untyped params.
                // Start counter high to avoid conflicts with a=1, b=2, etc.
                let mut trait_var_counter = 100u32;
                let hm_param_types: Vec<nostos_types::Type> = param_types.iter()
                    .enumerate()
                    .map(|(idx, pt)| {
                        if pt == "_" {
                            trait_var_counter += 1;
                            nostos_types::Type::Var(trait_var_counter)
                        } else if idx == 0 && impl_specifies_type_args {
                            // For concrete generic impls like Holder[Int]: Measurable,
                            // generalize the self type to use fresh type variables
                            // (e.g., Holder[?X] instead of Holder[Int]).
                            // Without this, multiple impls (Holder[Int], Holder[String])
                            // overwrite each other under the same key and the last wins.
                            let base_ty = self.type_name_to_type(pt);
                            if let nostos_types::Type::Named { ref name, ref args } = base_ty {
                                if !args.is_empty() {
                                    let generic_args: Vec<nostos_types::Type> = args.iter().map(|_| {
                                        trait_var_counter += 1;
                                        nostos_types::Type::Var(trait_var_counter)
                                    }).collect();
                                    nostos_types::Type::Named { name: name.clone(), args: generic_args }
                                } else {
                                    base_ty
                                }
                            } else {
                                base_ty
                            }
                        } else {
                            self.type_name_to_type(pt)
                        }
                    })
                    .collect();

                // Get return type from trait definition or method body analysis
                let hm_ret_type = if let Some(trait_info) = self.trait_defs.get(&trait_name) {
                    if let Some(trait_method) = trait_info.methods.iter().find(|m| m.name == method_name) {
                        let ret_type = &trait_method.return_type;
                        if ret_type != "()" && !ret_type.is_empty()
                            && !ret_type.chars().next().map(|c| c.is_lowercase()).unwrap_or(false)
                            && !ret_type.contains("->")
                        {
                            let self_sub = generic_self_type.as_deref()
                                .unwrap_or(unqualified_type_name.as_str());
                            let mut resolved_ret = ret_type.replace("Self", self_sub);
                            // Handle bare generic return types (e.g., "Either" → "Either[a, b]")
                            if !resolved_ret.contains('[') && !resolved_ret.contains('(') {
                                let ret_td = self.type_defs.get(&self.qualify_name(&resolved_ret))
                                    .or_else(|| self.type_defs.get(resolved_ret.as_str()));
                                if let Some(td) = ret_td {
                                    if !td.type_params.is_empty() {
                                        let vars: Vec<String> = td.type_params.iter()
                                            .map(|p| p.name.node.to_lowercase())
                                            .collect();
                                        resolved_ret = format!("{}[{}]", resolved_ret, vars.join(", "));
                                    }
                                }
                            }
                            self.type_name_to_type(&resolved_ret)
                        } else {
                            self.infer_return_type_from_method_body(method, &unqualified_type_name, &generic_self_type)
                                .unwrap_or_else(|| {
                                    trait_var_counter += 1;
                                    nostos_types::Type::Var(trait_var_counter)
                                })
                        }
                    } else {
                        trait_var_counter += 1;
                        nostos_types::Type::Var(trait_var_counter)
                    }
                } else {
                    trait_var_counter += 1;
                    nostos_types::Type::Var(trait_var_counter)
                };

                let fn_type = nostos_types::FunctionType {
                    required_params: {
                        if let Some(clause) = method.clauses.first() {
                            let req = clause.params.iter().filter(|p| p.default.is_none()).count();
                            if req < arity { Some(req) } else { None }
                        } else {
                            None
                        }
                    },
                    type_params: vec![],
                    params: hm_param_types,
                    ret: Box::new(hm_ret_type),
                    var_bounds: vec![],
                };

                // Use unified registration to ensure all key variants are registered consistently.
                self.register_ufcs_method(
                    &unqualified_type_name,
                    &qualified_type_name,
                    &method_name,
                    fn_type,
                    Some(method),
                );
            }
        }

        // In register_only mode, we've done validation + forward declaration.
        // Return early so that ALL trait impl methods are registered before
        // any bodies are compiled (fixes cross-trait method visibility).
        if register_only {
            return Ok(());
        }

        // Merge same-named methods into multi-clause functions.
        // When a trait impl has multiple clauses for the same method (e.g., pattern matching
        // on variant constructors like `speak(Cat) = "meow"` and `speak(Dog) = "woof"`),
        // they are parsed as separate FnDefs. We need to merge them into a single FnDef
        // with multiple clauses so that compile_fn_def generates proper multi-clause dispatch.
        let merged_methods: Vec<FnDef> = {
            let mut method_groups: Vec<(String, FnDef)> = Vec::new();
            for method in &impl_def.methods {
                let name = method.name.node.clone();
                if let Some(existing) = method_groups.iter_mut().find(|(n, _)| *n == name) {
                    // Merge clauses into existing group
                    existing.1.clauses.extend(method.clauses.clone());
                    // Extend the span to cover all clauses
                    existing.1.span = Span::new(existing.1.span.start, method.span.end);
                } else {
                    method_groups.push((name, method.clone()));
                }
            }
            method_groups.into_iter().map(|(_, def)| def).collect()
        };

        // Compile each method as a function with a special qualified name: Type.Trait.method
        // Use unqualified type name here because compile_fn_def will add module prefix
        // IMPORTANT: Strip type args from the type name for the function key.
        // This must match the forward declaration pass which also strips type args.
        let bare_unqualified_type_name = if let Some(bracket_pos) = unqualified_type_name.find('[') {
            &unqualified_type_name[..bracket_pos]
        } else {
            unqualified_type_name.as_str()
        };
        let mut method_names = Vec::new();
        for method in &merged_methods {
            let method_name = method.name.node.clone();
            // Use unqualified type name for method - compile_fn_def adds module prefix
            let local_method_name = format!("{}.{}.{}", bare_unqualified_type_name, trait_name, method_name);
            // The fully qualified method name (for registration)
            // For cross-module trait impls, the correct key uses the type's original module
            // prefix, not the current module prefix. E.g., "types.Shape.display.Displayable.display"
            // not "display.Shape.display.Displayable.display"
            let compile_fn_def_name = self.qualify_name(&local_method_name);
            let base_qualified_type_for_body = if let Some(bracket_pos) = qualified_type_name.find('[') {
                &qualified_type_name[..bracket_pos]
            } else {
                qualified_type_name.as_str()
            };
            let qualified_method_name = if base_qualified_type_for_body.contains('.') && qualified_type_name != self.qualify_name(&unqualified_type_name) {
                // Cross-module: type is from another module, use its qualified name
                format!("{}.{}.{}", base_qualified_type_for_body, trait_name, method_name)
            } else {
                compile_fn_def_name.clone()
            };

            // Create a modified FnDef with the method name.
            // For cross-module trait impls (type from another module), use the fully qualified
            // name so compile_fn_def registers the function under the correct key that
            // find_trait_method will look up.
            let is_cross_module = qualified_type_name.contains('.') && qualified_type_name != self.qualify_name(&unqualified_type_name);
            let fn_def_name = if is_cross_module {
                // Use qualified_method_name directly; we'll temporarily clear module_path
                // so compile_fn_def's qualify_name is a no-op
                qualified_method_name.clone()
            } else {
                local_method_name.clone()
            };
            let mut modified_def = method.clone();
            modified_def.name = Spanned::new(fn_def_name.clone(), method.name.span);
            modified_def.visibility = Visibility::Public; // Trait methods are always callable

            // Propagate return type from trait definition to implementation.
            // If the trait defines a return type (e.g., `area(self) -> Int`) but the impl
            // doesn't have one, inject it so type checking catches mismatches.
            // Inject return type from trait definition so type checking catches mismatches.
            // Skip unit "()", type variables (lowercase start), and function types (contain "->").
            // Self is resolved to the implementing type name (e.g., Self -> Counter).
            if let Some(trait_info) = self.trait_defs.get(&trait_name) {
                if let Some(trait_method) = trait_info.methods.iter().find(|m| m.name == method_name) {
                    let ret_type = &trait_method.return_type;
                    let is_injectable = ret_type != "()"
                        && !ret_type.chars().next().map(|c| c.is_lowercase()).unwrap_or(false)
                        && !ret_type.contains("->");
                    if is_injectable {
                        // Substitute Self with the implementing type.
                        // For generic types, use parameterized form to avoid TypeArityMismatch.
                        let self_sub = if let Some(ref gen_ty) = generic_self_type {
                            gen_ty.as_str()
                        } else {
                            unqualified_type_name.as_str()
                        };
                        let mut resolved_ret = ret_type.replace("Self", self_sub);
                        // If the resolved return type is a bare generic name (no type args),
                        // construct parameterized form with type variables to preserve the base
                        // type name while avoiding TypeArityMismatch.
                        // E.g., "Maybe" → "Maybe[a]", "Either" → "Either[a, b]"
                        if !resolved_ret.contains('[') && !resolved_ret.contains('(') {
                            let resolved_name = if let Some(imported) = self.imports.get(resolved_ret.as_str()) {
                                imported.as_str()
                            } else {
                                resolved_ret.as_str()
                            };
                            let qualified_ret_name = self.qualify_name(resolved_name);
                            let td = self.type_defs.get(&qualified_ret_name)
                                .or_else(|| self.type_defs.get(resolved_name))
                                .or_else(|| self.type_defs.get(resolved_ret.as_str()));
                            if let Some(td) = td {
                                if !td.type_params.is_empty() {
                                    let vars: Vec<String> = td.type_params.iter()
                                        .map(|p| p.name.node.to_lowercase())
                                        .collect();
                                    resolved_ret = format!("{}[{}]", resolved_ret, vars.join(", "));
                                }
                            }
                        }
                        if let Some(return_type_expr) = self.parse_return_type_expr(&resolved_ret) {
                            let mut injected = false;
                            for clause in &mut modified_def.clauses {
                                if clause.return_type.is_none() {
                                    clause.return_type = Some(return_type_expr.clone());
                                    injected = true;
                                }
                            }
                            if injected {
                                // Track that this method had its return type injected
                                // Use fn_def_name since that's what compile_fn_def sees as def.name.node
                                self.trait_return_type_injected.insert(fn_def_name.clone());
                            }
                        }
                    }
                }
            }

            // Inject parameter types from trait definition into impl methods
            // that lack their own type annotations. This ensures type checking catches
            // mismatches at call sites (e.g., passing String where trait says Int).
            if let Some(tpts) = trait_method_params.get(&method_name) {
                for clause in &mut modified_def.clauses {
                    for (i, param) in clause.params.iter_mut().enumerate() {
                        if param.ty.is_none() {
                            if let Some((pname, trait_ty)) = tpts.get(i) {
                                if pname == "self" && matches!(param.pattern, Pattern::Var(_)) {
                                    // For "self" parameter with simple Var pattern, inject
                                    // the implementing type. This is critical for type_check_fn
                                    // to resolve field access on self and catch return type
                                    // mismatches. Skip for Variant patterns like Circle(r) -
                                    // injecting the parent type breaks pattern matching dispatch.
                                    // For generic types, inject parameterized form (e.g., "Box[a]")
                                    // to preserve type info for field/trait dispatch.
                                    let self_type_str = if let Some(ref gen_ty) = generic_self_type {
                                        gen_ty.as_str()
                                    } else {
                                        unqualified_type_name.as_str()
                                    };
                                    if let Some(type_expr) = self.parse_return_type_expr(self_type_str) {
                                        param.ty = Some(type_expr);
                                    }
                                } else if trait_ty != "_" && !trait_ty.contains("->") {
                                    // Substitute Self with the implementing type.
                                    // For generic types, use parameterized form.
                                    let self_sub = if let Some(ref gen_ty) = generic_self_type {
                                        gen_ty.as_str()
                                    } else {
                                        unqualified_type_name.as_str()
                                    };
                                    let resolved_ty = trait_ty.replace("Self", self_sub);
                                    if let Some(type_expr) = self.parse_return_type_expr(&resolved_ty) {
                                        param.ty = Some(type_expr);
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // Set up param_types for Self-typed parameters before compiling
            // This allows type inference to work correctly for field access on self/other
            let saved_param_types = std::mem::take(&mut self.param_types);
            for clause in &method.clauses {
                for param in &clause.params {
                    // Check if this parameter's type is Self
                    let is_self_typed = param.ty.as_ref().map(|t| {
                        matches!(t, TypeExpr::Name(n) if n.node == "Self")
                    }).unwrap_or(false);

                    // For "self" parameter (first param in trait methods) or Self-typed params
                    if let Some(name) = self.pattern_binding_name(&param.pattern) {
                        if name == "self" || is_self_typed {
                            // For generic types, use parameterized form (e.g., "Box[a]")
                            // to preserve type name for field/trait dispatch.
                            let param_type_str = if let Some(ref gen_ty) = generic_self_type {
                                gen_ty.clone()
                            } else {
                                qualified_type_name.clone()
                            };
                            self.param_types.insert(name, self.type_name_to_type(&param_type_str));
                        }
                    }
                }
            }

            // For cross-module trait impls, temporarily clear module_path so
            // compile_fn_def's qualify_name doesn't add the wrong module prefix.
            // The modified_def.name already contains the fully qualified name.
            let saved_module_path_for_cross = if is_cross_module {
                Some(std::mem::take(&mut self.module_path))
            } else {
                None
            };

            self.compile_fn_def(&modified_def)?;

            // Restore module_path if it was saved for cross-module
            if let Some(saved_path) = saved_module_path_for_cross {
                self.module_path = saved_path;
            }

            // Restore param_types
            self.param_types = saved_param_types;

            // For cross-module trait impls, compile_fn_def registered the function under
            // the WRONG key (using current module prefix). Re-register under correct key.
            if compile_fn_def_name != qualified_method_name {
                let wrong_prefix = format!("{}/", compile_fn_def_name);
                let entries_to_move: Vec<(String, String)> = self.functions.keys()
                    .filter(|k| k.starts_with(&wrong_prefix))
                    .map(|k| {
                        let suffix = &k[compile_fn_def_name.len()..]; // e.g., "/Shape" or "/_"
                        (k.clone(), format!("{}{}", qualified_method_name, suffix))
                    })
                    .collect();

                for (wrong_key, correct_key) in entries_to_move {
                    if let Some(func) = self.functions.remove(&wrong_key) {
                        // Update function_indices to point to correct key
                        if let Some(idx) = self.function_indices.remove(&wrong_key) {
                            self.function_indices.insert(correct_key.clone(), idx);
                            if (idx as usize) < self.function_list.len() {
                                self.function_list[idx as usize] = correct_key.clone();
                            }
                        }
                        // Update functions_by_base index
                        let old_base = wrong_key.split('/').next().unwrap_or(&wrong_key);
                        if let Some(set) = self.functions_by_base.get_mut(old_base) {
                            set.remove(&wrong_key);
                        }
                        let new_base = correct_key.split('/').next().unwrap_or(&correct_key);
                        self.functions_by_base
                            .entry(new_base.to_string())
                            .or_insert_with(HashSet::new)
                            .insert(correct_key.clone());
                        // Also move fn_asts entry
                        if let Some(ast) = self.fn_asts.remove(&wrong_key) {
                            self.fn_asts.insert(correct_key.clone(), ast);
                        }
                        if let Some(imports) = self.fn_ast_imports.remove(&wrong_key) {
                            self.fn_ast_imports.insert(correct_key.clone(), imports);
                        }
                        self.functions.insert(correct_key, func);
                    }
                }
            }

            // Register {Type}.{Method} alias so HM inference can find trait methods
            // E.g., "Builder.Config.Builder.Configurable.withPort" → "Builder.Config.withPort"
            let type_method_alias = format!("{}.{}", qualified_type_name, method_name);
            self.trait_method_type_aliases.insert(
                qualified_method_name.clone(),
                type_method_alias,
            );

            method_names.push(qualified_method_name);
        }

        // Compile default methods from the trait definition that weren't overridden
        let impl_method_names_set: HashSet<String> = impl_def.methods.iter()
            .map(|m| m.name.node.clone())
            .collect();
        if let Some(trait_ast) = self.trait_defs_ast.get(&trait_name).cloned() {
            for trait_method in &trait_ast.methods {
                if let Some(ref default_body) = trait_method.default_impl {
                    if !impl_method_names_set.contains(&trait_method.name.node) {
                        // This default method was not overridden - compile it
                        let method_name = &trait_method.name.node;
                        let local_method_name = format!("{}.{}.{}", unqualified_type_name, trait_name, method_name);
                        let compile_fn_def_default_name = self.qualify_name(&local_method_name);
                        // Use correct qualified type name for cross-module trait impls
                        let qualified_method_name = if qualified_type_name.contains('.') && qualified_type_name != self.qualify_name(&unqualified_type_name) {
                            format!("{}.{}.{}", qualified_type_name, trait_name, method_name)
                        } else {
                            compile_fn_def_default_name.clone()
                        };

                        // Create a synthetic FnDef from the default implementation.
                        // For non-cross-module: use local_method_name (compile_fn_def will
                        // add the module prefix via qualify_name).
                        // For cross-module: use qualified_method_name (module_path is cleared
                        // below so qualify_name is a no-op).
                        let is_cross_module_default = qualified_type_name.contains('.')
                            && qualified_type_name != self.qualify_name(&unqualified_type_name);
                        let fn_def_name = if is_cross_module_default {
                            qualified_method_name.clone()
                        } else {
                            local_method_name.clone()
                        };
                        // Substitute Self with the implementing type in params and return type.
                        // Without this, functions like `double(self) -> Self` would keep
                        // the literal "Self" type, causing type mismatches.
                        let self_type_name = if let Some(ref gen_ty) = generic_self_type {
                            gen_ty.clone()
                        } else {
                            unqualified_type_name.clone()
                        };
                        fn subst_self_in_type_expr(te: &TypeExpr, replacement: &str, span: Span) -> TypeExpr {
                            match te {
                                TypeExpr::Name(ident) if ident.node == "Self" => {
                                    TypeExpr::Name(Spanned::new(replacement.to_string(), span))
                                }
                                TypeExpr::Generic(ident, args) => {
                                    let new_ident = if ident.node == "Self" {
                                        Spanned::new(replacement.to_string(), span)
                                    } else {
                                        ident.clone()
                                    };
                                    let new_args = args.iter()
                                        .map(|a| subst_self_in_type_expr(a, replacement, span))
                                        .collect();
                                    TypeExpr::Generic(new_ident, new_args)
                                }
                                TypeExpr::Function(params, ret) => {
                                    let new_params = params.iter()
                                        .map(|p| subst_self_in_type_expr(p, replacement, span))
                                        .collect();
                                    let new_ret = Box::new(subst_self_in_type_expr(ret, replacement, span));
                                    TypeExpr::Function(new_params, new_ret)
                                }
                                TypeExpr::Tuple(elems) => {
                                    let new_elems = elems.iter()
                                        .map(|e| subst_self_in_type_expr(e, replacement, span))
                                        .collect();
                                    TypeExpr::Tuple(new_elems)
                                }
                                other => other.clone(),
                            }
                        }
                        let subst_params: Vec<FnParam> = trait_method.params.iter().map(|p| {
                            FnParam {
                                pattern: p.pattern.clone(),
                                ty: p.ty.as_ref().map(|t| subst_self_in_type_expr(t, &self_type_name, trait_method.span)),
                                default: p.default.clone(),
                            }
                        }).collect();
                        let subst_return_type = trait_method.return_type.as_ref()
                            .map(|t| subst_self_in_type_expr(t, &self_type_name, trait_method.span));
                        let clause = FnClause {
                            params: subst_params,
                            guard: None,
                            return_type: subst_return_type,
                            body: default_body.clone(),
                            span: trait_method.span,
                        };
                        let synthetic_def = FnDef {
                            visibility: Visibility::Public,
                            doc: None,
                            decorators: vec![],
                            name: Spanned::new(fn_def_name.clone(), trait_method.name.span),
                            type_params: vec![],
                            clauses: vec![clause],
                            is_template: false,
                            span: trait_method.span,
                        };

                        // Forward declare the function
                        let param_types: Vec<String> = trait_method.params.iter()
                            .map(|p| p.ty.as_ref()
                                .map(|t| self.type_expr_to_string(t))
                                .unwrap_or_else(|| "_".to_string()))
                            .collect();
                        let arity = param_types.len();
                        let signature = param_types.join(",");
                        let full_name = format!("{}/{}", qualified_method_name, signature);

                        if !self.functions.contains_key(&full_name) {
                            let placeholder = FunctionValue {
                                name: full_name.clone(),
                                arity,
                                param_names: vec![],
                                code: Arc::new(Chunk::new()),
                                module: if self.module_path.is_empty() { None } else { Some(self.module_path.join(".")) },
                                source_span: None,
                                jit_code: None,
                                call_count: std::sync::atomic::AtomicU32::new(0),
                                debug_symbols: vec![],
                                source_code: None,
                                source_file: None,
                                doc: None,
                                signature: None,
                                param_types: param_types.clone(),
                                return_type: None,
                                required_params: None,
                            };
                            self.functions.insert(full_name.clone(), Arc::new(placeholder));
                            let fn_base = full_name.split('/').next().unwrap_or(&full_name);
                            self.functions_by_base
                                .entry(fn_base.to_string())
                                .or_insert_with(HashSet::new)
                                .insert(full_name.clone());
                        }
                        if !self.function_indices.contains_key(&full_name) {
                            let idx = self.function_list.len() as u16;
                            self.function_indices.insert(full_name.clone(), idx);
                            self.function_list.push(full_name.clone());
                        }

                        // Set up param_types for Self-typed parameters
                        let saved_param_types = std::mem::take(&mut self.param_types);
                        for param in &trait_method.params {
                            if let Some(name) = self.pattern_binding_name(&param.pattern) {
                                if name == "self" {
                                    self.param_types.insert(name, self.type_name_to_type(&qualified_type_name));
                                }
                            }
                        }

                        // For cross-module trait impls, temporarily clear module_path so
                        // compile_fn_def's qualify_name doesn't add the wrong module prefix.
                        // The synthetic_def.name already contains the fully qualified name.
                        let saved_module_path_for_default = if is_cross_module_default {
                            Some(std::mem::take(&mut self.module_path))
                        } else {
                            None
                        };

                        self.compile_fn_def(&synthetic_def)?;

                        // Restore module_path if it was saved
                        if let Some(saved_path) = saved_module_path_for_default {
                            self.module_path = saved_path;
                        }

                        self.param_types = saved_param_types;

                        // Register {Type}.{Method} alias for default methods too
                        let type_method_alias = format!("{}.{}", qualified_type_name, method_name);
                        self.trait_method_type_aliases.insert(
                            qualified_method_name.clone(),
                            type_method_alias,
                        );

                        method_names.push(qualified_method_name);
                    }
                }
            }
        }

        // Register the trait implementation with qualified type name.
        // Merge method_names with any existing entry (methods may be spread
        // across multiple impl blocks for the same type+trait pair).
        let key = (qualified_type_name.clone(), trait_name.clone());
        if let Some(existing) = self.trait_impls.get_mut(&key) {
            for name in method_names {
                if !existing.method_names.contains(&name) {
                    existing.method_names.push(name);
                }
            }
        } else {
            let impl_info = TraitImplInfo {
                type_name: qualified_type_name.clone(),
                trait_name: trait_name.clone(),
                method_names,
            };
            self.trait_impls.insert(key, impl_info);
        }

        Ok(())
    }

    /// Check that all trait implementations have all required methods.
    /// Uses pending_trait_completeness data recorded during compile_trait_impl_inner.
    /// Called AFTER all impl blocks for the current scope have been pre-registered.
    pub(super) fn check_trait_impl_completeness(&mut self) -> Result<(), CompileError> {
        let pending = std::mem::take(&mut self.pending_trait_completeness);
        for ((unqualified_type_name, trait_name), (seen_methods, span)) in &pending {
            if let Some(trait_info) = self.trait_defs.get(trait_name) {
                let methods = trait_info.methods.clone();
                for trait_method in &methods {
                    if trait_method.has_default {
                        continue;
                    }
                    if !seen_methods.contains(&trait_method.name) {
                        return Err(CompileError::MissingTraitMethod {
                            method: trait_method.name.clone(),
                            ty: unqualified_type_name.clone(),
                            trait_name: trait_name.clone(),
                            span: *span,
                        });
                    }
                }
            }
        }
        Ok(())
    }
}
