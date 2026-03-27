//! Multi-file compilation orchestration.
//!
//! Contains functions for compiling multi-module projects: compile_all,
//! module registration passes (forward declarations, type names, type defs,
//! metadata), use statement compilation, and visibility checking.

use super::*;

impl Compiler {
    /// Compile all pending functions.
    /// Returns (error, source_filename, source_code) on failure.
    pub fn compile_all(&mut self) -> Result<(), (CompileError, String, Arc<String>)> {
        let mut errors = self.compile_all_collecting_errors();
        // Filter out false positive type errors from monomorphized variants that were
        // instantiated with incomplete type information (type variables like ?N in the name).
        // These arise when polymorphic functions are monomorphized before all type variables
        // are resolved, leading to spurious StructuralMismatch errors.
        errors.retain(|(fn_name, e, _, _)| {
            if fn_name.contains('$') && fn_name.contains('?') {
                if let CompileError::TypeError { message, .. } = e {
                    if message.contains("type mismatch") {
                        return false; // suppress
                    }
                }
            }
            true
        });
        if errors.is_empty() {
            return Ok(());
        }
        // Prioritize DefinitionErrors (e.g., builtin shadowing) over TypeErrors,
        // since DefinitionErrors cause cascading type errors that are confusing.
        let best_error = errors.into_iter()
            .min_by_key(|(_, e, _, _)| match e {
                CompileError::DefinitionError { .. } => 0,
                _ => 1,
            })
            .unwrap();
        let (_fn_name, error, source_name, source) = best_error;
        Err((error, source_name, source))
    }

    /// Compile all pending functions, collecting all errors.
    /// Returns a vec of (function_name, error, source_filename, source_code) for functions that failed to compile.
    /// Functions that compile successfully get their signatures set.
    ///
    /// This uses two passes to handle dependencies:
    /// 1. First pass: compile all functions (some may get placeholder type 'a' for dependencies)
    /// 2. Second pass: re-run HM inference for functions with 'a' in their signatures
    pub fn compile_all_collecting_errors(&mut self) -> Vec<(String, CompileError, String, Arc<String>)> {
        let pending = std::mem::take(&mut self.pending_functions);
        let mut errors: Vec<(String, CompileError, String, Arc<String>)> = Vec::new();

        // Invalidate cached HM base env — types/traits/pending_fn_signatures will be rebuilt
        self.cached_hm_base_env = None;

        // Pre-build function signatures for type checking (done once, not per-function)
        // Preserve cached stdlib signatures (loaded from cache via register_function_signature_from_cache)
        // which start with "stdlib." - only clear non-stdlib entries.
        self.pending_fn_signatures.retain(|name, _| name.starts_with("stdlib."));
        // Start above 100 to avoid collisions with letter-based type param IDs
        // from type_name_to_type() which maps 'a'->1, 'b'->2, ..., 'z'->26.
        let mut counter = 100u32;
        for (fn_def, module_path, _, _, _, _) in &pending {
            let fn_name = if module_path.is_empty() {
                fn_def.name.node.clone()
            } else {
                format!("{}.{}", module_path.join("."), fn_def.name.node)
            };
            // Set module_path so type_name_to_type can resolve unqualified type names
            let saved_module_path = std::mem::replace(&mut self.module_path, module_path.clone());
            if let Some(clause) = fn_def.clauses.first() {
                // Build a map from declared type param names to the concrete type strings
                // they resolve to, so we can replace them after type conversion.
                // E.g., for fn[Option](x: Option), "Option" resolves to stdlib.list.Option
                // but should be treated as a type variable.
                let tp_name_set: std::collections::HashSet<String> = fn_def.type_params.iter()
                    .map(|tp| tp.name.node.clone())
                    .collect();
                // Map the resolved concrete type name to a Var (for post-processing).
                // E.g., for fn[Option](x: Option), type_name_to_type("Option") resolves to
                // Named { name: "stdlib.list.Option" }, so we map "stdlib.list.Option" → Var(id).
                let tp_resolved: std::collections::HashMap<String, nostos_types::Type> = if tp_name_set.is_empty() {
                    std::collections::HashMap::new()
                } else {
                    let mut tp_counter = 10000u32 + (counter * 10);
                    fn_def.type_params.iter().filter_map(|tp| {
                        tp_counter += 1;
                        let resolved_type = self.type_name_to_type(&tp.name.node);
                        // Only add to map if the name resolves to a Named type
                        // (i.e., it shadows a real type like Option, Result, etc.)
                        if let nostos_types::Type::Named { name, args } = &resolved_type {
                            if args.is_empty() {
                                return Some((name.clone(), nostos_types::Type::Var(tp_counter)));
                            }
                        }
                        // Also handle when the name stays as-is (lowercase type params
                        // that don't shadow anything - these are already Var from type_name_to_type)
                        None
                    }).collect()
                };

                let param_types: Vec<nostos_types::Type> = clause.params
                    .iter()
                    .map(|p| {
                        if let Some(ty_expr) = &p.ty {
                            let ty = self.type_name_to_type(&self.type_expr_to_string(ty_expr));
                            if tp_resolved.is_empty() {
                                ty
                            } else {
                                Self::replace_type_params_in_type(&ty, &tp_resolved)
                            }
                        } else {
                            // Create unique type variable for each untyped param
                            counter += 1;
                            nostos_types::Type::Var(counter)
                        }
                    })
                    .collect();
                let ret_ty = clause.return_type.as_ref()
                    .map(|ty| {
                        let resolved = self.type_name_to_type(&self.type_expr_to_string(ty));
                        if tp_resolved.is_empty() {
                            resolved
                        } else {
                            Self::replace_type_params_in_type(&resolved, &tp_resolved)
                        }
                    })
                    .unwrap_or_else(|| {
                        counter += 1;
                        nostos_types::Type::Var(counter)
                    });

                // Compute required_params for functions with optional parameters
                let required_count = clause.params.iter()
                    .filter(|p| p.default.is_none())
                    .count();
                let required_params = if required_count < clause.params.len() {
                    Some(required_count)
                } else {
                    None // All required
                };

                // Build qualified name with type suffix (e.g., "add/Int,Int" for typed params)
                // For overloaded functions, using actual types prevents collisions.
                // Untyped params and type params (like A, B in generics) use "_" as wildcard.
                let arity_suffix = if clause.params.is_empty() {
                    "/".to_string()
                } else {
                    let type_param_names: HashSet<&str> = fn_def.type_params.iter()
                        .map(|tp| tp.name.node.as_str())
                        .collect();
                    let parts: Vec<String> = clause.params.iter().map(|p| {
                        if let Some(ty_expr) = &p.ty {
                            let ty_str = self.type_expr_to_string(ty_expr);
                            // Use "_" for type parameters (e.g., A, B in pair[A,B])
                            if type_param_names.contains(ty_str.as_str()) {
                                "_".to_string()
                            } else {
                                ty_str
                            }
                        } else {
                            "_".to_string()
                        }
                    }).collect();
                    format!("/{}", parts.join(","))
                };
                let qualified_fn_name = format!("{}{}", fn_name, arity_suffix);

                // Convert AST type params to types module TypeParams
                let type_params: Vec<nostos_types::TypeParam> = fn_def.type_params.iter()
                    .map(|tp| nostos_types::TypeParam {
                        name: tp.name.node.clone(),
                        constraints: tp.constraints.iter().map(|c| c.node.clone()).collect(),
                    })
                    .collect();

                self.pending_fn_signatures.insert(
                    qualified_fn_name,
                    nostos_types::FunctionType { required_params,
                        type_params,
                        params: param_types,
                        ret: Box::new(ret_ty),
                        var_bounds: vec![],
                    },
                );
            }
            self.module_path = saved_module_path;
        }

        // TYPE INFERENCE PRE-PASS: Resolve type variables for mutual recursion support
        // This runs type inference in TWO SEPARATE PHASES to prevent cross-module type pollution:
        // 1. First, infer stdlib functions with only stdlib types (no user types)
        // 2. Then, infer user functions with all types (stdlib already resolved)
        // This prevents user type variables from polluting stdlib type inference.

        // Helper to check if a function is from stdlib
        let is_stdlib_fn = |module_path: &[String], source_name: &str| -> bool {
            module_path.first().map(|s| s == "stdlib").unwrap_or(false)
                || source_name.contains("stdlib/") || source_name.starts_with("stdlib")
        };

        // Helper to check if a type is from stdlib
        let is_stdlib_type = |name: &str| -> bool {
            name.starts_with("stdlib.") ||
            // Core types that are always available
            matches!(name, "Int" | "Float" | "String" | "Bool" | "List" | "Map" | "Set" |
                     "Option" | "Result" | "Unit" | "Char" | "Ordering" | "Json" |
                     "Buffer" | "Float64Array" | "Int64Array" | "Float32Array" |
                     "Duration" | "Instant" | "DateTime" | "Color" | "HttpMethod" |
                     "HttpRequest" | "HttpResponse" | "Request" | "PostgresConnection" | "PostgresRow" |
                     "Server" | "ServerHandle" | "WebSocket" | "WsMessage" | "Process")
        };

        // Separate functions into stdlib and user groups
        let (stdlib_fns, user_fns): (Vec<_>, Vec<_>) = pending.iter()
            .enumerate()
            .partition(|(_, (_, module_path, _, _, _, source_name))| is_stdlib_fn(module_path, source_name.as_str()));

        // Identify which pending_fn_signatures belong to stdlib
        let stdlib_fn_names: std::collections::HashSet<String> = stdlib_fns.iter()
            .filter_map(|(_, (fn_def, module_path, _, _, _, _))| {
                let fn_name = if module_path.is_empty() {
                    fn_def.name.node.clone()
                } else {
                    format!("{}.{}", module_path.join("."), fn_def.name.node)
                };
                fn_def.clauses.first().map(|clause| {
                    let arity_suffix = if clause.params.is_empty() {
                        "/".to_string()
                    } else {
                        format!("/{}", vec!["_"; clause.params.len()].join(","))
                    };
                    format!("{}{}", fn_name, arity_suffix)
                })
            })
            .collect();

        // PHASE 1: Infer stdlib functions (without user types)
        {
            let mut env = nostos_types::standard_env();

            // Register only stdlib types
            for (name, type_info) in &self.types {
                if !is_stdlib_type(name) {
                    continue; // Skip user types in phase 1
                }
                let type_params: Vec<nostos_types::TypeParam> = self.type_defs.get(name)
                    .map(|td| td.type_params.iter().map(|p| nostos_types::TypeParam {
                        name: p.name.node.clone(),
                        constraints: p.constraints.iter().map(|c| c.node.clone()).collect(),
                    }).collect())
                    .unwrap_or_default();

                match &type_info.kind {
                    TypeInfoKind::Record { fields, mutable } => {
                        let field_types: Vec<(String, nostos_types::Type, bool)> = fields
                            .iter()
                            .map(|(n, ty)| (n.clone(), Self::vars_to_type_params(&self.type_name_to_type(ty), &type_params), false))
                            .collect();
                        env.define_type(
                            name.clone(),
                            nostos_types::TypeDef::Record {
                                params: type_params,
                                fields: field_types,
                                is_mutable: *mutable,
                            },
                        );
                    }
                    TypeInfoKind::Reactive { fields } => {
                        let field_types: Vec<(String, nostos_types::Type, bool)> = fields
                            .iter()
                            .map(|(n, ty)| (n.clone(), Self::vars_to_type_params(&self.type_name_to_type(ty), &type_params), false))
                            .collect();
                        env.define_type(
                            name.clone(),
                            nostos_types::TypeDef::Record {
                                params: type_params,
                                fields: field_types,
                                is_mutable: true,
                            },
                        );
                    }
                    TypeInfoKind::Variant { constructors } | TypeInfoKind::ReactiveVariant { constructors } => {
                        let ctors: Vec<nostos_types::Constructor> = constructors
                            .iter()
                            .map(|(ctor_name, fields_info)| {
                                match fields_info {
                                    VariantFieldsInfo::Unit => {
                                        nostos_types::Constructor::Unit(ctor_name.clone())
                                    }
                                    VariantFieldsInfo::Positional(field_types) => {
                                        nostos_types::Constructor::Positional(
                                            ctor_name.clone(),
                                            field_types.iter().map(|ty| Self::vars_to_type_params(&self.type_name_to_type(ty), &type_params)).collect(),
                                        )
                                    }
                                    VariantFieldsInfo::Named(fields) => {
                                        nostos_types::Constructor::Named(
                                            ctor_name.clone(),
                                            fields.iter().map(|(name, ty)| (name.clone(), Self::vars_to_type_params(&self.type_name_to_type(ty), &type_params))).collect(),
                                        )
                                    }
                                }
                            })
                            .collect();
                        env.define_type(
                            name.clone(),
                            nostos_types::TypeDef::Variant {
                                params: type_params,
                                constructors: ctors,
                            },
                        );
                    }
                }
            }

            // Register already-compiled stdlib functions
            // Skip functions that start with "stdlib." - they'll be registered from
            // pending_fn_signatures below with proper type_params preserved.
            for (fn_name, fn_val) in &self.functions {
                if fn_name.starts_with("stdlib.") {
                    continue; // Registered from pending_fn_signatures with correct type_params
                }
                if let Some(sig) = fn_val.signature.as_ref() {
                    if let Some(fn_type) = self.parse_signature_string(sig) {
                        env.insert_function(fn_name.clone(), fn_type);
                    }
                }
            }

            // Register stdlib pending function signatures.
            // Include both compiled-from-source (in stdlib_fn_names) AND loaded-from-cache
            // (which start with "stdlib." prefix).
            for (fn_name, fn_type) in &self.pending_fn_signatures {
                let is_stdlib = stdlib_fn_names.contains(fn_name) || fn_name.starts_with("stdlib.");
                if is_stdlib {
                    env.insert_function(fn_name.clone(), fn_type.clone());
                }
            }

            // Register builtins from cached parsed signatures
            for &(name, sig, ref fn_type) in &self.cached_builtin_signatures {
                if !env.functions.contains_key(name) {
                    env.insert_function(name.to_string(), fn_type.clone());

                    // Register list methods with "List." prefix for UFCS type inference
                    if sig.starts_with("[a]") && !name.contains('.') {
                        let has_single_type_param = !sig.contains(" b")
                            && !sig.contains("(b");
                        let is_numeric_only = matches!(name, "sum" | "product");
                        let has_callback_param = sig.contains("(a ->");
                        if has_single_type_param && !is_numeric_only && !has_callback_param {
                            let list_qualified = format!("List.{}", name);
                            if !env.functions.contains_key(&list_qualified) {
                                env.insert_function(list_qualified, fn_type.clone());
                            }
                        }
                    }
                }
            }

            // Bind REPL local variables in the TypeEnv for HM inference
            // This allows expressions like `v * 2.0` to know that `v: testvec.Vec`
            // and dispatch correctly to scalar operations
            for (var_name, ty) in &self.local_types {
                env.bind(var_name.clone(), ty.clone(), false);
            }

            // Update next_var to avoid collisions with type variables in registered functions
            // Signatures use Var(1), Var(2) etc for type params 'a', 'b', so we need fresh vars
            // to start after the maximum var ID in any signature
            let max_var_in_functions = env.functions.values()
                .filter_map(|ft| ft.max_var_id())
                .filter(|&id| id != u32::MAX) // Sentinel for unknown types
                .max();
            if let Some(max_id) = max_var_in_functions {
                if env.next_var <= max_id {
                    env.next_var = max_id.saturating_add(1);
                }
            }

            // Infer ALL stdlib functions to ensure expression types are stored.
            // Even functions with complete type annotations need inference to store
            // types for inner expressions (e.g., method calls inside the function body).
            // The stdlib is only compiled once and cached, so this cost is acceptable.
            let mut ctx = InferCtx::new(&mut env);
            for (_, (fn_def, module_path, _, _, _, _)) in &stdlib_fns {
                // Use qualified name for batch inference (same as Phase 2)
                if !module_path.is_empty() {
                    let qualified_name = format!("{}.{}", module_path.join("."), fn_def.name.node);
                    let mut qualified_def = fn_def.clone();
                    qualified_def.name = nostos_syntax::Ident {
                        node: qualified_name,
                        span: fn_def.name.span,
                    };
                    let _ = ctx.infer_function(&qualified_def);
                } else {
                    let _ = ctx.infer_function(fn_def);
                }
            }

            // Solve stdlib constraints
            let _ = ctx.solve();

            // Transfer inferred expression types from stdlib inference.
            // With file_id in Span, spans are now unique across files.
            // Apply substitution to resolve type variables to concrete types.
            let stdlib_expr_types: HashMap<_, _> = ctx.take_expr_types()
                .into_iter()
                .map(|(span, ty)| (span, ctx.apply_full_subst(&ty)))
                .collect();
            self.inferred_expr_types.extend(stdlib_expr_types);

            // Apply full substitution (including TypeParam resolution) only to stdlib signatures
            for (fn_name, fn_type) in self.pending_fn_signatures.iter_mut() {
                if stdlib_fn_names.contains(fn_name) {
                    let resolved_params: Vec<nostos_types::Type> = fn_type.params.iter()
                        .map(|p| ctx.apply_full_subst(p))
                        .collect();
                    let resolved_ret = ctx.apply_full_subst(&fn_type.ret);
                    *fn_type = nostos_types::FunctionType {
                        type_params: fn_type.type_params.clone(),
                        params: resolved_params,
                        ret: Box::new(resolved_ret),
                        required_params: fn_type.required_params,
                        var_bounds: vec![],
                    };
                }
            }
        }

        // PHASE 2: Infer user functions (with all types, including resolved stdlib)
        {
            let mut env = nostos_types::standard_env();

            // Register ALL types (stdlib already resolved, user types now included)
            for (name, type_info) in &self.types {
                let type_params: Vec<nostos_types::TypeParam> = self.type_defs.get(name)
                    .map(|td| td.type_params.iter().map(|p| nostos_types::TypeParam {
                        name: p.name.node.clone(),
                        constraints: p.constraints.iter().map(|c| c.node.clone()).collect(),
                    }).collect())
                    .unwrap_or_default();

                match &type_info.kind {
                    TypeInfoKind::Record { fields, mutable } => {
                        let field_types: Vec<(String, nostos_types::Type, bool)> = fields
                            .iter()
                            .map(|(n, ty)| (n.clone(), Self::vars_to_type_params(&self.type_name_to_type(ty), &type_params), false))
                            .collect();
                        env.define_type(
                            name.clone(),
                            nostos_types::TypeDef::Record {
                                params: type_params,
                                fields: field_types,
                                is_mutable: *mutable,
                            },
                        );
                    }
                    TypeInfoKind::Reactive { fields } => {
                        let field_types: Vec<(String, nostos_types::Type, bool)> = fields
                            .iter()
                            .map(|(n, ty)| (n.clone(), Self::vars_to_type_params(&self.type_name_to_type(ty), &type_params), false))
                            .collect();
                        env.define_type(
                            name.clone(),
                            nostos_types::TypeDef::Record {
                                params: type_params,
                                fields: field_types,
                                is_mutable: true,
                            },
                        );
                    }
                    TypeInfoKind::Variant { constructors } | TypeInfoKind::ReactiveVariant { constructors } => {
                        let ctors: Vec<nostos_types::Constructor> = constructors
                            .iter()
                            .map(|(ctor_name, fields_info)| {
                                match fields_info {
                                    VariantFieldsInfo::Unit => {
                                        nostos_types::Constructor::Unit(ctor_name.clone())
                                    }
                                    VariantFieldsInfo::Positional(field_types) => {
                                        nostos_types::Constructor::Positional(
                                            ctor_name.clone(),
                                            field_types.iter().map(|ty| Self::vars_to_type_params(&self.type_name_to_type(ty), &type_params)).collect(),
                                        )
                                    }
                                    VariantFieldsInfo::Named(fields) => {
                                        nostos_types::Constructor::Named(
                                            ctor_name.clone(),
                                            fields.iter().map(|(name, ty)| (name.clone(), Self::vars_to_type_params(&self.type_name_to_type(ty), &type_params))).collect(),
                                        )
                                    }
                                }
                            })
                            .collect();
                        env.define_type(
                            name.clone(),
                            nostos_types::TypeDef::Variant {
                                params: type_params,
                                constructors: ctors,
                            },
                        );
                    }
                }
            }

            // Register type alias definitions (e.g., `type Score = Int`) in the HM env
            // so that alias expansion works during constraint solving (e.g., Score implements Num)
            for (alias_name, target_name) in &self.type_alias_targets {
                let target_type = self.type_name_to_type(target_name);
                env.define_type(
                    alias_name.clone(),
                    nostos_types::TypeDef::Alias {
                        params: vec![],
                        target: target_type,
                    },
                );
            }

            // Add type aliases (e.g., "Wrapper" -> "A.Wrapper") so that unqualified
            // type names used inside module functions resolve correctly during batch inference.
            self.add_type_aliases_to_env(&mut env);

            // Register already-compiled functions.
            // For stdlib functions: register them here since some stdlib functions (e.g.,
            // stdlib.rhtml.span) are in self.functions but NOT in pending_fn_signatures,
            // so without this registration they would be missing from the batch env.
            // For non-stdlib functions: only register if they have a compiled signature.
            for (fn_name, fn_val) in &self.functions {
                if fn_val.signature.is_some() {
                    if let Some(sig) = fn_val.signature.as_ref() {
                        if let Some(fn_type) = self.parse_signature_string(sig) {
                            // Preserve required_params from FunctionValue so optional params work
                            let mut ft = fn_type;
                            ft.required_params = fn_val.required_params;
                            if !env.functions.contains_key(fn_name) {
                                env.insert_function(fn_name.clone(), ft);
                            }
                        }
                    }
                }
            }

            // Register ALL pending function signatures (stdlib now resolved, user still has type vars).
            // These take precedence over the above (may have better type_params from HM inference).
            for (fn_name, fn_type) in &self.pending_fn_signatures {
                env.insert_function(fn_name.clone(), fn_type.clone());
            }

            // Register builtins from cached parsed signatures
            for &(name, sig, ref fn_type) in &self.cached_builtin_signatures {
                if !env.functions.contains_key(name) {
                    env.insert_function(name.to_string(), fn_type.clone());

                    // Register list methods with "List." prefix for UFCS type inference.
                    // Without this, check_pending_method_calls phase2 only finds
                    // String.take (a named BUILTIN) but not List.take, causing batch
                    // inference to incorrectly resolve generic receivers to String.
                    if sig.starts_with("[a]") && !name.contains('.') {
                        let has_single_type_param = !sig.contains(" b")
                            && !sig.contains("(b");
                        let is_numeric_only = matches!(name, "sum" | "product");
                        let has_callback_param = sig.contains("(a ->");
                        if has_single_type_param && !is_numeric_only && !has_callback_param {
                            let list_qualified = format!("List.{}", name);
                            if !env.functions.contains_key(&list_qualified) {
                                env.insert_function(list_qualified, fn_type.clone());
                            }
                        }
                    }
                }
            }

            // Update next_var to avoid collisions with type variables in registered functions
            let max_var_in_functions = env.functions.values()
                .filter_map(|ft| ft.max_var_id())
                .filter(|&id| id != u32::MAX)
                .max();
            if let Some(max_id) = max_var_in_functions {
                if env.next_var <= max_id {
                    env.next_var = max_id.saturating_add(1);
                }
            }

            // Bind REPL local variables in the TypeEnv for HM inference
            // This allows expressions like `v * 2.0` to know that `v: testvec.Vec`
            // and dispatch correctly to scalar operations
            for (var_name, ty) in &self.local_types {
                env.bind(var_name.clone(), ty.clone(), false);
            }

            // Bind module-level mutable variables (mvars) in the TypeEnv
            // This allows functions to reference mvars and have proper type inference
            for (mvar_name, mvar_info) in &self.mvars {
                let mvar_type = self.type_name_to_type(&mvar_info.type_name);
                env.bind(mvar_name.clone(), mvar_type, true);
            }

            // Copy function imports from compiler to TypeEnv for cross-module resolution
            // This allows imported functions like `query` from `stdlib.pool` to be found
            // during type inference even when called by their short name
            for (short_name, qualified_name) in &self.imports {
                // Only add function aliases (skip type aliases which are handled separately)
                // A name is likely a function if the qualified name contains a module path
                // and the base name starts with a lowercase letter
                let is_type = qualified_name.split('.').last()
                    .map(|s| s.chars().next().map(|c| c.is_uppercase()).unwrap_or(false))
                    .unwrap_or(false);
                if !is_type {
                    env.function_aliases.insert(short_name.clone(), qualified_name.clone());
                }
            }

            // Register user-defined traits in the type environment for batch inference.
            // This enables trait method lookup in function bodies that call trait methods
            // in function form (e.g., getId(x) where getId is a trait method).
            for (trait_name, trait_info) in &self.trait_defs {
                let methods: Vec<nostos_types::TraitMethod> = trait_info.methods.iter()
                    .map(|m| {
                        let mut params = vec![
                            ("self".to_string(), nostos_types::Type::TypeParam("Self".to_string()))
                        ];
                        for i in 1..m.param_count {
                            params.push((format!("arg{}", i), nostos_types::Type::TypeParam(format!("T{}", i))));
                        }
                        nostos_types::TraitMethod {
                            name: m.name.clone(),
                            params,
                            ret: if m.return_type.is_empty() || m.return_type == "_" {
                                nostos_types::Type::TypeParam("R".to_string())
                            } else {
                                self.type_name_to_type(&m.return_type)
                            },
                        }
                    })
                    .collect();

                env.traits.insert(trait_name.clone(), nostos_types::TraitDef {
                    name: trait_name.clone(),
                    supertraits: trait_info.super_traits.clone(),
                    required: methods,
                    defaults: vec![],
                });

                if let Some(dot_pos) = trait_name.rfind('.') {
                    let short_name = &trait_name[dot_pos + 1..];
                    if !env.traits.contains_key(short_name) {
                        if let Some(trait_def) = env.traits.get(trait_name).cloned() {
                            env.traits.insert(short_name.to_string(), trait_def);
                        }
                    }
                }
            }

            // Register trait method UFCS signatures for batch inference.
            for (fn_name, fn_type) in &self.trait_method_ufcs_signatures {
                if !env.functions.contains_key(fn_name) {
                    env.insert_function(fn_name.clone(), fn_type.clone());
                }
                // Register param names for named argument resolution in type inference.
                // UFCS keys are like "Server.setup/_, _" - extract base "Server.setup".
                let base_name = if let Some(slash_pos) = fn_name.find('/') {
                    &fn_name[..slash_pos]
                } else {
                    fn_name.as_str()
                };
                if !env.function_param_names.contains_key(base_name) {
                    let names = self.get_function_param_names(base_name);
                    let param_name_strings: Vec<String> = names.into_iter()
                        .filter_map(|n| n)
                        .collect();
                    if !param_name_strings.is_empty() {
                        env.function_param_names.insert(base_name.to_string(), param_name_strings);
                    }
                }
            }

            // Register trait methods under bare names with generic signatures.
            // This allows trait methods called in function form (e.g., getId(x))
            // to be resolved during batch inference.
            {
                let mut trait_var_base = 200u32;
                for trait_info in self.trait_defs.values() {
                    for method in &trait_info.methods {
                        let method_name = &method.name;
                        let mut params = Vec::new();
                        trait_var_base += 1;
                        params.push(nostos_types::Type::Var(trait_var_base));
                        for _ in 1..method.param_count {
                            trait_var_base += 1;
                            params.push(nostos_types::Type::Var(trait_var_base));
                        }
                        let ret = if method.return_type != "()" && !method.return_type.is_empty()
                            && !method.return_type.chars().next().map(|c| c.is_lowercase()).unwrap_or(false)
                            && !method.return_type.contains("->")
                        {
                            self.type_name_to_type(&method.return_type)
                        } else {
                            trait_var_base += 1;
                            nostos_types::Type::Var(trait_var_base)
                        };
                        let arity_suffix = if method.param_count == 0 {
                            "/".to_string()
                        } else {
                            format!("/{}", vec!["_"; method.param_count].join(","))
                        };
                        let fn_key = format!("{}{}", method_name, arity_suffix);
                        env.insert_function(fn_key, nostos_types::FunctionType {
                            required_params: None,
                            type_params: vec![],
                            params: params.clone(),
                            ret: Box::new(ret.clone()),
                            var_bounds: vec![],
                        });
                        // Also register bare name
                        if !env.functions.contains_key(method_name) {
                            env.insert_function(method_name.clone(), nostos_types::FunctionType {
                                required_params: None,
                                type_params: vec![],
                                params,
                                ret: Box::new(ret),
                                var_bounds: vec![],
                            });
                        }
                    }
                }
            }

            // Copy bare-name function entries from compiler's functions_by_base to TypeEnv.
            // This enables cross-module overload resolution in HM inference: e.g., when
            // math_ops.compute(Int) and str_ops.compute(String) are both imported,
            // the bare name "compute" must map to all overload keys.
            for (bare_name, keys) in &self.functions_by_base {
                if !bare_name.contains('.') {
                    for key in keys {
                        env.functions_by_base
                            .entry(bare_name.clone())
                            .or_default()
                            .insert(key.clone());
                    }
                }
            }

            // Register top-level bindings in TypeEnv so functions can reference them.
            // For annotated bindings, use the annotation. For unannotated bindings, infer the type.
            {
                let mut to_infer: Vec<(&str, &str, &nostos_syntax::Expr)> = Vec::new();
                for (binding_name, (binding, _, _)) in &self.top_level_bindings {
                    if let Pattern::Var(ident) = &binding.pattern {
                        let bind_name = ident.node.as_str();
                        if let Some(ty_expr) = &binding.ty {
                            let type_str = self.type_expr_to_string(ty_expr);
                            let binding_type = self.type_name_to_type(&type_str);
                            env.bind(bind_name.to_string(), binding_type.clone(), false);
                            // Also register with qualified name so Module.binding lookups work
                            if binding_name != bind_name {
                                env.bind(binding_name.clone(), binding_type, false);
                            }
                        } else {
                            to_infer.push((bind_name, binding_name.as_str(), &binding.value));
                        }
                    }
                }
                if !to_infer.is_empty() {
                    let mut tmp_env = env.clone();
                    let mut tmp_ctx = InferCtx::new(&mut tmp_env);
                    let mut inferred: Vec<(String, String, nostos_types::Type)> = Vec::new();
                    for (name, qualified, expr) in &to_infer {
                        if let Ok(ty) = tmp_ctx.infer_expr(expr) {
                            inferred.push((name.to_string(), qualified.to_string(), ty));
                        }
                    }
                    let _ = tmp_ctx.solve();
                    for (name, qualified, ty) in inferred {
                        let resolved = tmp_ctx.apply_full_subst(&ty);
                        env.bind(name.clone(), resolved.clone(), false);
                        // Also register with qualified name so Module.binding lookups work
                        if qualified != name {
                            env.bind(qualified, resolved, false);
                        }
                    }
                    // CRITICAL: Advance env.next_var past any fresh vars created by tmp_ctx.
                    // Without this, the main batch ctx (InferCtx::new(&mut env)) will reuse
                    // those var IDs for unrelated purposes, causing spurious unification with
                    // the Var types stored in env.bindings for top-level constants.
                    // Example: addFive = addN(1) stores Var(N) in env.bindings; if env.next_var
                    // is not advanced, the main ctx reuses Var(N) for a list element type Int,
                    // making addFive appear to be Int when passed to map() as a callback.
                    if tmp_env.next_var > env.next_var {
                        env.next_var = tmp_env.next_var;
                    }
                }
            }

            // Collect implicit conversion function names (e.g., "tensorFromList").
            // These follow the naming convention {targetLower}From{Source} and allow
            // the type checker to accept type mismatches that can be resolved by
            // auto-inserting a call to the conversion function.
            // Extract the bare function name (after last dot, before slash) from qualified keys.
            let implicit_fns: HashSet<String> = self.pending_fn_signatures.keys()
                .filter_map(|name| {
                    // Get base name: strip arity suffix (e.g., "candle.tensorFromList/_" -> "candle.tensorFromList")
                    let base = name.split('/').next().unwrap_or(name);
                    // Get local name: strip module prefix (e.g., "candle.tensorFromList" -> "tensorFromList")
                    let local = base.rsplit('.').next().unwrap_or(base);
                    // Match pattern: xFrom... where x starts lowercase
                    if let Some(from_pos) = local.find("From") {
                        if from_pos > 0 && local[..1].chars().next().map_or(false, |c| c.is_ascii_lowercase()) {
                            return Some(local.to_string());
                        }
                    }
                    None
                })
                .collect();

            // Register trait implementations so definitely_not_implements() is accurate
            {
                let builtin_traits = ["Hash", "Show", "Eq", "Copy"];
                for (type_name, _) in &self.types {
                    let for_type = nostos_types::Type::Named {
                        name: type_name.clone(),
                        args: vec![],
                    };
                    for trait_name in &builtin_traits {
                        env.impls.push(nostos_types::TraitImpl {
                            trait_name: trait_name.to_string(),
                            for_type: for_type.clone(),
                            constraints: vec![],
                        });
                    }
                }
                for (type_name, traits) in &self.type_traits {
                    let for_type = nostos_types::Type::Named {
                        name: type_name.clone(),
                        args: vec![],
                    };
                    for trait_name in traits {
                        if !builtin_traits.contains(&trait_name.as_str()) {
                            env.impls.push(nostos_types::TraitImpl {
                                trait_name: trait_name.clone(),
                                for_type: for_type.clone(),
                                constraints: vec![],
                            });
                        }
                    }
                }
            }

            // Infer ALL user functions to capture expression types for compilation.
            // Even functions with complete type annotations need body inference for
            // proper method dispatch and expression type recording.
            let mut ctx = InferCtx::new(&mut env);
            if !implicit_fns.is_empty() {
                ctx.set_known_implicit_fns(implicit_fns);
            }
            // Build base function_aliases from global imports (stdlib etc.)
            let base_function_aliases = ctx.env.function_aliases.clone();
            for (_, (fn_def, module_path, _, _, _, _)) in &user_fns {
                // Use qualified name for batch inference to prevent cross-module
                // type variable collision when different modules define functions
                // with the same bare name (e.g., IntOps.combine vs StrOps.combine).
                if !module_path.is_empty() {
                    // Set current module so resolve_type_name prefers this module's types.
                    // E.g., inside module B, `Point` resolves to `B.Point` not `A.Point`.
                    let module_key = module_path.join(".");
                    ctx.env.current_module = Some(module_key.clone());
                    // Restore base aliases then overlay this module's imports
                    ctx.env.function_aliases = base_function_aliases.clone();
                    if let Some(mod_imports) = self.module_imports.get(&module_key) {
                        for (short, qualified) in mod_imports {
                            ctx.env.function_aliases.insert(short.clone(), qualified.clone());
                        }
                    }
                    let qualified_name = format!("{}.{}", module_key, fn_def.name.node);
                    let mut qualified_def = fn_def.clone();
                    qualified_def.name = nostos_syntax::Ident {
                        node: qualified_name.clone(),
                        span: fn_def.name.span,
                    };
                    let _ = ctx.infer_function(&qualified_def);
                } else {
                    ctx.env.current_module = None;
                    // Restore base aliases then overlay top-level file's imports.
                    // This ensures that `use stdlib.rhtml.*` (which adds span → stdlib.rhtml.span)
                    // is visible in batch inference, not just the global self.imports (which has
                    // span → stdlib.list.span from the prelude). module_imports[""] is saved
                    // inside compile_items before add_module restores self.imports.
                    ctx.env.function_aliases = base_function_aliases.clone();
                    if let Some(mod_imports) = self.module_imports.get("") {
                        for (short, qualified) in mod_imports {
                            ctx.env.function_aliases.insert(short.clone(), qualified.clone());
                        }
                    }
                    let _ = ctx.infer_function(fn_def);
                }
            }
            ctx.env.current_module = None;

            // Solve user constraints - capture trait-bound errors for reporting
            let solve_result = ctx.solve();
            if let Err(ref e) = solve_result {
                use nostos_types::TypeError;
                // Only surface MissingTraitImpl errors for types that can NEVER implement
                // the trait (tuples, containers, Bool, etc.). User-defined Named types might
                // have custom trait implementations that the inference env doesn't know about.
                if let TypeError::MissingTraitImpl { ref ty, ref trait_name, ref resolved_type } = e {
                    // Use resolved Type enum when available for type classification
                    // Also check if the type name matches a declared type parameter
                    // in any of the user functions (e.g., fn[Elem, Result](...))
                    let is_declared_type_param = user_fns.iter().any(|(_, (fn_def, _, _, _, _, _))| {
                        fn_def.type_params.iter().any(|tp| tp.name.node == *ty)
                    });
                    let is_type_param = if let Some(ref rty) = resolved_type {
                        matches!(rty, nostos_types::Type::TypeParam(_))
                    } else {
                        ty.len() <= 2 && ty.chars().next().map_or(false, |c| c.is_alphabetic())
                    };
                    let is_type_param = is_type_param || is_declared_type_param;
                    let is_type_var = if let Some(ref rty) = resolved_type {
                        matches!(rty, nostos_types::Type::Var(_))
                    } else {
                        ty.starts_with('?')
                    };

                    // For declared type parameters (multi-char like Result, Elem),
                    // batch inference may create false Num/Ord constraints from call sites.
                    // Suppress these entirely — per-function inference handles them correctly.
                    if is_declared_type_param {
                        // Don't report — batch inference can't distinguish function-body vs call-site constraints
                    }
                    // For type parameters used with Num/Ord, check if the function has that trait bound.
                    else if is_type_param && matches!(trait_name.as_str(), "Num" | "Ord") {
                        let has_explicit_type_params = user_fns.iter().any(|(_, (fn_def, _, _, _, _, _))| {
                            !fn_def.type_params.is_empty()
                        });
                        let has_required_bound = if has_explicit_type_params {
                            user_fns.iter().any(|(_, (fn_def, _, _, _, _, _))| {
                                fn_def.type_params.iter().any(|tp| {
                                    tp.name.node == *ty &&
                                    tp.constraints.iter().any(|c| c.node == *trait_name)
                                })
                            })
                        } else {
                            true
                        };
                        if !has_required_bound {
                            let error_span = ctx.last_error_span().unwrap_or_else(|| Span::new(0, 0));
                            let op_kind = if trait_name == "Ord" { "comparison" } else { "arithmetic" };
                            let compile_error = CompileError::TypeError {
                                message: format!(
                                    "type parameter `{}` must have `{}` trait bound to use {} operations. \
                                     Add trait bound: `[{}: {}]`",
                                    ty, trait_name, op_kind, ty, trait_name
                                ),
                                span: error_span,
                            };
                            let (source_name, source) = user_fns.first()
                                .map(|(_, (_, _, _, _, source, source_name))| (source_name.clone(), source.clone()))
                                .unwrap_or_else(|| ("unknown".to_string(), Arc::new(String::new())));
                            errors.push(("".to_string(), compile_error, source_name, source));
                        }
                    }

                    // Use TypeEnv::definitely_not_implements() with the actual Type enum
                    let is_definitely_non_implementing = if let Some(ref resolved) = resolved_type {
                        !is_type_param && !is_type_var &&
                        ctx.env.definitely_not_implements(resolved, trait_name)
                    } else {
                        false
                    };
                    if is_definitely_non_implementing {
                        let error_span = ctx.last_error_span().unwrap_or_else(|| Span::new(0, 0));
                        let compile_error = self.convert_type_error(e.clone(), error_span);
                        // Find the source file for this error span
                        let (source_name, source) = user_fns.first()
                            .map(|(_, (_, _, _, _, source, source_name))| (source_name.clone(), source.clone()))
                            .unwrap_or_else(|| ("unknown".to_string(), Arc::new(String::new())));
                        errors.push(("".to_string(), compile_error, source_name, source));
                    }
                } else if let TypeError::Mismatch { ref expected, ref found } = e {
                    // Mismatch errors from pre-checks (e.g., unzip on non-tuple list)
                    // are definitive - always report them.
                    // BUT: if either side contains unresolved type variables (?NNN),
                    // this is likely a false positive from batch inference where type
                    // constraints weren't fully resolved. The per-function type_check_fn
                    // pass will catch real errors with fully-resolved types.
                    let has_type_var = |s: &str| s.contains('?');
                    let is_type_param = |s: &str| s.len() == 1 && s.chars().next().map(|c| c.is_alphabetic()).unwrap_or(false);
                    if !has_type_var(expected) && !has_type_var(found) && !is_type_param(expected) && !is_type_param(found) {
                        let error_span = ctx.last_error_span().unwrap_or_else(|| Span::new(0, 0));
                        let compile_error = self.convert_type_error(e.clone(), error_span);
                        let (source_name, source) = user_fns.first()
                            .map(|(_, (_, _, _, _, source, source_name))| (source_name.clone(), source.clone()))
                            .unwrap_or_else(|| ("unknown".to_string(), Arc::new(String::new())));
                        errors.push(("".to_string(), compile_error, source_name, source));
                    }
                } else if let TypeError::OccursCheck(ref var, ref ty_str) = e {
                    // OccursCheck errors indicate cyclic types (e.g., T = List[T]).
                    // In batch inference, recursive generic multi-clause functions can
                    // produce spurious occurs checks when TypeParam types share vars
                    // across clauses. Only report when the type string doesn't contain
                    // unresolved type variables (real cycles between concrete types).
                    // The per-function type_check_fn pass will catch real cycles.
                    let has_type_var = var.contains('?') || ty_str.contains('?');
                    if !has_type_var {
                        let error_span = ctx.last_error_span().unwrap_or_else(|| Span::new(0, 0));
                        let compile_error = self.convert_type_error(e.clone(), error_span);
                        let (source_name, source) = user_fns.first()
                            .map(|(_, (_, _, _, _, source, source_name))| (source_name.clone(), source.clone()))
                            .unwrap_or_else(|| ("unknown".to_string(), Arc::new(String::new())));
                        errors.push(("".to_string(), compile_error, source_name, source));
                    }
                } else if let TypeError::NoSuchField { ref ty, ref field, ref resolved_type } = e {
                    // NoSuchField from deferred_has_field is definitive only for
                    // built-in types where HM inference has complete field knowledge.
                    // For user-defined Named types, HM may not know about reactive
                    // fields, private fields, or module-specific fields - those are
                    // better handled by later compilation stages.
                    let is_builtin_type = if let Some(ref resolved) = resolved_type {
                        use nostos_types::Type;
                        matches!(resolved,
                            Type::List(_) | Type::Map(_, _) | Type::Set(_)
                            | Type::Tuple(_)
                            | Type::Int | Type::String | Type::Bool
                            | Type::Float | Type::Char | Type::Unit
                        ) && !matches!(resolved, Type::Var(_))
                    } else {
                        // String fallback
                        (ty.starts_with("List[") || ty.starts_with("Map[")
                            || ty.starts_with("Set[")
                            || (ty.starts_with('(') && !ty.contains("->"))
                            || ty == "Int" || ty == "String" || ty == "Bool"
                            || ty == "Float" || ty == "Char" || ty == "Unit")
                            && !ty.starts_with('?')
                    };
                    if is_builtin_type {
                        let error_span = ctx.last_error_span().unwrap_or_else(|| Span::new(0, 0));
                        let compile_error = self.convert_type_error(e.clone(), error_span);
                        let (source_name, source) = user_fns.first()
                            .map(|(_, (_, _, _, _, source, source_name))| (source_name.clone(), source.clone()))
                            .unwrap_or_else(|| ("unknown".to_string(), Arc::new(String::new())));
                        errors.push(("".to_string(), compile_error, source_name, source));
                    }
                }
            }

            // When solve() exits early (e.g., on a type error), some constraints
            // remain unprocessed. Extract Var→Type bindings from these remaining
            // constraints to supplement the substitution. This ensures that function
            // signatures in pending_fn_signatures get properly resolved during
            // enrichment, even when batch solve couldn't process everything.
            // Note: We do NOT supplement the global substitution here.
            // Doing so can corrupt overloaded function signatures by linking
            // their Var IDs through call-site fresh vars. Instead, each function's
            // enrichment uses clause_param_types/clause_ret_types as local fallback.

            // Transfer inferred expression types from user inference.
            // With file_id in Span, spans are now unique across files.
            // Apply substitution to resolve type variables to concrete types.
            let user_expr_types: HashMap<_, _> = ctx.take_expr_types()
                .into_iter()
                .map(|(span, ty)| {
                    let resolved = ctx.apply_full_subst(&ty);
                    (span, resolved)
                })
                .collect();
            self.inferred_expr_types.extend(user_expr_types);

            // Transfer implicit conversions from inference context.
            // These map expression Span -> conversion function name (e.g., "tensorFromList").
            for (span, fn_name) in ctx.implicit_conversions.iter() {
                self.implicit_conversions.insert(*span, fn_name.clone());
            }

            // Enrich user function signatures with inferred trait bounds.
            // After batch inference, function bodies may have discovered trait bounds
            // (e.g., add(x,y) = x+y discovers Num on the param vars). Extract these
            // bounds and add them as type_params so that callers get proper HasTrait
            // constraints when instantiating the function.
            // Also replace Var types with TypeParam types in the signature so that
            // instantiate_function can properly propagate the bounds to call sites.
            // Store root_var→letter mapping per function for post-subst replacement
            // of Var inside nested types (e.g., curried function returns).
            let mut fn_root_var_to_letter: std::collections::HashMap<String, std::collections::HashMap<u32, String>> = std::collections::HashMap::new();
            // Enrichment helper: extract trait bounds from batch ctx for a function's vars,
            // replacing Var types with TypeParam and adding type_params with constraints.
            fn enrich_fn_signature(
                fn_name: &str,
                fn_type: &mut nostos_types::FunctionType,
                ctx: &InferCtx,
                fn_root_var_to_letter: &mut std::collections::HashMap<String, std::collections::HashMap<u32, String>>,
            ) {
                if !fn_type.type_params.is_empty() {
                    return; // Already has explicit type_params
                }
                let mut root_var_to_letter: std::collections::HashMap<u32, String> = std::collections::HashMap::new();
                let mut letter_idx = 0u8;
                let mut new_type_params: Vec<nostos_types::TypeParam> = Vec::new();

                let mut all_original_vars: Vec<u32> = Vec::new();
                for param in &fn_type.params {
                    if let nostos_types::Type::Var(var_id) = param {
                        all_original_vars.push(*var_id);
                    }
                }
                if let nostos_types::Type::Var(var_id) = fn_type.ret.as_ref() {
                    all_original_vars.push(*var_id);
                }

                let mut original_to_root: std::collections::HashMap<u32, u32> = std::collections::HashMap::new();
                for &var_id in &all_original_vars {
                    let resolved = ctx.apply_full_subst(&nostos_types::Type::Var(var_id));
                    if let nostos_types::Type::Var(root_id) = resolved {
                        original_to_root.insert(var_id, root_id);

                        if !root_var_to_letter.contains_key(&root_id) {
                            let bounds: Vec<String> = ctx.get_trait_bounds(root_id)
                                .into_iter()
                                .cloned()
                                .filter(|b| !b.starts_with("HasMethod(") && !b.starts_with("HasField("))
                                .collect();

                            let mut unique_bounds: Vec<String> = Vec::new();
                            for b in bounds {
                                if !unique_bounds.contains(&b) {
                                    unique_bounds.push(b);
                                }
                            }

                            if !unique_bounds.is_empty() {
                                let letter = format!("{}", (b'a' + letter_idx) as char);
                                letter_idx += 1;
                                new_type_params.push(nostos_types::TypeParam {
                                    name: letter.clone(),
                                    constraints: unique_bounds,
                                });
                                root_var_to_letter.insert(root_id, letter);
                            }
                        }
                    }
                }

                if !new_type_params.is_empty() {
                    fn_root_var_to_letter.insert(fn_name.to_string(), root_var_to_letter.clone());

                    let new_params: Vec<nostos_types::Type> = fn_type.params.iter().map(|p| {
                        if let nostos_types::Type::Var(var_id) = p {
                            if let Some(&root_id) = original_to_root.get(var_id) {
                                if let Some(letter) = root_var_to_letter.get(&root_id) {
                                    return nostos_types::Type::TypeParam(letter.clone());
                                }
                            }
                        }
                        p.clone()
                    }).collect();
                    let new_ret = if let nostos_types::Type::Var(var_id) = fn_type.ret.as_ref() {
                        if let Some(&root_id) = original_to_root.get(var_id) {
                            if let Some(letter) = root_var_to_letter.get(&root_id) {
                                nostos_types::Type::TypeParam(letter.clone())
                            } else {
                                (*fn_type.ret).clone()
                            }
                        } else {
                            (*fn_type.ret).clone()
                        }
                    } else {
                        (*fn_type.ret).clone()
                    };

                    fn_type.type_params = new_type_params;
                    fn_type.params = new_params;
                    fn_type.ret = Box::new(new_ret);
                }
            }

            for (fn_name, fn_type) in self.pending_fn_signatures.iter_mut() {
                let is_stdlib = stdlib_fn_names.contains(fn_name) || fn_name.starts_with("stdlib.");
                if is_stdlib {
                    continue;
                }
                enrich_fn_signature(fn_name, fn_type, &ctx, &mut fn_root_var_to_letter);
            }

            // Apply full substitution (including TypeParam resolution) only to user signatures
            // Exclude both compiled-from-source stdlib (in stdlib_fn_names) AND loaded-from-cache
            // stdlib (which start with "stdlib." prefix).
            for (fn_name, fn_type) in self.pending_fn_signatures.iter_mut() {
                let is_stdlib = stdlib_fn_names.contains(fn_name) || fn_name.starts_with("stdlib.");
                if !is_stdlib {
                    // For functions with explicit type_params (e.g., identity[T], double[T: Num]),
                    // use apply_subst (Var resolution only) instead of apply_full_subst.
                    // apply_full_subst also resolves TypeParam through type_param_mappings,
                    // which at this point contains stale mappings from the LAST function
                    // inferred in the batch. This would incorrectly resolve TypeParam("T")
                    // in all functions to whatever T was in the last function (e.g., Int
                    // from `double`'s `x * 2`), breaking polymorphism for other functions
                    // like `identity[T]`.
                    let has_explicit_type_params = fn_type.type_params.iter()
                        .any(|tp| fn_type.params.iter().any(|p| *p == nostos_types::Type::TypeParam(tp.name.clone()))
                            || *fn_type.ret == nostos_types::Type::TypeParam(tp.name.clone()));
                    let mut resolved_params: Vec<nostos_types::Type> = fn_type.params.iter()
                        .map(|p| if has_explicit_type_params { ctx.env.apply_subst(p) } else { ctx.apply_full_subst(p) })
                        .collect();
                    let mut resolved_ret = if has_explicit_type_params { ctx.env.apply_subst(&fn_type.ret) } else { ctx.apply_full_subst(&fn_type.ret) };

                    // Fix: Batch inference can contaminate shared Var IDs (Var(1)='a', Var(2)='b',
                    // etc.) causing apply_full_subst to resolve them to Named("a"), Named("b")
                    // instead of keeping them as type variables. Normalize these back to Var IDs
                    // so Phase 3 can properly convert them to TypeParams.
                    fn normalize_leaked_type_params(ty: &nostos_types::Type, known_types: &HashMap<String, TypeInfo>) -> nostos_types::Type {
                        match ty {
                            nostos_types::Type::Named { name, args } if args.is_empty() => {
                                // Single lowercase letter not in known types → type parameter leaked as Named
                                if name.len() == 1 && name.chars().next().map(|c| c.is_ascii_lowercase()).unwrap_or(false)
                                    && !known_types.contains_key(name)
                                {
                                    let var_id = (name.chars().next().unwrap() as u32) - ('a' as u32) + 1;
                                    nostos_types::Type::Var(var_id)
                                } else {
                                    ty.clone()
                                }
                            }
                            nostos_types::Type::Named { name, args } => {
                                let new_args: Vec<nostos_types::Type> = args.iter()
                                    .map(|a| normalize_leaked_type_params(a, known_types))
                                    .collect();
                                nostos_types::Type::Named { name: name.clone(), args: new_args }
                            }
                            nostos_types::Type::List(e) => nostos_types::Type::List(Box::new(normalize_leaked_type_params(e, known_types))),
                            nostos_types::Type::Array(e) => nostos_types::Type::Array(Box::new(normalize_leaked_type_params(e, known_types))),
                            nostos_types::Type::Set(e) => nostos_types::Type::Set(Box::new(normalize_leaked_type_params(e, known_types))),
                            nostos_types::Type::IO(e) => nostos_types::Type::IO(Box::new(normalize_leaked_type_params(e, known_types))),
                            nostos_types::Type::Map(k, v) => nostos_types::Type::Map(
                                Box::new(normalize_leaked_type_params(k, known_types)),
                                Box::new(normalize_leaked_type_params(v, known_types)),
                            ),
                            nostos_types::Type::Tuple(elems) => nostos_types::Type::Tuple(
                                elems.iter().map(|e| normalize_leaked_type_params(e, known_types)).collect(),
                            ),
                            nostos_types::Type::Function(ft) => nostos_types::Type::Function(nostos_types::FunctionType {
                                type_params: ft.type_params.clone(),
                                params: ft.params.iter().map(|p| normalize_leaked_type_params(p, known_types)).collect(),
                                ret: Box::new(normalize_leaked_type_params(&ft.ret, known_types)),
                                required_params: ft.required_params,
                                var_bounds: ft.var_bounds.clone(),
                            }),
                            _ => ty.clone(),
                        }
                    }
                    for p in resolved_params.iter_mut() {
                        *p = normalize_leaked_type_params(p, &self.types);
                    }
                    resolved_ret = normalize_leaked_type_params(&resolved_ret, &self.types);

                    // Fallback: if resolved params/ret are still bare Vars (solve() may have
                    // exited early on an error from a different function), use LOCAL resolution
                    // from remaining constraints to recover the actual types.
                    // This builds a mini-substitution for each function independently,
                    // avoiding cross-contamination between overloaded functions.
                    if !has_explicit_type_params {
                        let has_unresolved = resolved_params.iter().any(|p| matches!(p, nostos_types::Type::Var(_)))
                            || matches!(&resolved_ret, nostos_types::Type::Var(_));
                        if has_unresolved {
                            // Collect this function's seed Var IDs (from pending_fn_signatures)
                            let mut seed_vars: Vec<u32> = Vec::new();
                            for p in &fn_type.params {
                                if let nostos_types::Type::Var(id) = p { seed_vars.push(*id); }
                            }
                            if let nostos_types::Type::Var(id) = fn_type.ret.as_ref() { seed_vars.push(*id); }

                            // Also add vars from clause_param_types and clause_ret_types
                            fn collect_var_ids(ty: &nostos_types::Type, out: &mut Vec<u32>) {
                                match ty {
                                    nostos_types::Type::Var(id) => out.push(*id),
                                    nostos_types::Type::List(inner) | nostos_types::Type::Set(inner) | nostos_types::Type::IO(inner) | nostos_types::Type::Array(inner) => collect_var_ids(inner, out),
                                    nostos_types::Type::Map(k, v) => { collect_var_ids(k, out); collect_var_ids(v, out); }
                                    nostos_types::Type::Tuple(elems) => { for e in elems { collect_var_ids(e, out); } }
                                    nostos_types::Type::Function(ft) => { for p in &ft.params { collect_var_ids(p, out); } collect_var_ids(&ft.ret, out); }
                                    nostos_types::Type::Named { args, .. } => { for a in args { collect_var_ids(a, out); } }
                                    _ => {}
                                }
                            }
                            for p in &fn_type.params {
                                if let nostos_types::Type::Var(id) = p {
                                    if let Some(clause_ty) = ctx.clause_param_types.get(id) {
                                        collect_var_ids(clause_ty, &mut seed_vars);
                                    }
                                }
                            }
                            if let nostos_types::Type::Var(id) = fn_type.ret.as_ref() {
                                if let Some(clause_ty) = ctx.clause_ret_types.get(id) {
                                    collect_var_ids(clause_ty, &mut seed_vars);
                                }
                            }

                            let local_subst = ctx.resolve_vars_locally(&seed_vars);

                            // Apply local resolution to clause types as fallback
                            for (i, orig_param) in fn_type.params.iter().enumerate() {
                                if let nostos_types::Type::Var(param_var_id) = orig_param {
                                    if matches!(&resolved_params[i], nostos_types::Type::Var(_)) {
                                        if let Some(clause_param) = ctx.clause_param_types.get(param_var_id) {
                                            resolved_params[i] = ctx.apply_local_subst(clause_param, &local_subst);
                                        }
                                    }
                                }
                            }
                            if let nostos_types::Type::Var(ret_var_id) = &resolved_ret {
                                if let Some(clause_ret) = ctx.clause_ret_types.get(ret_var_id) {
                                    resolved_ret = ctx.apply_local_subst(clause_ret, &local_subst);
                                }
                            }

                            // Re-normalize after local resolution: clause_param_types / clause_ret_types
                            // may contain leaked Named("a") from batch inference, re-contaminating the
                            // params/ret that were just normalized above. Clean them up again so Phase 3
                            // sees proper Var types and can convert them to TypeParams.
                            for p in resolved_params.iter_mut() {
                                *p = normalize_leaked_type_params(p, &self.types);
                            }
                            resolved_ret = normalize_leaked_type_params(&resolved_ret, &self.types);
                        }
                    }

                    // Replace Var IDs with TypeParam in resolved types.
                    // This handles:
                    // 1. Curried functions: adder(a) = b => a+b → TypeParam("a") in both outer
                    //    param AND inner Function return type
                    // 2. Multi-TypeParam functions: pair[A,B] where HM may have merged A=B
                    // 3. Nested types: List[a], Map[k,v], (a, b), etc.
                    fn restore_type_params(ty: &nostos_types::Type, var_map: &std::collections::HashMap<u32, String>) -> nostos_types::Type {
                        match ty {
                            nostos_types::Type::Var(id) => {
                                if let Some(tp_name) = var_map.get(id) {
                                    nostos_types::Type::TypeParam(tp_name.clone())
                                } else {
                                    ty.clone()
                                }
                            }
                            nostos_types::Type::Function(ft) => {
                                nostos_types::Type::Function(nostos_types::FunctionType {
                                    type_params: ft.type_params.clone(),
                                    params: ft.params.iter().map(|p| restore_type_params(p, var_map)).collect(),
                                    ret: Box::new(restore_type_params(&ft.ret, var_map)),
                                    required_params: ft.required_params,
                                    var_bounds: vec![],
                                })
                            }
                            nostos_types::Type::Tuple(elems) => {
                                nostos_types::Type::Tuple(elems.iter().map(|e| restore_type_params(e, var_map)).collect())
                            }
                            nostos_types::Type::List(elem) => {
                                nostos_types::Type::List(Box::new(restore_type_params(elem, var_map)))
                            }
                            nostos_types::Type::Map(k, v) => {
                                nostos_types::Type::Map(
                                    Box::new(restore_type_params(k, var_map)),
                                    Box::new(restore_type_params(v, var_map)),
                                )
                            }
                            nostos_types::Type::Set(elem) => {
                                nostos_types::Type::Set(Box::new(restore_type_params(elem, var_map)))
                            }
                            nostos_types::Type::Named { name, args } => {
                                nostos_types::Type::Named {
                                    name: name.clone(),
                                    args: args.iter().map(|a| restore_type_params(a, var_map)).collect(),
                                }
                            }
                            nostos_types::Type::IO(inner) => {
                                nostos_types::Type::IO(Box::new(restore_type_params(inner, var_map)))
                            }
                            nostos_types::Type::Array(elem) => {
                                nostos_types::Type::Array(Box::new(restore_type_params(elem, var_map)))
                            }
                            _ => ty.clone(),
                        }
                    }

                    let (final_params, final_ret) = if !fn_type.type_params.is_empty() {
                        // Get the root_var→letter mapping (stored during enrichment step)
                        let var_map = fn_root_var_to_letter.get(fn_name);

                        // For multi-TypeParam functions, also handle merged vars
                        // (where HM unified distinct TypeParams)
                        let mut restored_params = resolved_params.clone();
                        if fn_type.type_params.len() >= 2 {
                            // Only apply to Var params that have trait bounds (i.e., are in the
                            // root_var_to_letter map). Unconstrained Vars (like a third param `op`
                            // that isn't involved in arithmetic) should be left as Vars and
                            // converted to independent TypeParams in Phase 3.
                            let var_map_ref = fn_root_var_to_letter.get(fn_name);
                            let mut var_to_first_tp: std::collections::HashMap<u32, String> = std::collections::HashMap::new();
                            let mut tp_index = 0;
                            for (i, orig_param) in fn_type.params.iter().enumerate() {
                                if let nostos_types::Type::Var(var_id) = orig_param {
                                    // Check if this var's root has a letter mapping (i.e., has trait bounds)
                                    let has_letter = var_map_ref.map_or(false, |vm| {
                                        let resolved = ctx.apply_full_subst(&nostos_types::Type::Var(*var_id));
                                        if let nostos_types::Type::Var(root_id) = resolved {
                                            vm.contains_key(&root_id)
                                        } else {
                                            false
                                        }
                                    });
                                    if has_letter && tp_index < fn_type.type_params.len() {
                                        let tp_name = &fn_type.type_params[tp_index].name;
                                        if let Some(existing_tp) = var_to_first_tp.get(var_id) {
                                            if existing_tp != tp_name {
                                                restored_params[i] = nostos_types::Type::TypeParam(tp_name.clone());
                                            }
                                        } else {
                                            var_to_first_tp.insert(*var_id, tp_name.clone());
                                            restored_params[i] = nostos_types::Type::TypeParam(tp_name.clone());
                                        }
                                        tp_index += 1;
                                    }
                                }
                            }
                        }

                        // Apply recursive Var→TypeParam replacement to ALL resolved types
                        if let Some(var_map) = var_map {
                            let final_params = restored_params.iter()
                                .map(|p| restore_type_params(p, var_map))
                                .collect();
                            let final_ret = restore_type_params(&resolved_ret, var_map);
                            (final_params, final_ret)
                        } else {
                            (restored_params, resolved_ret)
                        }
                    } else {
                        (resolved_params, resolved_ret)
                    };

                    // Phase 3: Convert any remaining unresolved Vars to TypeParams.
                    // This handles functions like constFn(x) = (y) => x where the return
                    // type resolves to Function(Var, Var) with Vars inside nested types.
                    // Phase 1 only caught top-level Vars with trait bounds; this catches
                    // ALL remaining Vars to enable proper let-polymorphism in the second pass.
                    let (final_params, final_ret) = {
                        fn collect_all_vars(ty: &nostos_types::Type, vars: &mut Vec<u32>) {
                            match ty {
                                nostos_types::Type::Var(id) => {
                                    if !vars.contains(id) { vars.push(*id); }
                                }
                                nostos_types::Type::Function(ft) => {
                                    for p in &ft.params { collect_all_vars(p, vars); }
                                    collect_all_vars(&ft.ret, vars);
                                }
                                nostos_types::Type::List(e) | nostos_types::Type::Array(e) |
                                nostos_types::Type::Set(e) | nostos_types::Type::IO(e) => {
                                    collect_all_vars(e, vars);
                                }
                                nostos_types::Type::Map(k, v) => {
                                    collect_all_vars(k, vars); collect_all_vars(v, vars);
                                }
                                nostos_types::Type::Tuple(elems) => {
                                    for e in elems { collect_all_vars(e, vars); }
                                }
                                nostos_types::Type::Named { args, .. } => {
                                    for a in args { collect_all_vars(a, vars); }
                                }
                                _ => {}
                            }
                        }

                        let mut remaining_vars = Vec::new();
                        for p in &final_params { collect_all_vars(p, &mut remaining_vars); }
                        collect_all_vars(&final_ret, &mut remaining_vars);

                        if !remaining_vars.is_empty() && fn_type.type_params.is_empty() {
                            // Assign letters and create TypeParams for remaining Vars
                            let mut var_to_letter: std::collections::HashMap<u32, String> = std::collections::HashMap::new();
                            let mut new_type_params: Vec<nostos_types::TypeParam> = Vec::new();
                            let mut letter_idx = 0u8;

                            for var_id in &remaining_vars {
                                if !var_to_letter.contains_key(var_id) {
                                    let letter = format!("{}", (b'a' + letter_idx) as char);
                                    letter_idx += 1;

                                    // Collect ALL trait bounds for this var, including
                                    // user-defined traits (e.g., Scorable). These are needed
                                    // during monomorphization to check trait satisfaction.
                                    // Filter out internal HasMethod/HasField constraints.
                                    let bounds: Vec<String> = ctx.get_trait_bounds(*var_id)
                                        .into_iter()
                                        .cloned()
                                        .filter(|b| !b.starts_with("HasMethod(") && !b.starts_with("HasField("))
                                        .collect();
                                    let mut unique_bounds: Vec<String> = Vec::new();
                                    for b in bounds {
                                        if !unique_bounds.contains(&b) {
                                            unique_bounds.push(b);
                                        }
                                    }

                                    new_type_params.push(nostos_types::TypeParam {
                                        name: letter.clone(),
                                        constraints: unique_bounds,
                                    });
                                    var_to_letter.insert(*var_id, letter);
                                }
                            }

                            fn_type.type_params = new_type_params;
                            let fp: Vec<nostos_types::Type> = final_params.iter()
                                .map(|p| restore_type_params(p, &var_to_letter))
                                .collect();
                            let fr = restore_type_params(&final_ret, &var_to_letter);
                            (fp, fr)
                        } else if !remaining_vars.is_empty() && !fn_type.type_params.is_empty() {
                            // Function already has type_params but may have additional Vars
                            // (e.g., from nested Function types). Add new TypeParams for those.
                            let existing_letters: std::collections::HashSet<String> = fn_type.type_params.iter()
                                .map(|tp| tp.name.clone()).collect();
                            let mut var_to_letter: std::collections::HashMap<u32, String> = std::collections::HashMap::new();
                            // First, map existing TypeParam Vars
                            if let Some(existing_map) = fn_root_var_to_letter.get(fn_name) {
                                for (vid, letter) in existing_map {
                                    var_to_letter.insert(*vid, letter.clone());
                                }
                            }
                            let mut letter_idx = fn_type.type_params.len() as u8;
                            let mut added = false;
                            for var_id in &remaining_vars {
                                if !var_to_letter.contains_key(var_id) {
                                    let letter = format!("{}", (b'a' + letter_idx) as char);
                                    letter_idx += 1;
                                    if !existing_letters.contains(&letter) {
                                        let bounds: Vec<String> = ctx.get_trait_bounds(*var_id)
                                            .into_iter()
                                            .cloned()
                                            .filter(|b| !b.starts_with("HasMethod(") && !b.starts_with("HasField("))
                                            .collect();
                                        let mut unique_bounds: Vec<String> = Vec::new();
                                        for b in bounds {
                                            if !unique_bounds.contains(&b) {
                                                unique_bounds.push(b);
                                            }
                                        }
                                        fn_type.type_params.push(nostos_types::TypeParam {
                                            name: letter.clone(),
                                            constraints: unique_bounds,
                                        });
                                        added = true;
                                    }
                                    var_to_letter.insert(*var_id, letter);
                                }
                            }
                            if added || !var_to_letter.is_empty() {
                                let fp: Vec<nostos_types::Type> = final_params.iter()
                                    .map(|p| restore_type_params(p, &var_to_letter))
                                    .collect();
                                let fr = restore_type_params(&final_ret, &var_to_letter);
                                (fp, fr)
                            } else {
                                (final_params, final_ret)
                            }
                        } else {
                            (final_params, final_ret)
                        }
                    };

                    *fn_type = nostos_types::FunctionType {
                        type_params: fn_type.type_params.clone(),
                        params: final_params,
                        ret: Box::new(final_ret),
                        required_params: fn_type.required_params,
                        var_bounds: vec![],
                    };

                }
            }
        }

        // POST-ENRICHMENT FIX-UP: Resolve unresolved return types for wrapper functions.
        // When a function's return type is TypeParam("a") but all params are concrete,
        // the function is not truly generic - its return type just wasn't resolved during
        // batch inference (because instantiate_function creates fresh vars disconnected
        // from the originals). Re-infer these functions with resolved signatures.
        {
            let needs_reinfer: Vec<String> = self.pending_fn_signatures.iter()
                .filter(|(fn_name, fn_type)| {
                    let is_stdlib = fn_name.starts_with("stdlib.");
                    !is_stdlib && fn_type.type_params.iter().any(|tp| {
                        *fn_type.ret == nostos_types::Type::TypeParam(tp.name.clone())
                    }) && fn_type.params.iter().all(|p| {
                        !matches!(p, nostos_types::Type::TypeParam(_))
                    })
                })
                .map(|(name, _)| name.clone())
                .collect();

            if !needs_reinfer.is_empty() {
                // Build resolved function signature lookup (qualified -> return type)
                let resolved_returns: HashMap<String, nostos_types::Type> = self.pending_fn_signatures.iter()
                    .filter(|(_, ft)| !matches!(ft.ret.as_ref(), nostos_types::Type::TypeParam(_) | nostos_types::Type::Var(_)))
                    .map(|(name, ft)| (name.clone(), (*ft.ret).clone()))
                    .collect();


                // Also build bare name → return type lookup
                let mut bare_returns: HashMap<String, nostos_types::Type> = HashMap::new();
                for (fn_name, ret_ty) in &resolved_returns {
                    let base = fn_name.split('/').next().unwrap_or(fn_name);
                    if let Some(dot_pos) = base.rfind('.') {
                        let bare = &base[dot_pos + 1..];
                        let bare_key = if let Some(slash_pos) = fn_name.find('/') {
                            format!("{}{}", bare, &fn_name[slash_pos..])
                        } else {
                            bare.to_string()
                        };
                        bare_returns.entry(bare_key).or_insert_with(|| ret_ty.clone());
                    }
                }

                // For each function needing reinference, analyze its body AST
                for fn_name in &needs_reinfer {
                    let fn_info = pending.iter().find(|(fn_def, module_path, _, _, _, _)| {
                        let qualified = if module_path.is_empty() {
                            fn_def.name.node.clone()
                        } else {
                            format!("{}.{}", module_path.join("."), fn_def.name.node)
                        };
                        let clause = fn_def.clauses.first();
                        if let Some(clause) = clause {
                            let type_param_names: std::collections::HashSet<&str> = fn_def.type_params.iter()
                                .map(|tp| tp.name.node.as_str())
                                .collect();
                            let parts: Vec<String> = clause.params.iter().map(|p| {
                                if let Some(ty_expr) = &p.ty {
                                    let ty_str = self.type_expr_to_string(ty_expr);
                                    if type_param_names.contains(ty_str.as_str()) { "_".to_string() } else { ty_str }
                                } else { "_".to_string() }
                            }).collect();
                            let arity_suffix = if clause.params.is_empty() { "/".to_string() } else { format!("/{}", parts.join(",")) };
                            let full_name = format!("{}{}", qualified, arity_suffix);
                            &full_name == fn_name
                        } else {
                            false
                        }
                    });

                    if let Some((fn_def, module_path, _, _, _, _)) = fn_info {
                        if let Some(clause) = fn_def.clauses.first() {
                            // Try to resolve the return type by analyzing the body's last expression
                            let module_prefix = if module_path.is_empty() {
                                String::new()
                            } else {
                                format!("{}.", module_path.join("."))
                            };
                            if let Some(resolved_ret) = self.resolve_body_return_type(
                                &clause.body, &module_prefix, &resolved_returns, &bare_returns
                            ) {
                                if let Some(sig) = self.pending_fn_signatures.get_mut(fn_name) {
                                    sig.ret = Box::new(resolved_ret);
                                    sig.type_params.clear();
                                }
                            }
                        }
                    }
                }
            }
        }

        // POST-ENRICHMENT FIX-UP: For functions with template decorators, the decorator
        // transforms the function body at compile time, potentially changing the return type.
        // The batch inference ran on the UNDECORATED body (e.g., `myFunc(x:Int) = 0` → ret=Int),
        // but after decoration the body may return a different type (e.g., → ret=String).
        // Reset the return type to an unresolved Var so that type_check_fn (which runs on the
        // DECORATED body) doesn't get a stale Int constraint.
        {
            let decorated_fn_names: std::collections::HashSet<String> = pending.iter()
                .filter(|(fn_def, _, _, _, _, _)| !fn_def.decorators.is_empty())
                .map(|(fn_def, module_path, _, _, _, _)| {
                    if module_path.is_empty() {
                        fn_def.name.node.clone()
                    } else {
                        format!("{}.{}", module_path.join("."), fn_def.name.node)
                    }
                })
                .collect();

            if !decorated_fn_names.is_empty() {
                for (fn_name, fn_type) in self.pending_fn_signatures.iter_mut() {
                    let base_name = fn_name.split('/').next().unwrap_or(fn_name);
                    if decorated_fn_names.contains(base_name) {
                        // Only reset if return type is fully concrete (no Vars/TypeParams).
                        // If it's still a Var, the type is already appropriately unresolved.
                        // If it's a TypeParam, there's some generic return type - keep it.
                        let ret_is_concrete = !matches!(fn_type.ret.as_ref(),
                            nostos_types::Type::Var(_) | nostos_types::Type::TypeParam(_));
                        if ret_is_concrete {
                            // Use Var(u32::MAX) as sentinel for "return type unknown due to decoration"
                            fn_type.ret = Box::new(nostos_types::Type::Var(u32::MAX));
                        }
                    }
                }
            }
        }

        // Pre-populate fn_asts with all pending functions so they can see each other
        // This is critical for multi-file modules where functions from different files
        // need to call each other (e.g., main.nos calling benchmark.nos in the same module)
        for (fn_def, module_path, fn_imports, _, _, _) in &pending {
            let base_name = if module_path.is_empty() {
                fn_def.name.node.clone()
            } else {
                format!("{}.{}", module_path.join("."), fn_def.name.node)
            };
            if let Some(clause) = fn_def.clauses.first() {
                let param_types: Vec<String> = clause.params.iter()
                    .map(|p| p.ty.as_ref()
                        .map(|t| self.type_expr_to_string(t))
                        .unwrap_or_else(|| "_".to_string()))
                    .collect();
                let signature = param_types.join(",");
                let name = format!("{}/{}", base_name, signature);
                // Insert a placeholder in fn_asts so has_function_with_base can find it
                self.fn_asts.insert(name.clone(), fn_def.clone());
                // Use the module-specific imports (not self.imports which may be from a different module)
                self.fn_ast_imports.insert(name.clone(), fn_imports.clone());
                // Update fn_asts_by_base index
                let fn_base = name.split('/').next().unwrap_or(&name);
                self.fn_asts_by_base
                    .entry(fn_base.to_string())
                    .or_insert_with(HashSet::new)
                    .insert(name);
            }
        }

        // First pass: compile all functions
        // Collect function info for type validation later (fn_name -> (fn_def, imports, source_name, source))
        let mut pending_fn_info: Vec<(String, FnDef, HashMap<String, String>, String, Arc<String>)> = Vec::new();

        // Save polymorphic_fns state before compilation to detect newly discovered polymorphic functions
        let poly_fns_before: std::collections::HashSet<String> = self.polymorphic_fns.clone();

        // Store all function compilation info for potential recompilation pass
        let mut fn_compile_info: Vec<(FnDef, Vec<String>, HashMap<String, String>, Vec<usize>, Arc<String>, String)> = Vec::new();

        for (fn_def, module_path, imports, line_starts, source, source_name) in pending {
            let saved_path = self.module_path.clone();
            let saved_imports = self.imports.clone();
            let saved_line_starts = self.line_starts.clone();
            let saved_source = self.current_source.clone();
            let saved_source_name = self.current_source_name.clone();

            // Build qualified function name
            let fn_name = if module_path.is_empty() {
                fn_def.name.node.clone()
            } else {
                format!("{}.{}", module_path.join("."), fn_def.name.node)
            };

            // Store for type validation
            pending_fn_info.push((fn_name.clone(), fn_def.clone(), imports.clone(), source_name.clone(), source.clone()));
            // Store for potential recompilation
            fn_compile_info.push((fn_def.clone(), module_path.clone(), imports.clone(), line_starts.clone(), source.clone(), source_name.clone()));

            self.module_path = module_path;
            // Merge imports instead of replacing, to be safe.
            self.imports.extend(imports);
            self.line_starts = line_starts;
            self.current_source = Some(source.clone());
            self.current_source_name = Some(source_name.clone());

            // Continue compiling other functions even if one fails
            match self.compile_fn_def(&fn_def) {
                Ok(()) => {}
                Err(e) => {
                    errors.push((fn_name, e, source_name.clone(), source.clone()));
                }
            }

            self.module_path = saved_path;
            self.imports = saved_imports;
            self.line_starts = saved_line_starts;
            self.current_source = saved_source;
            self.current_source_name = saved_source_name;
        }

        // Recompilation pass: if new polymorphic functions were discovered during the first pass,
        // propagate polymorphism transitively and recompile concrete callers.
        // This handles cross-module cases where a caller is compiled before its polymorphic callee.
        // Loop until no new polymorphic functions are discovered (handles transitive chains like
        // main → alpha.run → beta.doubleAll where each level needs to be marked polymorphic).
        let mut current_poly_set = poly_fns_before.clone();
        for _poly_iter in 0..5 {
            let new_poly_fns: Vec<String> = self.polymorphic_fns.difference(&current_poly_set).cloned().collect();
            if new_poly_fns.is_empty() {
                break;
            }
            // Update current set so next iteration only looks at newly added
            current_poly_set = self.polymorphic_fns.clone();

            // Extract base names of newly-polymorphic functions for matching
            // e.g., "helper.process/_" → "helper.process"
            let new_poly_bases: Vec<String> = new_poly_fns.iter()
                .map(|n| n.split('/').next().unwrap_or(n).to_string())
                .collect();

            // Helper: compute fn_key and fn_key_wildcard for a function in fn_compile_info
            let compute_fn_keys = |fn_def: &FnDef, module_path: &[String], compiler: &Self| -> (String, String, String) {
                let fn_name = if module_path.is_empty() {
                    fn_def.name.node.clone()
                } else {
                    format!("{}.{}", module_path.join("."), fn_def.name.node)
                };
                let arity = fn_def.clauses[0].params.len();
                let fn_key_wildcard = if arity == 0 {
                    format!("{}/", fn_name)
                } else {
                    format!("{}/{}", fn_name, vec!["_"; arity].join(","))
                };
                let fn_key = {
                    let mut param_types: Vec<String> = vec!["_".to_string(); arity];
                    for clause in &fn_def.clauses {
                        for (i, param) in clause.params.iter().enumerate() {
                            if let Some(ty) = &param.ty {
                                let ty_str = compiler.type_expr_to_string(ty);
                                if param_types[i] == "_" {
                                    param_types[i] = ty_str;
                                }
                            }
                        }
                    }
                    let base_name = if module_path.is_empty() {
                        fn_def.name.node.clone()
                    } else {
                        format!("{}.{}", module_path.join("."), fn_def.name.node)
                    };
                    format!("{}/{}", base_name, param_types.join(","))
                };
                (fn_name, fn_key, fn_key_wildcard)
            };

            // Helper: check if a function's code calls any of the new poly functions
            let calls_any_poly = |fn_key: &str, fn_key_wildcard: &str, compiler: &Self| -> bool {
                if let Some(func_val) = compiler.functions.get(fn_key).or_else(|| compiler.functions.get(fn_key_wildcard)) {
                    let code = &func_val.code;
                    code.code.iter().any(|instr| {
                        let func_idx = match instr {
                            Instruction::CallDirect(_, idx, _) => Some(*idx),
                            Instruction::TailCallDirect(idx, _) => Some(*idx),
                            _ => None,
                        };
                        if let Some(idx) = func_idx {
                            if let Some(called_name) = compiler.function_list.get(idx as usize) {
                                let called_base = called_name.split('/').next().unwrap_or(called_name);
                                new_poly_bases.iter().any(|b| called_base == b)
                            } else {
                                false
                            }
                        } else {
                            false
                        }
                    })
                } else {
                    false
                }
            };

            // PHASE 1: Mark all transitive polymorphic functions FIRST.
            // This must happen before recompilation so that when concrete callers
            // are recompiled and trigger monomorphization, the entire transitive
            // chain of polymorphic functions is already known.
            for (fn_def, module_path, imports, _line_starts, _source, source_name) in &fn_compile_info {
                let (fn_name, fn_key, fn_key_wildcard) = compute_fn_keys(fn_def, module_path, self);
                let is_fn_stdlib = source_name.contains("stdlib/") || source_name.starts_with("stdlib");

                if is_fn_stdlib || self.polymorphic_fns.contains(&fn_key) || self.polymorphic_fns.contains(&fn_key_wildcard) {
                    continue;
                }

                if !calls_any_poly(&fn_key, &fn_key_wildcard, self) {
                    continue;
                }

                let has_untyped_params = fn_def.clauses.first()
                    .map(|c| c.params.iter().any(|p| p.ty.is_none()))
                    .unwrap_or(false);

                if has_untyped_params {
                    self.polymorphic_fns.insert(fn_key.clone());
                    self.polymorphic_fns.insert(fn_key_wildcard.clone());
                    // Compute HM-inferred signature for the now-polymorphic function
                    let saved_path2 = std::mem::replace(&mut self.module_path, module_path.clone());
                    let saved_imports2 = self.imports.clone();
                    self.imports.extend(imports.clone());
                    let sig = self.infer_signature(fn_def);
                    // Populate fn_type_params from HM-inferred trait bounds
                    let lookup_key = if self.functions.contains_key(&fn_key) { &fn_key } else { &fn_key_wildcard };
                    if sig.contains("=>") && !self.fn_type_params.contains_key(lookup_key) {
                        if let Some(fn_type) = self.parse_signature_string(&sig) {
                            if !fn_type.type_params.is_empty() {
                                let type_params: Vec<TypeParam> = fn_type.type_params.iter()
                                    .map(|tp| TypeParam {
                                        name: Spanned { node: tp.name.clone(), span: Span::default() },
                                        constraints: tp.constraints.iter()
                                            .map(|c| Spanned { node: c.clone(), span: Span::default() })
                                            .collect(),
                                    })
                                    .collect();
                                self.fn_type_params.insert(lookup_key.clone(), type_params);
                            }
                        }
                    }
                    if let Some(fn_val) = self.functions.get(lookup_key) {
                        let mut new_fn_val = (**fn_val).clone();
                        new_fn_val.signature = Some(sig);
                        self.functions.insert(lookup_key.clone(), Arc::new(new_fn_val));
                    }
                    self.module_path = saved_path2;
                    self.imports = saved_imports2;
                }
            }

            // PHASE 2: Recompile concrete callers (functions with all typed params
            // that call a newly-polymorphic function). Now that all transitive
            // polymorphic functions are marked, monomorphization chains will propagate fully.
            // Re-check against the FULL current poly set (including Phase 1 additions from this iter).
            let all_poly_bases: Vec<String> = self.polymorphic_fns.difference(&poly_fns_before).map(|n| n.split('/').next().unwrap_or(n).to_string()).collect();

            // First pass: identify which fn_compile_info indices need recompilation.
            // We check both direct calls AND embedded lambda/closure constants recursively,
            // since lambdas like `s => doubleFromStr(s)` are compiled as FunctionValue
            // constants inside the parent function's chunk.
            let mut recompile_indices: Vec<usize> = Vec::new();
            for (idx, (fn_def, module_path, _imports, _line_starts, _source, source_name)) in fn_compile_info.iter().enumerate() {
                let (_fn_name, fn_key, fn_key_wildcard) = compute_fn_keys(fn_def, module_path, self);
                let is_fn_stdlib = source_name.contains("stdlib/") || source_name.starts_with("stdlib");

                if is_fn_stdlib || self.polymorphic_fns.contains(&fn_key) || self.polymorphic_fns.contains(&fn_key_wildcard) {
                    continue;
                }

                // Check if this function (including embedded lambda constants) calls any poly function
                let calls_poly = if let Some(func_val) = self.functions.get(&fn_key).or_else(|| self.functions.get(&fn_key_wildcard)) {
                    let mut chunks_to_check: Vec<&Chunk> = vec![&func_val.code];
                    let mut found = false;
                    while let Some(chunk) = chunks_to_check.pop() {
                        for instr in &chunk.code {
                            let func_idx = match instr {
                                Instruction::CallDirect(_, idx, _) => Some(*idx),
                                Instruction::TailCallDirect(idx, _) => Some(*idx),
                                _ => None,
                            };
                            if let Some(fidx) = func_idx {
                                if let Some(called_name) = self.function_list.get(fidx as usize) {
                                    let called_base = called_name.split('/').next().unwrap_or(called_name);
                                    if all_poly_bases.iter().any(|b| called_base == b) {
                                        found = true;
                                        break;
                                    }
                                }
                            }
                        }
                        if found { break; }
                        // Also check embedded lambda/closure functions in constants
                        for constant in &chunk.constants {
                            if let Value::Function(fv) = constant {
                                chunks_to_check.push(&fv.code);
                            }
                        }
                    }
                    found
                } else {
                    false
                };

                if !calls_poly {
                    continue;
                }
                recompile_indices.push(idx);
            }

            // Second pass: recompile identified functions
            for idx in recompile_indices {
                let (fn_def, module_path, _imports, line_starts, source, source_name) = &fn_compile_info[idx];
                let (fn_name, _fn_key, _fn_key_wildcard) = compute_fn_keys(fn_def, module_path, self);

                let saved_path = std::mem::replace(&mut self.module_path, module_path.clone());
                let saved_imports = self.imports.clone();
                self.imports.extend(_imports.clone());
                let saved_line_starts = std::mem::replace(&mut self.line_starts, line_starts.clone());
                let saved_source = std::mem::replace(&mut self.current_source, Some(source.clone()));
                let saved_source_name = std::mem::replace(&mut self.current_source_name, Some(source_name.clone()));

                // Recompile to replace CallDirect/TailCallDirect with monomorphized variant calls.
                if let Err(e) = self.compile_fn_def(fn_def) {
                    if matches!(e, CompileError::TypeError { .. }
                        | CompileError::TraitBoundNotSatisfied { .. }
                        | CompileError::UnresolvedTraitMethod { .. }) {
                        errors.push((fn_name.clone(), e, source_name.clone(), source.clone()));
                    }
                }

                self.module_path = saved_path;
                self.imports = saved_imports;
                self.line_starts = saved_line_starts;
                self.current_source = saved_source;
                self.current_source_name = saved_source_name;
            }
        }

        // Additional recompilation pass: retry functions that FAILED during the first pass
        // if their error involves any of the newly-polymorphic functions. This handles cases
        // where main() calls a newly-polymorphic function and monomorphization failed because
        // the function's code wasn't available yet (e.g., lambdas calling polymorphic functions).
        let all_new_poly_fns: Vec<String> = self.polymorphic_fns.difference(&poly_fns_before).cloned().collect();
        if !all_new_poly_fns.is_empty() && !errors.is_empty() {
            let new_poly_bases: Vec<String> = all_new_poly_fns.iter()
                .map(|n| n.split('/').next().unwrap_or(n).to_string())
                .collect();
            let error_fn_names: Vec<String> = errors.iter()
                .filter(|(_, e, _, _)| {
                    // Check if the error is UnresolvedTraitMethod or related to a poly function
                    if let CompileError::UnresolvedTraitMethod { method, .. } = e {
                        let method_base = method.split('/').next().unwrap_or(method);
                        new_poly_bases.iter().any(|b| method_base == b)
                    } else {
                        false
                    }
                })
                .map(|(name, _, _, _)| name.clone())
                .collect();

            for fn_name in &error_fn_names {
                // Find the function in fn_compile_info
                for (fn_def, module_path, imports, line_starts, source, source_name) in &fn_compile_info {
                    let compiled_name = if module_path.is_empty() {
                        fn_def.name.node.clone()
                    } else {
                        format!("{}.{}", module_path.join("."), fn_def.name.node)
                    };
                    if &compiled_name != fn_name {
                        continue;
                    }

                    let saved_path = std::mem::replace(&mut self.module_path, module_path.clone());
                    let saved_imports = self.imports.clone();
                    self.imports.extend(imports.clone());
                    let saved_line_starts = std::mem::replace(&mut self.line_starts, line_starts.clone());
                    let saved_source = std::mem::replace(&mut self.current_source, Some(source.clone()));
                    let saved_source_name = std::mem::replace(&mut self.current_source_name, Some(source_name.clone()));

                    if self.compile_fn_def(fn_def).is_ok() {
                        // Successfully recompiled - remove from errors
                        errors.retain(|(name, _, _, _)| name != fn_name);
                    }

                    self.module_path = saved_path;
                    self.imports = saved_imports;
                    self.line_starts = saved_line_starts;
                    self.current_source = saved_source;
                    self.current_source_name = saved_source_name;
                    break;
                }
            }
        }

        // General retry pass: some UnresolvedTraitMethod errors are caused by compilation
        // order, not by actual bugs. For example, when main.nos imports ops.nos and calls
        // ops.myMapper(words, w => w.reverse()), the lambda parameter type can't be resolved
        // if ops.myMapper hasn't been compiled yet (its signature is still None). Now that all
        // functions have been compiled and have their HM-inferred signatures, retry to see if
        // the error resolves. This is more general than the poly-fn-specific retry above.
        {
            let retry_fn_names: Vec<String> = errors.iter()
                .filter(|(_, e, _, _)| matches!(e, CompileError::UnresolvedTraitMethod { .. }))
                .map(|(name, _, _, _)| name.clone())
                .collect();
            for fn_name in &retry_fn_names {
                for (fn_def, module_path, imports, line_starts, source, source_name) in &fn_compile_info {
                    let compiled_name = if module_path.is_empty() {
                        fn_def.name.node.clone()
                    } else {
                        format!("{}.{}", module_path.join("."), fn_def.name.node)
                    };
                    if &compiled_name != fn_name {
                        continue;
                    }

                    let saved_path = std::mem::replace(&mut self.module_path, module_path.clone());
                    let saved_imports = self.imports.clone();
                    self.imports.extend(imports.clone());
                    let saved_line_starts = std::mem::replace(&mut self.line_starts, line_starts.clone());
                    let saved_source = std::mem::replace(&mut self.current_source, Some(source.clone()));
                    let saved_source_name = std::mem::replace(&mut self.current_source_name, Some(source_name.clone()));

                    if self.compile_fn_def(fn_def).is_ok() {
                        errors.retain(|(name, _, _, _)| name != fn_name);
                    }

                    self.module_path = saved_path;
                    self.imports = saved_imports;
                    self.line_starts = saved_line_starts;
                    self.current_source = saved_source;
                    self.current_source_name = saved_source_name;
                    break;
                }
            }
        }

        // Stranded polymorphic function pass: functions that were marked polymorphic
        // (UnresolvedTraitMethod during compilation) but have ZERO concrete monomorphized
        // variants. This can happen when a function like total(p) = p.price * p.qty.toFloat()
        // is only referenced first-class (passed to .map()) and never called directly.
        // These functions are never monomorphized, so they remain as empty-code placeholders
        // and cause "Instruction pointer out of bounds" at runtime.
        //
        // After all compilation passes, pending_fn_signatures may have been updated with
        // concrete types (e.g., model.total's param resolved to model.Product via HM inference).
        // Re-run compile_fn_def for stranded functions with their updated signature, which
        // will now have has_generic_hm_signature=false and compile successfully.
        {
            let stranded: Vec<String> = self.polymorphic_fns.iter()
                .filter(|fn_name| {
                    // Check if this function has empty code (placeholder)
                    let has_empty_code = self.functions.get(*fn_name)
                        .map(|f| f.code.code.is_empty())
                        .unwrap_or(false);
                    if !has_empty_code {
                        return false;
                    }
                    // Check if no concrete variants exist
                    let fn_base = fn_name.split('/').next().unwrap_or(fn_name);
                    let has_variants = self.function_variants.get(fn_base)
                        .map(|s| !s.is_empty())
                        .unwrap_or(false);
                    !has_variants
                })
                .cloned()
                .collect();

            for fn_name in &stranded {
                // Find in fn_compile_info by base name
                let fn_base = fn_name.split('/').next().unwrap_or(fn_name);
                // Get last component (the actual function name without module prefix)
                let fn_local = fn_base.rsplit('.').next().unwrap_or(fn_base);

                for (fn_def, module_path, imports, line_starts, source, source_name) in &fn_compile_info {
                    let compiled_name = if module_path.is_empty() {
                        fn_def.name.node.clone()
                    } else {
                        format!("{}.{}", module_path.join("."), fn_def.name.node)
                    };
                    if fn_def.name.node != fn_local && compiled_name != fn_base {
                        continue;
                    }
                    // Also verify the module path matches
                    let expected_module = fn_base.trim_end_matches(&format!(".{}", fn_local));
                    let actual_module = module_path.join(".");
                    if expected_module != actual_module && !fn_base.is_empty() && fn_base.contains('.') && expected_module != fn_base {
                        continue;
                    }

                    let saved_path = std::mem::replace(&mut self.module_path, module_path.clone());
                    let saved_imports = self.imports.clone();
                    self.imports.extend(imports.clone());
                    let saved_line_starts = std::mem::replace(&mut self.line_starts, line_starts.clone());
                    let saved_source = std::mem::replace(&mut self.current_source, Some(source.clone()));
                    let saved_source_name = std::mem::replace(&mut self.current_source_name, Some(source_name.clone()));

                    // Set the stranded flag so the compiler uses best-effort dispatch
                    // (e.g., generic numeric toFloat instead of triggering monomorphization)
                    let saved_stranded = self.compiling_stranded_fn;
                    self.compiling_stranded_fn = true;

                    // Try recompilation - if it succeeds, the function now has real code
                    let _ = self.compile_fn_def(fn_def);

                    self.compiling_stranded_fn = saved_stranded;

                    self.module_path = saved_path;
                    self.imports = saved_imports;
                    self.line_starts = saved_line_starts;
                    self.current_source = saved_source;
                    self.current_source_name = saved_source_name;
                    break;
                }
            }
        }

        // Clear pending_functions now that we've processed them
        // Note: pending_fn_signatures is kept until after the third pass type-check
        self.pending_functions.clear();

        // Collect pending function names for the third pass (needed for filtering)
        let pending_fn_names: std::collections::HashSet<String> = pending_fn_info
            .iter()
            .map(|(fn_name, _, _, _, _)| fn_name.clone())
            .collect();

        // Validate parameter type annotations for functions compiled in THIS pass
        // This catches undefined types like `greet(p: Person)` where Person doesn't exist
        // Skip stdlib files - they're tested separately and have complex inter-dependencies
        for (fn_name, fn_def, imports, source_name, source) in pending_fn_info {
            // Skip validation for stdlib files
            if source_name.contains("stdlib/") || source_name.starts_with("stdlib") {
                continue;
            }

            // Temporarily set imports for this function so validate_type_expr can check imported types
            let saved_imports = std::mem::replace(&mut self.imports, imports);

            // Get the module path from the function name (e.g., "stdlib.rhtml.wrapNode" -> ["stdlib", "rhtml"])
            let module_path: Vec<String> = fn_name.split('.')
                .take(fn_name.matches('.').count())  // All but the last segment
                .map(|s| s.to_string())
                .collect();
            let saved_path = std::mem::replace(&mut self.module_path, module_path);

            // Build set of explicitly declared type params (e.g., [El, Key, Val] from fn[El, Key, Val])
            let fn_type_param_names: std::collections::HashSet<String> = fn_def.type_params
                .iter()
                .map(|tp| tp.name.node.clone())
                .collect();

            if let Some(clause) = fn_def.clauses.first() {
                for param in &clause.params {
                    if let Some(ty_expr) = &param.ty {
                        if let Some(err) = self.validate_type_expr_with_type_params(ty_expr, &fn_name, &fn_type_param_names) {
                            errors.push((fn_name.clone(), err, source_name.clone(), source.clone()));
                        }
                    }
                }
                // Also check return type annotation
                if let Some(ret_ty) = &clause.return_type {
                    if let Some(err) = self.validate_type_expr_with_type_params(ret_ty, &fn_name, &fn_type_param_names) {
                        errors.push((fn_name.clone(), err, source_name.clone(), source.clone()));
                    }
                }
            }

            self.imports = saved_imports;
            self.module_path = saved_path;
        }

        // Second pass: re-run HM inference for functions with type variables in their signatures
        // This handles cases like bar23() = bar() + 1 where bar was compiled after bar23's first inference
        // Run multiple iterations until no more signatures change (handles dependency order issues)
        let fn_names: Vec<String> = self.functions.keys().cloned().collect();
        let max_iterations = 5; // Prevent infinite loops
        for _iteration in 0..max_iterations {
            let mut changed = false;
            for fn_name in &fn_names {
                // Skip stdlib functions - they have proper type annotations
                if fn_name.starts_with("stdlib.") {
                    continue;
                }
                // Clone just the signature string (cheap) to release borrow early
                let sig_to_check = self.functions.get(fn_name)
                    .and_then(|fv| fv.signature.clone());

                if let Some(sig) = sig_to_check {
                    // Check if signature has unresolved compiler-generated type variables
                    // (?N format, e.g., ?1, ?23). Only these indicate genuinely unresolved types.
                    // DO NOT match standalone lowercase letters (a, b, c) — these are TypeParam
                    // letters from the enrichment phase and represent intentional polymorphism.
                    // Re-inferring polymorphic functions is wasteful and causes massive slowdowns
                    // on large projects (80+ functions × 5 iterations = hundreds of unnecessary
                    // try_hm_inference calls, each creating a full inference context).
                    let has_type_var = sig.contains('?');

                    if has_type_var {
                        // Set module_path from function name for correct type resolution
                        let module_path: Vec<String> = fn_name.split('.')
                            .take(fn_name.matches('.').count())
                            .map(|s| s.to_string())
                            .collect();
                        let saved_path = std::mem::replace(&mut self.module_path, module_path);

                        // Restore the function's original imports so that names like "span"
                        // resolve to the correct module (e.g., stdlib.rhtml.span, not stdlib.list.span).
                        // Without this, self.imports reflects whichever module was compiled last,
                        // which may have different aliases than the function's defining module.
                        let saved_imports = self.imports.clone();
                        if let Some(fn_imports) = self.fn_ast_imports.get(fn_name).cloned() {
                            self.imports = fn_imports;
                        }

                        // Try HM inference again now that all dependencies are compiled
                        // Clone the fn_ast to avoid borrow conflict with &mut self in try_hm_inference
                        let inferred = self.fn_asts.get(fn_name).cloned()
                            .and_then(|fn_ast| self.try_hm_inference(&fn_ast).0);

                        if let Some((inferred_sig, _expr_types)) = inferred {
                            // Check if signature actually changed
                            if inferred_sig != sig {
                                // Update the function's signature - need to clone and replace
                                // since Arc::get_mut won't work if there are other references
                                if let Some(fn_val) = self.functions.get(fn_name) {
                                    let mut new_fn_val = (**fn_val).clone();
                                    new_fn_val.signature = Some(inferred_sig);
                                    self.functions.insert(fn_name.clone(), Arc::new(new_fn_val));
                                    changed = true;
                                }
                            }
                        }

                        self.imports = saved_imports;
                        self.module_path = saved_path;
                    }
                }
            }
            if !changed {
                break;
            }
        }

        // NOTE: No pre-pass needed here. The Second Pass (above) already re-runs HM inference
        // for functions with type variables and updates self.functions. type_check_fn (below)
        // prefers self.functions entries over pending_fn_signatures for functions that have
        // inferred signatures. So validate's "Int -> Int" signature from self.functions will
        // be used by type_check_fn when checking main.

        // Third pass: Type check functions that were just compiled (in pending_fn_info)
        // This catches errors like bar() = bar2() + "x" where bar2() returns ()
        // IMPORTANT: Only check functions from this compile pass, not ALL functions.
        // Otherwise, old broken functions (that haven't been recompiled yet) would
        // cause errors even when compiling unrelated functions.
        // Note: pending_fn_names was collected earlier, before the validation loop
        // Collect resolved expression types from per-function inference to supplement
        // the batch inference results. The batch inference uses separate type variables
        // for each function's definition, so lambda params in HOF calls may be unresolved.
        // The per-function pass properly resolves them through call-site instantiation.
        let mut additional_expr_types: Vec<(Span, nostos_types::Type)> = Vec::new();
        for (fn_name, fn_ast) in &self.fn_asts {
            // Only type-check functions that were just compiled in this pass
            // Use base name without signature for comparison
            let base_name = fn_name.split('/').next().unwrap_or(fn_name);
            if !pending_fn_names.contains(base_name) && !pending_fn_names.contains(fn_name) {
                continue;
            }

            // Skip REPL wrappers - these are temporary functions that may contain errors
            // from previous REPL inputs. We don't want old errors to affect new inputs.
            // Also skip var thunks which have local bindings injected that HM inference doesn't see.
            if fn_name.starts_with("__repl_eval_") || fn_name.starts_with("__repl_var_") || fn_name.starts_with("__repl_tuple_") {
                continue;
            }

            // Skip stdlib functions - they have complex inter-dependencies that cause
            // type inference to be slow. Stdlib is tested separately.
            if fn_name.starts_with("stdlib.") ||
               fn_name.starts_with("List.") || fn_name.starts_with("String.") ||
               fn_name.starts_with("Math.") || fn_name.starts_with("Map.") ||
               fn_name.starts_with("Set.") || fn_name.starts_with("Json.") {
                continue;
            }

            // Skip monomorphized variants (contain $) - they're derived from
            // already-validated generic functions; HM inference can produce false
            // positives on specialized function types.
            if fn_name.contains('$') {
                continue;
            }

            // Set module_path from function name for correct type resolution
            let fn_base = fn_name.split('/').next().unwrap_or(fn_name);
            let module_path: Vec<String> = fn_base.split('.')
                .take(fn_base.matches('.').count())
                .map(|s| s.to_string())
                .collect();
            let saved_path = std::mem::replace(&mut self.module_path, module_path);

            // Pre-process the function AST to apply arity-overload rewriting.
            // This mirrors the preprocessing done in compile_fn_def before type_check_fn.
            // Without this, functions with different-arity local function overloads (like
            // `combine(x) = x*2` and `combine(x,y) = x+y`) would fail here with
            // "Wrong number of arguments" because fn_asts stores the original unprocessed AST.
            let preprocessed_fn_ast;
            let fn_ast_for_check = if let Some(p) = Self::preprocess_fn_def_for_arity_overloads(fn_ast) {
                preprocessed_fn_ast = p;
                &preprocessed_fn_ast
            } else {
                fn_ast
            };

            // Run type checking with full knowledge of all function signatures
            match self.type_check_fn(fn_ast_for_check, fn_name) {
            Ok(resolved_types) => {
                // Collect resolved types from per-function inference
                additional_expr_types.extend(resolved_types);
            }
            Err(e) => {
                // Report type errors - only filter truly spurious ones from inference limitations
                let should_report = match &e {
                    // All type error filtering now happens in type_check_fn via
                    // TypeError::should_suppress() and should_suppress_field_error(),
                    // which operate on structured Type values instead of parsing
                    // serialized error message strings.
                    CompileError::TypeError { .. } => true,
                    // UnresolvedTraitMethod from type_check_fn is NOT a hard error.
                    // It happens when a function calls methods on polymorphic parameters
                    // (e.g., `revId(x) = { y = id(x); y.reverse() }`). The actual body
                    // compilation handles this by marking the function as polymorphic
                    // and triggering monomorphization at call sites.
                    CompileError::UnresolvedTraitMethod { .. } => false,
                    _ => true,
                };
                if should_report {
                    // Use base function name without signature for error reporting
                    let base_name = fn_name.split('/').next().unwrap_or(fn_name).to_string();
                    // Get source info from fn_sources for proper error reporting
                    let (source_name, source) = self.fn_sources.get(fn_name)
                        .cloned()
                        .unwrap_or_else(|| ("unknown".to_string(), Arc::new(String::new())));
                    errors.push((base_name, e, source_name, source));
                }
            } // end Err(e)
            } // end match

            self.module_path = saved_path;
        }

        // Report structural type errors from try_hm_inference.
        {
            let mut reported_fns: HashSet<String> = HashSet::new();
            for (fn_name, type_err) in std::mem::take(&mut self.hm_inference_errors) {
                // Skip errors where top-level types are unresolved (noise from batch inference).
                // But keep errors where top-level constructors differ (e.g., List vs Function)
                // even if inner types have unresolved vars.
                let suppress = match &type_err {
                    nostos_types::TypeError::StructuralMismatch(t1, t2) |
                    nostos_types::TypeError::UnificationFailed(t1, t2) => {
                        fn is_top_level_unresolved(ty: &nostos_types::Type) -> bool {
                            match ty {
                                nostos_types::Type::Var(_) | nostos_types::Type::TypeParam(_) => true,
                                nostos_types::Type::Named { name, .. } => name.starts_with('?'),
                                _ => false,
                            }
                        }
                        // Also suppress when both types share the same top-level constructor
                        // but one has unresolved TypeParams inside (e.g., List[TypeParam("a")] vs List[Int]).
                        // Keep errors where constructors differ (e.g., List vs Int) as those are real.
                        fn same_top_constructor(t1: &nostos_types::Type, t2: &nostos_types::Type) -> bool {
                            use nostos_types::Type;
                            matches!((t1, t2),
                                (Type::List(_), Type::List(_)) |
                                (Type::Set(_), Type::Set(_)) |
                                (Type::Map(_, _), Type::Map(_, _)) |
                                (Type::Array(_), Type::Array(_)) |
                                (Type::IO(_), Type::IO(_)) |
                                (Type::Tuple(_), Type::Tuple(_))
                            ) || matches!((t1, t2), (Type::Named { name: n1, .. }, Type::Named { name: n2, .. }) if n1 == n2)
                        }
                        is_top_level_unresolved(t1) || is_top_level_unresolved(t2)
                            || (same_top_constructor(t1, t2) && (t1.has_any_type_var() || t2.has_any_type_var()))
                    }
                    _ => type_err.should_suppress(),
                };
                if suppress { continue; }
                // For monomorphized variants ($), suppress numeric coercion errors.
                // When a polymorphic function like `doubleAll(xs) = xs.map(x => x * 2)`
                // is monomorphized with Float, HM sees `Float * Int` and errors.
                // The VM handles mixed Float/Int arithmetic at runtime, so suppress.
                if fn_name.contains('$') {
                    let is_numeric_coercion = match &type_err {
                        nostos_types::TypeError::UnificationFailed(t1, t2) |
                        nostos_types::TypeError::StructuralMismatch(t1, t2) => {
                            fn is_numeric(t: &nostos_types::Type) -> bool {
                                matches!(t, nostos_types::Type::Int | nostos_types::Type::Int64 |
                                    nostos_types::Type::Float | nostos_types::Type::Float64 |
                                    nostos_types::Type::Float32 | nostos_types::Type::BigInt |
                                    nostos_types::Type::Decimal)
                            }
                            is_numeric(t1) && is_numeric(t2)
                        }
                        nostos_types::TypeError::Mismatch { expected, found } => {
                            fn is_numeric_str(s: &str) -> bool {
                                matches!(s, "Int" | "Int64" | "Float" | "Float64" |
                                    "Float32" | "BigInt" | "Decimal")
                            }
                            is_numeric_str(expected) && is_numeric_str(found)
                        }
                        _ => false,
                    };
                    if is_numeric_coercion { continue; }
                }
                let base_name = fn_name.split('/').next().unwrap_or(&fn_name).to_string();
                if !reported_fns.insert(base_name.clone()) { continue; }
                use super::clean_type_vars;
                let error_message = match &type_err {
                    nostos_types::TypeError::StructuralMismatch(a, b) =>
                        format!("type mismatch: expected `{}`, found `{}`",
                            clean_type_vars(&a.display()), clean_type_vars(&b.display())),
                    nostos_types::TypeError::UnificationFailed(a, b) =>
                        format!("type mismatch: expected `{}`, found `{}`",
                            clean_type_vars(&a.display()), clean_type_vars(&b.display())),
                    nostos_types::TypeError::Mismatch { expected: a, found: b } =>
                        format!("type mismatch: expected `{}`, found `{}`", clean_type_vars(a), clean_type_vars(b)),
                    nostos_types::TypeError::MissingTraitImpl { ty, trait_name, .. } =>
                        format!("Type error: {} does not implement {}",
                            clean_type_vars(ty), clean_type_vars(trait_name)),
                    other => format!("{:?}", other),
                };
                let fn_key = self.fn_sources.keys()
                    .find(|k| k.starts_with(&fn_name) || k.split('/').next() == Some(&fn_name))
                    .cloned();
                if let Some(key) = fn_key {
                    if let Some((source_name, source)) = self.fn_sources.get(&key).cloned() {
                        if let Some(fn_ast) = self.fn_asts.get(&key) {
                            let span = fn_ast.name.span;
                            let compile_error = CompileError::TypeError { message: error_message, span };
                            errors.push((fn_name.clone(), compile_error, source_name, source));
                        }
                    }
                }
            }
        }

        // Merge resolved expression types from per-function inference.
        // Overwrite entries that are still unresolved (contain Var or TypeParam) from batch inference.
        for (span, ty) in additional_expr_types {
            let should_overwrite = self.inferred_expr_types.get(&span)
                .map(|existing| {
                    // Overwrite if existing entry contains ANY unresolved type variables
                    fn has_unresolved(ty: &nostos_types::Type) -> bool {
                        match ty {
                            nostos_types::Type::Var(_) | nostos_types::Type::TypeParam(_) => true,
                            nostos_types::Type::List(inner) | nostos_types::Type::Set(inner) |
                            nostos_types::Type::IO(inner) | nostos_types::Type::Array(inner) => has_unresolved(inner),
                            nostos_types::Type::Map(k, v) => has_unresolved(k) || has_unresolved(v),
                            nostos_types::Type::Tuple(elems) => elems.iter().any(has_unresolved),
                            nostos_types::Type::Named { args, .. } => args.iter().any(has_unresolved),
                            nostos_types::Type::Function(ft) => ft.params.iter().any(has_unresolved) || has_unresolved(&ft.ret),
                            _ => false,
                        }
                    }
                    has_unresolved(existing)
                })
                .unwrap_or(true); // Also add if not present at all
            if should_overwrite {
                self.inferred_expr_types.insert(span, ty);
            }
        }

        // NOTE: Don't clear pending_fn_signatures yet - the REPL needs them to get variable types
        // after compilation via get_function_return_type_hm()
        // self.pending_fn_signatures.clear();

        // Check for mvar safety violations (prevents runtime deadlocks)
        let mvar_errors = self.check_mvar_deadlocks();
        for msg in mvar_errors {
            // Extract function name from error message if possible
            let fn_name = msg.split('`').nth(1).unwrap_or("unknown").to_string();
            let (source_name, source) = self.fn_sources.iter()
                .find(|(k, _)| k.contains(&fn_name))
                .map(|(_, v)| v.clone())
                .unwrap_or_else(|| ("unknown".to_string(), Arc::new(String::new())));
            errors.push((
                fn_name,
                CompileError::MvarSafetyViolation {
                    message: msg,
                    span: Span::default()
                },
                source_name,
                source,
            ));
        }


        errors
    }

    /// Compile a module and add it to the current compilation context.
    pub fn add_module(&mut self, module: &Module, module_path: Vec<String>, source: Arc<String>, source_name: String) -> Result<(), CompileError> {
        // Update line_starts for this file
        self.line_starts = vec![0];
        for (i, c) in source.char_indices() {
            if c == '\n' {
                self.line_starts.push(i + 1);
            }
        }

        self.current_source = Some(source);
        self.current_source_name = Some(source_name.clone());

        // Register this module path and all its prefixes as known modules
        // e.g., for ["math", "vector"], register both "math" and "math.vector"
        if !module_path.is_empty() {
            let mut prefix = String::new();
            for component in &module_path {
                if !prefix.is_empty() {
                    prefix.push('.');
                }
                prefix.push_str(component);
                self.known_modules.insert(prefix.clone());
            }
        }

        // Set module path
        self.module_path = module_path;

        // Save imports so that `use` statements inside this file-based module don't leak
        // to other modules (prevents transitive import leaking across files).
        // This mirrors what compile_module_def() does for inline modules.
        let saved_imports = self.imports.clone();
        let saved_import_sources = self.import_sources.clone();

        // Compile items
        self.compile_items(&module.items)?;

        // Save per-module imports for type_check_fn in compile_all_collecting_errors.
        // For top-level files (empty module_path), save under empty-string key.
        // For named modules, save under module name (e.g., "mymod").
        // type_check_fn uses this to set up function_aliases when re-type-checking functions.
        {
            let module_key = if self.module_path.is_empty() {
                String::new()
            } else {
                self.module_path.join(".")
            };
            let fn_imports: std::collections::HashMap<String, String> = self.imports.iter()
                .filter(|(_, v)| {
                    // Function imports have qualified names where last segment starts lowercase
                    v.split('.').last()
                        .map(|s| s.chars().next().map(|c| c.is_ascii_lowercase()).unwrap_or(false))
                        .unwrap_or(false)
                })
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect();
            if !fn_imports.is_empty() {
                self.module_imports.insert(module_key, fn_imports);
            }
        }

        // Restore imports to prevent transitive leaking between file-based modules
        self.imports = saved_imports;
        self.import_sources = saved_import_sources;

        // Reset module path
        self.module_path = Vec::new();

        Ok(())
    }

    /// Add module metadata (types, traits) without compiling functions.
    /// Used when loading functions from bytecode cache.
    pub fn add_module_metadata_only(&mut self, module: &Module, module_path: Vec<String>, source: Arc<String>, source_name: String) -> Result<(), CompileError> {
        // Update line_starts for this file
        self.line_starts = vec![0];
        for (i, c) in source.char_indices() {
            if c == '\n' {
                self.line_starts.push(i + 1);
            }
        }

        self.current_source = Some(source);
        self.current_source_name = Some(source_name.clone());

        // Register this module path and all its prefixes as known modules
        if !module_path.is_empty() {
            let mut prefix = String::new();
            for component in &module_path {
                if !prefix.is_empty() {
                    prefix.push('.');
                }
                prefix.push_str(component);
                self.known_modules.insert(prefix.clone());
            }
        }

        // Set module path
        self.module_path = module_path;

        // Save imports so that `use` statements inside this file-based module don't leak
        // to other modules (prevents transitive import leaking across files).
        let saved_imports = self.imports.clone();
        let saved_import_sources = self.import_sources.clone();

        // Compile only metadata items (no functions)
        self.compile_items_metadata_only(&module.items)?;

        // Restore imports to prevent transitive leaking between file-based modules
        self.imports = saved_imports;
        self.import_sources = saved_import_sources;

        // Reset module path
        self.module_path = Vec::new();

        Ok(())
    }

    /// Register module exports (function/type names) without compiling.
    /// Used for two-pass stdlib compilation to ensure all exports are known
    /// before any use statements are processed.
    pub fn register_module_forward_declarations(&mut self, module: &Module, module_path: Vec<String>) -> Result<(), CompileError> {
        use nostos_syntax::ast::{Item, Visibility};

        // Register this module path as known
        if !module_path.is_empty() {
            let mut prefix = String::new();
            for component in &module_path {
                if !prefix.is_empty() {
                    prefix.push('.');
                }
                prefix.push_str(component);
                self.known_modules.insert(prefix.clone());
            }
        }

        // Set module path for qualified name generation
        let old_module_path = std::mem::replace(&mut self.module_path, module_path);

        // Register all public function names
        for item in &module.items {
            if let Item::FnDef(fn_def) = item {
                if !fn_def.is_template {
                    let qualified_name = self.qualify_name(&fn_def.name.node);
                    // Register visibility so use statements can find this function.
                    // For multi-clause functions (e.g., pub f(A) = ...; f(B) = ...),
                    // only the first clause may have `pub`. Never downgrade from Public.
                    let existing = self.function_visibility.get(&qualified_name);
                    if !matches!(existing, Some(Visibility::Public)) {
                        self.function_visibility.insert(qualified_name, fn_def.visibility.clone());
                    }
                }
            }
            // Also register public type names
            if let Item::TypeDef(type_def) = item {
                let qualified_name = self.qualify_name(&type_def.name.node);
                // Type visibility is tracked via type_defs, which are registered during actual compilation
                // But we need the name known for imports
                if matches!(type_def.visibility, Visibility::Public) {
                    self.function_visibility.insert(qualified_name, type_def.visibility.clone());
                }
            }
            // Register public trait definitions so cross-file `use module.*` can find them
            if let Item::TraitDef(trait_def) = item {
                if matches!(trait_def.visibility, Visibility::Public) {
                    let qualified_name = self.qualify_name(&trait_def.name.node);
                    let super_traits: Vec<String> = trait_def.super_traits
                        .iter()
                        .map(|t| t.node.clone())
                        .collect();
                    let methods: Vec<TraitMethodInfo> = trait_def.methods
                        .iter()
                        .map(|m| {
                            let param_types: Vec<(String, String)> = m.params.iter()
                                .map(|p| {
                                    let pname = self.pattern_binding_name(&p.pattern)
                                        .unwrap_or_else(|| "_".to_string());
                                    let ptype = p.ty.as_ref()
                                        .map(|ty| {
                                            let s = self.type_expr_to_string(ty);
                                            // Normalize lowercase "self" to "Self" so downstream
                                            // .replace("Self", impl_type) works for params like `other: self`
                                            if s == "self" { "Self".to_string() } else { s }
                                        })
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
                                required_params: {
                                    let req = m.params.iter().filter(|p| p.default.is_none()).count();
                                    if req < m.params.len() { Some(req) } else { None }
                                },
                            }
                        })
                        .collect();
                    self.trait_defs.insert(qualified_name.clone(), TraitInfo {
                        name: qualified_name,
                        visibility: trait_def.visibility,
                        super_traits,
                        methods,
                    });
                }
            }
            // Recurse into nested modules to register their functions, types, and traits
            if let Item::ModuleDef(module_def) = item {
                self.register_nested_module_forward_declarations(module_def);
            }
        }

        // Restore module path
        self.module_path = old_module_path;

        Ok(())
    }

    /// Recursively register forward declarations (functions, types, traits) from
    /// nested module definitions. Handles arbitrary depth of module nesting.
    pub(super) fn register_nested_module_forward_declarations(&mut self, module_def: &nostos_syntax::ast::ModuleDef) {
        use nostos_syntax::ast::{Item, Visibility};

        let saved = self.module_path.clone();
        self.module_path.push(module_def.name.node.clone());

        // Register nested module path as known
        let nested_path = self.module_path.join(".");
        self.known_modules.insert(nested_path);

        for inner_item in &module_def.items {
            if let Item::FnDef(fn_def) = inner_item {
                if !fn_def.is_template {
                    let qualified_name = self.qualify_name(&fn_def.name.node);
                    // For multi-clause functions, never downgrade from Public
                    let existing = self.function_visibility.get(&qualified_name);
                    if !matches!(existing, Some(Visibility::Public)) {
                        self.function_visibility.insert(qualified_name, fn_def.visibility.clone());
                    }
                }
            }
            if let Item::TypeDef(type_def) = inner_item {
                let qualified_name = self.qualify_name(&type_def.name.node);
                if matches!(type_def.visibility, Visibility::Public) {
                    self.function_visibility.insert(qualified_name, type_def.visibility.clone());
                }
            }
            if let Item::TraitDef(trait_def) = inner_item {
                if matches!(trait_def.visibility, Visibility::Public) {
                    let qualified_name = self.qualify_name(&trait_def.name.node);
                    let super_traits: Vec<String> = trait_def.super_traits
                        .iter()
                        .map(|t| t.node.clone())
                        .collect();
                    let methods: Vec<TraitMethodInfo> = trait_def.methods
                        .iter()
                        .map(|m| {
                            let param_types: Vec<(String, String)> = m.params.iter()
                                .map(|p| {
                                    let pname = self.pattern_binding_name(&p.pattern)
                                        .unwrap_or_else(|| "_".to_string());
                                    let ptype = p.ty.as_ref()
                                        .map(|ty| {
                                            let s = self.type_expr_to_string(ty);
                                            if s == "self" { "Self".to_string() } else { s }
                                        })
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
                                required_params: {
                                    let req = m.params.iter().filter(|p| p.default.is_none()).count();
                                    if req < m.params.len() { Some(req) } else { None }
                                },
                            }
                        })
                        .collect();
                    self.trait_defs.insert(qualified_name.clone(), TraitInfo {
                        name: qualified_name,
                        visibility: trait_def.visibility,
                        super_traits,
                        methods,
                    });
                }
            }
            // Recurse into deeper nested modules
            if let Item::ModuleDef(inner_module_def) = inner_item {
                self.register_nested_module_forward_declarations(inner_module_def);
            }
        }

        self.module_path = saved;
    }

    /// Pre-register type names and visibility from a module.
    /// Called BEFORE pre_register_module_metadata so that `use module.*` statements
    /// Pre-register top-level value bindings (e.g., `pub pi = 3.14159`) from a module
    /// so that `use module.pi` in other modules can find them regardless of compilation order.
    /// This must be called for ALL modules before any use-statement processing (Pass 1.5b).
    pub fn pre_register_module_bindings(&mut self, module: &Module, module_path: Vec<String>) {
        use nostos_syntax::ast::{Item, Pattern};

        let old_module_path = std::mem::replace(&mut self.module_path, module_path);

        for item in &module.items {
            if let Item::Binding(binding) = item {
                if let Pattern::Var(ident) = &binding.pattern {
                    let qualified_name = self.qualify_name(&ident.node);
                    if !self.top_level_bindings.contains_key(&qualified_name) {
                        self.top_level_bindings.insert(
                            qualified_name,
                            (binding.clone(), self.module_path.clone(), self.imports.clone()),
                        );
                    }
                }
            }
        }

        self.module_path = old_module_path;
    }

    /// in other modules can find types from this module via type_visibility.
    /// This is critical for cross-module type references: when canvas.nos has
    /// `use types.*` and `type DrawCmd = Draw(Shape, Color)`, the Color type from
    /// types.nos must be visible in type_visibility before canvas.nos processes
    /// its use statement.
    pub fn pre_register_module_type_names(&mut self, module: &Module, module_path: Vec<String>) {
        use nostos_syntax::ast::{Item, Visibility};

        let old_module_path = std::mem::replace(&mut self.module_path, module_path);

        for item in &module.items {
            if let Item::TypeDef(type_def) = item {
                let qualified_name = self.qualify_name(&type_def.name.node);
                // Register visibility so compile_use_stmt can find this type
                self.type_visibility.insert(qualified_name.clone(), type_def.visibility);
                // Register constructor names so they're recognized as constructors
                if let nostos_syntax::ast::TypeBody::Variant(variants) = &type_def.body {
                    for v in variants {
                        let qualified_ctor = self.qualify_name(&v.name.node);
                        self.known_constructors.insert(qualified_ctor);
                        self.known_constructors.insert(v.name.node.clone());
                    }
                } else if let nostos_syntax::ast::TypeBody::Record(_) = &type_def.body {
                    self.known_constructors.insert(qualified_name.clone());
                } else if let nostos_syntax::ast::TypeBody::Alias(target_ty) = &type_def.body {
                    // Detect self-referential alias: `type Foo = Foo`
                    // The parser treats single-constructor unit variants with the same name
                    // as the type as type aliases. Detect and register them as constructors.
                    let target_name_str = match target_ty {
                        nostos_syntax::TypeExpr::Name(ident) => ident.node.as_str(),
                        _ => "",
                    };
                    if target_name_str == type_def.name.node.as_str() {
                        // Self-referential alias = single unit-variant constructor
                        self.known_constructors.insert(qualified_name.clone());
                        self.known_constructors.insert(type_def.name.node.clone());
                    }
                }
                // Also register in function_visibility if public (for use stmt resolution)
                if matches!(type_def.visibility, Visibility::Public) {
                    self.function_visibility.insert(qualified_name, type_def.visibility);
                }
            }
            // Handle nested module types (recursively for arbitrary depth)
            if let Item::ModuleDef(module_def) = item {
                self.pre_register_nested_module_type_names(module_def);
            }
        }

        self.module_path = old_module_path;
    }

    /// Recursively register type names from nested module definitions.
    /// Handles arbitrary depth of module nesting.
    pub(super) fn pre_register_nested_module_type_names(&mut self, module_def: &nostos_syntax::ast::ModuleDef) {
        use nostos_syntax::ast::{Item, Visibility};

        let saved = self.module_path.clone();
        self.module_path.push(module_def.name.node.clone());

        for inner_item in &module_def.items {
            if let Item::TypeDef(type_def) = inner_item {
                let qualified_name = self.qualify_name(&type_def.name.node);
                self.type_visibility.insert(qualified_name.clone(), type_def.visibility);
                if let nostos_syntax::ast::TypeBody::Variant(variants) = &type_def.body {
                    for v in variants {
                        let qualified_ctor = self.qualify_name(&v.name.node);
                        self.known_constructors.insert(qualified_ctor);
                        self.known_constructors.insert(v.name.node.clone());
                    }
                } else if let nostos_syntax::ast::TypeBody::Record(_) = &type_def.body {
                    self.known_constructors.insert(qualified_name.clone());
                } else if let nostos_syntax::ast::TypeBody::Alias(target_ty) = &type_def.body {
                    // Detect self-referential alias: `type Foo = Foo`
                    let target_name_str = match target_ty {
                        nostos_syntax::TypeExpr::Name(ident) => ident.node.as_str(),
                        _ => "",
                    };
                    if target_name_str == type_def.name.node.as_str() {
                        self.known_constructors.insert(qualified_name.clone());
                        self.known_constructors.insert(type_def.name.node.clone());
                    }
                }
                if matches!(type_def.visibility, Visibility::Public) {
                    self.function_visibility.insert(qualified_name, type_def.visibility);
                }
            }
            // Recurse into deeper nested modules
            if let Item::ModuleDef(inner_module_def) = inner_item {
                self.pre_register_nested_module_type_names(inner_module_def);
            }
        }

        self.module_path = saved;
    }

    /// Compile all type definitions from a module, processing use statements first
    /// so imported types can be resolved. This populates self.type_defs which is
    /// needed by pre_register_trait_impl to determine if a type is generic.
    /// Must be called for ALL modules before pre_register_module_metadata to ensure
    /// cross-module type definitions are available for trait impl registration.
    pub fn compile_module_type_defs_only(&mut self, module: &Module, module_path: Vec<String>, source: std::sync::Arc<String>, source_name: String) -> Result<(), CompileError> {
        use nostos_syntax::ast::Item;

        // Update line_starts for error reporting
        self.line_starts = vec![0];
        for (i, c) in source.char_indices() {
            if c == '\n' {
                self.line_starts.push(i + 1);
            }
        }

        self.current_source = Some(source);
        self.current_source_name = Some(source_name);

        // Register module path
        if !module_path.is_empty() {
            let mut prefix = String::new();
            for component in &module_path {
                if !prefix.is_empty() {
                    prefix.push('.');
                }
                prefix.push_str(component);
                self.known_modules.insert(prefix.clone());
            }
        }

        let old_module_path = std::mem::replace(&mut self.module_path, module_path);
        let saved_imports = self.imports.clone();

        // Process use statements first (needed to resolve imported type names)
        for item in &module.items {
            if let Item::Use(use_stmt) = item {
                let _ = self.compile_use_stmt(use_stmt);
            }
        }

        // Register templates (needed for type decorators like @withTypeName)
        for item in &module.items {
            if let Item::FnDef(fn_def) = item {
                if fn_def.is_template {
                    let qualified_name = self.qualify_name(&fn_def.name.node);
                    self.templates.insert(fn_def.name.node.clone(), fn_def.clone());
                    self.templates.insert(qualified_name, fn_def.clone());
                }
            }
        }

        // Compile type definitions (idempotent - safe to call again later)
        for item in &module.items {
            if let Item::TypeDef(type_def) = item {
                self.compile_type_def(type_def)?;
            }
        }

        // Handle nested modules
        for item in &module.items {
            if let Item::ModuleDef(module_def) = item {
                self.compile_nested_module_type_defs_only(module_def)?;
            }
        }

        // Restore state - imports will be properly set up in pre_register_module_metadata
        self.imports = saved_imports;
        self.module_path = old_module_path;
        Ok(())
    }

    /// Recursively compile type definitions from nested module definitions.
    pub(super) fn compile_nested_module_type_defs_only(&mut self, module_def: &nostos_syntax::ast::ModuleDef) -> Result<(), CompileError> {
        use nostos_syntax::ast::Item;

        let saved = self.module_path.clone();
        self.module_path.push(module_def.name.node.clone());

        // Process use statements
        for item in &module_def.items {
            if let Item::Use(use_stmt) = item {
                let _ = self.compile_use_stmt(use_stmt);
            }
        }

        // Register templates (needed for type decorators)
        for item in &module_def.items {
            if let Item::FnDef(fn_def) = item {
                if fn_def.is_template {
                    let qualified_name = self.qualify_name(&fn_def.name.node);
                    self.templates.insert(fn_def.name.node.clone(), fn_def.clone());
                    self.templates.insert(qualified_name, fn_def.clone());
                }
            }
        }

        // Compile type definitions
        for item in &module_def.items {
            if let Item::TypeDef(type_def) = item {
                self.compile_type_def(type_def)?;
            }
        }

        // Recurse into nested modules
        for item in &module_def.items {
            if let Item::ModuleDef(inner_module_def) = item {
                self.compile_nested_module_type_defs_only(inner_module_def)?;
            }
        }

        self.module_path = saved;
        Ok(())
    }

    /// Pre-register module metadata: use statements, types, traits, and trait impls.
    /// This registers trait implementations (type_traits, trait_impls) without compiling
    /// method bodies. Used in multi-file projects to ensure all trait impls are visible
    /// across modules before any function bodies are compiled.
    pub fn pre_register_module_metadata(&mut self, module: &Module, module_path: Vec<String>, source: std::sync::Arc<String>, source_name: String) -> Result<(), CompileError> {
        use nostos_syntax::ast::Item;

        // Update line_starts for error reporting
        self.line_starts = vec![0];
        for (i, c) in source.char_indices() {
            if c == '\n' {
                self.line_starts.push(i + 1);
            }
        }

        self.current_source = Some(source);
        self.current_source_name = Some(source_name);

        // Register module path
        if !module_path.is_empty() {
            let mut prefix = String::new();
            for component in &module_path {
                if !prefix.is_empty() {
                    prefix.push('.');
                }
                prefix.push_str(component);
                self.known_modules.insert(prefix.clone());
            }
        }

        let old_module_path = std::mem::replace(&mut self.module_path, module_path);

        // Save imports so that `use` statements inside this file-based module don't leak
        // to other modules during metadata pre-registration.
        let saved_imports = self.imports.clone();
        let saved_import_sources = self.import_sources.clone();

        // Pre-pass: collect names of local (inline) modules
        let local_module_names: std::collections::HashSet<String> = module.items.iter()
            .filter_map(|item| {
                if let Item::ModuleDef(module_def) = item {
                    Some(module_def.name.node.clone())
                } else {
                    None
                }
            })
            .collect();

        // Pre-register top-level pub bindings so that `use module.*` in other modules
        // can find them. Only register the binding name and metadata; the full compilation
        // (type checking, inline code gen) happens later in compile_items.
        for item in &module.items {
            if let Item::Binding(binding) = item {
                if let Pattern::Var(ident) = &binding.pattern {
                    let qualified_name = self.qualify_name(&ident.node);
                    if !self.top_level_bindings.contains_key(&qualified_name) {
                        self.top_level_bindings.insert(
                            qualified_name,
                            (binding.clone(), self.module_path.clone(), self.imports.clone()),
                        );
                    }
                }
            }
        }

        // Process use statements (needed for import resolution in trait impls)
        let mut deferred_use_stmts: Vec<&nostos_syntax::ast::UseStmt> = Vec::new();
        for item in &module.items {
            if let Item::Use(use_stmt) = item {
                let module_path_str = use_stmt.path.first().map(|id| id.node.as_str()).unwrap_or("");
                if local_module_names.contains(module_path_str) {
                    deferred_use_stmts.push(use_stmt);
                } else {
                    self.compile_use_stmt(use_stmt)?;
                }
            }
        }

        // Register templates (needed for type decorators)
        for item in &module.items {
            if let Item::FnDef(fn_def) = item {
                if fn_def.is_template {
                    self.templates.insert(fn_def.name.node.clone(), fn_def.clone());
                }
            }
        }

        // Process type definitions (needed for cross-module constructor visibility
        // in trait impls - compile_type_def is idempotent so Pass 2 won't re-process)
        for item in &module.items {
            if let Item::TypeDef(type_def) = item {
                self.compile_type_def(type_def)?;
            }
        }

        // Process trait definitions
        for item in &module.items {
            if let Item::TraitDef(trait_def) = item {
                self.compile_trait_def(trait_def)?;
            }
        }

        // Process nested modules FIRST: register their types, traits, and trait impls.
        // This must happen BEFORE deferred use stmts and top-level trait impls,
        // because `use Geo.*` needs nested traits registered, and top-level trait impls
        // like `Shape: HasArea` need to find traits imported from nested modules.
        for item in &module.items {
            if let Item::ModuleDef(module_def) = item {
                self.pre_register_nested_module_metadata(module_def)?;
            }
        }

        // Process deferred use statements (e.g., `use Geo.*` after Geo's traits are registered)
        for use_stmt in deferred_use_stmts {
            self.compile_use_stmt(use_stmt)?;
        }

        // Pre-register trait implementations at top level (register type_traits + forward
        // declare methods, but do NOT compile method bodies). This runs AFTER nested modules
        // and deferred use stmts so that traits from nested modules are visible.
        for item in &module.items {
            if let Item::TraitImpl(trait_impl) = item {
                self.pre_register_trait_impl(trait_impl)?;
            }
        }

        // Save per-module imports for batch inference before restoring.
        // This allows batch inference to set up correct function_aliases per module.
        if !self.module_path.is_empty() {
            let module_key = self.module_path.join(".");
            // Only store function imports (not type imports)
            let fn_imports: HashMap<String, String> = self.imports.iter()
                .filter(|(_, v)| {
                    // Function imports have qualified names where last segment starts lowercase
                    v.split('.').last()
                        .map(|s| s.chars().next().map(|c| c.is_ascii_lowercase()).unwrap_or(false))
                        .unwrap_or(false)
                })
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect();
            self.module_imports.insert(module_key, fn_imports);
        }

        // Restore imports to prevent transitive leaking between file-based modules
        self.imports = saved_imports;
        self.import_sources = saved_import_sources;

        self.module_path = old_module_path;

        Ok(())
    }

    /// Recursively register metadata (types, traits, trait impls) from nested module
    /// definitions. Handles arbitrary depth of module nesting.
    pub(super) fn pre_register_nested_module_metadata(&mut self, module_def: &nostos_syntax::ast::ModuleDef) -> Result<(), CompileError> {
        use nostos_syntax::ast::Item;

        let saved = self.module_path.clone();
        self.module_path.push(module_def.name.node.clone());

        // Process use statements FIRST so that imported types are available
        // for trait impl registration. Without this, `use Types.Vec2` inside
        // module Ops isn't processed, and `Vec2: Scalable` resolves Vec2 to
        // "Ops.Vec2" instead of "Types.Vec2", creating an empty placeholder
        // function that's never overwritten by the actual compilation.
        let saved_imports = self.imports.clone();
        for inner_item in &module_def.items {
            if let Item::Use(use_stmt) = inner_item {
                // Only process if the imported module is already known
                // (e.g., Types is already registered from an earlier nested module)
                let _ = self.compile_use_stmt(use_stmt);
            }
        }

        // Register type definitions inside nested module
        for inner_item in &module_def.items {
            if let Item::TypeDef(type_def) = inner_item {
                self.compile_type_def(type_def)?;
            }
        }

        // Register trait definitions inside nested module
        for inner_item in &module_def.items {
            if let Item::TraitDef(trait_def) = inner_item {
                self.compile_trait_def(trait_def)?;
            }
        }

        // Now process trait impls (traits and imports are now registered)
        for inner_item in &module_def.items {
            if let Item::TraitImpl(trait_impl) = inner_item {
                self.pre_register_trait_impl(trait_impl)?;
            }
        }

        // Restore imports - don't let nested module imports leak into parent scope
        self.imports = saved_imports;

        // Recurse into deeper nested modules
        for inner_item in &module_def.items {
            if let Item::ModuleDef(inner_module_def) = inner_item {
                self.pre_register_nested_module_metadata(inner_module_def)?;
            }
        }

        self.module_path = saved;
        Ok(())
    }

    /// Compile only metadata items: use statements, type definitions, traits.
    /// Skips function definitions - used when loading from cache.
    pub(super) fn compile_items_metadata_only(&mut self, items: &[Item]) -> Result<(), CompileError> {
        use nostos_syntax::ast::Item;

        // Pre-pass: collect names of local (inline) modules
        let local_module_names: std::collections::HashSet<String> = items.iter()
            .filter_map(|item| {
                if let Item::ModuleDef(module_def) = item {
                    Some(module_def.name.node.clone())
                } else {
                    None
                }
            })
            .collect();

        // First pass: process use statements for external modules
        let mut deferred_use_stmts: Vec<&nostos_syntax::ast::UseStmt> = Vec::new();
        for item in items {
            if let Item::Use(use_stmt) = item {
                let module_path = use_stmt.path.first().map(|id| id.node.as_str()).unwrap_or("");
                if local_module_names.contains(module_path) {
                    deferred_use_stmts.push(use_stmt);
                } else {
                    self.compile_use_stmt(use_stmt)?;
                }
            }
        }

        // Second pass (a): register templates (needed for type decorators)
        for item in items {
            if let Item::FnDef(fn_def) = item {
                if fn_def.is_template {
                    self.templates.insert(fn_def.name.node.clone(), fn_def.clone());
                }
            }
        }

        // Second pass (b): collect type definitions
        for item in items {
            if let Item::TypeDef(type_def) = item {
                self.compile_type_def(type_def)?;
            }
        }

        // Third pass: compile trait definitions
        for item in items {
            if let Item::TraitDef(trait_def) = item {
                self.compile_trait_def(trait_def)?;
            }
        }

        // Fourth pass: compile trait implementations
        for item in items {
            if let Item::TraitImpl(trait_impl) = item {
                self.compile_trait_impl(trait_impl)?;
            }
        }

        // Check trait impl completeness after all impl blocks have been compiled
        self.check_trait_impl_completeness()?;

        // Fourth pass (b): compile generated items from type decorators
        let generated = std::mem::take(&mut self.generated_items);
        for item in &generated {
            match item {
                Item::TraitImpl(trait_impl) => {
                    self.compile_trait_impl(trait_impl)?;
                }
                Item::FnDef(fn_def) => {
                    self.compile_fn_def(fn_def)?;
                }
                _ => {}
            }
        }

        // Fifth pass: process nested modules (metadata only)
        for item in items {
            if let Item::ModuleDef(module_def) = item {
                self.compile_module_def_metadata_only(module_def)?;
            }
        }

        // Process deferred use statements
        for use_stmt in deferred_use_stmts {
            self.compile_use_stmt(use_stmt)?;
        }

        // Sixth pass: process mvar definitions
        for item in items {
            if let Item::MvarDef(mvar_def) = item {
                self.compile_mvar_def(mvar_def)?;
            }
        }

        // Skip function definitions - they're loaded from cache

        Ok(())
    }

    /// Register function signatures for all functions in the module.
    /// This is called during Phase 1 to make UFCS methods available
    /// before compiling function bodies in Phase 2.
    pub(super) fn register_function_signatures(&mut self, items: &[Item]) -> Result<(), CompileError> {
        use nostos_syntax::ast::Item;

        for item in items {
            match item {
                Item::FnDef(fn_def) => {
                    self.register_fn_def_signature(fn_def)?;
                }
                Item::ModuleDef(module_def) => {
                    // Process nested modules recursively
                    let parent_path = self.module_path.clone();
                    self.module_path.push(module_def.name.node.clone());
                    let full_path = self.module_path.join(".");
                    self.known_modules.insert(full_path);
                    self.register_function_signatures(&module_def.items)?;
                    self.module_path = parent_path;
                }
                _ => {}
            }
        }
        Ok(())
    }
}
