//! Pattern matching and match expression compilation.
//!
//! Contains functions for compiling match expressions, pattern tests,
//! exhaustiveness checking, try/catch compilation, and pattern binding
//! type extraction.

use super::*;

impl Compiler {
    /// Compile a match expression.
    pub(super) fn compile_match(&mut self, scrutinee: &Expr, arms: &[MatchArm], is_tail: bool, line: usize) -> Result<Reg, CompileError> {
        // Get scrutinee type for exhaustiveness checking and type propagation
        // Prefer structural Type from HM inference, fall back to string
        let scrut_type_structural = self.inferred_expr_types.get(&scrutinee.span()).cloned();
        let scrut_type = self.expr_type_name(scrutinee);
        if let Some(ref ty) = scrut_type {
            self.check_match_exhaustiveness(ty, scrut_type_structural.as_ref(), arms, scrutinee.span())?;
        }

        let scrut_reg = self.compile_expr_tail(scrutinee, false)?;
        let dst = self.alloc_reg();
        let mut end_jumps = Vec::new();
        let mut last_arm_fail_jump: Option<usize> = None;
        // Jumps that need to go to the next arm (pattern fail or guard fail)
        let mut jumps_to_next_arm: Vec<usize> = Vec::new();

        for (i, arm) in arms.iter().enumerate() {
            let is_last = i == arms.len() - 1;

            // Patch any jumps from previous arm that should go to this arm
            let arm_start = self.chunk.code.len();
            for jump in jumps_to_next_arm.drain(..) {
                self.chunk.patch_jump(jump, arm_start);
            }

            // Save locals and local_types before processing arm (pattern bindings should be scoped to this arm)
            let saved_locals = self.locals.clone();
            let saved_local_types = self.local_types.clone();

            // Try to match the pattern
            let (match_success, bindings) = self.compile_pattern_test(&arm.pattern, scrut_reg)?;

            // If pattern fails, jump to next arm (or panic if last)
            let pattern_fail_jump = self.chunk.emit(Instruction::JumpIfFalse(match_success, 0), 0);

            // Bind pattern variables with type info from pattern
            for (name, reg, is_float) in bindings {
                self.locals.insert(name, LocalInfo { reg, is_float, mutable: false, is_cell: false });
            }

            // Set types for pattern variables based on scrutinee type
            // Try structural type matching first (more reliable), fall back to string-based
            let mut binding_types = Vec::new();
            let used_structural = if let Some(scrut_ty) = self.inferred_expr_types.get(&scrutinee.span()) {
                let scrut_ty = scrut_ty.clone(); // Clone to avoid borrow issues
                self.extract_pattern_binding_types_structural(&arm.pattern, &scrut_ty, &mut binding_types)
            } else {
                false
            };
            if !used_structural {
                if let Some(ref ty) = scrut_type {
                    binding_types = self.extract_pattern_binding_types(&arm.pattern, ty);
                }
            }
            for (var_name, var_type) in binding_types {
                self.local_types.insert(var_name, self.type_name_to_type(&var_type));
            }

            // Compile guard if present
            let guard_fail_jump = if let Some(guard) = &arm.guard {
                let guard_reg = self.compile_expr_tail(guard, false)?;
                Some(self.chunk.emit(Instruction::JumpIfFalse(guard_reg, 0), 0))
            } else {
                None
            };

            // Compile arm body - pass is_tail for tail calls, but always move result
            let body_reg = self.compile_expr_tail(&arm.body, is_tail)?;
            self.chunk.emit(Instruction::Move(dst, body_reg), 0);

            // After body, jump to end of match
            end_jumps.push(self.chunk.emit(Instruction::Jump(0), 0));

            // Handle jumps to next arm
            if is_last {
                // Last arm: pattern fail or guard fail should panic
                last_arm_fail_jump = Some(pattern_fail_jump);
                if let Some(guard_jump) = guard_fail_jump {
                    // For last arm with guard, need to also jump to panic on guard fail
                    // Patch pattern fail to same location as guard fail
                    jumps_to_next_arm.push(guard_jump);
                }
            } else {
                // Not last arm: pattern fail or guard fail should try next arm
                jumps_to_next_arm.push(pattern_fail_jump);
                if let Some(guard_jump) = guard_fail_jump {
                    jumps_to_next_arm.push(guard_jump);
                }
            }

            // Restore locals and local_types after arm (pattern bindings shouldn't leak to next arm)
            self.locals = saved_locals;
            self.local_types = saved_local_types;
        }

        // Patch remaining jumps (from last arm failures) to panic location
        let panic_location = self.chunk.code.len();
        for jump in jumps_to_next_arm.drain(..) {
            self.chunk.patch_jump(jump, panic_location);
        }

        // If last arm failed, emit panic for non-exhaustive match
        if let Some(fail_jump) = last_arm_fail_jump {
            self.chunk.patch_jump(fail_jump, self.chunk.code.len());
            let msg_idx = self.chunk.add_constant(Value::String(Arc::new("Non-exhaustive match: no pattern matched".to_string())));
            let msg_reg = self.alloc_reg();
            self.chunk.emit(Instruction::LoadConst(msg_reg, msg_idx), line);
            self.chunk.emit(Instruction::Panic(msg_reg), line);
        }

        // Patch all end jumps
        let end_target = self.chunk.code.len();
        for jump in end_jumps {
            self.chunk.patch_jump(jump, end_target);
        }

        Ok(dst)
    }

    /// Extract type bindings for pattern variables given the scrutinee type.
    /// For example, pattern `(a, _)` with type `(Int, Int)` returns `[("a", "Int")]`
    pub(super) fn extract_pattern_binding_types(&self, pattern: &Pattern, scrut_type: &str) -> Vec<(String, String)> {
        let mut result = Vec::new();
        self.extract_pattern_binding_types_inner(pattern, scrut_type, &mut result);
        result
    }

    pub(super) fn extract_pattern_binding_types_inner(&self, pattern: &Pattern, ty: &str, result: &mut Vec<(String, String)>) {
        match pattern {
            Pattern::Var(ident) => {
                // Bind variable to the type
                if !ty.is_empty() && ty != "Tuple" {
                    result.push((ident.node.clone(), ty.to_string()));
                }
            }
            Pattern::Tuple(pats, _) => {
                // Parse tuple type: "(Int, Int)" -> ["Int", "Int"]
                if ty.starts_with('(') && ty.ends_with(')') {
                    let inner = &ty[1..ty.len() - 1];
                    let elem_types = Self::split_type_args(inner);
                    for (i, pat) in pats.iter().enumerate() {
                        if i < elem_types.len() {
                            self.extract_pattern_binding_types_inner(pat, &elem_types[i], result);
                        }
                    }
                }
            }
            Pattern::Wildcard(_) => {
                // Wildcards don't bind anything
            }
            // Literal patterns don't bind anything
            Pattern::Int(_, _) | Pattern::Int8(_, _) | Pattern::Int16(_, _) | Pattern::Int32(_, _) |
            Pattern::UInt8(_, _) | Pattern::UInt16(_, _) | Pattern::UInt32(_, _) | Pattern::UInt64(_, _) |
            Pattern::BigInt(_, _) | Pattern::Float(_, _) | Pattern::Float32(_, _) | Pattern::Decimal(_, _) |
            Pattern::String(_, _) | Pattern::Char(_, _) | Pattern::Bool(_, _) | Pattern::Unit(_) => {
                // Literals don't bind anything
            }
            Pattern::Variant(ctor_ident, fields, _) => {
                // For variants, look up the constructor to get field types, then substitute
                // type parameters with concrete type args from the scrutinee type.
                // E.g., for type Outcome[a] = Success(a) | Failure(String):
                //   matching Outcome[Int] against Failure(m) -> m has type String (not Int)
                //   matching Outcome[Int] against Success(v) -> v has type Int (from param a)
                if let VariantPatternFields::Positional(patterns) = fields {
                    if !patterns.is_empty() {
                        // Extract type arguments from the scrutinee type string
                        let type_args: Vec<String> = if let Some(bracket_pos) = ty.find('[') {
                            if ty.ends_with(']') {
                                let inner = &ty[bracket_pos + 1..ty.len() - 1];
                                Self::split_type_args(inner)
                            } else {
                                vec![]
                            }
                        } else {
                            vec![]
                        };

                        // Look up the constructor to get its raw field types
                        let ctor_name = &ctor_ident.node;
                        let mut type_names_sorted: Vec<_> = self.types.keys().collect();
                        type_names_sorted.sort();
                        for ty_name in type_names_sorted {
                            if let Some(info) = self.types.get(ty_name) {
                                if let TypeInfoKind::Variant { constructors } = &info.kind {
                                    if let Some((_, field_types)) = constructors.iter().find(|(name, _)| name == ctor_name) {
                                        // Build a type param -> type arg substitution map
                                        // Get type params from type_defs if available, fallback to TypeInfo
                                        let base_type_name = if ty.contains('[') {
                                            &ty[..ty.find('[').unwrap()]
                                        } else {
                                            ty
                                        };
                                        let type_param_names: Vec<String> = self.type_defs.get(base_type_name)
                                            .or_else(|| self.type_defs.get(ty_name))
                                            .map(|td| td.type_params.iter().map(|tp| tp.name.node.clone()).collect())
                                            .unwrap_or_else(|| {
                                                // Fallback: use type_param_names from TypeInfo (populated from cache)
                                                info.type_param_names.clone()
                                            });

                                        for (i, pat) in patterns.iter().enumerate() {
                                            if let Some(raw_field_ty) = field_types.get_type(i) {
                                                // Substitute type parameters in the field type
                                                let resolved_field_ty = if !type_args.is_empty() && !type_param_names.is_empty() {
                                                    let mut resolved = raw_field_ty.clone();
                                                    for (param_name, arg_val) in type_param_names.iter().zip(type_args.iter()) {
                                                        // Use word-boundary-aware substitution to avoid corrupting
                                                        // type names that contain the param letter (e.g., "Map" contains "a")
                                                        resolved = Self::substitute_single_type_param(&resolved, param_name, arg_val);
                                                    }
                                                    resolved
                                                } else {
                                                    raw_field_ty.clone()
                                                };
                                                self.extract_pattern_binding_types_inner(pat, &resolved_field_ty, result);
                                            }
                                        }
                                        return;
                                    }
                                }
                            }
                        }

                        // Fallback: if constructor not found in self.types, use type args directly
                        // (this handles cases where type info isn't available)
                        if type_args.len() == 1 && patterns.len() == 1 {
                            self.extract_pattern_binding_types_inner(&patterns[0], &type_args[0], result);
                            return;
                        }
                    }
                }
            }
            Pattern::Record(fields, _) => {
                // For record patterns, we'd need field type info
                for field in fields {
                    match field {
                        RecordPatternField::Named(_, pat) => {
                            self.extract_pattern_binding_types_inner(pat, "", result);
                        }
                        RecordPatternField::Punned(ident) => {
                            // Punned binding - we don't know the field type
                            result.push((ident.node.clone(), String::new()));
                        }
                        RecordPatternField::Rest(_) => {}
                    }
                }
            }
            Pattern::List(list_pat, _) => {
                // For list patterns, extract element type from List[T]
                let elem_type = if ty.starts_with("List[") && ty.ends_with(']') {
                    ty[5..ty.len() - 1].to_string()
                } else {
                    String::new()
                };
                match list_pat {
                    ListPattern::Empty => {}
                    ListPattern::Cons(head_pats, tail) => {
                        for pat in head_pats {
                            self.extract_pattern_binding_types_inner(pat, &elem_type, result);
                        }
                        if let Some(tail_pat) = tail {
                            // Tail has the same type as the whole list
                            self.extract_pattern_binding_types_inner(tail_pat, ty, result);
                        }
                    }
                }
            }
            Pattern::Map(_, _) | Pattern::Or(_, _) | Pattern::Pin(_, _) |
            Pattern::Set(_, _) | Pattern::StringCons(_, _) | Pattern::Range(_, _, _, _) => {
                // Not handling these for now
            }
        }
    }

    /// Extract type bindings using structural Type matching (more reliable than string parsing).
    /// Returns true if structural matching succeeded, false if we should fall back to string-based.
    pub(super) fn extract_pattern_binding_types_structural(&self, pattern: &Pattern, ty: &nostos_types::Type, result: &mut Vec<(String, String)>) -> bool {
        use nostos_types::Type;

        // Only proceed if the type is fully resolved (no type variables or params)
        if !self.is_type_structurally_resolved(ty) {
            return false;
        }

        match pattern {
            Pattern::Var(ident) => {
                let ty_str = ty.display();
                // Skip empty or generic "Tuple" type names
                if !ty_str.is_empty() && !matches!(ty, Type::Tuple(_) if ty_str == "Tuple") {
                    result.push((ident.node.clone(), ty_str));
                }
                true
            }
            Pattern::Tuple(pats, _) => {
                if let Type::Tuple(elem_types) = ty {
                    for (i, pat) in pats.iter().enumerate() {
                        if i < elem_types.len() {
                            // Recursively extract, but fall back to string if structural fails
                            if !self.extract_pattern_binding_types_structural(pat, &elem_types[i], result) {
                                self.extract_pattern_binding_types_inner(pat, &elem_types[i].display(), result);
                            }
                        }
                    }
                    true
                } else {
                    false
                }
            }
            Pattern::Wildcard(_) => true,
            // Literal patterns don't bind anything
            Pattern::Int(_, _) | Pattern::Int8(_, _) | Pattern::Int16(_, _) | Pattern::Int32(_, _) |
            Pattern::UInt8(_, _) | Pattern::UInt16(_, _) | Pattern::UInt32(_, _) | Pattern::UInt64(_, _) |
            Pattern::BigInt(_, _) | Pattern::Float(_, _) | Pattern::Float32(_, _) | Pattern::Decimal(_, _) |
            Pattern::String(_, _) | Pattern::Char(_, _) | Pattern::Bool(_, _) | Pattern::Unit(_) => true,
            Pattern::List(list_pat, _) => {
                if let Type::List(elem_ty) = ty {
                    match list_pat {
                        ListPattern::Empty => {}
                        ListPattern::Cons(head_pats, tail) => {
                            for pat in head_pats {
                                if !self.extract_pattern_binding_types_structural(pat, elem_ty, result) {
                                    self.extract_pattern_binding_types_inner(pat, &elem_ty.display(), result);
                                }
                            }
                            if let Some(tail_pat) = tail {
                                // Tail has the same type as the whole list
                                if !self.extract_pattern_binding_types_structural(tail_pat, ty, result) {
                                    self.extract_pattern_binding_types_inner(tail_pat, &ty.display(), result);
                                }
                            }
                        }
                    }
                    true
                } else {
                    false
                }
            }
            Pattern::Variant(ctor_ident, fields, _) => {
                // For Named types with type arguments, look up the constructor to get
                // actual field types rather than assuming type args map to field types.
                // E.g., Outcome[Int] = Success(a) | Failure(String) - for Failure(m),
                // m should be String (not Int from the type arg).
                if let Type::Named { name, args } = ty {
                    if let VariantPatternFields::Positional(patterns) = fields {
                        let ctor_name = &ctor_ident.node;
                        // Try to look up constructor field types and substitute type params
                        let base_name = name.as_str();
                        let mut found = false;
                        let mut type_names_sorted: Vec<_> = self.types.keys().collect();
                        type_names_sorted.sort();
                        for ty_name in &type_names_sorted {
                            if let Some(info) = self.types.get(*ty_name) {
                                if let TypeInfoKind::Variant { constructors } = &info.kind {
                                    if let Some((_, field_types)) = constructors.iter().find(|(n, _)| n == ctor_name) {
                                        // Get type parameter names from type_defs, falling back to TypeInfo
                                        let type_param_names: Vec<String> = self.type_defs.get(base_name)
                                            .or_else(|| self.type_defs.get(*ty_name))
                                            .map(|td| td.type_params.iter().map(|tp| tp.name.node.clone()).collect())
                                            .unwrap_or_else(|| {
                                                // Fallback: use type_param_names from TypeInfo (populated from cache)
                                                info.type_param_names.clone()
                                            });

                                        for (i, pat) in patterns.iter().enumerate() {
                                            if let Some(raw_field_ty) = field_types.get_type(i) {
                                                // Check if the raw field type is a type parameter
                                                let param_idx = type_param_names.iter().position(|p| p == raw_field_ty);
                                                if let Some(idx) = param_idx {
                                                    // Field type is a type parameter - use the corresponding type arg
                                                    if idx < args.len() {
                                                        if !self.extract_pattern_binding_types_structural(pat, &args[idx], result) {
                                                            self.extract_pattern_binding_types_inner(pat, &args[idx].display(), result);
                                                        }
                                                    }
                                                } else {
                                                    // Field type is concrete (e.g., String) - use as-is
                                                    self.extract_pattern_binding_types_inner(pat, raw_field_ty, result);
                                                }
                                            }
                                        }
                                        found = true;
                                        break;
                                    }
                                }
                            }
                        }
                        if found {
                            return true;
                        }

                        // Fallback: if constructor not found, use type args directly
                        // (original behavior for simple cases)
                        if args.len() == 1 && patterns.len() == 1 {
                            if !self.extract_pattern_binding_types_structural(&patterns[0], &args[0], result) {
                                self.extract_pattern_binding_types_inner(&patterns[0], &args[0].display(), result);
                            }
                            return true;
                        }
                    }
                }
                // Fall back to string-based for complex variant patterns
                false
            }
            // For other patterns, fall back to string-based
            _ => false,
        }
    }

    /// Check if match arms exhaustively cover all cases of a type.
    /// Returns Ok(()) if exhaustive, Err with missing patterns otherwise.
    /// Uses structural Type when available for more reliable checking.
    pub(super) fn check_match_exhaustiveness(&self, scrut_type: &str, scrut_type_structural: Option<&nostos_types::Type>, arms: &[MatchArm], span: Span) -> Result<(), CompileError> {
        // Check for wildcard or variable pattern (catches all)
        for arm in arms {
            if self.pattern_is_catch_all(&arm.pattern) {
                return Ok(());
            }
        }

        // Check for Bool type
        if scrut_type == "Bool" {
            // Helper to check if a pattern covers a specific bool value (including inside Or patterns)
            fn pattern_covers_bool(pattern: &Pattern, target: bool) -> bool {
                match pattern {
                    Pattern::Bool(b, _) => *b == target,
                    Pattern::Or(patterns, _) => patterns.iter().any(|p| pattern_covers_bool(p, target)),
                    Pattern::Var(_) | Pattern::Wildcard(_) => true,
                    _ => false,
                }
            }

            let has_true = arms.iter().any(|arm| pattern_covers_bool(&arm.pattern, true));
            let has_false = arms.iter().any(|arm| pattern_covers_bool(&arm.pattern, false));

            if !has_true || !has_false {
                let mut missing = Vec::new();
                if !has_true { missing.push("true".to_string()); }
                if !has_false { missing.push("false".to_string()); }
                return Err(CompileError::TypeError {
                    message: format!("non-exhaustive patterns: {} not covered", missing.join(", ")),
                    span,
                });
            }
            return Ok(());
        }

        // Helper to extract variant names from a pattern (including inside Or patterns)
        fn extract_variant_names<'a>(pattern: &'a Pattern, names: &mut Vec<&'a str>) {
            match pattern {
                Pattern::Variant(ident, _, _) => names.push(ident.node.as_str()),
                Pattern::Or(patterns, _) => {
                    for p in patterns {
                        extract_variant_names(p, names);
                    }
                }
                _ => {}
            }
        }

        // Check for variant types
        if let Some(type_info) = self.types.get(scrut_type) {
            if let TypeInfoKind::Variant { constructors } = &type_info.kind {
                let all_ctors: std::collections::HashSet<&str> = constructors
                    .iter()
                    .map(|(name, _)| name.as_str())
                    .collect();

                let mut covered_names = Vec::new();
                for arm in arms {
                    extract_variant_names(&arm.pattern, &mut covered_names);
                }
                let covered_ctors: std::collections::HashSet<&str> = covered_names.into_iter().collect();

                let missing: Vec<String> = all_ctors
                    .difference(&covered_ctors)
                    .map(|s| s.to_string())
                    .collect();

                if !missing.is_empty() {
                    return Err(CompileError::TypeError {
                        message: format!("non-exhaustive patterns: `{}` not covered", missing.join("`, `")),
                        span,
                    });
                }
            }
        }

        // Check for Option type (special case - very common)
        // Use structural type first, fall back to string matching
        let is_option = scrut_type_structural.map(|ty| self.as_option_type(ty).is_some()).unwrap_or(false)
            || scrut_type.starts_with("Option[") || scrut_type == "Option";
        if is_option {
            // Helper to check if a pattern covers a specific variant (including inside Or patterns)
            fn pattern_covers_variant(pattern: &Pattern, variant_name: &str) -> bool {
                match pattern {
                    Pattern::Variant(ident, _, _) => ident.node == variant_name,
                    Pattern::Or(patterns, _) => patterns.iter().any(|p| pattern_covers_variant(p, variant_name)),
                    Pattern::Var(_) | Pattern::Wildcard(_) => true,
                    _ => false,
                }
            }

            let has_some = arms.iter().any(|arm| pattern_covers_variant(&arm.pattern, "Some"));
            let has_none = arms.iter().any(|arm| pattern_covers_variant(&arm.pattern, "None"));

            if !has_some || !has_none {
                let mut missing = Vec::new();
                if !has_some { missing.push("Some(_)".to_string()); }
                if !has_none { missing.push("None".to_string()); }
                return Err(CompileError::TypeError {
                    message: format!("non-exhaustive patterns: `{}` not covered", missing.join("`, `")),
                    span,
                });
            }
        }

        // For other types (Int, String, List, etc.), we can't check exhaustiveness
        // without a wildcard pattern - but we don't error here because these types
        // have infinite values
        Ok(())
    }

    /// Check if a pattern catches all values (wildcard or variable binding).
    pub(super) fn pattern_is_catch_all(&self, pattern: &Pattern) -> bool {
        match pattern {
            Pattern::Wildcard(_) => true,
            Pattern::Var(_) => true,
            Pattern::Or(patterns, _) => patterns.iter().any(|p| self.pattern_is_catch_all(p)),
            _ => false,
        }
    }

    /// Compile a try/catch/finally expression.
    pub(super) fn compile_try(
        &mut self,
        try_expr: &Expr,
        catch_arms: &[MatchArm],
        finally_expr: Option<&Expr>,
        is_tail: bool,
    ) -> Result<Reg, CompileError> {
        let dst = self.alloc_reg();

        // For finally blocks, we need a flag to track if we should rethrow after finally
        // This only adds overhead when finally exists
        let rethrow_flag = if finally_expr.is_some() {
            let flag = self.alloc_reg();
            // Initialize to false - no rethrow pending
            let false_idx = self.chunk.add_constant(Value::Bool(false));
            self.chunk.emit(Instruction::LoadConst(flag, false_idx), 0);
            Some(flag)
        } else {
            None
        };

        // 1. Push exception handler - offset will be patched later
        let handler_idx = self.chunk.emit(Instruction::PushHandler(0), 0);

        // 2. Compile the try body
        let try_result = self.compile_expr_tail(try_expr, false)?;
        self.chunk.emit(Instruction::Move(dst, try_result), 0);

        // 3. Pop the handler (success path)
        self.chunk.emit(Instruction::PopHandler, 0);

        // 4. Jump past the catch block (success path)
        let skip_catch_jump = self.chunk.emit(Instruction::Jump(0), 0);

        // 5. CATCH BLOCK START - patch the handler to jump here
        let catch_start = self.chunk.code.len();
        self.chunk.patch_jump(handler_idx, catch_start);

        // 6. Get the exception value
        let exc_reg = self.alloc_reg();
        self.chunk.emit(Instruction::GetException(exc_reg), 0);

        // 7. Pattern match on the exception (similar to compile_match)
        // We need to track jumps to re-throw if no pattern matches
        let mut end_jumps = Vec::new();
        let mut rethrow_jumps = Vec::new();

        for (i, arm) in catch_arms.iter().enumerate() {
            let is_last = i == catch_arms.len() - 1;

            // Save locals before processing arm (pattern bindings should be scoped to this arm)
            let saved_locals = self.locals.clone();

            // Try to match the pattern
            let (match_success, bindings) = self.compile_pattern_test(&arm.pattern, exc_reg)?;

            // Always emit JumpIfFalse, even for last arm (to handle no-match case)
            let next_arm_jump = self.chunk.emit(Instruction::JumpIfFalse(match_success, 0), 0);

            // Bind pattern variables with type info from pattern
            for (name, reg, is_float) in bindings {
                self.locals.insert(name, LocalInfo { reg, is_float, mutable: false, is_cell: false });
            }

            // Compile guard if present
            if let Some(guard) = &arm.guard {
                let guard_reg = self.compile_expr_tail(guard, false)?;
                // If guard fails, jump to next arm (or rethrow for last arm)
                let guard_jump = self.chunk.emit(Instruction::JumpIfFalse(guard_reg, 0), 0);
                if is_last {
                    rethrow_jumps.push(guard_jump);
                } else {
                    // Patch to same location as pattern mismatch
                    rethrow_jumps.push(guard_jump);
                }
            }

            // Compile catch arm body
            let body_reg = self.compile_expr_tail(&arm.body, is_tail && finally_expr.is_none())?;
            self.chunk.emit(Instruction::Move(dst, body_reg), 0);

            // Jump to end (skip other arms and rethrow)
            end_jumps.push(self.chunk.emit(Instruction::Jump(0), 0));

            // Patch jump to next arm (or rethrow block for last arm)
            if is_last {
                rethrow_jumps.push(next_arm_jump);
            } else {
                let next_target = self.chunk.code.len();
                self.chunk.patch_jump(next_arm_jump, next_target);
            }

            // Restore locals after arm (pattern bindings shouldn't leak to next arm or subsequent code)
            self.locals = saved_locals;
        }

        // 7.5 Re-throw block: if no pattern matched, handle rethrow
        let rethrow_start = self.chunk.code.len();
        for jump in rethrow_jumps {
            self.chunk.patch_jump(jump, rethrow_start);
        }

        if let Some(flag) = rethrow_flag {
            // With finally: set flag to true and jump to finally, then rethrow after
            let true_idx = self.chunk.add_constant(Value::Bool(true));
            self.chunk.emit(Instruction::LoadConst(flag, true_idx), 0);
            // Fall through to after_catch where finally is compiled
        } else {
            // No finally: rethrow immediately
            self.chunk.emit(Instruction::Throw(exc_reg), 0);
        }

        // 8. Patch all end jumps from catch arms
        let after_catch = self.chunk.code.len();
        for jump in end_jumps {
            self.chunk.patch_jump(jump, after_catch);
        }

        // 9. Patch the skip_catch_jump to land here
        self.chunk.patch_jump(skip_catch_jump, after_catch);

        // 10. Compile finally block if present
        if let Some(finally) = finally_expr {
            // Finally is executed for both success and exception paths
            // Its result is discarded; the try/catch result is preserved in dst
            self.compile_expr_tail(finally, false)?;

            // After finally, check if we need to rethrow
            if let Some(flag) = rethrow_flag {
                let done_jump = self.chunk.emit(Instruction::JumpIfFalse(flag, 0), 0);
                // Rethrow the preserved exception
                self.chunk.emit(Instruction::Throw(exc_reg), 0);
                // Patch done_jump to here (normal exit)
                let done_target = self.chunk.code.len();
                self.chunk.patch_jump(done_jump, done_target);
            }
        }

        Ok(dst)
    }

    /// Compile error propagation: expr?
    /// If expr throws, re-throw the exception. Otherwise return its value.
    pub(super) fn compile_try_propagate(&mut self, inner_expr: &Expr) -> Result<Reg, CompileError> {
        let dst = self.alloc_reg();

        // 1. Push exception handler
        let handler_idx = self.chunk.emit(Instruction::PushHandler(0), 0);

        // 2. Compile the inner expression
        let result = self.compile_expr_tail(inner_expr, false)?;
        self.chunk.emit(Instruction::Move(dst, result), 0);

        // 3. Pop handler on success
        self.chunk.emit(Instruction::PopHandler, 0);

        // 4. Jump past the re-throw
        let skip_rethrow_jump = self.chunk.emit(Instruction::Jump(0), 0);

        // 5. RE-THROW BLOCK - patch handler to jump here
        let rethrow_start = self.chunk.code.len();
        self.chunk.patch_jump(handler_idx, rethrow_start);

        // 6. Get exception and re-throw it
        let exc_reg = self.alloc_reg();
        self.chunk.emit(Instruction::GetException(exc_reg), 0);
        self.chunk.emit(Instruction::Throw(exc_reg), 0);

        // 7. Patch skip_rethrow_jump to land here
        let after_rethrow = self.chunk.code.len();
        self.chunk.patch_jump(skip_rethrow_jump, after_rethrow);

        Ok(dst)
    }

    /// Compile a pattern test and return (success_reg, bindings).
    /// Bindings are (name, reg, is_float) tuples.
    pub(super) fn compile_pattern_test(&mut self, pattern: &Pattern, scrut_reg: Reg) -> Result<(Reg, Vec<(String, Reg, bool)>), CompileError> {
        let success_reg = self.alloc_reg();
        let mut bindings: Vec<(String, Reg, bool)> = Vec::new();

        match pattern {
            Pattern::Wildcard(_) => {
                self.chunk.emit(Instruction::LoadTrue(success_reg), 0);
            }
            Pattern::Var(ident) => {
                // Check if variable already exists as immutable binding
                if let Some(existing_info) = self.locals.get(&ident.node).copied() {
                    if !existing_info.mutable {
                        // Immutable variable: use as constraint (test equality)
                        self.chunk.emit(Instruction::Eq(success_reg, scrut_reg, existing_info.reg), 0);
                    } else {
                        // Mutable variable: rebind it
                        self.chunk.emit(Instruction::LoadTrue(success_reg), 0);
                        self.chunk.emit(Instruction::Move(existing_info.reg, scrut_reg), 0);
                    }
                } else {
                    // New variable: create binding
                    self.chunk.emit(Instruction::LoadTrue(success_reg), 0);
                    let var_reg = self.alloc_reg();
                    self.chunk.emit(Instruction::Move(var_reg, scrut_reg), 0);
                    // Type unknown for plain var pattern, will be updated by variant context
                    bindings.push((ident.node.clone(), var_reg, false));
                }
            }
            Pattern::Unit(_) => {
                self.chunk.emit(Instruction::TestUnit(success_reg, scrut_reg), 0);
            }
            Pattern::Int(n, _) => {
                let const_idx = self.chunk.add_constant(Value::Int64(*n));
                self.chunk.emit(Instruction::TestConst(success_reg, scrut_reg, const_idx), 0);
            }
            Pattern::Int8(n, _) => {
                let const_idx = self.chunk.add_constant(Value::Int8(*n));
                self.chunk.emit(Instruction::TestConst(success_reg, scrut_reg, const_idx), 0);
            }
            Pattern::Int16(n, _) => {
                let const_idx = self.chunk.add_constant(Value::Int16(*n));
                self.chunk.emit(Instruction::TestConst(success_reg, scrut_reg, const_idx), 0);
            }
            Pattern::Int32(n, _) => {
                let const_idx = self.chunk.add_constant(Value::Int32(*n));
                self.chunk.emit(Instruction::TestConst(success_reg, scrut_reg, const_idx), 0);
            }
            Pattern::UInt8(n, _) => {
                let const_idx = self.chunk.add_constant(Value::UInt8(*n));
                self.chunk.emit(Instruction::TestConst(success_reg, scrut_reg, const_idx), 0);
            }
            Pattern::UInt16(n, _) => {
                let const_idx = self.chunk.add_constant(Value::UInt16(*n));
                self.chunk.emit(Instruction::TestConst(success_reg, scrut_reg, const_idx), 0);
            }
            Pattern::UInt32(n, _) => {
                let const_idx = self.chunk.add_constant(Value::UInt32(*n));
                self.chunk.emit(Instruction::TestConst(success_reg, scrut_reg, const_idx), 0);
            }
            Pattern::UInt64(n, _) => {
                let const_idx = self.chunk.add_constant(Value::UInt64(*n));
                self.chunk.emit(Instruction::TestConst(success_reg, scrut_reg, const_idx), 0);
            }
            Pattern::Float(f, _) => {
                let const_idx = self.chunk.add_constant(Value::Float64(*f));
                self.chunk.emit(Instruction::TestConst(success_reg, scrut_reg, const_idx), 0);
            }
            Pattern::Float32(f, _) => {
                let const_idx = self.chunk.add_constant(Value::Float32(*f));
                self.chunk.emit(Instruction::TestConst(success_reg, scrut_reg, const_idx), 0);
            }
            Pattern::BigInt(s, _) => {
                use num_bigint::BigInt;
                let big = s.parse::<BigInt>().unwrap_or_default();
                let const_idx = self.chunk.add_constant(Value::BigInt(Arc::new(big)));
                self.chunk.emit(Instruction::TestConst(success_reg, scrut_reg, const_idx), 0);
            }
            Pattern::Decimal(s, _) => {
                use rust_decimal::Decimal;
                let dec = s.parse::<Decimal>().unwrap_or_default();
                let const_idx = self.chunk.add_constant(Value::Decimal(dec));
                self.chunk.emit(Instruction::TestConst(success_reg, scrut_reg, const_idx), 0);
            }
            Pattern::Char(c, _) => {
                let const_idx = self.chunk.add_constant(Value::Char(*c));
                self.chunk.emit(Instruction::TestConst(success_reg, scrut_reg, const_idx), 0);
            }
            Pattern::Bool(b, _) => {
                let const_idx = self.chunk.add_constant(Value::Bool(*b));
                self.chunk.emit(Instruction::TestConst(success_reg, scrut_reg, const_idx), 0);
            }
            Pattern::String(s, _) => {
                let const_idx = self.chunk.add_constant(Value::String(Arc::new(s.clone())));
                self.chunk.emit(Instruction::TestConst(success_reg, scrut_reg, const_idx), 0);
            }
            Pattern::Variant(ctor, fields, _) => {
                // Store constructor name in constants for exact string matching
                // Use local constructor name (without module path) to match how variants are created
                // Variants are created with local tags, so patterns must also use local tags
                let local_ctor = ctor.node.rsplit('.').next().unwrap_or(&ctor.node).to_string();
                let ctor_idx = self.chunk.add_constant(Value::String(Arc::new(local_ctor.clone())));
                self.chunk.emit(Instruction::TestTag(success_reg, scrut_reg, ctor_idx), 0);

                // Look up field types for this constructor - try both qualified and local names
                let qualified_ctor = self.qualify_name(&ctor.node);
                let field_types = {
                    let qt = self.get_constructor_field_types(&qualified_ctor);
                    if !qt.is_empty() { qt } else { self.get_constructor_field_types(&local_ctor) }
                };

                // Extract and bind fields - only if tag matches (guard with conditional jump)
                match fields {
                    VariantPatternFields::Unit => {}
                    VariantPatternFields::Positional(patterns) => {
                        // Jump past field extraction if tag doesn't match
                        let skip_jump = self.chunk.emit(Instruction::JumpIfFalse(success_reg, 0), 0);

                        for (i, pat) in patterns.iter().enumerate() {
                            let field_reg = self.alloc_reg();
                            self.chunk.emit(Instruction::GetVariantField(field_reg, scrut_reg, i as u8), 0);
                            let (sub_success, mut sub_bindings) = self.compile_pattern_test(pat, field_reg)?;
                            // AND the sub-pattern's success with our overall success
                            self.chunk.emit(Instruction::And(success_reg, success_reg, sub_success), 0);

                            // Update type info based on constructor's field type
                            let is_float = field_types.get(i)
                                .map(|t| matches!(t.as_str(), "Float" | "Float32" | "Float64"))
                                .unwrap_or(false);
                            for binding in &mut sub_bindings {
                                binding.2 = is_float;
                            }
                            bindings.append(&mut sub_bindings);
                        }

                        // Patch the skip jump to land here
                        let after_extract = self.chunk.code.len();
                        self.chunk.patch_jump(skip_jump, after_extract);
                    }
                    VariantPatternFields::Named(nfields) => {
                        // Jump past field extraction if tag doesn't match
                        let skip_jump = self.chunk.emit(Instruction::JumpIfFalse(success_reg, 0), 0);

                        // Get field names from type definition to map names to indices
                        let ctor_field_names = {
                            let qn = self.get_constructor_field_names(&qualified_ctor);
                            if !qn.is_empty() { qn } else { self.get_constructor_field_names(&local_ctor) }
                        };
                        // Build name -> index map
                        let name_to_idx: std::collections::HashMap<&str, usize> = ctor_field_names
                            .iter()
                            .enumerate()
                            .map(|(i, name)| (name.as_str(), i))
                            .collect();

                        for field in nfields {
                            match field {
                                RecordPatternField::Punned(ident) => {
                                    // Point(x) means bind field "x" to variable "x"
                                    let field_reg = self.alloc_reg();
                                    if let Some(&idx) = name_to_idx.get(ident.node.as_str()) {
                                        self.chunk.emit(Instruction::GetVariantField(field_reg, scrut_reg, idx as u8), 0);
                                    } else {
                                        // Fallback: field name not found, use index 0 (will likely fail at runtime)
                                        self.chunk.emit(Instruction::GetVariantField(field_reg, scrut_reg, 0), 0);
                                    }
                                    // Determine if field is float type
                                    let is_float = name_to_idx.get(ident.node.as_str())
                                        .and_then(|&idx| field_types.get(idx))
                                        .map(|t| matches!(t.as_str(), "Float" | "Float32" | "Float64"))
                                        .unwrap_or(false);
                                    bindings.push((ident.node.clone(), field_reg, is_float));
                                }
                                RecordPatternField::Named(ident, pat) => {
                                    // Point(x: a) means bind field "x" to the result of matching pattern
                                    let field_reg = self.alloc_reg();
                                    if let Some(&idx) = name_to_idx.get(ident.node.as_str()) {
                                        self.chunk.emit(Instruction::GetVariantField(field_reg, scrut_reg, idx as u8), 0);
                                    } else {
                                        // Fallback: field name not found
                                        self.chunk.emit(Instruction::GetVariantField(field_reg, scrut_reg, 0), 0);
                                    }
                                    let (sub_success, mut sub_bindings) = self.compile_pattern_test(pat, field_reg)?;
                                    // AND the sub-pattern's success with our overall success
                                    self.chunk.emit(Instruction::And(success_reg, success_reg, sub_success), 0);

                                    // Update type info for float fields
                                    let is_float = name_to_idx.get(ident.node.as_str())
                                        .and_then(|&idx| field_types.get(idx))
                                        .map(|t| matches!(t.as_str(), "Float" | "Float32" | "Float64"))
                                        .unwrap_or(false);
                                    for binding in &mut sub_bindings {
                                        binding.2 = is_float;
                                    }
                                    bindings.append(&mut sub_bindings);
                                }
                                RecordPatternField::Rest(_) => {
                                    // Ignore rest pattern for now - it just matches remaining fields
                                }
                            }
                        }

                        // Patch the skip jump to land here
                        let after_extract = self.chunk.code.len();
                        self.chunk.patch_jump(skip_jump, after_extract);
                    }
                }
            }
            Pattern::List(list_pattern, _) => {
                match list_pattern {
                    ListPattern::Empty => {
                        self.chunk.emit(Instruction::TestNil(success_reg, scrut_reg), 0);
                    }
                    ListPattern::Cons(head_patterns, tail) => {
                        let n = head_patterns.len();

                        if tail.is_some() {
                            // Pattern like [a, b | t] - check list has at least n elements
                            // Check length >= n
                            let len_reg = self.alloc_reg();
                            self.chunk.emit(Instruction::Length(len_reg, scrut_reg), 0);
                            let n_reg = self.alloc_reg();
                            let n_idx = self.chunk.add_constant(Value::Int64(n as i64));
                            self.chunk.emit(Instruction::LoadConst(n_reg, n_idx), 0);
                            self.chunk.emit(Instruction::GeInt(success_reg, len_reg, n_reg), 0);
                        } else {
                            // Pattern like [a, b, c] - check list has exactly n elements
                            let len_reg = self.alloc_reg();
                            self.chunk.emit(Instruction::Length(len_reg, scrut_reg), 0);
                            let n_reg = self.alloc_reg();
                            let n_idx = self.chunk.add_constant(Value::Int64(n as i64));
                            self.chunk.emit(Instruction::LoadConst(n_reg, n_idx), 0);
                            self.chunk.emit(Instruction::Eq(success_reg, len_reg, n_reg), 0);
                        }

                        // Guard the Decons block: skip if success_reg is false (list too short)
                        // This prevents "Cannot decons empty list" errors when pattern doesn't match
                        let skip_decons_jump = self.chunk.emit(Instruction::JumpIfFalse(success_reg, 0), 0);

                        // Decons each head element
                        let mut current_list = scrut_reg;
                        for (i, head_pat) in head_patterns.iter().enumerate() {
                            let head_reg = self.alloc_reg();
                            let tail_reg = self.alloc_reg();
                            self.chunk.emit(Instruction::Decons(head_reg, tail_reg, current_list), 0);

                            let (head_success, mut head_bindings) = self.compile_pattern_test(head_pat, head_reg)?;
                            // AND the head pattern's success with our overall success
                            self.chunk.emit(Instruction::And(success_reg, success_reg, head_success), 0);
                            bindings.append(&mut head_bindings);

                            // If this is the last head pattern and there's a tail pattern
                            if i == n - 1 {
                                if let Some(tail_pat) = tail {
                                    let (tail_success, mut tail_bindings) = self.compile_pattern_test(tail_pat, tail_reg)?;
                                    // AND the tail pattern's success with our overall success
                                    self.chunk.emit(Instruction::And(success_reg, success_reg, tail_success), 0);
                                    bindings.append(&mut tail_bindings);
                                }
                            } else {
                                current_list = tail_reg;
                            }
                        }

                        // Patch jump to skip past the Decons block
                        self.chunk.patch_jump(skip_decons_jump, self.chunk.code.len());
                    }
                }
            }
            Pattern::StringCons(string_pattern, _) => {
                match string_pattern {
                    StringPattern::Empty => {
                        // Empty string pattern: test if string is ""
                        self.chunk.emit(Instruction::TestEmptyString(success_reg, scrut_reg), 0);
                    }
                    StringPattern::Cons(prefix_strings, tail_pat) => {
                        // String cons pattern like ["hello", "world" | rest]
                        // Concatenate prefix strings to form the expected prefix
                        let prefix: String = prefix_strings.concat();

                        // Use optimized TestStringPrefix instruction
                        // This does prefix match + tail extraction in one operation
                        let tail_reg = self.alloc_reg();
                        let prefix_idx = self.chunk.add_constant(Value::String(Arc::new(prefix)));
                        self.chunk.emit(Instruction::TestStringPrefix(success_reg, tail_reg, scrut_reg, prefix_idx), 0);

                        // Guard: skip tail pattern if prefix didn't match
                        let skip_tail_jump = self.chunk.emit(Instruction::JumpIfFalse(success_reg, 0), 0);

                        // Compile tail pattern binding
                        let (tail_success, mut tail_bindings) = self.compile_pattern_test(tail_pat, tail_reg)?;
                        self.chunk.emit(Instruction::And(success_reg, success_reg, tail_success), 0);
                        bindings.append(&mut tail_bindings);

                        // Patch skip jump
                        self.chunk.patch_jump(skip_tail_jump, self.chunk.code.len());
                    }
                }
            }
            Pattern::Tuple(patterns, _) => {
                self.chunk.emit(Instruction::LoadTrue(success_reg), 0);

                // Use batch destructure for common cases (pairs and triples)
                let elem_regs: Vec<Reg> = (0..patterns.len()).map(|_| self.alloc_reg()).collect();

                if patterns.len() == 2 {
                    // Optimized pair destructure - single heap lookup
                    self.chunk.emit(Instruction::DestructurePair(elem_regs[0], elem_regs[1], scrut_reg), 0);
                } else if patterns.len() == 3 {
                    // Optimized triple destructure - single heap lookup
                    self.chunk.emit(Instruction::DestructureTriple(elem_regs[0], elem_regs[1], elem_regs[2], scrut_reg), 0);
                } else {
                    // Fallback for other sizes
                    for (i, &elem_reg) in elem_regs.iter().enumerate() {
                        self.chunk.emit(Instruction::GetTupleField(elem_reg, scrut_reg, i as u8), 0);
                    }
                }

                // Now compile sub-patterns
                for (i, pat) in patterns.iter().enumerate() {
                    let (sub_success, mut sub_bindings) = self.compile_pattern_test(pat, elem_regs[i])?;
                    // AND the sub-pattern's success with our overall success
                    self.chunk.emit(Instruction::And(success_reg, success_reg, sub_success), 0);
                    bindings.append(&mut sub_bindings);
                }
            }
            Pattern::Map(entries, _) => {
                // 1. Check if it is a map
                self.chunk.emit(Instruction::IsMap(success_reg, scrut_reg), 0);
                
                // Jump to end if not a map
                let type_fail_jump = self.chunk.emit(Instruction::JumpIfFalse(success_reg, 0), 0);
                
                for (key_expr, val_pat) in entries {
                    // Compile key expression to a register
                    let key_reg = self.compile_expr_tail(key_expr, false)?;
                    
                    // Check if key exists
                    let exists_reg = self.alloc_reg();
                    self.chunk.emit(Instruction::MapContainsKey(exists_reg, scrut_reg, key_reg), 0);
                    
                    // Update success_reg
                    self.chunk.emit(Instruction::And(success_reg, success_reg, exists_reg), 0);
                    
                    // Guard: if key doesn't exist, skip value check (to avoid panic in MapGet)
                    let skip_val_jump = self.chunk.emit(Instruction::JumpIfFalse(exists_reg, 0), 0);
                    
                    // Get value
                    let val_reg = self.alloc_reg();
                    self.chunk.emit(Instruction::MapGet(val_reg, scrut_reg, key_reg), 0);
                    
                    // Match value against pattern
                    let (sub_success, mut sub_bindings) = self.compile_pattern_test(val_pat, val_reg)?;
                    self.chunk.emit(Instruction::And(success_reg, success_reg, sub_success), 0);
                    bindings.append(&mut sub_bindings);
                    
                    // Patch skip val jump
                    let after_val = self.chunk.code.len();
                    self.chunk.patch_jump(skip_val_jump, after_val);
                }
                
                // Patch type check jump
                let after_checks = self.chunk.code.len();
                self.chunk.patch_jump(type_fail_jump, after_checks);
            }
            Pattern::Set(elements, span) => {
                // 1. Check if it is a set
                self.chunk.emit(Instruction::IsSet(success_reg, scrut_reg), 0);
                
                // Jump to end if not a set
                let type_fail_jump = self.chunk.emit(Instruction::JumpIfFalse(success_reg, 0), 0);
                
                for elem_pat in elements {
                    // We need to compile the pattern to a value to check existence
                    // Only support literals and pinned variables for now
                    let val_reg = match elem_pat {
                        Pattern::Int(n, _) => {
                            let r = self.alloc_reg();
                            let idx = self.chunk.add_constant(Value::Int64(*n));
                            self.chunk.emit(Instruction::LoadConst(r, idx), 0);
                            r
                        }
                        Pattern::String(s, _) => {
                            let r = self.alloc_reg();
                            let idx = self.chunk.add_constant(Value::String(Arc::new(s.clone())));
                            self.chunk.emit(Instruction::LoadConst(r, idx), 0);
                            r
                        }
                        Pattern::Bool(b, _) => {
                            let r = self.alloc_reg();
                            if *b { self.chunk.emit(Instruction::LoadTrue(r), 0); }
                            else { self.chunk.emit(Instruction::LoadFalse(r), 0); }
                            r
                        }
                        Pattern::Char(c, _) => {
                            let r = self.alloc_reg();
                            let idx = self.chunk.add_constant(Value::Char(*c));
                            self.chunk.emit(Instruction::LoadConst(r, idx), 0);
                            r
                        }
                        Pattern::Pin(expr, _) => {
                            self.compile_expr_tail(expr, false)?
                        }
                        _ => {
                            return Err(CompileError::InvalidPattern {
                                span: *span,
                                context: "Set patterns only support literals and pinned variables".to_string(),
                            });
                        }
                    };
                    
                    // Check if value exists in set
                    let exists_reg = self.alloc_reg();
                    self.chunk.emit(Instruction::SetContains(exists_reg, scrut_reg, val_reg), 0);
                    
                    // Update success_reg
                    self.chunk.emit(Instruction::And(success_reg, success_reg, exists_reg), 0);
                }
                
                // Patch type check jump
                let after_checks = self.chunk.code.len();
                self.chunk.patch_jump(type_fail_jump, after_checks);
            }
            Pattern::Pin(expr, span) => {
                // Pin pattern: evaluate the expression and compare with scrutinee
                // ^expected matches if scrutinee == expected
                let pin_val_reg = self.compile_expr_tail(expr, false)?;

                // Compare scrutinee with pinned value
                self.chunk.emit(
                    Instruction::Eq(success_reg, scrut_reg, pin_val_reg),
                    self.span_line(*span)
                );

                // Pin doesn't create bindings - just constrains the match
            }
            Pattern::Or(patterns, _span) => {
                // Or pattern: match if ANY sub-pattern matches
                // "a" | "b" | "c" -> matches if value is "a" OR "b" OR "c"

                // Start with false - will OR in each sub-pattern's result
                self.chunk.emit(Instruction::LoadFalse(success_reg), 0);

                for sub_pattern in patterns {
                    let (sub_success, sub_bindings) = self.compile_pattern_test(sub_pattern, scrut_reg)?;

                    // OR the sub-pattern's success with our overall success
                    self.chunk.emit(Instruction::Or(success_reg, success_reg, sub_success), 0);

                    // For now, collect bindings from all sub-patterns
                    // Note: In proper implementation, all alternatives should bind the same variables
                    // and we should only keep bindings from the first matching alternative
                    bindings.extend(sub_bindings);
                }
            }
            Pattern::Range(start, end, inclusive, span) => {
                // Range pattern: start..end (exclusive) or start..=end (inclusive)
                // Match if scrut >= start AND scrut < end (or <= for inclusive)
                let line = self.span_line(*span);

                // Load start constant
                let start_reg = self.alloc_reg();
                let start_idx = self.chunk.add_constant(Value::Int64(*start));
                self.chunk.emit(Instruction::LoadConst(start_reg, start_idx), line);

                // Load end constant
                let end_reg = self.alloc_reg();
                let end_idx = self.chunk.add_constant(Value::Int64(*end));
                self.chunk.emit(Instruction::LoadConst(end_reg, end_idx), line);

                // Check scrut >= start
                let ge_start = self.alloc_reg();
                self.chunk.emit(Instruction::GeInt(ge_start, scrut_reg, start_reg), line);

                // Check scrut < end (exclusive) or scrut <= end (inclusive)
                let le_end = self.alloc_reg();
                if *inclusive {
                    self.chunk.emit(Instruction::LeInt(le_end, scrut_reg, end_reg), line);
                } else {
                    self.chunk.emit(Instruction::LtInt(le_end, scrut_reg, end_reg), line);
                }

                // Both conditions must be true
                self.chunk.emit(Instruction::And(success_reg, ge_start, le_end), line);
            }
            _ => {
                return Err(CompileError::NotImplemented {
                    feature: format!("pattern: {:?}", pattern),
                    span: pattern.span(),
                });
            }
        }

        Ok((success_reg, bindings))
    }
}
