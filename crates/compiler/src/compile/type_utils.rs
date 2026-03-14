//! Type string parsing and conversion utilities.
//!
//! These are helper methods on Compiler that convert between string
//! representations of types and the Type enum used by the type system.

use super::*;

impl Compiler {
    /// Resolve an unqualified type name to its fully-qualified form.
    /// Checks module_path first, then imports, then falls back to the original name.
    pub(super) fn resolve_user_type_name(&self, name: &str) -> String {
        if name.contains('.') {
            return name.to_string();
        }
        // 1. Try qualifying with current module path
        if !self.module_path.is_empty() {
            let qualified = format!("{}.{}", self.module_path.join("."), name);
            if self.types.contains_key(&qualified) {
                return qualified;
            }
        }
        // 2. Try resolving through imports (e.g., "use Types.*" maps IntBox -> Types.IntBox)
        if let Some(imported) = self.imports.get(name) {
            if self.types.contains_key(imported) {
                return imported.clone();
            }
        }
        // 3. Fall back to original name
        name.to_string()
    }

    pub(super) fn type_name_to_type(&self, ty: &str) -> nostos_types::Type {
        let ty = ty.trim();

        // Resolve type aliases before any other processing
        // e.g., "Score" -> "Int", "Palette" -> "List[Color]", "Predicate" -> "(Int) -> Bool"
        let resolved = self.resolve_type_alias_name(ty);
        if resolved != ty {
            return self.type_name_to_type(&resolved);
        }

        // Handle list shorthand syntax: [a] means List[a]
        if ty.starts_with('[') && ty.ends_with(']') {
            let inner = &ty[1..ty.len() - 1].trim();
            let elem_type = self.type_name_to_type(inner);
            return nostos_types::Type::List(Box::new(elem_type));
        }

        // Handle tuple types: (A, B, C), grouped expressions: (S), but NOT function types: (A) -> B
        // Also exclude "()" which is Unit
        if ty.starts_with('(') && ty.ends_with(')') && ty != "()" {
            // First, check if the entire string is wrapped in one matching pair of parens.
            // e.g., "(S)" or "(a, ((a) -> a))" are wrapped, but "(a) -> (a, a)" is NOT
            // (the first '(' closes at position 2, not at the end).
            let mut depth = 0i32;
            let mut first_close_pos = 0usize;
            for (i, ch) in ty.char_indices() {
                match ch {
                    '(' => depth += 1,
                    ')' => {
                        depth -= 1;
                        if depth == 0 {
                            first_close_pos = i;
                            break;
                        }
                    }
                    _ => {}
                }
            }

            if first_close_pos == ty.len() - 1 {
                // The entire string is wrapped in one pair of parens
                let inner = &ty[1..ty.len() - 1];

                // Check for depth-0 comma to distinguish tuple from grouped expression
                let mut check_depth = 0i32;
                let mut has_depth0_comma = false;
                for ch in inner.chars() {
                    match ch {
                        '(' | '[' | '{' => check_depth += 1,
                        ')' | ']' | '}' => check_depth -= 1,
                        ',' if check_depth == 0 => { has_depth0_comma = true; break; }
                        _ => {}
                    }
                }

                if has_depth0_comma {
                    // Parse comma-separated elements at depth 0
                    let mut elems = Vec::new();
                    let mut current = String::new();
                    let mut elem_depth = 0;
                    for ch in inner.chars() {
                        match ch {
                            '(' | '[' | '{' => {
                                elem_depth += 1;
                                current.push(ch);
                            }
                            ')' | ']' | '}' => {
                                elem_depth -= 1;
                                current.push(ch);
                            }
                            ',' if elem_depth == 0 => {
                                if !current.trim().is_empty() {
                                    elems.push(self.type_name_to_type(current.trim()));
                                }
                                current.clear();
                            }
                            _ => current.push(ch),
                        }
                    }
                    if !current.trim().is_empty() {
                        elems.push(self.type_name_to_type(current.trim()));
                    }
                    if elems.len() >= 2 {
                        return nostos_types::Type::Tuple(elems);
                    }
                    // Single element in parens with comma (shouldn't happen, but handle gracefully)
                    if elems.len() == 1 {
                        return elems.into_iter().next().expect("single element");
                    }
                } else {
                    // No comma - single type in parens like "(S)" - unwrap
                    return self.type_name_to_type(inner);
                }
            }
            // If first_close_pos != ty.len()-1, the first '(' closes before the end.
            // This is something like "(A) -> (B, C)" - fall through to function type parsing.
        }

        // Handle function type syntax: "(params) -> ret" or "param -> ret"
        // This includes parenthesized function types like "(() -> a)"
        if let Some(func_type) = self.parse_function_type_string(ty) {
            return func_type;
        }

        // Check for parameterized type syntax: Name[Args]
        if let Some(bracket_pos) = ty.find('[') {
            if ty.ends_with(']') {
                let name = ty[..bracket_pos].trim();
                let args_str = &ty[bracket_pos + 1..ty.len() - 1];

                // Parse type arguments (handle nested brackets for things like List[List[Int]])
                let args = self.parse_type_args(args_str);
                // Handle built-in parameterized types
                return match name {
                    "List" if args.len() == 1 => {
                        nostos_types::Type::List(Box::new(args.into_iter().next().expect("List should have 1 type arg")))
                    }
                    "Array" if args.len() == 1 => {
                        nostos_types::Type::Array(Box::new(args.into_iter().next().expect("Array should have 1 type arg")))
                    }
                    "Set" if args.len() == 1 => {
                        nostos_types::Type::Set(Box::new(args.into_iter().next().expect("Set should have 1 type arg")))
                    }
                    "Map" if args.len() == 2 => {
                        let mut iter = args.into_iter();
                        let key = iter.next().expect("Map should have key type arg");
                        let val = iter.next().expect("Map should have value type arg");
                        nostos_types::Type::Map(Box::new(key), Box::new(val))
                    }
                    "IO" if args.len() == 1 => {
                        nostos_types::Type::IO(Box::new(args.into_iter().next().expect("IO should have 1 type arg")))
                    }
                    _ => {
                        // Resolve unqualified type names using module context and imports
                        let resolved_name = self.resolve_user_type_name(name);
                        nostos_types::Type::Named {
                            name: resolved_name,
                            args,
                        }
                    },
                };
            }
        }

        // Handle space-separated parameterized types like "Map k v", "List a", "Set a"
        // These come from BUILTINS signatures
        // Must split respecting parentheses, e.g., "Option (String, String)" -> ["Option", "(String, String)"]
        let parts = self.split_type_args_by_space(ty);
        if parts.len() >= 2 {
            let name = parts[0].as_str();
            let args: Vec<nostos_types::Type> = parts[1..].iter()
                .map(|arg| self.type_name_to_type(arg))
                .collect();

            return match name {
                "List" if args.len() == 1 => {
                    nostos_types::Type::List(Box::new(args.into_iter().next().expect("List should have 1 type arg")))
                }
                "Array" if args.len() == 1 => {
                    nostos_types::Type::Array(Box::new(args.into_iter().next().expect("Array should have 1 type arg")))
                }
                "Set" if args.len() == 1 => {
                    nostos_types::Type::Set(Box::new(args.into_iter().next().expect("Set should have 1 type arg")))
                }
                "Map" if args.len() == 2 => {
                    let mut iter = args.into_iter();
                    let key = iter.next().expect("Map should have key type arg");
                    let val = iter.next().expect("Map should have value type arg");
                    nostos_types::Type::Map(Box::new(key), Box::new(val))
                }
                "IO" if args.len() == 1 => {
                    nostos_types::Type::IO(Box::new(args.into_iter().next().expect("IO should have 1 type arg")))
                }
                _ => {
                    // Resolve unqualified type names using module context and imports
                    let resolved_name = self.resolve_user_type_name(name);
                    nostos_types::Type::Named {
                        name: resolved_name,
                        args,
                    }
                },
            };
        }

        match ty {
            "Int" | "Int64" => nostos_types::Type::Int,
            "Int8" => nostos_types::Type::Int8,
            "Int16" => nostos_types::Type::Int16,
            "Int32" => nostos_types::Type::Int32,
            "UInt8" => nostos_types::Type::UInt8,
            "UInt16" => nostos_types::Type::UInt16,
            "UInt32" => nostos_types::Type::UInt32,
            "UInt64" => nostos_types::Type::UInt64,
            "Float" | "Float64" => nostos_types::Type::Float,
            "Float32" => nostos_types::Type::Float32,
            "BigInt" => nostos_types::Type::BigInt,
            "Decimal" => nostos_types::Type::Decimal,
            "Bool" => nostos_types::Type::Bool,
            "Char" => nostos_types::Type::Char,
            "String" => nostos_types::Type::String,
            "Pid" => nostos_types::Type::Pid,
            "Ref" => nostos_types::Type::Ref,
            "()" | "Unit" => nostos_types::Type::Unit,
            "?" | "_" => nostos_types::Type::Var(u32::MAX), // Unknown/untyped param
            _ => {
                // "Self" is always a type parameter (from trait definitions)
                if ty == "Self" {
                    return nostos_types::Type::TypeParam("Self".to_string());
                }
                // Check if this is a type variable (single lowercase letter)
                // Type variables like 'a', 'b', 'c' are used in polymorphic signatures
                if ty.len() == 1 && ty.chars().next().map(|c| c.is_ascii_lowercase()).unwrap_or(false) {
                    // Convert to a consistent type variable ID based on the letter
                    // 'a' -> 1, 'b' -> 2, etc.
                    let var_id = (ty.chars().next().expect("single char type should have a char") as u32) - ('a' as u32) + 1;
                    nostos_types::Type::Var(var_id)
                } else if ty.len() == 1 && ty.chars().next().map(|c| c.is_ascii_uppercase()).unwrap_or(false) {
                    // Single uppercase letter could be a type parameter (T, U, V)
                    // OR a single-letter type name. Check if it's a known type first.
                    if self.types.contains_key(ty) {
                        nostos_types::Type::Named { name: ty.to_string(), args: vec![] }
                    } else {
                        // Also check module-qualified version (e.g., "B" -> "M.B")
                        let qualified = self.qualify_name(ty);
                        if qualified != ty && self.types.contains_key(&qualified) {
                            nostos_types::Type::Named { name: qualified, args: vec![] }
                        } else if let Some(qualified_name) = self.imports.get(ty) {
                            // Check imports (e.g., "use M.*" maps "B" -> "M.B")
                            if self.types.contains_key(qualified_name) {
                                nostos_types::Type::Named { name: qualified_name.clone(), args: vec![] }
                            } else {
                                nostos_types::Type::TypeParam(ty.to_string())
                            }
                        } else {
                            // Check if any registered type has this as its short name
                            // (handles cases where module types exist but aren't imported)
                            let found_qualified = self.types.keys()
                                .find(|k| {
                                    k.rsplit('.').next() == Some(ty)
                                })
                                .cloned();
                            if let Some(qualified_name) = found_qualified {
                                nostos_types::Type::Named { name: qualified_name, args: vec![] }
                            } else {
                                nostos_types::Type::TypeParam(ty.to_string())
                            }
                        }
                    }
                } else {
                    // Resolve using module context and imports
                    let resolved_name = self.resolve_user_type_name(ty);
                    nostos_types::Type::Named { name: resolved_name, args: vec![] }
                }
            }
        }
    }

    /// Add type aliases to an inference environment, with current module's types taking priority.
    /// This ensures that within-module type references like `Config` resolve to the local
    /// module's `mod_b.Config` rather than another module's `mod_a.Config`.
    pub(super) fn add_type_aliases_to_env(&self, env: &mut nostos_types::TypeEnv) {
        let mut type_names: Vec<_> = self.types.keys().collect();
        type_names.sort();
        let current_module_prefix = if self.module_path.is_empty() {
            String::new()
        } else {
            format!("{}.", self.module_path.join("."))
        };
        // First pass: add non-current-module aliases
        for type_name in &type_names {
            if let Some(dot_pos) = type_name.rfind('.') {
                let short_name = &type_name[dot_pos + 1..];
                if !current_module_prefix.is_empty() && type_name.starts_with(&current_module_prefix) {
                    continue;
                }
                if !env.type_aliases.contains_key(short_name) {
                    env.add_type_alias(short_name.to_string(), type_name.to_string());
                }
            }
        }
        // Second pass: current module's types override (take priority)
        if !current_module_prefix.is_empty() {
            for type_name in &type_names {
                if type_name.starts_with(&current_module_prefix) {
                    if let Some(dot_pos) = type_name.rfind('.') {
                        let short_name = &type_name[dot_pos + 1..];
                        env.add_type_alias(short_name.to_string(), type_name.to_string());
                    }
                }
            }
        }
    }

    /// Parse comma-separated type arguments, handling nested brackets and parentheses.
    pub(super) fn parse_type_args(&self, args_str: &str) -> Vec<nostos_types::Type> {
        let mut args = Vec::new();
        let mut current = String::new();
        let mut depth = 0;

        for ch in args_str.chars() {
            match ch {
                '[' | '(' => {
                    depth += 1;
                    current.push(ch);
                }
                ']' | ')' => {
                    depth -= 1;
                    current.push(ch);
                }
                ',' if depth == 0 => {
                    if !current.trim().is_empty() {
                        args.push(self.type_name_to_type(current.trim()));
                    }
                    current.clear();
                }
                _ => current.push(ch),
            }
        }

        // Don't forget the last argument
        if !current.trim().is_empty() {
            args.push(self.type_name_to_type(current.trim()));
        }

        args
    }

    /// Split type string by spaces while respecting parentheses and brackets.
    /// E.g., "Option (String, String)" -> ["Option", "(String, String)"]
    /// E.g., "Map k v" -> ["Map", "k", "v"]
    pub(super) fn split_type_args_by_space(&self, ty: &str) -> Vec<String> {
        let mut parts = Vec::new();
        let mut current = String::new();
        let mut depth = 0;

        for ch in ty.chars() {
            match ch {
                '(' | '[' | '{' => {
                    depth += 1;
                    current.push(ch);
                }
                ')' | ']' | '}' => {
                    depth -= 1;
                    current.push(ch);
                }
                ' ' if depth == 0 => {
                    if !current.is_empty() {
                        parts.push(current.clone());
                        current.clear();
                    }
                }
                _ => current.push(ch),
            }
        }

        if !current.is_empty() {
            parts.push(current);
        }

        parts
    }

    /// Parse a signature string like "String -> Int" or "Int -> String -> ()" into a FunctionType.
    /// This is used to register built-in functions for type inference.
    pub(super) fn parse_signature_string(&self, sig: &str) -> Option<nostos_types::FunctionType> {
        let sig = sig.trim();

        // Parse constraint syntax: "Show a, Eq b => a -> b -> Bool"
        let (type_params, sig_without_constraints) = if let Some(idx) = sig.find("=>") {
            let constraint_part = sig[..idx].trim();
            let sig_part = sig[idx + 2..].trim();

            // Parse constraints: "Show a, Eq b" or "Show a" or "HasField(0,c) a"
            // Split on commas at depth 0 (respecting parentheses in HasField(name,param))
            let mut type_param_map: std::collections::HashMap<String, Vec<String>> = std::collections::HashMap::new();

            let mut clauses = Vec::new();
            let mut depth = 0i32;
            let mut start = 0;
            for (i, ch) in constraint_part.char_indices() {
                match ch {
                    '(' | '[' => depth += 1,
                    ')' | ']' => depth -= 1,
                    ',' if depth == 0 => {
                        clauses.push(constraint_part[start..i].trim());
                        start = i + 1;
                    }
                    _ => {}
                }
            }
            clauses.push(constraint_part[start..].trim());

            for clause in clauses {
                // Parse "Show a" or "HasField(0,c) a"
                // Find last whitespace to split trait_name from type_param
                let parts: Vec<&str> = clause.rsplitn(2, ' ').collect();
                if parts.len() == 2 {
                    let type_param_name = parts[0].to_string();
                    let trait_name = parts[1].to_string();
                    type_param_map.entry(type_param_name).or_insert_with(Vec::new).push(trait_name);
                }
            }

            // Convert to TypeParam vec
            let mut type_params = Vec::new();
            for (name, constraints) in type_param_map {
                type_params.push(nostos_types::TypeParam {
                    name,
                    constraints,
                });
            }

            (type_params, sig_part)
        } else {
            (vec![], sig)
        };

        // Split by " -> " to get parameter and return types
        // Need to be careful with nested types like "Map k v -> k"
        let parts: Vec<&str> = self.split_arrow_types(sig_without_constraints);

        if parts.is_empty() {
            return None;
        }

        // Last part is the return type, rest are parameters
        let ret_str = parts.last()?;
        let param_strs = &parts[..parts.len() - 1];

        // Filter out "()" which means no params in signature syntax (e.g., "() -> Pid")
        let params: Vec<nostos_types::Type> = param_strs
            .iter()
            .filter(|s| s.trim() != "()")
            .map(|s| self.type_name_to_type(s.trim()))
            .collect();

        let ret = self.type_name_to_type(ret_str.trim());

        Some(nostos_types::FunctionType { required_params: None,
            type_params,
            params,
            ret: Box::new(ret),
            var_bounds: vec![],
        })
    }

    /// Split a signature string by " -> " while respecting nested brackets.
    pub(super) fn split_arrow_types<'a>(&self, sig: &'a str) -> Vec<&'a str> {
        let mut parts = Vec::new();
        let mut depth: i32 = 0;
        let mut start = 0;
        let bytes = sig.as_bytes();
        let mut i = 0;

        while i < bytes.len() {
            match bytes[i] {
                b'[' | b'(' | b'{' => depth += 1,
                b']' | b')' | b'}' => depth = (depth - 1).max(0),
                b'-' if depth == 0 && i + 2 < bytes.len() && bytes[i + 1] == b'>' => {
                    // Found " -> " at depth 0
                    let part = &sig[start..i].trim();
                    if !part.is_empty() {
                        parts.push(*part);
                    }
                    i += 2; // Skip "->"
                    // Skip whitespace after ->
                    while i < bytes.len() && bytes[i] == b' ' {
                        i += 1;
                    }
                    start = i;
                    continue;
                }
                _ => {}
            }
            i += 1;
        }

        // Don't forget the last part (return type)
        let last = sig[start..].trim();
        if !last.is_empty() {
            parts.push(last);
        }

        parts
    }

    /// Parse a function type string like "() -> Int", "a -> b", or "(() -> a)".
    /// Returns None if the string doesn't represent a function type.
    pub(super) fn parse_function_type_string(&self, ty: &str) -> Option<nostos_types::Type> {
        let ty = ty.trim();

        // Handle parenthesized type: if entire string is wrapped in parens, unwrap and recurse
        // But be careful: "()" is Unit, not an empty paren group
        // And "(a, b)" is a tuple, not a paren group
        if ty.starts_with('(') && ty.ends_with(')') && ty != "()" {
            // Check if the parens are balanced and wrap the entire expression
            let inner = &ty[1..ty.len() - 1];
            let mut depth = 0;
            let mut is_wrapped = true;
            for (i, c) in inner.char_indices() {
                match c {
                    '(' | '[' | '{' => depth += 1,
                    ')' | ']' | '}' => {
                        depth -= 1;
                        if depth < 0 {
                            is_wrapped = false;
                            break;
                        }
                    }
                    ',' if depth == 0 => {
                        // Found a comma at depth 0 - this is a tuple, not a wrapped type
                        is_wrapped = false;
                        break;
                    }
                    _ => {}
                }
                // If we find "->" at depth 0, we need to check if it's in the middle or at the end
                if depth == 0 && i + 2 < inner.len() && &inner[i..i+2] == "->" {
                    // There's an arrow inside, so this is a wrapped function type
                    // Continue checking for balanced parens
                }
            }

            // If the outer parens wrap the entire expression and it's balanced
            if is_wrapped && depth == 0 {
                // Try to parse the inner content as a function type
                if let Some(inner_type) = self.parse_function_type_string(inner) {
                    return Some(inner_type);
                }
                // If inner isn't a function type, the outer parens might just be grouping
                // Fall through to check for arrow at this level
            }
        }

        // Look for " -> " at depth 0 to identify function types
        let mut depth = 0;
        let bytes = ty.as_bytes();
        for i in 0..bytes.len() {
            match bytes[i] {
                b'(' | b'[' | b'{' => depth += 1,
                b')' | b']' | b'}' => depth = (depth - 1).max(0),
                b'-' if depth == 0 && i + 1 < bytes.len() && bytes[i + 1] == b'>' => {
                    // Found " -> " at depth 0 - this is a function type
                    let params_str = ty[..i].trim();
                    let ret_str = ty[i + 2..].trim();

                    // Parse parameter types
                    let params = if params_str == "()" || params_str.is_empty() {
                        // No parameters
                        vec![]
                    } else if params_str.starts_with('(') && params_str.ends_with(')') {
                        // Multiple params or single param in parens: "(a, b)" or "(a)"
                        let inner = &params_str[1..params_str.len() - 1];
                        self.parse_type_args(inner)
                    } else {
                        // Single param without parens: "a"
                        vec![self.type_name_to_type(params_str)]
                    };

                    let ret = self.type_name_to_type(ret_str);

                    return Some(nostos_types::Type::Function(nostos_types::FunctionType { required_params: None,
                        type_params: vec![],
                        params,
                        ret: Box::new(ret),
                        var_bounds: vec![],
                    }));
                }
                _ => {}
            }
        }

        None
    }
}
