//! Type inference for Nostos.
//!
//! Implements Hindley-Milner type inference with extensions for:
//! - Records (structural typing)
//! - Variants (sum types)
//! - Traits (type class constraints)
//!
//! The algorithm works in two phases:
//! 1. Generate constraints by walking the AST
//! 2. Solve constraints through unification

use crate::{Constructor, FunctionType, RecordType, Type, TypeDef, TypeError, TypeEnv};
use nostos_syntax::ast::{
    BinOp, Binding, Expr, FnClause, FnDef, Item, MatchArm, Module, Pattern, RecordField,
    RecordPatternField, Stmt, TypeExpr, UnaryOp, VariantPatternFields,
};
use std::collections::HashMap;

/// A type constraint generated during inference.
#[derive(Debug, Clone)]
pub enum Constraint {
    /// Two types must be equal
    Equal(Type, Type),
    /// A type must implement a trait
    HasTrait(Type, String),
    /// A type must have a field
    HasField(Type, String, Type),
}

/// Type inference context.
pub struct InferCtx<'a> {
    pub env: &'a mut TypeEnv,
    pub constraints: Vec<Constraint>,
}

impl<'a> InferCtx<'a> {
    pub fn new(env: &'a mut TypeEnv) -> Self {
        Self {
            env,
            constraints: Vec::new(),
        }
    }

    /// Generate a fresh type variable.
    pub fn fresh(&mut self) -> Type {
        self.env.fresh_var()
    }

    /// Add an equality constraint.
    pub fn unify(&mut self, t1: Type, t2: Type) {
        self.constraints.push(Constraint::Equal(t1, t2));
    }

    /// Add a trait constraint.
    pub fn require_trait(&mut self, ty: Type, trait_name: &str) {
        self.constraints
            .push(Constraint::HasTrait(ty, trait_name.to_string()));
    }

    /// Add a field constraint.
    pub fn require_field(&mut self, ty: Type, field: &str, field_ty: Type) {
        self.constraints
            .push(Constraint::HasField(ty, field.to_string(), field_ty));
    }

    /// Solve all constraints through unification.
    pub fn solve(&mut self) -> Result<(), TypeError> {
        while let Some(constraint) = self.constraints.pop() {
            match constraint {
                Constraint::Equal(t1, t2) => {
                    self.unify_types(&t1, &t2)?;
                }
                Constraint::HasTrait(ty, trait_name) => {
                    let resolved = self.env.apply_subst(&ty);
                    if !matches!(resolved, Type::Var(_)) {
                        if !self.env.implements(&resolved, &trait_name) {
                            return Err(TypeError::MissingTraitImpl {
                                ty: resolved.display(),
                                trait_name,
                            });
                        }
                    }
                    // If it's still a variable, we defer the check
                }
                Constraint::HasField(ty, field, expected_ty) => {
                    let resolved = self.env.apply_subst(&ty);
                    match &resolved {
                        Type::Record(rec) => {
                            if let Some((_, actual_ty, _)) =
                                rec.fields.iter().find(|(n, _, _)| n == &field)
                            {
                                self.unify_types(actual_ty, &expected_ty)?;
                            } else {
                                return Err(TypeError::NoSuchField {
                                    ty: resolved.display(),
                                    field,
                                });
                            }
                        }
                        Type::Var(_) => {
                            // Defer: we don't know the type yet
                            self.constraints
                                .push(Constraint::HasField(resolved, field, expected_ty));
                        }
                        _ => {
                            return Err(TypeError::NoSuchField {
                                ty: resolved.display(),
                                field,
                            });
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// Unify two types, updating the substitution.
    fn unify_types(&mut self, t1: &Type, t2: &Type) -> Result<(), TypeError> {
        let t1 = self.env.apply_subst(t1);
        let t2 = self.env.apply_subst(t2);

        match (&t1, &t2) {
            // Same type - nothing to do
            _ if t1 == t2 => Ok(()),

            // Variable on the left
            (Type::Var(id), _) => {
                if t2.contains_var(*id) {
                    return Err(TypeError::OccursCheck(
                        format!("?{}", id),
                        t2.display(),
                    ));
                }
                self.env.substitution.insert(*id, t2);
                Ok(())
            }

            // Variable on the right
            (_, Type::Var(id)) => {
                if t1.contains_var(*id) {
                    return Err(TypeError::OccursCheck(
                        format!("?{}", id),
                        t1.display(),
                    ));
                }
                self.env.substitution.insert(*id, t1);
                Ok(())
            }

            // Tuples
            (Type::Tuple(elems1), Type::Tuple(elems2)) => {
                if elems1.len() != elems2.len() {
                    return Err(TypeError::ArityMismatch {
                        expected: elems1.len(),
                        found: elems2.len(),
                    });
                }
                for (e1, e2) in elems1.iter().zip(elems2.iter()) {
                    self.unify_types(e1, e2)?;
                }
                Ok(())
            }

            // Lists
            (Type::List(elem1), Type::List(elem2)) => self.unify_types(elem1, elem2),

            // Arrays
            (Type::Array(elem1), Type::Array(elem2)) => self.unify_types(elem1, elem2),

            // Sets
            (Type::Set(elem1), Type::Set(elem2)) => self.unify_types(elem1, elem2),

            // Maps
            (Type::Map(k1, v1), Type::Map(k2, v2)) => {
                self.unify_types(k1, k2)?;
                self.unify_types(v1, v2)
            }

            // Functions
            (Type::Function(f1), Type::Function(f2)) => {
                if f1.params.len() != f2.params.len() {
                    return Err(TypeError::ArityMismatch {
                        expected: f1.params.len(),
                        found: f2.params.len(),
                    });
                }
                for (p1, p2) in f1.params.iter().zip(f2.params.iter()) {
                    self.unify_types(p1, p2)?;
                }
                self.unify_types(&f1.ret, &f2.ret)
            }

            // Records (structural)
            (Type::Record(rec1), Type::Record(rec2)) => {
                // Build field maps
                let fields1: HashMap<_, _> = rec1
                    .fields
                    .iter()
                    .map(|(n, t, m)| (n.clone(), (t.clone(), *m)))
                    .collect();
                let fields2: HashMap<_, _> = rec2
                    .fields
                    .iter()
                    .map(|(n, t, m)| (n.clone(), (t.clone(), *m)))
                    .collect();

                // Check all fields in rec1 exist in rec2
                for (name, (ty1, _)) in &fields1 {
                    if let Some((ty2, _)) = fields2.get(name) {
                        self.unify_types(ty1, ty2)?;
                    } else {
                        return Err(TypeError::MissingField(name.clone()));
                    }
                }

                // Check no extra fields in rec2
                for name in fields2.keys() {
                    if !fields1.contains_key(name) {
                        return Err(TypeError::ExtraField(name.clone()));
                    }
                }

                Ok(())
            }

            // Named types with same name
            (
                Type::Named { name: n1, args: a1 },
                Type::Named { name: n2, args: a2 },
            ) if n1 == n2 => {
                if a1.len() != a2.len() {
                    return Err(TypeError::TypeArityMismatch {
                        expected: a1.len(),
                        found: a2.len(),
                    });
                }
                for (arg1, arg2) in a1.iter().zip(a2.iter()) {
                    self.unify_types(arg1, arg2)?;
                }
                Ok(())
            }

            // IO
            (Type::IO(inner1), Type::IO(inner2)) => self.unify_types(inner1, inner2),

            // Mismatch
            _ => Err(TypeError::UnificationFailed(t1.display(), t2.display())),
        }
    }

    // =========================================================================
    // AST Type Inference
    // =========================================================================

    /// Convert an AST TypeExpr to an internal Type.
    pub fn type_from_ast(&mut self, ty: &TypeExpr) -> Type {
        match ty {
            TypeExpr::Name(ident) => {
                let name = &ident.node;
                match name.as_str() {
                    "Int" => Type::Int,
                    "Float" => Type::Float,
                    "Bool" => Type::Bool,
                    "Char" => Type::Char,
                    "String" => Type::String,
                    "Pid" => Type::Pid,
                    "Ref" => Type::Ref,
                    "Never" => Type::Never,
                    _ => Type::Named {
                        name: name.clone(),
                        args: vec![],
                    },
                }
            }
            TypeExpr::Generic(name, args) => {
                let type_args: Vec<_> = args.iter().map(|a| self.type_from_ast(a)).collect();
                let name_str = &name.node;
                match name_str.as_str() {
                    "List" if type_args.len() == 1 => Type::List(Box::new(type_args[0].clone())),
                    "Array" if type_args.len() == 1 => Type::Array(Box::new(type_args[0].clone())),
                    "Set" if type_args.len() == 1 => Type::Set(Box::new(type_args[0].clone())),
                    "Map" if type_args.len() == 2 => Type::Map(
                        Box::new(type_args[0].clone()),
                        Box::new(type_args[1].clone()),
                    ),
                    "IO" if type_args.len() == 1 => Type::IO(Box::new(type_args[0].clone())),
                    _ => Type::Named {
                        name: name_str.clone(),
                        args: type_args,
                    },
                }
            }
            TypeExpr::Function(params, ret) => {
                let param_types: Vec<_> = params.iter().map(|p| self.type_from_ast(p)).collect();
                let ret_type = self.type_from_ast(ret);
                Type::Function(FunctionType {
                    type_params: vec![],
                    params: param_types,
                    ret: Box::new(ret_type),
                })
            }
            TypeExpr::Record(fields) => {
                let field_types: Vec<_> = fields
                    .iter()
                    .map(|(name, ty)| (name.node.clone(), self.type_from_ast(ty), false))
                    .collect();
                Type::Record(RecordType {
                    name: None,
                    fields: field_types,
                })
            }
            TypeExpr::Tuple(elems) => {
                let elem_types: Vec<_> = elems.iter().map(|e| self.type_from_ast(e)).collect();
                Type::Tuple(elem_types)
            }
            TypeExpr::Unit => Type::Unit,
        }
    }

    /// Infer the type of an expression.
    pub fn infer_expr(&mut self, expr: &Expr) -> Result<Type, TypeError> {
        match expr {
            // Literals - integers
            Expr::Int(_, _) => Ok(Type::Int),
            Expr::Int8(_, _) => Ok(Type::Int8),
            Expr::Int16(_, _) => Ok(Type::Int16),
            Expr::Int32(_, _) => Ok(Type::Int32),
            // Unsigned integers
            Expr::UInt8(_, _) => Ok(Type::UInt8),
            Expr::UInt16(_, _) => Ok(Type::UInt16),
            Expr::UInt32(_, _) => Ok(Type::UInt32),
            Expr::UInt64(_, _) => Ok(Type::UInt64),
            // Floats
            Expr::Float(_, _) => Ok(Type::Float),
            Expr::Float32(_, _) => Ok(Type::Float32),
            // Arbitrary precision
            Expr::BigInt(_, _) => Ok(Type::BigInt),
            Expr::Decimal(_, _) => Ok(Type::Decimal),
            // Other literals
            Expr::Bool(_, _) => Ok(Type::Bool),
            Expr::Char(_, _) => Ok(Type::Char),
            Expr::String(_, _) => Ok(Type::String),
            Expr::Unit(_) => Ok(Type::Unit),

            // Variable lookup
            Expr::Var(ident) => {
                let name = &ident.node;
                // First check if it's a constructor
                if let Some(ty) = self.lookup_constructor(name) {
                    return Ok(ty);
                }
                // Then check bindings
                if let Some((ty, _)) = self.env.lookup(name) {
                    Ok(ty.clone())
                } else if let Some(sig) = self.env.functions.get(name) {
                    Ok(Type::Function(sig.clone()))
                } else {
                    Err(TypeError::UnknownIdent(name.clone()))
                }
            }

            // Binary operations
            Expr::BinOp(left, op, right, _) => self.infer_binop(left, *op, right),

            // Unary operations
            Expr::UnaryOp(op, operand, _) => {
                let operand_ty = self.infer_expr(operand)?;
                match op {
                    UnaryOp::Neg => {
                        // Neg works on Int or Float
                        let result_ty = self.fresh();
                        self.unify(operand_ty.clone(), result_ty.clone());
                        // We need to check it's numeric at solve time
                        Ok(result_ty)
                    }
                    UnaryOp::Not => {
                        self.unify(operand_ty, Type::Bool);
                        Ok(Type::Bool)
                    }
                }
            }

            // Function call
            Expr::Call(func, args, _) => {
                let func_ty = self.infer_expr(func)?;
                let mut arg_types = Vec::new();
                for arg in args {
                    arg_types.push(self.infer_expr(arg)?);
                }

                let ret_ty = self.fresh();
                let expected_func_ty = Type::Function(FunctionType {
                    type_params: vec![],
                    params: arg_types,
                    ret: Box::new(ret_ty.clone()),
                });

                self.unify(func_ty, expected_func_ty);
                Ok(ret_ty)
            }

            // Lambda
            Expr::Lambda(params, body, _) => {
                // Save current bindings
                let saved_bindings = self.env.bindings.clone();

                let mut param_types = Vec::new();
                for param in params {
                    let param_ty = self.fresh();
                    self.infer_pattern(param, &param_ty)?;
                    param_types.push(param_ty);
                }

                let body_ty = self.infer_expr(body)?;

                // Restore bindings
                self.env.bindings = saved_bindings;

                Ok(Type::Function(FunctionType {
                    type_params: vec![],
                    params: param_types,
                    ret: Box::new(body_ty),
                }))
            }

            // If expression
            Expr::If(cond, then_branch, else_branch, _) => {
                let cond_ty = self.infer_expr(cond)?;
                self.unify(cond_ty, Type::Bool);

                let then_ty = self.infer_expr(then_branch)?;
                let else_ty = self.infer_expr(else_branch)?;
                self.unify(then_ty.clone(), else_ty);

                Ok(then_ty)
            }

            // Match expression
            Expr::Match(scrutinee, arms, _) => {
                let scrutinee_ty = self.infer_expr(scrutinee)?;
                let result_ty = self.fresh();

                for arm in arms {
                    self.infer_match_arm(arm, &scrutinee_ty, &result_ty)?;
                }

                Ok(result_ty)
            }

            // Tuple
            Expr::Tuple(elems, _) => {
                let mut elem_types = Vec::new();
                for elem in elems {
                    elem_types.push(self.infer_expr(elem)?);
                }
                Ok(Type::Tuple(elem_types))
            }

            // List
            Expr::List(elems, tail, _) => {
                let elem_ty = self.fresh();

                for elem in elems {
                    let ty = self.infer_expr(elem)?;
                    self.unify(ty, elem_ty.clone());
                }

                if let Some(tail_expr) = tail {
                    let tail_ty = self.infer_expr(tail_expr)?;
                    self.unify(tail_ty, Type::List(Box::new(elem_ty.clone())));
                }

                Ok(Type::List(Box::new(elem_ty)))
            }

            // Map
            Expr::Map(pairs, _) => {
                let key_ty = self.fresh();
                let val_ty = self.fresh();

                for (k, v) in pairs {
                    let k_ty = self.infer_expr(k)?;
                    let v_ty = self.infer_expr(v)?;
                    self.unify(k_ty, key_ty.clone());
                    self.unify(v_ty, val_ty.clone());
                }

                self.require_trait(key_ty.clone(), "Hash");
                self.require_trait(key_ty.clone(), "Eq");

                Ok(Type::Map(Box::new(key_ty), Box::new(val_ty)))
            }

            // Set
            Expr::Set(elems, _) => {
                let elem_ty = self.fresh();

                for elem in elems {
                    let ty = self.infer_expr(elem)?;
                    self.unify(ty, elem_ty.clone());
                }

                self.require_trait(elem_ty.clone(), "Hash");
                self.require_trait(elem_ty.clone(), "Eq");

                Ok(Type::Set(Box::new(elem_ty)))
            }

            // Record construction
            Expr::Record(name, fields, _) => {
                let type_name = &name.node;
                if let Some(TypeDef::Record {
                    fields: def_fields, ..
                }) = self.env.lookup_type(type_name).cloned()
                {
                    let mut provided = HashMap::new();
                    for field in fields {
                        match field {
                            RecordField::Positional(expr) => {
                                // Positional args match in order
                                let ty = self.infer_expr(expr)?;
                                let idx = provided.len();
                                if idx < def_fields.len() {
                                    let (fname, fty, _) = &def_fields[idx];
                                    self.unify(ty, fty.clone());
                                    provided.insert(fname.clone(), ());
                                }
                            }
                            RecordField::Named(fname, expr) => {
                                let ty = self.infer_expr(expr)?;
                                if let Some((_, fty, _)) =
                                    def_fields.iter().find(|(n, _, _)| n == &fname.node)
                                {
                                    self.unify(ty, fty.clone());
                                    provided.insert(fname.node.clone(), ());
                                } else {
                                    return Err(TypeError::NoSuchField {
                                        ty: type_name.clone(),
                                        field: fname.node.clone(),
                                    });
                                }
                            }
                        }
                    }

                    Ok(Type::Named {
                        name: type_name.clone(),
                        args: vec![],
                    })
                } else {
                    Err(TypeError::UnknownType(type_name.clone()))
                }
            }

            // Field access
            Expr::FieldAccess(expr, field, _) => {
                let expr_ty = self.infer_expr(expr)?;
                let field_ty = self.fresh();
                self.require_field(expr_ty, &field.node, field_ty.clone());
                Ok(field_ty)
            }

            // Index access
            Expr::Index(container, index, _) => {
                let container_ty = self.infer_expr(container)?;
                let index_ty = self.infer_expr(index)?;
                let elem_ty = self.fresh();

                // Container could be List, Array, Map, or String
                let list_ty = Type::List(Box::new(elem_ty.clone()));
                let _array_ty = Type::Array(Box::new(elem_ty.clone()));

                // For now, assume Int index for lists/arrays
                // This is simplified - a full impl would handle Map[K, V]
                self.unify(index_ty, Type::Int);

                // Try to unify with List or Array
                // We'll handle this via constraint solving
                self.constraints.push(Constraint::Equal(
                    container_ty,
                    list_ty,
                ));

                Ok(elem_ty)
            }

            // Block
            Expr::Block(stmts, _) => {
                let saved_bindings = self.env.bindings.clone();
                let mut result_ty = Type::Unit;

                for (i, stmt) in stmts.iter().enumerate() {
                    let is_last = i == stmts.len() - 1;
                    match stmt {
                        Stmt::Expr(expr) => {
                            let ty = self.infer_expr(expr)?;
                            if is_last {
                                result_ty = ty;
                            }
                        }
                        Stmt::Let(binding) => {
                            self.infer_binding(binding)?;
                        }
                        Stmt::Assign(target, expr, _) => {
                            let expr_ty = self.infer_expr(expr)?;
                            match target {
                                nostos_syntax::ast::AssignTarget::Var(ident) => {
                                    if let Some((var_ty, mutable)) = self.env.lookup(&ident.node) {
                                        if !mutable {
                                            return Err(TypeError::ImmutableBinding(
                                                ident.node.clone(),
                                            ));
                                        }
                                        self.unify(expr_ty, var_ty.clone());
                                    } else {
                                        return Err(TypeError::UnknownIdent(ident.node.clone()));
                                    }
                                }
                                nostos_syntax::ast::AssignTarget::Field(obj, field) => {
                                    let obj_ty = self.infer_expr(obj)?;
                                    self.require_field(obj_ty, &field.node, expr_ty);
                                }
                                nostos_syntax::ast::AssignTarget::Index(obj, idx) => {
                                    let _obj_ty = self.infer_expr(obj)?;
                                    let _idx_ty = self.infer_expr(idx)?;
                                    // Index assignment - simplified for now
                                }
                            }
                        }
                    }
                }

                self.env.bindings = saved_bindings;
                Ok(result_ty)
            }

            // Method call
            Expr::MethodCall(receiver, method, args, _) => {
                let receiver_ty = self.infer_expr(receiver)?;
                let mut arg_types = vec![receiver_ty.clone()];
                for arg in args {
                    arg_types.push(self.infer_expr(arg)?);
                }

                // Look up method based on receiver type
                let ret_ty = self.fresh();
                // For now, we just return a fresh var
                // A full impl would look up method in trait/impl
                let _ = method;
                Ok(ret_ty)
            }

            // Try expression (error propagation)
            Expr::Try_(inner, _) => {
                let inner_ty = self.infer_expr(inner)?;
                let ok_ty = self.fresh();
                let err_ty = self.fresh();

                // inner should be Result[T, E]
                self.unify(
                    inner_ty,
                    Type::Named {
                        name: "Result".to_string(),
                        args: vec![ok_ty.clone(), err_ty],
                    },
                );

                Ok(ok_ty)
            }

            // Do block (IO monad)
            Expr::Do(stmts, _) => {
                let saved_bindings = self.env.bindings.clone();
                let mut result_ty = Type::Unit;

                for (i, stmt) in stmts.iter().enumerate() {
                    let is_last = i == stmts.len() - 1;
                    match stmt {
                        nostos_syntax::ast::DoStmt::Bind(pat, expr) => {
                            let expr_ty = self.infer_expr(expr)?;
                            // expr should have type IO[T]
                            let inner_ty = self.fresh();
                            self.unify(expr_ty, Type::IO(Box::new(inner_ty.clone())));
                            self.infer_pattern(pat, &inner_ty)?;
                        }
                        nostos_syntax::ast::DoStmt::Expr(expr) => {
                            let ty = self.infer_expr(expr)?;
                            if is_last {
                                // Last expr should be IO[T]
                                let inner_ty = self.fresh();
                                self.unify(ty, Type::IO(Box::new(inner_ty.clone())));
                                result_ty = inner_ty;
                            }
                        }
                    }
                }

                self.env.bindings = saved_bindings;
                Ok(Type::IO(Box::new(result_ty)))
            }

            // Spawn
            Expr::Spawn(_, func, args, _) => {
                let func_ty = self.infer_expr(func)?;
                let mut arg_types = Vec::new();
                for arg in args {
                    arg_types.push(self.infer_expr(arg)?);
                }

                let ret_ty = self.fresh();
                let expected = Type::Function(FunctionType {
                    type_params: vec![],
                    params: arg_types,
                    ret: Box::new(ret_ty),
                });
                self.unify(func_ty, expected);

                Ok(Type::Pid)
            }

            // Send
            Expr::Send(pid, msg, _) => {
                let pid_ty = self.infer_expr(pid)?;
                let _msg_ty = self.infer_expr(msg)?;
                self.unify(pid_ty, Type::Pid);
                Ok(Type::Unit)
            }

            // Receive
            Expr::Receive(arms, timeout, _) => {
                let result_ty = self.fresh();
                let msg_ty = self.fresh();

                for arm in arms {
                    self.infer_match_arm(arm, &msg_ty, &result_ty)?;
                }

                if let Some((timeout_expr, timeout_body)) = timeout {
                    let timeout_ty = self.infer_expr(timeout_expr)?;
                    self.unify(timeout_ty, Type::Int);
                    let body_ty = self.infer_expr(timeout_body)?;
                    self.unify(body_ty, result_ty.clone());
                }

                Ok(result_ty)
            }

            // Record update
            Expr::RecordUpdate(name, base, fields, _) => {
                let base_ty = self.infer_expr(base)?;
                let expected_ty = Type::Named {
                    name: name.node.clone(),
                    args: vec![],
                };
                self.unify(base_ty, expected_ty.clone());

                for field in fields {
                    if let RecordField::Named(fname, expr) = field {
                        let field_ty = self.infer_expr(expr)?;
                        self.require_field(expected_ty.clone(), &fname.node, field_ty);
                    }
                }

                Ok(expected_ty)
            }

            // Try/catch
            Expr::Try(body, catch_arms, finally, _) => {
                let body_ty = self.infer_expr(body)?;
                let result_ty = self.fresh();
                self.unify(body_ty.clone(), result_ty.clone());

                let err_ty = self.fresh();
                for arm in catch_arms {
                    self.infer_match_arm(arm, &err_ty, &result_ty)?;
                }

                if let Some(finally_expr) = finally {
                    let _ = self.infer_expr(finally_expr)?;
                }

                Ok(result_ty)
            }

            // Quote/Splice - macro-related
            Expr::Quote(inner, _) => {
                let _ = self.infer_expr(inner)?;
                Ok(Type::Named {
                    name: "Expr".to_string(),
                    args: vec![],
                })
            }
            Expr::Splice(inner, _) => {
                let inner_ty = self.infer_expr(inner)?;
                self.unify(
                    inner_ty,
                    Type::Named {
                        name: "Expr".to_string(),
                        args: vec![],
                    },
                );
                // Result depends on context
                Ok(self.fresh())
            }

            // While loop: condition must be Bool, returns Unit
            Expr::While(cond, body, _) => {
                let cond_ty = self.infer_expr(cond)?;
                self.unify(cond_ty, Type::Bool);
                let _ = self.infer_expr(body)?;
                Ok(Type::Unit)
            }

            // For loop: iterates from start to end (both Int), returns Unit
            Expr::For(var, start, end, body, _) => {
                let start_ty = self.infer_expr(start)?;
                let end_ty = self.infer_expr(end)?;
                self.unify(start_ty, Type::Int);
                self.unify(end_ty, Type::Int);
                // Add loop variable to scope
                let saved_bindings = self.env.bindings.clone();
                self.env.bind(var.node.clone(), Type::Int, false);
                let _ = self.infer_expr(body)?;
                self.env.bindings = saved_bindings;
                Ok(Type::Unit)
            }

            // Break: optional value, returns Never (doesn't produce a value normally)
            Expr::Break(value, _) => {
                if let Some(val) = value {
                    let _ = self.infer_expr(val)?;
                }
                Ok(Type::Unit) // Break doesn't continue execution normally
            }

            // Continue: returns Never (doesn't produce a value normally)
            Expr::Continue(_) => Ok(Type::Unit),
        }
    }

    /// Infer types for a binary operation.
    fn infer_binop(&mut self, left: &Expr, op: BinOp, right: &Expr) -> Result<Type, TypeError> {
        let left_ty = self.infer_expr(left)?;
        let right_ty = self.infer_expr(right)?;

        match op {
            // Arithmetic: both operands same numeric type, result same type
            BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Mod | BinOp::Pow => {
                self.unify(left_ty.clone(), right_ty);
                Ok(left_ty)
            }

            // Comparison: both operands same type with Ord, result Bool
            BinOp::Lt | BinOp::Gt | BinOp::LtEq | BinOp::GtEq => {
                self.unify(left_ty.clone(), right_ty);
                self.require_trait(left_ty, "Ord");
                Ok(Type::Bool)
            }

            // Equality: both operands same type with Eq, result Bool
            BinOp::Eq | BinOp::NotEq => {
                self.unify(left_ty.clone(), right_ty);
                self.require_trait(left_ty, "Eq");
                Ok(Type::Bool)
            }

            // Logical: both operands Bool, result Bool
            BinOp::And | BinOp::Or => {
                self.unify(left_ty, Type::Bool);
                self.unify(right_ty, Type::Bool);
                Ok(Type::Bool)
            }

            // Concat: lists or strings
            BinOp::Concat => {
                self.unify(left_ty.clone(), right_ty);
                Ok(left_ty)
            }

            // Pipe: f |> g means g(f)
            BinOp::Pipe => {
                let result_ty = self.fresh();
                let expected_func = Type::Function(FunctionType {
                    type_params: vec![],
                    params: vec![left_ty],
                    ret: Box::new(result_ty.clone()),
                });
                self.unify(right_ty, expected_func);
                Ok(result_ty)
            }
        }
    }

    /// Infer types for a pattern, binding variables in the environment.
    pub fn infer_pattern(&mut self, pat: &Pattern, expected: &Type) -> Result<(), TypeError> {
        match pat {
            Pattern::Wildcard(_) => Ok(()),

            Pattern::Var(ident) => {
                self.env.bind(ident.node.clone(), expected.clone(), false);
                Ok(())
            }

            Pattern::Int(_, _) => {
                self.unify(expected.clone(), Type::Int);
                Ok(())
            }

            Pattern::Int8(_, _) => {
                self.unify(expected.clone(), Type::Int8);
                Ok(())
            }

            Pattern::Int16(_, _) => {
                self.unify(expected.clone(), Type::Int16);
                Ok(())
            }

            Pattern::Int32(_, _) => {
                self.unify(expected.clone(), Type::Int32);
                Ok(())
            }

            Pattern::UInt8(_, _) => {
                self.unify(expected.clone(), Type::UInt8);
                Ok(())
            }

            Pattern::UInt16(_, _) => {
                self.unify(expected.clone(), Type::UInt16);
                Ok(())
            }

            Pattern::UInt32(_, _) => {
                self.unify(expected.clone(), Type::UInt32);
                Ok(())
            }

            Pattern::UInt64(_, _) => {
                self.unify(expected.clone(), Type::UInt64);
                Ok(())
            }

            Pattern::BigInt(_, _) => {
                self.unify(expected.clone(), Type::BigInt);
                Ok(())
            }

            Pattern::Float(_, _) => {
                self.unify(expected.clone(), Type::Float);
                Ok(())
            }

            Pattern::Float32(_, _) => {
                self.unify(expected.clone(), Type::Float32);
                Ok(())
            }

            Pattern::Decimal(_, _) => {
                self.unify(expected.clone(), Type::Decimal);
                Ok(())
            }

            Pattern::String(_, _) => {
                self.unify(expected.clone(), Type::String);
                Ok(())
            }

            Pattern::Char(_, _) => {
                self.unify(expected.clone(), Type::Char);
                Ok(())
            }

            Pattern::Bool(_, _) => {
                self.unify(expected.clone(), Type::Bool);
                Ok(())
            }

            Pattern::Unit(_) => {
                self.unify(expected.clone(), Type::Unit);
                Ok(())
            }

            Pattern::Tuple(elems, _) => {
                let elem_types: Vec<_> = elems.iter().map(|_| self.fresh()).collect();
                self.unify(expected.clone(), Type::Tuple(elem_types.clone()));
                for (elem, ty) in elems.iter().zip(elem_types.iter()) {
                    self.infer_pattern(elem, ty)?;
                }
                Ok(())
            }

            Pattern::List(list_pat, _) => {
                let elem_ty = self.fresh();
                self.unify(expected.clone(), Type::List(Box::new(elem_ty.clone())));

                match list_pat {
                    nostos_syntax::ast::ListPattern::Empty => {}
                    nostos_syntax::ast::ListPattern::Cons(elems, tail) => {
                        for elem in elems {
                            self.infer_pattern(elem, &elem_ty)?;
                        }
                        if let Some(tail_pat) = tail {
                            self.infer_pattern(tail_pat, &Type::List(Box::new(elem_ty)))?;
                        }
                    }
                }
                Ok(())
            }

            Pattern::Record(fields, _) => {
                for field in fields {
                    match field {
                        RecordPatternField::Punned(name) => {
                            let field_ty = self.fresh();
                            self.require_field(expected.clone(), &name.node, field_ty.clone());
                            self.env.bind(name.node.clone(), field_ty, false);
                        }
                        RecordPatternField::Named(name, pat) => {
                            let field_ty = self.fresh();
                            self.require_field(expected.clone(), &name.node, field_ty.clone());
                            self.infer_pattern(pat, &field_ty)?;
                        }
                        RecordPatternField::Rest(_) => {}
                    }
                }
                Ok(())
            }

            Pattern::Variant(name, fields, _) => {
                // Look up constructor
                if let Some(ctor_ty) = self.lookup_constructor(&name.node) {
                    match fields {
                        VariantPatternFields::Unit => {
                            self.unify(expected.clone(), ctor_ty);
                        }
                        VariantPatternFields::Positional(pats) => {
                            // Constructor should be a function from fields to the type
                            if let Type::Function(f) = &ctor_ty {
                                for (pat, param_ty) in pats.iter().zip(f.params.iter()) {
                                    self.infer_pattern(pat, param_ty)?;
                                }
                                self.unify(expected.clone(), (*f.ret).clone());
                            } else {
                                self.unify(expected.clone(), ctor_ty);
                            }
                        }
                        VariantPatternFields::Named(fields) => {
                            for field in fields {
                                match field {
                                    RecordPatternField::Punned(fname) => {
                                        let field_ty = self.fresh();
                                        self.env.bind(fname.node.clone(), field_ty, false);
                                    }
                                    RecordPatternField::Named(fname, pat) => {
                                        let field_ty = self.fresh();
                                        self.infer_pattern(pat, &field_ty)?;
                                        let _ = fname;
                                    }
                                    RecordPatternField::Rest(_) => {}
                                }
                            }
                            self.unify(expected.clone(), ctor_ty);
                        }
                    }
                    Ok(())
                } else {
                    Err(TypeError::UnknownConstructor(name.node.clone()))
                }
            }

            Pattern::Or(pats, _) => {
                for pat in pats {
                    self.infer_pattern(pat, expected)?;
                }
                Ok(())
            }

            Pattern::Pin(expr, _) => {
                let expr_ty = self.infer_expr(expr)?;
                self.unify(expected.clone(), expr_ty);
                Ok(())
            }
        }
    }

    /// Look up a constructor and return its type.
    fn lookup_constructor(&mut self, name: &str) -> Option<Type> {
        // Search through all variant types for this constructor
        let types_clone = self.env.types.clone();
        for (type_name, def) in &types_clone {
            if let TypeDef::Variant { params, constructors } = def {
                for ctor in constructors {
                    let matches = match ctor {
                        Constructor::Unit(n) => n == name,
                        Constructor::Positional(n, _) => n == name,
                        Constructor::Named(n, _) => n == name,
                    };
                    if !matches {
                        continue;
                    }

                    // Create fresh type variables for each type parameter
                    let substitution: HashMap<String, Type> = params
                        .iter()
                        .map(|p| (p.name.clone(), self.fresh()))
                        .collect();

                    // Apply substitution to get instantiated types
                    let instantiate = |ty: &Type| -> Type {
                        Self::substitute_type_params(ty, &substitution)
                    };

                    // Get the type args for the result type
                    let type_args: Vec<Type> = params
                        .iter()
                        .map(|p| substitution.get(&p.name).cloned().unwrap_or_else(|| Type::TypeParam(p.name.clone())))
                        .collect();

                    let result_ty = Type::Named {
                        name: type_name.clone(),
                        args: type_args,
                    };

                    match ctor {
                        Constructor::Unit(_) => {
                            return Some(result_ty);
                        }
                        Constructor::Positional(_, ctor_params) => {
                            if ctor_params.is_empty() {
                                return Some(result_ty);
                            } else {
                                let instantiated_params: Vec<_> = ctor_params
                                    .iter()
                                    .map(instantiate)
                                    .collect();
                                return Some(Type::Function(FunctionType {
                                    type_params: vec![],
                                    params: instantiated_params,
                                    ret: Box::new(result_ty),
                                }));
                            }
                        }
                        Constructor::Named(_, fields) => {
                            let instantiated_params: Vec<_> = fields
                                .iter()
                                .map(|(_, ty)| instantiate(ty))
                                .collect();
                            if instantiated_params.is_empty() {
                                return Some(result_ty);
                            } else {
                                return Some(Type::Function(FunctionType {
                                    type_params: vec![],
                                    params: instantiated_params,
                                    ret: Box::new(result_ty),
                                }));
                            }
                        }
                    }
                }
            }
        }
        None
    }

    /// Substitute type parameters in a type with concrete types.
    fn substitute_type_params(ty: &Type, subst: &HashMap<String, Type>) -> Type {
        match ty {
            Type::TypeParam(name) => {
                subst.get(name).cloned().unwrap_or_else(|| ty.clone())
            }
            Type::Tuple(elems) => {
                Type::Tuple(elems.iter().map(|t| Self::substitute_type_params(t, subst)).collect())
            }
            Type::List(elem) => {
                Type::List(Box::new(Self::substitute_type_params(elem, subst)))
            }
            Type::Array(elem) => {
                Type::Array(Box::new(Self::substitute_type_params(elem, subst)))
            }
            Type::Set(elem) => {
                Type::Set(Box::new(Self::substitute_type_params(elem, subst)))
            }
            Type::Map(k, v) => {
                Type::Map(
                    Box::new(Self::substitute_type_params(k, subst)),
                    Box::new(Self::substitute_type_params(v, subst)),
                )
            }
            Type::Function(f) => {
                Type::Function(FunctionType {
                    type_params: f.type_params.clone(),
                    params: f.params.iter().map(|t| Self::substitute_type_params(t, subst)).collect(),
                    ret: Box::new(Self::substitute_type_params(&f.ret, subst)),
                })
            }
            Type::Named { name, args } => {
                Type::Named {
                    name: name.clone(),
                    args: args.iter().map(|t| Self::substitute_type_params(t, subst)).collect(),
                }
            }
            Type::IO(inner) => {
                Type::IO(Box::new(Self::substitute_type_params(inner, subst)))
            }
            Type::Record(rec) => {
                Type::Record(RecordType {
                    name: rec.name.clone(),
                    fields: rec.fields.iter().map(|(n, t, m)| {
                        (n.clone(), Self::substitute_type_params(t, subst), *m)
                    }).collect(),
                })
            }
            _ => ty.clone(),
        }
    }

    /// Infer types for a match arm.
    fn infer_match_arm(
        &mut self,
        arm: &MatchArm,
        scrutinee_ty: &Type,
        result_ty: &Type,
    ) -> Result<(), TypeError> {
        let saved_bindings = self.env.bindings.clone();

        // Infer pattern against scrutinee
        self.infer_pattern(&arm.pattern, scrutinee_ty)?;

        // Check guard if present
        if let Some(guard) = &arm.guard {
            let guard_ty = self.infer_expr(guard)?;
            self.unify(guard_ty, Type::Bool);
        }

        // Infer body
        let body_ty = self.infer_expr(&arm.body)?;
        self.unify(body_ty, result_ty.clone());

        self.env.bindings = saved_bindings;
        Ok(())
    }

    /// Infer types for a binding.
    pub fn infer_binding(&mut self, binding: &Binding) -> Result<Type, TypeError> {
        let value_ty = self.infer_expr(&binding.value)?;

        // If there's a type annotation, unify with it
        if let Some(ty_expr) = &binding.ty {
            let annotated_ty = self.type_from_ast(ty_expr);
            self.unify(value_ty.clone(), annotated_ty);
        }

        // Bind pattern
        self.infer_pattern(&binding.pattern, &value_ty)?;

        Ok(value_ty)
    }

    /// Infer types for a function definition.
    pub fn infer_function(&mut self, func: &FnDef) -> Result<FunctionType, TypeError> {
        let name = &func.name.node;

        // For now, use first clause to determine signature
        if let Some(clause) = func.clauses.first() {
            let (param_types, ret_ty) = self.infer_clause(clause)?;

            let func_ty = FunctionType {
                type_params: vec![],
                params: param_types,
                ret: Box::new(ret_ty),
            };

            // Register function in environment
            self.env.functions.insert(name.clone(), func_ty.clone());

            Ok(func_ty)
        } else {
            Err(TypeError::ArityMismatch {
                expected: 1,
                found: 0,
            })
        }
    }

    /// Infer types for a function clause.
    fn infer_clause(&mut self, clause: &FnClause) -> Result<(Vec<Type>, Type), TypeError> {
        let saved_bindings = self.env.bindings.clone();

        let mut param_types = Vec::new();
        for param in &clause.params {
            let param_ty = if let Some(ty_expr) = &param.ty {
                self.type_from_ast(ty_expr)
            } else {
                self.fresh()
            };
            self.infer_pattern(&param.pattern, &param_ty)?;
            param_types.push(param_ty);
        }

        // Check guard if present
        if let Some(guard) = &clause.guard {
            let guard_ty = self.infer_expr(guard)?;
            self.unify(guard_ty, Type::Bool);
        }

        // Infer body
        let body_ty = self.infer_expr(&clause.body)?;

        // If there's a return type annotation, unify with it
        let ret_ty = if let Some(ret_expr) = &clause.return_type {
            let annotated = self.type_from_ast(ret_expr);
            self.unify(body_ty, annotated.clone());
            annotated
        } else {
            body_ty
        };

        self.env.bindings = saved_bindings;
        Ok((param_types, ret_ty))
    }

    /// Infer types for a module.
    pub fn infer_module(&mut self, module: &Module) -> Result<(), TypeError> {
        // First pass: collect type and function definitions
        for item in &module.items {
            match item {
                Item::TypeDef(td) => {
                    self.register_type_def(td)?;
                }
                Item::FnDef(fd) => {
                    // Register a placeholder for recursive calls
                    let ret_ty = self.fresh();
                    let param_types: Vec<_> = fd.clauses.first().map(|c| {
                        c.params.iter().map(|_| self.fresh()).collect()
                    }).unwrap_or_default();

                    self.env.functions.insert(
                        fd.name.node.clone(),
                        FunctionType {
                            type_params: vec![],
                            params: param_types,
                            ret: Box::new(ret_ty),
                        },
                    );
                }
                _ => {}
            }
        }

        // Second pass: infer function bodies
        for item in &module.items {
            match item {
                Item::FnDef(fd) => {
                    self.infer_function(fd)?;
                }
                Item::Binding(binding) => {
                    self.infer_binding(binding)?;
                }
                _ => {}
            }
        }

        Ok(())
    }

    /// Register a type definition in the environment.
    fn register_type_def(
        &mut self,
        td: &nostos_syntax::ast::TypeDef,
    ) -> Result<(), TypeError> {
        let name = td.name.node.clone();
        let params: Vec<_> = td
            .type_params
            .iter()
            .map(|p| crate::TypeParam {
                name: p.name.node.clone(),
                constraints: p.constraints.iter().map(|c| c.node.clone()).collect(),
            })
            .collect();

        let def = match &td.body {
            nostos_syntax::ast::TypeBody::Record(fields) => TypeDef::Record {
                params,
                fields: fields
                    .iter()
                    .map(|f| (f.name.node.clone(), self.type_from_ast(&f.ty), !f.private))
                    .collect(),
                is_mutable: td.mutable,
            },
            nostos_syntax::ast::TypeBody::Variant(variants) => TypeDef::Variant {
                params,
                constructors: variants
                    .iter()
                    .map(|v| {
                        let vname = v.name.node.clone();
                        match &v.fields {
                            nostos_syntax::ast::VariantFields::Unit => Constructor::Unit(vname),
                            nostos_syntax::ast::VariantFields::Positional(types) => {
                                Constructor::Positional(
                                    vname,
                                    types.iter().map(|t| self.type_from_ast(t)).collect(),
                                )
                            }
                            nostos_syntax::ast::VariantFields::Named(fields) => Constructor::Named(
                                vname,
                                fields
                                    .iter()
                                    .map(|f| (f.name.node.clone(), self.type_from_ast(&f.ty)))
                                    .collect(),
                            ),
                        }
                    })
                    .collect(),
            },
            nostos_syntax::ast::TypeBody::Alias(ty) => TypeDef::Alias {
                params,
                target: self.type_from_ast(ty),
            },
            nostos_syntax::ast::TypeBody::Empty => {
                return Ok(()); // Never type
            }
        };

        self.env.define_type(name, def);
        Ok(())
    }
}

/// Convenience function to infer and solve types for an expression.
pub fn infer_expr_type(env: &mut TypeEnv, expr: &Expr) -> Result<Type, TypeError> {
    let mut ctx = InferCtx::new(env);
    let ty = ctx.infer_expr(expr)?;
    ctx.solve()?;
    Ok(ctx.env.apply_subst(&ty))
}

/// Convenience function to type-check a module.
pub fn check_module(env: &mut TypeEnv, module: &Module) -> Result<(), TypeError> {
    let mut ctx = InferCtx::new(env);
    ctx.infer_module(module)?;
    ctx.solve()
}

#[cfg(test)]
mod tests {
    use super::*;
    use nostos_syntax::ast::Span;

    // Helper to create a spanned identifier
    fn ident(name: &str) -> nostos_syntax::ast::Ident {
        nostos_syntax::ast::Spanned::new(name.to_string(), Span::default())
    }

    // Helper to create expression with default span
    fn span() -> Span {
        Span::default()
    }

    // =========================================================================
    // Unification Tests
    // =========================================================================

    #[test]
    fn test_unify_same() {
        let mut env = TypeEnv::new();
        let mut ctx = InferCtx::new(&mut env);
        ctx.unify(Type::Int, Type::Int);
        assert!(ctx.solve().is_ok());
    }

    #[test]
    fn test_unify_var() {
        let mut env = TypeEnv::new();
        let var = env.fresh_var();
        let mut ctx = InferCtx::new(&mut env);
        ctx.unify(var.clone(), Type::Int);
        assert!(ctx.solve().is_ok());
        assert_eq!(ctx.env.apply_subst(&var), Type::Int);
    }

    #[test]
    fn test_unify_mismatch() {
        let mut env = TypeEnv::new();
        let mut ctx = InferCtx::new(&mut env);
        ctx.unify(Type::Int, Type::String);
        assert!(matches!(ctx.solve(), Err(TypeError::UnificationFailed(_, _))));
    }

    #[test]
    fn test_unify_lists() {
        let mut env = TypeEnv::new();
        let var = env.fresh_var();
        let mut ctx = InferCtx::new(&mut env);
        ctx.unify(
            Type::List(Box::new(var.clone())),
            Type::List(Box::new(Type::Int)),
        );
        assert!(ctx.solve().is_ok());
        assert_eq!(ctx.env.apply_subst(&var), Type::Int);
    }

    #[test]
    fn test_occurs_check() {
        let mut env = TypeEnv::new();
        let var = env.fresh_var();
        let mut ctx = InferCtx::new(&mut env);
        // Try to unify ?0 with List[?0] - should fail
        ctx.unify(var.clone(), Type::List(Box::new(var.clone())));
        assert!(matches!(ctx.solve(), Err(TypeError::OccursCheck(_, _))));
    }

    #[test]
    fn test_unify_tuples() {
        let mut env = TypeEnv::new();
        let var1 = env.fresh_var();
        let var2 = env.fresh_var();
        let mut ctx = InferCtx::new(&mut env);
        ctx.unify(
            Type::Tuple(vec![var1.clone(), var2.clone()]),
            Type::Tuple(vec![Type::Int, Type::String]),
        );
        assert!(ctx.solve().is_ok());
        assert_eq!(ctx.env.apply_subst(&var1), Type::Int);
        assert_eq!(ctx.env.apply_subst(&var2), Type::String);
    }

    #[test]
    fn test_unify_functions() {
        let mut env = TypeEnv::new();
        let var = env.fresh_var();
        let mut ctx = InferCtx::new(&mut env);

        let func1 = Type::Function(FunctionType {
            type_params: vec![],
            params: vec![Type::Int],
            ret: Box::new(var.clone()),
        });
        let func2 = Type::Function(FunctionType {
            type_params: vec![],
            params: vec![Type::Int],
            ret: Box::new(Type::Bool),
        });

        ctx.unify(func1, func2);
        assert!(ctx.solve().is_ok());
        assert_eq!(ctx.env.apply_subst(&var), Type::Bool);
    }

    // =========================================================================
    // Literal Expression Tests
    // =========================================================================

    #[test]
    fn test_infer_int_literal() {
        let mut env = TypeEnv::new();
        let expr = Expr::Int(42, span());
        let ty = infer_expr_type(&mut env, &expr).unwrap();
        assert_eq!(ty, Type::Int);
    }

    #[test]
    fn test_infer_float_literal() {
        let mut env = TypeEnv::new();
        let expr = Expr::Float(3.14, span());
        let ty = infer_expr_type(&mut env, &expr).unwrap();
        assert_eq!(ty, Type::Float);
    }

    #[test]
    fn test_infer_bool_literal() {
        let mut env = TypeEnv::new();
        let expr = Expr::Bool(true, span());
        let ty = infer_expr_type(&mut env, &expr).unwrap();
        assert_eq!(ty, Type::Bool);
    }

    #[test]
    fn test_infer_string_literal() {
        let mut env = TypeEnv::new();
        let expr = Expr::String(nostos_syntax::ast::StringLit::Plain("hello".to_string()), span());
        let ty = infer_expr_type(&mut env, &expr).unwrap();
        assert_eq!(ty, Type::String);
    }

    #[test]
    fn test_infer_unit_literal() {
        let mut env = TypeEnv::new();
        let expr = Expr::Unit(span());
        let ty = infer_expr_type(&mut env, &expr).unwrap();
        assert_eq!(ty, Type::Unit);
    }

    // =========================================================================
    // Variable Tests
    // =========================================================================

    #[test]
    fn test_infer_variable() {
        let mut env = TypeEnv::new();
        env.bind("x".to_string(), Type::Int, false);
        let expr = Expr::Var(ident("x"));
        let ty = infer_expr_type(&mut env, &expr).unwrap();
        assert_eq!(ty, Type::Int);
    }

    #[test]
    fn test_infer_unknown_variable() {
        let mut env = TypeEnv::new();
        let expr = Expr::Var(ident("unknown"));
        let result = infer_expr_type(&mut env, &expr);
        assert!(matches!(result, Err(TypeError::UnknownIdent(_))));
    }

    // =========================================================================
    // Binary Operation Tests
    // =========================================================================

    #[test]
    fn test_infer_addition() {
        let mut env = TypeEnv::new();
        let expr = Expr::BinOp(
            Box::new(Expr::Int(1, span())),
            BinOp::Add,
            Box::new(Expr::Int(2, span())),
            span(),
        );
        let ty = infer_expr_type(&mut env, &expr).unwrap();
        assert_eq!(ty, Type::Int);
    }

    #[test]
    fn test_infer_comparison() {
        let mut env = crate::standard_env();
        let expr = Expr::BinOp(
            Box::new(Expr::Int(1, span())),
            BinOp::Lt,
            Box::new(Expr::Int(2, span())),
            span(),
        );
        let ty = infer_expr_type(&mut env, &expr).unwrap();
        assert_eq!(ty, Type::Bool);
    }

    #[test]
    fn test_infer_equality() {
        let mut env = crate::standard_env();
        let expr = Expr::BinOp(
            Box::new(Expr::Int(1, span())),
            BinOp::Eq,
            Box::new(Expr::Int(2, span())),
            span(),
        );
        let ty = infer_expr_type(&mut env, &expr).unwrap();
        assert_eq!(ty, Type::Bool);
    }

    #[test]
    fn test_infer_logical() {
        let mut env = TypeEnv::new();
        let expr = Expr::BinOp(
            Box::new(Expr::Bool(true, span())),
            BinOp::And,
            Box::new(Expr::Bool(false, span())),
            span(),
        );
        let ty = infer_expr_type(&mut env, &expr).unwrap();
        assert_eq!(ty, Type::Bool);
    }

    #[test]
    fn test_infer_type_mismatch_binop() {
        let mut env = TypeEnv::new();
        let expr = Expr::BinOp(
            Box::new(Expr::Int(1, span())),
            BinOp::Add,
            Box::new(Expr::String(nostos_syntax::ast::StringLit::Plain("hello".to_string()), span())),
            span(),
        );
        let result = infer_expr_type(&mut env, &expr);
        assert!(matches!(result, Err(TypeError::UnificationFailed(_, _))));
    }

    // =========================================================================
    // Unary Operation Tests
    // =========================================================================

    #[test]
    fn test_infer_negation() {
        let mut env = TypeEnv::new();
        let expr = Expr::UnaryOp(UnaryOp::Neg, Box::new(Expr::Int(42, span())), span());
        let ty = infer_expr_type(&mut env, &expr).unwrap();
        assert_eq!(ty, Type::Int);
    }

    #[test]
    fn test_infer_not() {
        let mut env = TypeEnv::new();
        let expr = Expr::UnaryOp(UnaryOp::Not, Box::new(Expr::Bool(true, span())), span());
        let ty = infer_expr_type(&mut env, &expr).unwrap();
        assert_eq!(ty, Type::Bool);
    }

    // =========================================================================
    // Tuple Tests
    // =========================================================================

    #[test]
    fn test_infer_tuple() {
        let mut env = TypeEnv::new();
        let expr = Expr::Tuple(
            vec![
                Expr::Int(1, span()),
                Expr::String(nostos_syntax::ast::StringLit::Plain("hello".to_string()), span()),
            ],
            span(),
        );
        let ty = infer_expr_type(&mut env, &expr).unwrap();
        assert_eq!(ty, Type::Tuple(vec![Type::Int, Type::String]));
    }

    // =========================================================================
    // List Tests
    // =========================================================================

    #[test]
    fn test_infer_empty_list() {
        let mut env = TypeEnv::new();
        let expr = Expr::List(vec![], None, span());
        let ty = infer_expr_type(&mut env, &expr).unwrap();
        // Empty list has a fresh variable element type
        assert!(matches!(ty, Type::List(_)));
    }

    #[test]
    fn test_infer_int_list() {
        let mut env = TypeEnv::new();
        let expr = Expr::List(
            vec![Expr::Int(1, span()), Expr::Int(2, span()), Expr::Int(3, span())],
            None,
            span(),
        );
        let ty = infer_expr_type(&mut env, &expr).unwrap();
        assert_eq!(ty, Type::List(Box::new(Type::Int)));
    }

    #[test]
    fn test_infer_heterogeneous_list_fails() {
        let mut env = TypeEnv::new();
        let expr = Expr::List(
            vec![
                Expr::Int(1, span()),
                Expr::String(nostos_syntax::ast::StringLit::Plain("hello".to_string()), span()),
            ],
            None,
            span(),
        );
        let result = infer_expr_type(&mut env, &expr);
        assert!(matches!(result, Err(TypeError::UnificationFailed(_, _))));
    }

    // =========================================================================
    // Lambda Tests
    // =========================================================================

    #[test]
    fn test_infer_lambda_identity() {
        let mut env = TypeEnv::new();
        let expr = Expr::Lambda(
            vec![Pattern::Var(ident("x"))],
            Box::new(Expr::Var(ident("x"))),
            span(),
        );
        let ty = infer_expr_type(&mut env, &expr).unwrap();
        if let Type::Function(f) = ty {
            assert_eq!(f.params.len(), 1);
            // Return type should equal param type
        } else {
            panic!("Expected function type");
        }
    }

    #[test]
    fn test_infer_lambda_add() {
        let mut env = TypeEnv::new();
        let expr = Expr::Lambda(
            vec![Pattern::Var(ident("x"))],
            Box::new(Expr::BinOp(
                Box::new(Expr::Var(ident("x"))),
                BinOp::Add,
                Box::new(Expr::Int(1, span())),
                span(),
            )),
            span(),
        );
        let ty = infer_expr_type(&mut env, &expr).unwrap();
        if let Type::Function(f) = ty {
            assert_eq!(f.params, vec![Type::Int]);
            assert_eq!(*f.ret, Type::Int);
        } else {
            panic!("Expected function type");
        }
    }

    // =========================================================================
    // Function Call Tests
    // =========================================================================

    #[test]
    fn test_infer_function_call() {
        let mut env = TypeEnv::new();
        // Register a function: f : Int -> Bool
        env.functions.insert(
            "f".to_string(),
            FunctionType {
                type_params: vec![],
                params: vec![Type::Int],
                ret: Box::new(Type::Bool),
            },
        );
        let expr = Expr::Call(
            Box::new(Expr::Var(ident("f"))),
            vec![Expr::Int(42, span())],
            span(),
        );
        let ty = infer_expr_type(&mut env, &expr).unwrap();
        assert_eq!(ty, Type::Bool);
    }

    #[test]
    fn test_infer_function_call_wrong_arg_type() {
        let mut env = TypeEnv::new();
        // Register a function: f : Int -> Bool
        env.functions.insert(
            "f".to_string(),
            FunctionType {
                type_params: vec![],
                params: vec![Type::Int],
                ret: Box::new(Type::Bool),
            },
        );
        let expr = Expr::Call(
            Box::new(Expr::Var(ident("f"))),
            vec![Expr::String(nostos_syntax::ast::StringLit::Plain("hello".to_string()), span())],
            span(),
        );
        let result = infer_expr_type(&mut env, &expr);
        assert!(matches!(result, Err(TypeError::UnificationFailed(_, _))));
    }

    // =========================================================================
    // If Expression Tests
    // =========================================================================

    #[test]
    fn test_infer_if() {
        let mut env = TypeEnv::new();
        let expr = Expr::If(
            Box::new(Expr::Bool(true, span())),
            Box::new(Expr::Int(1, span())),
            Box::new(Expr::Int(2, span())),
            span(),
        );
        let ty = infer_expr_type(&mut env, &expr).unwrap();
        assert_eq!(ty, Type::Int);
    }

    #[test]
    fn test_infer_if_branches_must_match() {
        let mut env = TypeEnv::new();
        let expr = Expr::If(
            Box::new(Expr::Bool(true, span())),
            Box::new(Expr::Int(1, span())),
            Box::new(Expr::String(nostos_syntax::ast::StringLit::Plain("hello".to_string()), span())),
            span(),
        );
        let result = infer_expr_type(&mut env, &expr);
        assert!(matches!(result, Err(TypeError::UnificationFailed(_, _))));
    }

    #[test]
    fn test_infer_if_condition_must_be_bool() {
        let mut env = TypeEnv::new();
        let expr = Expr::If(
            Box::new(Expr::Int(1, span())),
            Box::new(Expr::Int(1, span())),
            Box::new(Expr::Int(2, span())),
            span(),
        );
        let result = infer_expr_type(&mut env, &expr);
        assert!(matches!(result, Err(TypeError::UnificationFailed(_, _))));
    }

    // =========================================================================
    // Match Expression Tests
    // =========================================================================

    #[test]
    fn test_infer_match() {
        let mut env = TypeEnv::new();
        let expr = Expr::Match(
            Box::new(Expr::Int(1, span())),
            vec![
                MatchArm {
                    pattern: Pattern::Int(0, span()),
                    guard: None,
                    body: Expr::String(nostos_syntax::ast::StringLit::Plain("zero".to_string()), span()),
                    span: span(),
                },
                MatchArm {
                    pattern: Pattern::Wildcard(span()),
                    guard: None,
                    body: Expr::String(nostos_syntax::ast::StringLit::Plain("other".to_string()), span()),
                    span: span(),
                },
            ],
            span(),
        );
        let ty = infer_expr_type(&mut env, &expr).unwrap();
        assert_eq!(ty, Type::String);
    }

    #[test]
    fn test_infer_match_with_binding() {
        let mut env = TypeEnv::new();
        let expr = Expr::Match(
            Box::new(Expr::Int(42, span())),
            vec![MatchArm {
                pattern: Pattern::Var(ident("n")),
                guard: None,
                body: Expr::BinOp(
                    Box::new(Expr::Var(ident("n"))),
                    BinOp::Add,
                    Box::new(Expr::Int(1, span())),
                    span(),
                ),
                span: span(),
            }],
            span(),
        );
        let ty = infer_expr_type(&mut env, &expr).unwrap();
        assert_eq!(ty, Type::Int);
    }

    // =========================================================================
    // Block Tests
    // =========================================================================

    #[test]
    fn test_infer_block() {
        let mut env = TypeEnv::new();
        let expr = Expr::Block(
            vec![
                Stmt::Expr(Expr::Int(1, span())),
                Stmt::Expr(Expr::String(nostos_syntax::ast::StringLit::Plain("result".to_string()), span())),
            ],
            span(),
        );
        let ty = infer_expr_type(&mut env, &expr).unwrap();
        assert_eq!(ty, Type::String);
    }

    #[test]
    fn test_infer_block_with_let() {
        let mut env = TypeEnv::new();
        let expr = Expr::Block(
            vec![
                Stmt::Let(Binding {
                    mutable: false,
                    pattern: Pattern::Var(ident("x")),
                    ty: None,
                    value: Expr::Int(42, span()),
                    span: span(),
                }),
                Stmt::Expr(Expr::BinOp(
                    Box::new(Expr::Var(ident("x"))),
                    BinOp::Add,
                    Box::new(Expr::Int(1, span())),
                    span(),
                )),
            ],
            span(),
        );
        let ty = infer_expr_type(&mut env, &expr).unwrap();
        assert_eq!(ty, Type::Int);
    }

    // =========================================================================
    // Pattern Tests
    // =========================================================================

    #[test]
    fn test_infer_tuple_pattern() {
        let mut env = TypeEnv::new();
        let expr = Expr::Match(
            Box::new(Expr::Tuple(
                vec![Expr::Int(1, span()), Expr::String(nostos_syntax::ast::StringLit::Plain("hello".to_string()), span())],
                span(),
            )),
            vec![MatchArm {
                pattern: Pattern::Tuple(
                    vec![Pattern::Var(ident("a")), Pattern::Var(ident("b"))],
                    span(),
                ),
                guard: None,
                body: Expr::Var(ident("a")),
                span: span(),
            }],
            span(),
        );
        let ty = infer_expr_type(&mut env, &expr).unwrap();
        assert_eq!(ty, Type::Int);
    }

    #[test]
    fn test_infer_list_pattern() {
        let mut env = TypeEnv::new();
        let expr = Expr::Match(
            Box::new(Expr::List(
                vec![Expr::Int(1, span()), Expr::Int(2, span())],
                None,
                span(),
            )),
            vec![
                MatchArm {
                    pattern: Pattern::List(
                        nostos_syntax::ast::ListPattern::Cons(
                            vec![Pattern::Var(ident("h"))],
                            Some(Box::new(Pattern::Var(ident("t")))),
                        ),
                        span(),
                    ),
                    guard: None,
                    body: Expr::Var(ident("h")),
                    span: span(),
                },
            ],
            span(),
        );
        let ty = infer_expr_type(&mut env, &expr).unwrap();
        assert_eq!(ty, Type::Int);
    }

    // =========================================================================
    // Constructor Tests
    // =========================================================================

    #[test]
    fn test_infer_option_none() {
        let mut env = crate::standard_env();
        let expr = Expr::Var(ident("None"));
        let ty = infer_expr_type(&mut env, &expr).unwrap();
        // None has type Option[?T] for some unknown T
        if let Type::Named { name, args } = ty {
            assert_eq!(name, "Option");
            assert_eq!(args.len(), 1);
        } else {
            panic!("Expected Option type, got {:?}", ty);
        }
    }

    #[test]
    fn test_infer_option_some() {
        let mut env = crate::standard_env();
        let expr = Expr::Call(
            Box::new(Expr::Var(ident("Some"))),
            vec![Expr::Int(42, span())],
            span(),
        );
        let ty = infer_expr_type(&mut env, &expr).unwrap();
        // Some(42) should have type Option (with Int inner type)
        assert!(matches!(ty, Type::Named { name, .. } if name == "Option"));
    }

    // =========================================================================
    // Pipe Operator Tests
    // =========================================================================

    #[test]
    fn test_infer_pipe() {
        let mut env = TypeEnv::new();
        // Register a function: double : Int -> Int
        env.functions.insert(
            "double".to_string(),
            FunctionType {
                type_params: vec![],
                params: vec![Type::Int],
                ret: Box::new(Type::Int),
            },
        );
        let expr = Expr::BinOp(
            Box::new(Expr::Int(21, span())),
            BinOp::Pipe,
            Box::new(Expr::Var(ident("double"))),
            span(),
        );
        let ty = infer_expr_type(&mut env, &expr).unwrap();
        assert_eq!(ty, Type::Int);
    }

    // =========================================================================
    // Type Annotation Tests
    // =========================================================================

    #[test]
    fn test_infer_with_annotation() {
        let mut env = TypeEnv::new();
        let expr = Expr::Block(
            vec![
                Stmt::Let(Binding {
                    mutable: false,
                    pattern: Pattern::Var(ident("x")),
                    ty: Some(TypeExpr::Name(ident("Int"))),
                    value: Expr::Int(42, span()),
                    span: span(),
                }),
                Stmt::Expr(Expr::Var(ident("x"))),
            ],
            span(),
        );
        let ty = infer_expr_type(&mut env, &expr).unwrap();
        assert_eq!(ty, Type::Int);
    }

    #[test]
    fn test_infer_annotation_mismatch() {
        let mut env = TypeEnv::new();
        let expr = Expr::Block(
            vec![Stmt::Let(Binding {
                mutable: false,
                pattern: Pattern::Var(ident("x")),
                ty: Some(TypeExpr::Name(ident("String"))),
                value: Expr::Int(42, span()),
                span: span(),
            })],
            span(),
        );
        let result = infer_expr_type(&mut env, &expr);
        assert!(matches!(result, Err(TypeError::UnificationFailed(_, _))));
    }

    // =========================================================================
    // Map and Set Tests
    // =========================================================================

    #[test]
    fn test_infer_map() {
        let mut env = crate::standard_env();
        let expr = Expr::Map(
            vec![
                (
                    Expr::String(nostos_syntax::ast::StringLit::Plain("a".to_string()), span()),
                    Expr::Int(1, span()),
                ),
                (
                    Expr::String(nostos_syntax::ast::StringLit::Plain("b".to_string()), span()),
                    Expr::Int(2, span()),
                ),
            ],
            span(),
        );
        let ty = infer_expr_type(&mut env, &expr).unwrap();
        assert_eq!(
            ty,
            Type::Map(Box::new(Type::String), Box::new(Type::Int))
        );
    }

    #[test]
    fn test_infer_set() {
        let mut env = crate::standard_env();
        let expr = Expr::Set(
            vec![Expr::Int(1, span()), Expr::Int(2, span()), Expr::Int(3, span())],
            span(),
        );
        let ty = infer_expr_type(&mut env, &expr).unwrap();
        assert_eq!(ty, Type::Set(Box::new(Type::Int)));
    }

    // =========================================================================
    // Advanced Type Inference Tests
    // =========================================================================

    #[test]
    fn test_infer_higher_order_function() {
        // map : (a -> b) -> List[a] -> List[b]
        // map(f, xs) should infer correct types
        let mut env = TypeEnv::new();
        // Create a lambda: f => xs => ... (simplified)
        let expr = Expr::Lambda(
            vec![Pattern::Var(ident("f"))],
            Box::new(Expr::Lambda(
                vec![Pattern::Var(ident("xs"))],
                Box::new(Expr::Var(ident("xs"))), // Just return xs for now
                span(),
            )),
            span(),
        );
        let ty = infer_expr_type(&mut env, &expr).unwrap();
        if let Type::Function(f) = ty {
            assert_eq!(f.params.len(), 1);
            if let Type::Function(inner) = *f.ret {
                assert_eq!(inner.params.len(), 1);
            } else {
                panic!("Expected nested function type");
            }
        } else {
            panic!("Expected function type");
        }
    }

    #[test]
    fn test_infer_curried_function() {
        // Curried addition with concrete constraint: x => y => x + y + 0
        // The 0 forces the type to be Int
        let mut env = TypeEnv::new();
        let expr = Expr::Lambda(
            vec![Pattern::Var(ident("x"))],
            Box::new(Expr::Lambda(
                vec![Pattern::Var(ident("y"))],
                Box::new(Expr::BinOp(
                    Box::new(Expr::BinOp(
                        Box::new(Expr::Var(ident("x"))),
                        BinOp::Add,
                        Box::new(Expr::Var(ident("y"))),
                        span(),
                    )),
                    BinOp::Add,
                    Box::new(Expr::Int(0, span())), // Constrains to Int
                    span(),
                )),
                span(),
            )),
            span(),
        );
        let ty = infer_expr_type(&mut env, &expr).unwrap();
        if let Type::Function(f) = ty {
            assert_eq!(f.params.len(), 1);
            assert_eq!(f.params[0], Type::Int); // Now constrained to Int
            if let Type::Function(inner) = *f.ret {
                assert_eq!(inner.params.len(), 1);
                assert_eq!(inner.params[0], Type::Int);
                assert_eq!(*inner.ret, Type::Int);
            } else {
                panic!("Expected nested function type");
            }
        } else {
            panic!("Expected function type");
        }
    }

    #[test]
    fn test_infer_nested_list() {
        let mut env = TypeEnv::new();
        // [[1, 2], [3, 4]]
        let inner1 = Expr::List(
            vec![Expr::Int(1, span()), Expr::Int(2, span())],
            None,
            span(),
        );
        let inner2 = Expr::List(
            vec![Expr::Int(3, span()), Expr::Int(4, span())],
            None,
            span(),
        );
        let expr = Expr::List(vec![inner1, inner2], None, span());
        let ty = infer_expr_type(&mut env, &expr).unwrap();
        assert_eq!(ty, Type::List(Box::new(Type::List(Box::new(Type::Int)))));
    }

    #[test]
    fn test_infer_match_with_guard() {
        let mut env = crate::standard_env();
        // match x { n when n > 0 -> "positive", _ -> "non-positive" }
        env.bind("x".to_string(), Type::Int, false);
        let expr = Expr::Match(
            Box::new(Expr::Var(ident("x"))),
            vec![
                MatchArm {
                    pattern: Pattern::Var(ident("n")),
                    guard: Some(Expr::BinOp(
                        Box::new(Expr::Var(ident("n"))),
                        BinOp::Gt,
                        Box::new(Expr::Int(0, span())),
                        span(),
                    )),
                    body: Expr::String(nostos_syntax::ast::StringLit::Plain("positive".to_string()), span()),
                    span: span(),
                },
                MatchArm {
                    pattern: Pattern::Wildcard(span()),
                    guard: None,
                    body: Expr::String(nostos_syntax::ast::StringLit::Plain("non-positive".to_string()), span()),
                    span: span(),
                },
            ],
            span(),
        );
        let ty = infer_expr_type(&mut env, &expr).unwrap();
        assert_eq!(ty, Type::String);
    }

    #[test]
    fn test_infer_lambda_with_multiple_params() {
        let mut env = TypeEnv::new();
        // (a, b, c) => a + b + c + 0  (0 constrains to Int)
        let expr = Expr::Lambda(
            vec![
                Pattern::Var(ident("a")),
                Pattern::Var(ident("b")),
                Pattern::Var(ident("c")),
            ],
            Box::new(Expr::BinOp(
                Box::new(Expr::BinOp(
                    Box::new(Expr::BinOp(
                        Box::new(Expr::Var(ident("a"))),
                        BinOp::Add,
                        Box::new(Expr::Var(ident("b"))),
                        span(),
                    )),
                    BinOp::Add,
                    Box::new(Expr::Var(ident("c"))),
                    span(),
                )),
                BinOp::Add,
                Box::new(Expr::Int(0, span())), // Constrains to Int
                span(),
            )),
            span(),
        );
        let ty = infer_expr_type(&mut env, &expr).unwrap();
        if let Type::Function(f) = ty {
            assert_eq!(f.params.len(), 3);
            assert_eq!(f.params, vec![Type::Int, Type::Int, Type::Int]);
            assert_eq!(*f.ret, Type::Int);
        } else {
            panic!("Expected function type");
        }
    }

    #[test]
    fn test_infer_nested_if() {
        let mut env = TypeEnv::new();
        // if true then (if false then 1 else 2) else 3
        let expr = Expr::If(
            Box::new(Expr::Bool(true, span())),
            Box::new(Expr::If(
                Box::new(Expr::Bool(false, span())),
                Box::new(Expr::Int(1, span())),
                Box::new(Expr::Int(2, span())),
                span(),
            )),
            Box::new(Expr::Int(3, span())),
            span(),
        );
        let ty = infer_expr_type(&mut env, &expr).unwrap();
        assert_eq!(ty, Type::Int);
    }

    #[test]
    fn test_infer_function_returning_function() {
        let mut env = TypeEnv::new();
        // Register: make_adder : Int -> (Int -> Int)
        env.functions.insert(
            "make_adder".to_string(),
            FunctionType {
                type_params: vec![],
                params: vec![Type::Int],
                ret: Box::new(Type::Function(FunctionType {
                    type_params: vec![],
                    params: vec![Type::Int],
                    ret: Box::new(Type::Int),
                })),
            },
        );
        // make_adder(5)
        let expr = Expr::Call(
            Box::new(Expr::Var(ident("make_adder"))),
            vec![Expr::Int(5, span())],
            span(),
        );
        let ty = infer_expr_type(&mut env, &expr).unwrap();
        if let Type::Function(f) = ty {
            assert_eq!(f.params, vec![Type::Int]);
            assert_eq!(*f.ret, Type::Int);
        } else {
            panic!("Expected function type");
        }
    }

    #[test]
    fn test_infer_chained_pipe() {
        let mut env = TypeEnv::new();
        // Register functions
        env.functions.insert(
            "add_one".to_string(),
            FunctionType {
                type_params: vec![],
                params: vec![Type::Int],
                ret: Box::new(Type::Int),
            },
        );
        env.functions.insert(
            "to_string".to_string(),
            FunctionType {
                type_params: vec![],
                params: vec![Type::Int],
                ret: Box::new(Type::String),
            },
        );
        // 5 |> add_one |> to_string
        let expr = Expr::BinOp(
            Box::new(Expr::BinOp(
                Box::new(Expr::Int(5, span())),
                BinOp::Pipe,
                Box::new(Expr::Var(ident("add_one"))),
                span(),
            )),
            BinOp::Pipe,
            Box::new(Expr::Var(ident("to_string"))),
            span(),
        );
        let ty = infer_expr_type(&mut env, &expr).unwrap();
        assert_eq!(ty, Type::String);
    }

    #[test]
    fn test_infer_polymorphic_function() {
        // Test that we can call monomorphic functions
        let mut env = TypeEnv::new();
        // id_int : Int -> Int (monomorphic version)
        env.functions.insert(
            "id_int".to_string(),
            FunctionType {
                type_params: vec![],
                params: vec![Type::Int],
                ret: Box::new(Type::Int),
            },
        );
        // id_int(42) should be Int
        let expr = Expr::Call(
            Box::new(Expr::Var(ident("id_int"))),
            vec![Expr::Int(42, span())],
            span(),
        );
        let ty = infer_expr_type(&mut env, &expr).unwrap();
        assert_eq!(ty, Type::Int);
    }

    #[test]
    fn test_infer_complex_block() {
        let mut env = TypeEnv::new();
        // {
        //   let x = 1
        //   let y = 2
        //   let z = x + y
        //   z * 2
        // }
        let expr = Expr::Block(
            vec![
                Stmt::Let(Binding {
                    mutable: false,
                    pattern: Pattern::Var(ident("x")),
                    ty: None,
                    value: Expr::Int(1, span()),
                    span: span(),
                }),
                Stmt::Let(Binding {
                    mutable: false,
                    pattern: Pattern::Var(ident("y")),
                    ty: None,
                    value: Expr::Int(2, span()),
                    span: span(),
                }),
                Stmt::Let(Binding {
                    mutable: false,
                    pattern: Pattern::Var(ident("z")),
                    ty: None,
                    value: Expr::BinOp(
                        Box::new(Expr::Var(ident("x"))),
                        BinOp::Add,
                        Box::new(Expr::Var(ident("y"))),
                        span(),
                    ),
                    span: span(),
                }),
                Stmt::Expr(Expr::BinOp(
                    Box::new(Expr::Var(ident("z"))),
                    BinOp::Mul,
                    Box::new(Expr::Int(2, span())),
                    span(),
                )),
            ],
            span(),
        );
        let ty = infer_expr_type(&mut env, &expr).unwrap();
        assert_eq!(ty, Type::Int);
    }

    #[test]
    fn test_infer_or_pattern() {
        let mut env = TypeEnv::new();
        // match x { 0 | 1 | 2 -> "small", _ -> "big" }
        env.bind("x".to_string(), Type::Int, false);
        let expr = Expr::Match(
            Box::new(Expr::Var(ident("x"))),
            vec![
                MatchArm {
                    pattern: Pattern::Or(
                        vec![
                            Pattern::Int(0, span()),
                            Pattern::Int(1, span()),
                            Pattern::Int(2, span()),
                        ],
                        span(),
                    ),
                    guard: None,
                    body: Expr::String(nostos_syntax::ast::StringLit::Plain("small".to_string()), span()),
                    span: span(),
                },
                MatchArm {
                    pattern: Pattern::Wildcard(span()),
                    guard: None,
                    body: Expr::String(nostos_syntax::ast::StringLit::Plain("big".to_string()), span()),
                    span: span(),
                },
            ],
            span(),
        );
        let ty = infer_expr_type(&mut env, &expr).unwrap();
        assert_eq!(ty, Type::String);
    }

    #[test]
    fn test_infer_result_ok() {
        let mut env = crate::standard_env();
        // Ok(42)
        let expr = Expr::Call(
            Box::new(Expr::Var(ident("Ok"))),
            vec![Expr::Int(42, span())],
            span(),
        );
        let ty = infer_expr_type(&mut env, &expr).unwrap();
        if let Type::Named { name, args } = ty {
            assert_eq!(name, "Result");
            assert_eq!(args.len(), 2); // Result[T, E]
        } else {
            panic!("Expected Result type");
        }
    }

    #[test]
    fn test_infer_result_err() {
        let mut env = crate::standard_env();
        // Err("error")
        let expr = Expr::Call(
            Box::new(Expr::Var(ident("Err"))),
            vec![Expr::String(nostos_syntax::ast::StringLit::Plain("error".to_string()), span())],
            span(),
        );
        let ty = infer_expr_type(&mut env, &expr).unwrap();
        if let Type::Named { name, args } = ty {
            assert_eq!(name, "Result");
            assert_eq!(args.len(), 2);
        } else {
            panic!("Expected Result type");
        }
    }

    #[test]
    fn test_infer_float_arithmetic() {
        let mut env = TypeEnv::new();
        // 3.14 * 2.0 + 1.0
        let expr = Expr::BinOp(
            Box::new(Expr::BinOp(
                Box::new(Expr::Float(3.14, span())),
                BinOp::Mul,
                Box::new(Expr::Float(2.0, span())),
                span(),
            )),
            BinOp::Add,
            Box::new(Expr::Float(1.0, span())),
            span(),
        );
        let ty = infer_expr_type(&mut env, &expr).unwrap();
        assert_eq!(ty, Type::Float);
    }

    #[test]
    fn test_infer_mixed_int_float_error() {
        let mut env = TypeEnv::new();
        // 3.14 + 1 (should fail - can't mix Float and Int)
        let expr = Expr::BinOp(
            Box::new(Expr::Float(3.14, span())),
            BinOp::Add,
            Box::new(Expr::Int(1, span())),
            span(),
        );
        let result = infer_expr_type(&mut env, &expr);
        assert!(matches!(result, Err(TypeError::UnificationFailed(_, _))));
    }

    #[test]
    fn test_infer_char_literal() {
        let mut env = TypeEnv::new();
        let expr = Expr::Char('x', span());
        let ty = infer_expr_type(&mut env, &expr).unwrap();
        assert_eq!(ty, Type::Char);
    }

    #[test]
    fn test_infer_deeply_nested_tuple() {
        let mut env = TypeEnv::new();
        // ((1, 2), (3, 4))
        let inner1 = Expr::Tuple(
            vec![Expr::Int(1, span()), Expr::Int(2, span())],
            span(),
        );
        let inner2 = Expr::Tuple(
            vec![Expr::Int(3, span()), Expr::Int(4, span())],
            span(),
        );
        let expr = Expr::Tuple(vec![inner1, inner2], span());
        let ty = infer_expr_type(&mut env, &expr).unwrap();
        let expected = Type::Tuple(vec![
            Type::Tuple(vec![Type::Int, Type::Int]),
            Type::Tuple(vec![Type::Int, Type::Int]),
        ]);
        assert_eq!(ty, expected);
    }

    #[test]
    fn test_infer_unify_nested_types() {
        let mut env = TypeEnv::new();
        let var1 = env.fresh_var();
        let var2 = env.fresh_var();
        let mut ctx = InferCtx::new(&mut env);

        // Unify List[Tuple[?0, ?1]] with List[Tuple[Int, String]]
        ctx.unify(
            Type::List(Box::new(Type::Tuple(vec![var1.clone(), var2.clone()]))),
            Type::List(Box::new(Type::Tuple(vec![Type::Int, Type::String]))),
        );
        assert!(ctx.solve().is_ok());
        assert_eq!(ctx.env.apply_subst(&var1), Type::Int);
        assert_eq!(ctx.env.apply_subst(&var2), Type::String);
    }

    #[test]
    fn test_infer_function_arity_mismatch() {
        let mut env = TypeEnv::new();
        env.functions.insert(
            "add".to_string(),
            FunctionType {
                type_params: vec![],
                params: vec![Type::Int, Type::Int],
                ret: Box::new(Type::Int),
            },
        );
        // add(1) - missing one argument
        let expr = Expr::Call(
            Box::new(Expr::Var(ident("add"))),
            vec![Expr::Int(1, span())],
            span(),
        );
        let result = infer_expr_type(&mut env, &expr);
        assert!(matches!(result, Err(TypeError::ArityMismatch { .. })));
    }
}
