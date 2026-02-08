//! Nostos Compiler
//!
//! Compiles AST to bytecode with:
//! - Type inference and checking
//! - Tail call detection
//! - Closure conversion
//! - Pattern match compilation

#![allow(
    clippy::result_large_err,
    clippy::collapsible_if,
    clippy::type_complexity,
    clippy::map_entry,
    clippy::only_used_in_recursion,
    clippy::unnecessary_map_or,
    clippy::collapsible_match,
    clippy::clone_on_copy,
    clippy::redundant_closure,
    clippy::or_fun_call,
    clippy::identity_op,
    clippy::needless_borrow,
    clippy::unnecessary_cast,
    clippy::redundant_pattern_matching,
    clippy::option_map_unit_fn,
    clippy::manual_map,
    clippy::skip_while_next,
    clippy::bool_comparison,
    clippy::if_same_then_else,
    clippy::needless_return,
    clippy::doc_lazy_continuation,
    clippy::redundant_guards,
    clippy::match_single_binding,
    clippy::useless_format,
    clippy::match_like_matches_macro,
    clippy::filter_map_identity,
    clippy::iter_kv_map,
    clippy::manual_unwrap_or,
    clippy::unnecessary_get_then_check,
    clippy::match_ref_pats,
    clippy::iter_skip_next,
    clippy::unnecessary_lazy_evaluations,
    clippy::needless_borrowed_reference,
    clippy::manual_is_variant_and,
    clippy::let_and_return,
    clippy::needless_range_loop,
    clippy::single_match,
    clippy::redundant_else,
    clippy::map_entry,
    clippy::for_kv_map,
    clippy::nonminimal_bool,
    clippy::question_mark,
    clippy::comparison_chain,
    clippy::unwrap_or_default,
    clippy::unused_enumerate_index,
    clippy::needless_option_as_deref,
    clippy::needless_splitn,
    clippy::iter_overeager_cloned,
    clippy::op_ref,
    unused_variables
)]

pub mod compile;

pub use compile::*;
