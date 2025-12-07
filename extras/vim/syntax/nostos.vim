" Vim syntax file
" Language: Nostos
" Maintainer: Nostos Team
" Latest Revision: December 2024

if exists("b:current_syntax")
  finish
endif

" Keywords
syn keyword nostosKeyword if then else match end receive when where type trait spawn
syn keyword nostosKeyword nextgroup=nostosFunction skipwhite

" Boolean
syn keyword nostosBoolean true false

" Built-in functions
syn keyword nostosBuiltin self println print show sleep assert_eq

" Standard modules
syn match nostosModule "\<\(Http\|File\|Server\|Array\|List\|Map\|Set\|Json\|Math\|Time\|Regex\)\>"

" Types (capitalized identifiers)
syn match nostosType "\<[A-Z][a-zA-Z0-9_]*\>"

" Primitive types
syn keyword nostosPrimType Int Float Bool String Unit Self Num

" Function definitions (name followed by parentheses and =)
syn match nostosFunction "\<[a-z_][a-zA-Z0-9_]*\>\s*(" contains=nostosFunctionName
syn match nostosFunctionName "\<[a-z_][a-zA-Z0-9_]*\>" contained

" Module method calls
syn match nostosModuleMethod "\<[A-Z][a-zA-Z0-9_]*\>\.\<[a-z_][a-zA-Z0-9_]*\>"

" Field access
syn match nostosFieldAccess "\.\<[a-z_][a-zA-Z0-9_]*\>"

" Operators
syn match nostosOperator "="
syn match nostosOperator "+"
syn match nostosOperator "-"
syn match nostosOperator "\*"
syn match nostosOperator "/"
syn match nostosOperator "++"
syn match nostosOperator "<-"
syn match nostosOperator "->"
syn match nostosOperator "=>"
syn match nostosOperator "<"
syn match nostosOperator ">"
syn match nostosOperator "<="
syn match nostosOperator ">="
syn match nostosOperator "=="
syn match nostosOperator "!="
syn match nostosOperator "|"
syn match nostosOperator ":"

" Numbers
syn match nostosNumber "\<\d\+\>"
syn match nostosFloat "\<\d\+\.\d*\>"

" Strings
syn region nostosString start='"' end='"' skip='\\"' contains=nostosEscape
syn match nostosEscape "\\." contained

" Comments
syn match nostosComment "#.*$"

" Delimiters
syn match nostosDelimiter "[(),{}\[\]]"

" Wildcards in patterns
syn match nostosWildcard "\<_\>"

" Highlighting links
hi def link nostosKeyword Keyword
hi def link nostosBoolean Boolean
hi def link nostosBuiltin Function
hi def link nostosModule Structure
hi def link nostosType Type
hi def link nostosPrimType Type
hi def link nostosFunction Function
hi def link nostosFunctionName Function
hi def link nostosModuleMethod Function
hi def link nostosFieldAccess Identifier
hi def link nostosOperator Operator
hi def link nostosNumber Number
hi def link nostosFloat Float
hi def link nostosString String
hi def link nostosEscape SpecialChar
hi def link nostosComment Comment
hi def link nostosDelimiter Delimiter
hi def link nostosWildcard Special

let b:current_syntax = "nostos"
