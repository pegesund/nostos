(function() {
  const NOSTOS_KEYWORDS = {
    keyword: 'if then else match end receive when where type trait spawn for to break continue try catch finally after module pub var',
    built_in: 'self println print show sleep assert_eq panic',
    literal: 'true false None Some Ok Err',
    type: 'Int Float Bool String Unit Self Num UserId Email Meters Seconds'
  };

  const languageDefinition = {
    name: 'Nostos',
    aliases: ['nos'],
    keywords: NOSTOS_KEYWORDS,
    contains: [
      hljs.COMMENT('#', '$'), // Single line comments
      hljs.QUOTE_STRING_MODE, // String literals
      {
        className: 'string', // Char literals
        begin: "'", end: "'",
        illegal: '\n',
        contains: [hljs.BACKSLASH_ESCAPE]
      },
      {
        className: 'number',
        variants: [
          { begin: '\\b0x[a-fA-F0-9]+\\b' }, // Hex
          { begin: '\\b0b[01]+\\b' },      // Binary
          { begin: '\\b\\d(_\\d)*(\\.\\d+)?([eE][+-]?\\d+)?\\b' } // Numbers with optional underscores, decimals, and scientific notation
        ],
        relevance: 0
      },
      // --- Operators ---
      // Multi-character operators first for higher precedence
      { className: 'operator', begin: '<-', relevance: 10 },
      { className: 'operator', begin: '->', relevance: 10 },
      { className: 'operator', begin: '=>', relevance: 10 },
      { className: 'operator', begin: '==', relevance: 10 },
      { className: 'operator', begin: '!=', relevance: 10 },
      { className: 'operator', begin: '<=', relevance: 10 },
      { className: 'operator', begin: '>=', relevance: 10 },
      { className: 'operator', begin: '\\+\\+', relevance: 10 }, // ++ concatenation
      // Single character operators
      { className: 'operator', begin: '[\\*\\/\\%\\|\\:\\&\\=\\^\\~\\!\\+\\-]', relevance: 0 }, // Individual symbols
      // --- End Operators ---
      {
        className: 'title.class', // User-defined types (capitalized)
        begin: '\\b[A-Z][a-zA-Z0-9_]*\\b',
        relevance: 0
      },
      {
        className: 'function', // Function names (lowercase followed by () or =)
        begin: '\\b[a-z_][a-zA-Z0-9_]*\\s*(?=(\\s*\\(|\\s*=))',
        relevance: 0
      },
      {
        className: 'variable', // Wildcard in patterns and variables
        begin: '\\b_\\b'
      },
      {
        className: 'variable.constant', // For built-in modules/constants (Http, File, etc.)
        begin: '\\b(Http|File|Server|Array|List|Map|Set|Json|Math|Time|Regex|Dir|Base64|Url|Encoding)\\b'
      }
    ]
  };

  hljs.registerLanguage('nostos', function() { return languageDefinition; });
})();
