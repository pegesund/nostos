# Nostos Vim/Neovim Syntax Highlighting

Syntax highlighting for the Nostos programming language.

## Features

- Keywords: `if`, `then`, `else`, `match`, `end`, `receive`, `when`, `where`, `type`, `trait`, `spawn`
- Types: `Int`, `Float`, `Bool`, `String`, `Self`, and custom types (capitalized)
- Built-ins: `self`, `println`, `print`, `show`, `sleep`, `assert_eq`
- Modules: `Http`, `File`, `Server`, `Array`, `List`, `Map`, `Set`, `Json`, `Math`, `Time`, `Regex`
- Operators: `+`, `-`, `*`, `/`, `++`, `<-`, `->`, `=>`, `<`, `>`, `==`, etc.
- Comments: `# comment`
- Strings: `"hello"`
- Numbers: integers and floats

## Installation

### Manual Installation (Vim/Neovim)

Copy or symlink the files to your vim config:

```bash
# For Vim
mkdir -p ~/.vim/syntax ~/.vim/ftdetect
cp syntax/nostos.vim ~/.vim/syntax/
cp ftdetect/nostos.vim ~/.vim/ftdetect/

# For Neovim
mkdir -p ~/.config/nvim/syntax ~/.config/nvim/ftdetect
cp syntax/nostos.vim ~/.config/nvim/syntax/
cp ftdetect/nostos.vim ~/.config/nvim/ftdetect/
```

Or symlink (changes auto-update):

```bash
# For Neovim
mkdir -p ~/.config/nvim/syntax ~/.config/nvim/ftdetect
ln -s /path/to/nostos/extras/vim/syntax/nostos.vim ~/.config/nvim/syntax/
ln -s /path/to/nostos/extras/vim/ftdetect/nostos.vim ~/.config/nvim/ftdetect/
```

### Using vim-plug

Add to your `.vimrc` or `init.vim`:

```vim
Plug '/path/to/nostos/extras/vim'
```

### Using lazy.nvim

Add to your LazyVim plugins:

```lua
{ dir = "/path/to/nostos/extras/vim" }
```

## Usage

Open any `.nos` file and syntax highlighting will be applied automatically.

To manually set the filetype:
```vim
:set filetype=nostos
```
