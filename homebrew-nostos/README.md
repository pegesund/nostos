# Homebrew Tap for Nostos

This is the official Homebrew tap for [Nostos](https://github.com/pegesund/nostos).

## Installation

```bash
brew tap pegesund/nostos
brew install nostos
```

## Updating

```bash
brew update
brew upgrade nostos
```

## From Source

If you prefer to build from source:

```bash
git clone https://github.com/pegesund/nostos.git
cd nostos
cargo build --release
```

## For Maintainers

### Creating a Release

1. Build binaries for each platform:
   ```bash
   # On Mac ARM (M1/M2/M3)
   cargo build --release
   tar -czvf nostos-0.1.0-aarch64-apple-darwin.tar.gz -C target/release nostos nostos-lsp -C ../../ stdlib
   
   # On Mac Intel
   cargo build --release
   tar -czvf nostos-0.1.0-x86_64-apple-darwin.tar.gz -C target/release nostos nostos-lsp -C ../../ stdlib
   
   # On Linux x86_64
   cargo build --release
   tar -czvf nostos-0.1.0-x86_64-unknown-linux-gnu.tar.gz -C target/release nostos nostos-lsp -C ../../ stdlib
   ```

2. Upload to GitHub Releases

3. Get SHA256 checksums:
   ```bash
   shasum -a 256 nostos-*.tar.gz
   ```

4. Update `Formula/nostos.rb` with new version, URLs, and checksums
