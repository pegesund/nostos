# HTML Templating

Nostos provides a functional HTML templating system with the `Html(...)` syntax. Inside `Html(...)`, bare tag names like `div`, `h1`, and `p` are automatically resolved to their `stdlib.html` equivalents, giving you a clean, JSX-like syntax without the noise of fully qualified names.

## Basic Usage

You only need to import `Html` and `render`. Inside `Html(...)`, all tag functions are automatically available.

```nostos
use stdlib.html.{Html, render}

main() = {
    # Build an HTML tree - no need to import div, h1, p!
    page = Html(
        div([
            h1("Welcome to Nostos!"),
            p("This is a paragraph.")
        ])
    )

    # Render to string
    html = render(page)
    println(html)
    # Output: <div><h1>Welcome to Nostos!</h1><p>This is a paragraph.</p></div>
}
```

**Note:** The `Html(...)` wrapper enables scoped name resolution. Without it, you'd need to write `stdlib.html.div` everywhere. The magic of `Html(...)` is that you get access to 40+ tag functions with just a 2-item import!

## Tag Functions

Most tag functions are overloaded - they accept either a list of children or a string:

### Overloaded Tags

```nostos
# Most tags are overloaded - same function name, different argument types
div([...])      # <div>...</div>  - with child elements
div("text")     # <div>text</div> - with text content

span([...])     # <span>...</span>
span("text")    # <span>text</span>

p([...])        # <p>...</p>
p("text")       # <p>text</p>

h1([...])       # <h1>...</h1>
h1("title")     # <h1>title</h1>

li([...])       # <li>...</li>
li("item")      # <li>item</li>

# ... and many more: h2, h3, h4, h5, h6, th, td, button, label
```

### Container-Only Tags

```nostos
# Some tags only accept children (no text overload)
ul([...])       # <ul>...</ul>
ol([...])       # <ol>...</ol>
table([...])    # <table>...</table>
thead([...])    # <thead>...</thead>
tbody([...])    # <tbody>...</tbody>
tr([...])       # <tr>...</tr>
nav([...])      # <nav>...</nav>
header([...])   # <header>...</header>
footer([...])   # <footer>...</footer>
section([...])  # <section>...</section>
article([...])  # <article>...</article>
form([...])     # <form>...</form>
```

### Text-Only Tags

```nostos
# Some tags only accept text
title("text")   # <title>text</title>
strong("text")  # <strong>text</strong>
em("text")      # <em>text</em>
code("text")    # <code>text</code>
pre("text")     # <pre>text</pre>
small("text")   # <small>text</small>
```

### Self-Closing Tags

```nostos
# Self-closing tags
br()            # <br />
hr()            # <hr />
img(attrs)      # <img ... />
input(attrs)    # <input ... />
meta(attrs)     # <meta ... />
linkTag(attrs)  # <link ... />
```

## Attributes

Attributes are passed as a list of `(name, value)` tuples.

```nostos
use stdlib.html.{Html, render}

main() = {
    page = Html(
        div([
            # Link with href attribute - a(attrs, text) is overloaded
            a([("href", "https://example.com"), ("class", "link")], "Click here"),

            # Image with attributes
            img([("src", "/logo.png"), ("alt", "Logo"), ("width", "100")])
        ])
    )

    println(render(page))
    # <div><a href="https://example.com" class="link">Click here</a><img src="/logo.png" alt="Logo" width="100" /></div>
}
```

**Tip:** For custom elements or elements with attributes and children, use `el(tag, attrs, children)`.

## Automatic HTML Escaping

Text content is automatically escaped to prevent XSS attacks.

```nostos
use stdlib.html.{Html, render}

main() = {
    # User input with malicious content
    userInput = "<script>alert('xss')</script>"

    page = Html(p(userInput))
    println(render(page))
    # Safe output: <p>&lt;script&gt;alert(&#39;xss&#39;)&lt;/script&gt;</p>
}
```

For raw HTML (use with caution!), use `raw()`:

```nostos
use stdlib.html.{Html, render}

main() = {
    # Only use raw() for trusted content!
    trustedHtml = "<strong>Bold</strong>"

    page = Html(div([raw(trustedHtml)]))
    println(render(page))
    # Output: <div><strong>Bold</strong></div>
}
```

## Dynamic Content

Use `map` and list operations to build dynamic HTML.

```nostos
use stdlib.html.{Html, render}

main() = {
    items = ["Apple", "Banana", "Cherry"]

    page = Html(
        ul(items.map(item => li(item)))
    )

    println(render(page))
    # <ul><li>Apple</li><li>Banana</li><li>Cherry</li></ul>
}
```

## Components

Create reusable components as functions. Each component uses `Html(...)` for automatic tag resolution - no need to import individual tags!

```nostos
use stdlib.html.{Html, render}

# A reusable card component - uses Html(...) for tag resolution
# Type annotations help choose h2(String) over h2(List[Html])
card(title: String, content: String) = Html(
    div([h2(title), p(content)])
)

# Page that uses the card component
main() = {
    page = Html(
        div([
            h1("My Page"),
            card("Welcome", "Thanks for visiting!"),
            card("Features", "Check out our cool features.")
        ])
    )

    println(render(page))
    # <div><h1>My Page</h1><div><h2>Welcome</h2><p>Thanks...</p></div>...</div>
}
```

**Tip:** Add type annotations to component parameters (e.g., `title: String`) when using overloaded tag functions. This helps the compiler choose between `h2(String)` and `h2(List[Html])`.

## Conditional Rendering

Use `if/else` expressions and `empty()` for conditional content.

```nostos
use stdlib.html.{Html, render}

main() = {
    isLoggedIn = true
    userName = "Alice"

    page = Html(
        div([
            if isLoggedIn then
                span("Welcome, " ++ userName ++ "!")
            else
                span("Please log in"),

            # empty() renders to nothing
            if isLoggedIn then p("You have 3 notifications") else empty()
        ])
    )

    println(render(page))
}
```

## Full Page Example

Here's a complete example building a simple web page.

```nostos
use stdlib.html.{Html, render}

main() = {
    page = Html(
        el("html", [("lang", "en")], [
            head([
                meta([("charset", "UTF-8")]),
                title("My Nostos Page")
            ]),
            body([
                div([
                    h1("Welcome to Nostos"),
                    p("A functional programming language with HTML templating."),
                    ul([
                        li([a([("href", "/about")], "About")]),
                        li([a([("href", "/docs")], "Documentation")]),
                        li([a([("href", "/examples")], "Examples")])
                    ])
                ])
            ])
        ])
    )

    println(render(page))
}
```

## Buffer-Optimized Rendering

The `render` function uses buffer-based string building internally, avoiding the overhead of repeated string concatenation. This makes rendering efficient even for large HTML trees.

**Implementation detail:** The VM provides `Buffer.new()`, `Buffer.append()`, and `Buffer.toString()` instructions that the `render` function uses for O(n) string building instead of O(n²) concatenation.
