# Nostos Improvement Notes

Issues discovered during development that should be addressed.

## MVars with Function Types Cause Deadlock

**Problem**: When stdlib modules contain MVars with function types, importing those modules causes the server/program to deadlock during initialization.

**Example of problematic code**:
```nostos
# These cause deadlock when in stdlib:
mvar eventHandlers: List[(String, () -> Unit)] = []
mvar componentRenderers: List[(String, (RenderState) -> (Html, RenderState))] = []
mvar reactiveHandlers: List[(String, (String, RenderState) -> RenderState)] = []
```

**Symptoms**:
- Server hangs on startup without printing anything
- No error message, just silent deadlock
- Works fine when same MVars are defined in user code (non-stdlib)

**Workaround**: Don't use function types in MVars that are defined in stdlib modules. Users can define their own MVars with function types in their application code.

**Root Cause**: Unknown - needs investigation. Possibly related to how stdlib modules are initialized in parallel or how function types interact with the MVar initialization order.

**Discovered**: While implementing reactive web framework in stdlib/reactiveWeb.nos

## Multiline Record Types Not Supported

**Problem**: Record type definitions must be on a single line.

**This fails**:
```nostos
type ReactiveApp = {
    renderPage: (RenderState) -> (Html, RenderState),
    handleEvent: (String, RenderState) -> RenderState
}
```

**This works**:
```nostos
type ReactiveApp = { renderPage: (RenderState) -> (Html, RenderState), handleEvent: (String, RenderState) -> RenderState }
```

**Priority**: Low - single-line works fine

## Missing String.drop Function

**Problem**: `String.drop(n)` doesn't exist, must use `String.substring` instead.

**Workaround**:
```nostos
# Instead of: String.drop(path, 10)
# Use: String.substring(path, 10, String.length(path))
```

**Priority**: Low - easy workaround exists

## HTMX Integration Notes

**Idiomorph Extension Issues**: The idiomorph extension (`hx-ext="morph"` with `hx-swap="morph:outerHTML"`) did not work correctly in our setup. Forms would only submit once, then subsequent clicks wouldn't trigger HTMX.

**Working Configuration**:
```nostos
# Forms use standard HTMX swap (no idiomorph)
Element("form", [
    ("action", "/increment"),
    ("method", "post"),
    ("hx-post", "/increment"),
    ("hx-target", "body"),
    ("hx-swap", "outerHTML")  # NOT morph:outerHTML
], [...])
```

**Partial Updates**: For HTMX to work properly with body swaps:
1. Check for `HX-Request` header in server
2. Return just the `<body>` element for HTMX requests
3. Return full HTML page for initial/navigation requests

**Example pattern**:
```nostos
isHtmxRequest(req) = getParam(req.headers, "HX-Request") == "true"

handleRequest(req) = {
    html = if isHtmxRequest(req) then renderBody() else renderFullPage()
    respondHtml(req, render(html))
}
```

**Discovered**: While building Selenium tests for reactive web framework
