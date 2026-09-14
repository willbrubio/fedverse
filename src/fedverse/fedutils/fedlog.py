import logging
import sys

# Configure once, at the pipeline entry point (not inside library modules).
# Route status to stderr so stdout stays clean for actual data output.
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",       # We handle our own prefixes; skip logging's default noise
    stream=sys.stderr,
)

# Suppress verbose font subsetting / PDF export noise from downstream libraries.
# Keep the fedverse logger at INFO so project status messages still show.
for _name in (
    "fontTools",
    "fontTools.subset",
    "matplotlib",
    "matplotlib.font_manager",
    "matplotlib.backends",
):
    logging.getLogger(_name).setLevel(logging.WARNING)

log = logging.getLogger("fedverse")


# A tiny helper class instead of scattered print()s.
# fixed vocabulary for different steps
# symbol zoo. Indentation encodes depth; the glyph flavors the line.
class Status:
    def __init__(self):
        self._depth = 0  # current nesting level, drives indentation


    def _emit(self, glyph, msg, level=logging.INFO):
        indent = "  " * self._depth
        log.log(level, f"{indent}{glyph} {msg}")


    # Top-level stage: "==> Processing FR1 assays"
    def step(self, msg):
        self._depth = 0
        self._emit("==>", msg)
        self._depth = 1  # subsequent sub-messages nest under this step


    # A sub-item under the current step: "  -> loading zip"
    def sub(self, msg):
        self._emit("->", msg)


    # Terminal states get distinct, greppable prefixes so logs are scannable
    def ok(self, msg):
        self._emit("[ok]", msg)


    def warn(self, msg):
        self._emit("[warn]", msg, level=logging.WARNING)


    # Use this inside except blocks — pass exc_info to capture the traceback
    # that a bare `except: pass` would otherwise have eaten.
    def fail(self, msg, exc_info=False):
        self._emit("[fail]", msg, level=logging.ERROR)
        if exc_info:
            log.exception(msg)  # dumps the full traceback at ERROR level


    
    ######################## For printing tables ########################
    def _emit_block(self, rendered, level=logging.INFO):
        indent = "  " * (self._depth + 1)
        block = "\n".join(indent + line for line in rendered.splitlines())
        log.log(level, block)


    # Print a statistical summary of a DataFrame for a quick sanity check mid-pipeline. 
    # Transposes to best print onto terminal
    def preview(self, df, n=None, msg=None, round_to=3):
        caption = msg or "preview"

        # Guard against non-DataFrame input (None, a Series, a stray list) 
        try:
            n_rows, n_cols = df.shape
        except AttributeError:
            self._emit(
                "[preview]",
                f"{caption}: not a DataFrame ({type(df).__name__})",
                level=logging.WARNING,
            )
            return

        # Lead with the true shape of the full frame — describe() only reports on
        # a subset of columns (numeric by default), so the caller still needs the
        # real row/column counts to know what they're looking at.
        self._emit("[preview]", f"{caption} — {n_rows} rows x {n_cols} cols")

        # describe() defaults to numeric columns only. For a metrics frame that's
        # exactly right (filename/string columns aren't meaningfully summarizable),
        # but it means an all-object frame would summarize object columns instead,
        # or an empty frame yields nothing — so guard against an empty result
        # rather than emitting a blank block.
        summary = df.describe().T
        if summary.empty:
            self._emit(
                "[preview]",
                f"{caption}: nothing to summarize (no numeric columns?)",
                level=logging.WARNING,
            )
            return

        # Round for readability — raw describe() output carries full float
        # precision (e.g. 184.56458934...) which is noise for a glance-check.
        summary = summary.round(round_to)

        # describe().T is naturally narrow, but its ROW labels are your original
        # column names, which can be long ("Left Poke with Pellet"). line_width
        # still guards against an unusually wide stat block; it rarely triggers here.
        rendered = summary.to_string(line_width=100)

        # to_string() is multi-line; indent EVERY line one level under the caption
        # so the whole block stays aligned. Indenting only the first line would
        # leave the rest hugging the left margin.
        self._emit_block(rendered)

    # Show a DataFrame VERBATIM (unlike preview(), which summarizes via describe).
    # This is for small result tables — per-group counts, mappings — where the rows
    # themselves are the answer, not their statistics.
    def table(self, df, msg=None, max_rows=20):
        caption = msg or "table"

        # Same non-DataFrame guard as preview(): a stray None/Series/list degrades
        # to a warning instead of throwing on the .shape/.to_string() calls below.
        try:
            n_rows, n_cols = df.shape
        except AttributeError:
            self._emit(
                "[table]",
                f"{caption}: not a DataFrame ({type(df).__name__})",
                level=logging.WARNING,
            )
            return

        # Caption carries the TRUE shape so a truncated view can't mislead.
        self._emit("[table]", f"{caption} — {n_rows} rows x {n_cols} cols")

        # Cap rows so an accidentally-large frame can't flood the log; small
        # summaries (your group counts) pass through untouched.
        shown = df.head(max_rows) if max_rows is not None else df
        self._emit_block(shown.to_string())

        # Be explicit when the view was cut — otherwise it reads as the full table.
        if max_rows is not None and n_rows > max_rows:
            self._emit_block(f"... {n_rows - max_rows} more rows")


status = Status()