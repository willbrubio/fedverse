import logging
import sys

# Configure once, at the pipeline entry point (not inside library modules).
# Route status to stderr so stdout stays clean for actual data output.
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",       # We handle our own prefixes; skip logging's default noise
    stream=sys.stderr,
)
log = logging.getLogger("fedlib")


# A tiny helper class instead of scattered print()s.
# fixed vocabulary for different steps
# symbol zoo. Indentation encodes depth; the glyph just flavors the line.
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


status = Status()