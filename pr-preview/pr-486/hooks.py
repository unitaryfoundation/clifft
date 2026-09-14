import logging


class _SuppressNanobindAnnotationWarnings(logging.Filter):
    # nanobind-bound parameters lack Python type annotations; griffe warns
    # about this when inspecting the compiled extension. These are not real
    # doc errors, so drop them so mkdocs build --strict passes.
    # Scoped to WARNING level and griffe/mkdocstrings origins only so that
    # ERROR-level records and unrelated loggers are never silenced.
    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno > logging.WARNING:
            return True
        if not record.name.startswith(
            ("griffe", "mkdocs.plugins.griffe", "mkdocs.plugins.mkdocstrings")
        ):
            return True
        return "No type or annotation for parameter" not in record.getMessage()


def on_config(config):
    filt = _SuppressNanobindAnnotationWarnings()
    for handler in logging.getLogger("mkdocs").handlers:
        handler.addFilter(filt)
    return config
