import logging


class _SuppressGriffeNanobindWarnings(logging.Filter):
    # nanobind-bound parameters lack Python type annotations; griffe warns
    # about this when inspecting the compiled extension. These are not real
    # doc errors, so drop them so mkdocs build --strict passes.
    def filter(self, record: logging.LogRecord) -> bool:
        return "No type or annotation for parameter" not in record.getMessage()


def on_config(config):
    filt = _SuppressGriffeNanobindWarnings()
    for handler in logging.getLogger("mkdocs").handlers:
        handler.addFilter(filt)
    return config
