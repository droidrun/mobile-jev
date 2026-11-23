import logging

from mobilerun.logs.handlers import CLILogHandler, LogLevel, configure_logging


def _setup_cli_logging(debug: bool, level: LogLevel | None = None) -> CLILogHandler:
    """Configure the mobilerun logger with a CLILogHandler."""
    handler = CLILogHandler()
    handler.setFormatter(
        logging.Formatter("%(levelname)s %(name)s %(message)s", "%H:%M:%S")
        if debug
        else logging.Formatter("%(message)s")
    )
    configure_logging(debug=debug, handler=handler, level=level)
    return handler
