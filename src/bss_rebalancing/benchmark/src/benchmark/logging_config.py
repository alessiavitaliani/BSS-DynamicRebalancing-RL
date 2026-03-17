import logging
from dataclasses import dataclass
from pathlib import Path


@dataclass
class LoggingConfig:
    level: int = logging.INFO
    log_dir: str = "logs"
    run_id: int | None = None
    console: bool = True
    logger_name: str = "bss"  # top-level namespace; set per script


def init_logging(config: LoggingConfig) -> None:
    """
    Initialise logging for the given namespace.

    Calling this again with the same logger_name clears existing handlers
    and reconfigures, so it is safe to call once per run.
    """
    Path(config.log_dir).mkdir(parents=True, exist_ok=True)

    root = logging.getLogger(config.logger_name)
    root.setLevel(config.level)

    # Clear existing handlers to allow safe re-init (e.g. different run_id)
    root.handlers.clear()

    fmt = logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s")

    log_file = Path(config.log_dir) / f"{config.logger_name}.log"

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)

    if config.console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(fmt)
        root.addHandler(console_handler)

    # Prevent messages propagating to the root Python logger
    root.propagate = False


def get_logger(name: str, logger_name: str = "bss") -> logging.Logger:
    """
    Get a child logger under the given top-level namespace.

    Args:
        name: sub-component name, e.g. "train", "env", "benchmark".
        logger_name: top-level namespace used in init_logging.
    """
    return logging.getLogger(f"{logger_name}.{name}")
