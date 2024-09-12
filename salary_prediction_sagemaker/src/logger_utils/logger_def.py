import logging
import logging.config
import json
import os

# Ensure the logs directory exists
os.makedirs("logs", exist_ok=True)

script_dir = os.path.dirname(__file__)
log_config_file: str = os.path.join(script_dir, "logging_config.json")
# log_config_file:str = 'metal_audio_to_text/src/logger_utils/logging_config.json'

# Load the logging configuration
with open(log_config_file, "r") as f:
    log_config = json.load(f)
    logging.config.dictConfig(log_config)

# Create loggers
root_logger = logging.getLogger("")  # Root logger
console_logger = logging.getLogger("consoleLogger")
dev_logger = logging.getLogger("devLogger")
logger = root_logger if os.getenv("ENVIRONMENT") == "prod" else dev_logger

# Guideline:
# CRITICAL (50): Indicates a very serious error, the program may be unable to continue running. (logging.critical("A critical error occurred"))
# ERROR (40): Indicates a more serious problem that prevented the program from performing a certain task. (logging.error("An error occurred"))
# WARNING (30): Indicates something unexpected happened or a potential problem in the near future (e.g., 'disk space low'). The program is still working as expected. (logging.warning("This is a warning message"))
# INFO (20): Indicates general information about program execution, such as confirming that things are working as expected. (logging.info("This is an informational message"))
# DEBUG (10): Provides detailed information, typically of interest only when diagnosing problems. Used for debugging purposes. (logging.debug("This is a debug message"))
# NOTSET (0): This level is used to indicate that a handler or logger has no specific level set. If a logger's level is set to NOTSET, it will be treated as DEBUG. (logging.getLogger().setLevel(logging.NOTSET))
# ----------------------
# Example usage
# logger.info("This is an info message for the root logger")
# logger.error("This is an error message for the root logger")

# console_logger.info("This is an info message for the console logger")

# dev_logger.debug("This is a debug message for the dev logger")
# dev_logger.info("This is an info message for the dev logger")
# dev_logger.error("This is an error message for the dev logger")
