import logging
import logging.config
import os

DEBUG_LOG_FILENAME = os.environ.get("ZIZOULOG", ".zizou-processing.log")

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {  # The formatter name, it can be anything that I wish
            "format": "%(asctime)s:%(name)s:%(process)d:%(lineno)d "
            "%(levelname)s %(message)s",  #  What to add in the message
            "datefmt": "%Y-%m-%d %H:%M:%S",  # How to display dates
        },
        "json": {  # The formatter name
            "()": "pythonjsonlogger.jsonlogger.JsonFormatter",  # The class to instantiate!
            # Json is more complex, but easier to read, display all attributes!
            "format": """
                    asctime: %(asctime)s
                    filename: %(filename)s
                    funcName: %(funcName)s
                    levelname: %(levelname)s
                    lineno: %(lineno)d
                    message: %(message)s
                    module: %(module)s
                    name: %(name)s
                    pathname: %(pathname)s
                """,
            "datefmt": "%Y-%m-%d %H:%M:%S",  # How to display dates
        },
    },
    "handlers": {
        "logfile": {  # The handler name
            "formatter": "json",  # Refer to the formatter defined above
            "level": "DEBUG",  # FILTER: Only ERROR and CRITICAL logs
            "class": "logging.handlers.RotatingFileHandler",  # OUTPUT: Which class to use
            "filename": DEBUG_LOG_FILENAME,  # Param for class above. Defines filename to use, load it from constant
            "backupCount": 2,  # Param for class above. Defines how many log files to keep as it grows
        },
        "simple": {  # The handler name
            "formatter": "default",  # Refer to the formatter defined above
            "level": "WARNING",
            "class": "logging.StreamHandler",  # OUTPUT: Same as above, stream to console
            "stream": "ext://sys.stdout",
        },
    },
    "loggers": {
        "zizou": {  # The name of the logger, this SHOULD match your module!
            "level": "INFO",
            "handlers": [
                "simple",
                "logfile",
            ],
        },
    },
    "root": {
        "level": "INFO",
        "handlers": [
            "logfile",
            "simple",
        ],
    },
}


def setup_logging():
    logging.config.dictConfig(LOGGING_CONFIG)
