import logging
import logging.config

ERROR_LOG_FILENAME = ".zizou-processing-errors.log"

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {      
        "default": {  # The formatter name, it can be anything that I wish
            "format": "%(asctime)s:%(name)s:%(process)d:%(lineno)d " "%(levelname)s %(message)s",  #  What to add in the message
            "datefmt": "%Y-%m-%d %H:%M:%S",  # How to display dates
        },
        "json": {  # The formatter name
         "()": "pythonjsonlogger.jsonlogger.JsonFormatter",  # The class to instantiate!
            # Json is more complex, but easier to read, display all attributes!
            "format": """
                    asctime: %(asctime)s
                    created: %(created)f
                    filename: %(filename)s
                    funcName: %(funcName)s
                    levelname: %(levelname)s
                    levelno: %(levelno)s
                    lineno: %(lineno)d
                    message: %(message)s
                    module: %(module)s
                    msec: %(msecs)d
                    name: %(name)s
                    pathname: %(pathname)s
                    process: %(process)d
                    processName: %(processName)s
                    relativeCreated: %(relativeCreated)d
                    thread: %(thread)d
                    threadName: %(threadName)s
                    exc_info: %(exc_info)s
                """,
            "datefmt": "%Y-%m-%d %H:%M:%S",  # How to display dates
        },
    }, 
    "handlers": {
        "logfile": {  # The handler name
            "formatter": "json",  # Refer to the formatter defined above
            "level": "ERROR",  # FILTER: Only ERROR and CRITICAL logs
            "class": "logging.handlers.RotatingFileHandler",  # OUTPUT: Which class to use
            "filename": ERROR_LOG_FILENAME,  # Param for class above. Defines filename to use, load it from constant
            "backupCount": 2,  # Param for class above. Defines how many log files to keep as it grows
        }, 
        "simple": {  # The handler name
            "formatter": "default",  # Refer to the formatter defined above
            "class": "logging.StreamHandler",  # OUTPUT: Same as above, stream to console
            "stream": "ext://sys.stdout",
        },
    },
    "loggers": { 
        "zizou": {  # The name of the logger, this SHOULD match your module!
            "level": "DEBUG",  # FILTER: only INFO logs onwards from "tryceratops" logger
            "handlers": [
                "simple",  # Refer the handler defined above
            ],
        },
    },
    "root": {
        "level": "ERROR",  # FILTER: only INFO logs onwards
        "handlers": [
            "logfile",  # Refer the handler defined above
        ]
    },
}

logging.config.dictConfig(LOGGING_CONFIG)