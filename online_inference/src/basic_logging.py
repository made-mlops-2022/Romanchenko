log_conf = {
    "version": 1,
    "formatters": {
        "stdout_formatter": {
            "format": "%(asctime)s\t%(levelname)s\t%(funcName)s\t%(message)s",
        },
    },
    "handlers": {
        "stream_handler": {
            "level": "DEBUG",
            "formatter": "stdout_formatter",
            "class": "logging.StreamHandler",
        },
    },
    "loggers": {
        "": {
            "level": "DEBUG",
            "handlers": ["stream_handler"],
        }
    }
}
