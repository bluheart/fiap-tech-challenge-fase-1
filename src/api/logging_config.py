from pythonjsonlogger import jsonlogger
import logging
import datetime
import sys

class CustomJsonFormatter(jsonlogger.JsonFormatter):
    def add_fields(self, log_record, record, message_dict):
        super().add_fields(log_record, record, message_dict)
        log_record['timestamp'] = datetime.utcnow().isoformat()
        log_record['level'] = record.levelname
        log_record['service'] = 'mlp-api'
        log_record['logger'] = record.name
        if not log_record.get('message'):
            log_record['message'] = record.getMessage()


def setup_logging(level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger("api")
    logger.setLevel(getattr(logging, level.upper()))

    handler = logging.StreamHandler(sys.stdout)
    formatter = CustomJsonFormatter('%(timestamp)s %(level)s %(service)s %(message)s %(logger)s')
    handler.setFormatter(formatter)

    logger.handlers = []
    logger.addHandler(handler)
    logger.propagate = False
    return logger

logger = setup_logging()