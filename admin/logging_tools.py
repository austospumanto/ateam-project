import logging
import logging.config
from logging import DEBUG, INFO, WARNING, ERROR


def setup_logging():
    # Imported here to avoid cyclical import issues
    from admin.config import project_config

    config = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'simple': {
                'format': '%(asctime)s %(levelname)-8s %(name)-15s %(message)s'
            }
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'level': project_config.LOG_LEVEL,
                'formatter': 'simple'
            }
        },
        'root': {
            'level': DEBUG,
            'handlers': ['console']
        },
        'loggers': {
            'retry': {'level': INFO},
            'urllib3': {'level': WARNING},
            'requests': {'level': WARNING},
        }
    }

    logging.config.dictConfig(config)
