import logging
import os


def set_logging_level(logging_level, exp_label, start_date):
    logging_level = logging_level.lower()

    if logging_level == "critical":
        level = logging.CRITICAL
    elif logging_level == "warning":
        level = logging.WARNING
    elif logging_level == "info":
        level = logging.INFO
    else:
        level = logging.DEBUG

    filename = os.path.join('logs',start_date+'_'+exp_label+'.log')

    logging.basicConfig(
        filename=filename,
        format='%(asctime)s [%(filename)s]: %(message)s', 
        datefmt='%Y-%m-%d:%H:%M:%S', 
        level=level
    )
