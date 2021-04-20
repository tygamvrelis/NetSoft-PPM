#!/usr/bin/python
# Author: Tyler Gamvrelis
# General utilities

# Standard library imports
import logging
import os
import sys

def get_price_uri():
    return '/resources/prices'

def get_bid_uri():
    return '/resources/bids'

def get_logs_dir():
    ppm_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        os.pardir
    )
    logs_dir = os.path.join(ppm_path, 'app', 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    return logs_dir

def get_metrics_log_path():
    """Returns the path where logs of metrics are located."""
    logs_dir = get_logs_dir()
    METRICS_LOG_PATH = os.path.join(
        logs_dir,
        'metrics.log'
    )
    return METRICS_LOG_PATH

def get_schema_dir():
    """Returns the path where schemas are located."""
    SCHEMA_DIR = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'schemas'
    )
    return SCHEMA_DIR

def get_script_path():
    """Gets the path where the script is running."""
    return os.path.dirname(os.path.realpath(sys.argv[0]))

def setup_logger(log_level, fname):
    """
    Sets up the logger.

    Sources:
        - https://stackoverflow.com/questions/6386698/how-to-write-to-a-file-using-the-logging-python-module
        - https://stackoverflow.com/questions/14058453/making-python-loggers-output-all-messages-to-stdout-in-addition-to-log-file

    Args:
        log_level : str
            String indicating the log level
        fname : str
            Name of application file setting up the logger
    """
    fname_base = os.path.basename(fname)
    fname_base = os.path.splitext(fname_base)[0]
    logs_dir = get_logs_dir()
    log_name = os.path.join(logs_dir, fname_base + '.log')
    if log_level is not None:
        numeric_level = getattr(logging, log_level.upper(), None)
        if not isinstance(numeric_level, int):
            raise ValueError('Invalid log level: %s' % log_level)
        logging.basicConfig(
            filename=log_name,
            filemode='w',
            format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
            datefmt='%H:%M:%S',
            level=numeric_level
        )
        if numeric_level >= logging.DEBUG:
            # As long as logging is not disabled, always output INFO to stdout
            handler = logging.StreamHandler(sys.stdout)
            handler.setLevel(logging.INFO)
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            root = logging.getLogger()
            root.addHandler(handler)
