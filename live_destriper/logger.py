import logging
import os
import sys 

def get_logger(log_path):
    logger_name = f"destriper_{log_path}"
    logger = logging.getLogger(name=logger_name)

    # set level of logs
    logger.setLevel(logging.DEBUG)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    if len(logger.handlers) == 0:
        # Set file and console handlers
        logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
        FileOutputHandler = logging.FileHandler(filename = log_path)
        FileOutputHandler.setFormatter(logFormatter)
        ConsoleHandler = logging.StreamHandler(sys.stdout)
        ConsoleHandler.setFormatter(logFormatter)
        logger.addHandler(FileOutputHandler)
        logger.addHandler(ConsoleHandler)
        logger.propagate = False 
    return logger 

def get_default_logger():
    logger = logging.getLogger(name="destriper")
    logger.setLevel(logging.DEBUG)

    if not logger.hasHandlers():
        logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
        ConsoleHandler = logging.StreamHandler(sys.stdout)
        ConsoleHandler.setFormatter(logFormatter)
        logger.addHandler(ConsoleHandler)
        logger.propagate = False 
    return logger