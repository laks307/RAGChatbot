import logging

def configure_logging(name = 'chatbot'):
    logger = logging.getLogger(name)
    logger.setLevel("DEBUG")

    log = logging.StreamHandler()
    log.setLevel("DEBUG")

    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] : %(message)s")
    log.setFormatter(formatter)
    # logger.addHandler(log)

    if not logger.hasHandlers():
        logger.addHandler(log)

    return logger

logger = configure_logging()
logger.error('login working')