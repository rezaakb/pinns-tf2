import logging

def get_pylogger(name: str = __name__) -> logging.Logger:
    """Initializes a TensorFlow 2 compatible python command line logger.

    :param name: The name of the logger, defaults to ``__name__``.

    :return: A logger object.
    """
    logger = logging.getLogger(name)

    return logger
