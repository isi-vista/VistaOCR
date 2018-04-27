import datetime
import logging


def setup_custom_logger(name, filename, path="/tmp"):
    logFormatter = logging.Formatter("%(asctime)s [%(filename)s]  %(message)s")

    fileHandler = logging.FileHandler(
        "{0}/{1}.log".format(path, filename + '_' + datetime.datetime.now().strftime('%y_%m_%d_%H_%M_%S')))
    fileHandler.setFormatter(logFormatter)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.addHandler(fileHandler)
    logger.addHandler(consoleHandler)
    return logger
