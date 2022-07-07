import logging

class Logger:
    def __init__(self, appname, level = 'DEBUG'):
        self.logger = logging.getLogger(appname)
        hdlr = logging.FileHandler('/var/tmp/{}.log'.format(appname))
        formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
        hdlr.setFormatter(formatter)
        self.logger.addHandler(hdlr)
        self.logger.setLevel(logging.WARNING)
        if level == 'DEBUG':
            self.logger.setLevel(logging.DEBUG)
        elif level == 'INFO':
            self.logger.setLevel(logging.INFO)
        elif level == 'ERROR':
            self.logger.setLevel(logging.ERROR)
