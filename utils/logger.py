import logging 
import sys

class CustomFormatter(logging.Formatter):
    """
    Using custom formatter to inlcude dictionary type 
    """

    def __init__(self,
                 fmt='%(asctime)s | [%(levelname)s] | %(filename)s:%(funcName)s:line %(lineno)d | %(message)s',
                 datefmt='%Y-%m-%d %H:%M:%S',
                 style="%"):

        logging.Formatter.__init__(self, fmt=fmt, datefmt=datefmt, style=style)

    def format(self, record):
        logmsg = super(CustomFormatter, self).format(record)
        return {'msg': logmsg,
                'args':record.args}

class PythonLogger():
    def __init__(self):
        # Create a custom logger 
        self.logger = logging.getLogger('supervised-segmentation')
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False # to remove duplicate logs
        
        # Create handlers 
        self.stream_handler = logging.StreamHandler(stream=sys.stdout)
        
        # Create formatters 
        s_format = logging.Formatter(fmt='%(asctime)s | [%(levelname)s] | %(filename)s:%(funcName)s:line %(lineno)d | %(message)s',
                                     datefmt='%Y-%m-%d %H:%M:%S')
        self.stream_handler.setFormatter(s_format)
        #self.stream_handler.setFormatter(CustomFormatter)
        
        # add handler to logger
        if (self.logger.hasHandlers()):
                self.logger.handlers.clear()
        self.logger.addHandler(self.stream_handler)
        
    def send_log_debug(self, msg, error_dict):
        self.logger.debug(msg, error_dict)
