import logging
import datetime

__all__ = ['Log']
class Log:
    """
    Log class, the default filename is time.log
    usage:
        logger = Log(filename)
        logger.info(string)
    """
    def __init__(self,filename=None):
        if filename is None:
            self.filename = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M") + ".log"
        else:
            self.filename = filename + ".log"
        self.logger = logging.Logger('bin')
        self.logger.setLevel(logging.DEBUG)

        self.srhandle = logging.StreamHandler()

        self.fihandle = logging.FileHandler(filename=self.filename,mode='w')

        self.srhandle.setLevel(logging.DEBUG)
        self.fihandle.setLevel(logging.DEBUG)
        
        self.fmt = logging.Formatter('%(asctime)10s|   %(message)s',datefmt="%Y-%m-%d_%H:%M")
        self.srhandle.setFormatter(self.fmt)
        self.fihandle.setFormatter(self.fmt)

        self.logger.addHandler(self.srhandle)
        self.logger.addHandler(self.fihandle)

    def info(self,message):
        self.logger.info(message)

