import logging
import yaml
import sys
import os


class Config:
    """Loads parameters from config yaml file"""

    def __init__(self, path_to_config):

        self.path_to_config = path_to_config

        if os.path.isfile(path_to_config):
            pass
        else:
            self.path_to_config = '../{}'.format(self.path_to_config)

        with open(self.path_to_config, "r") as f:
            self.dictionary = yaml.load(f.read(), Loader=yaml.FullLoader)

        for k, v in self.dictionary.items():
            setattr(self, k, v)

    def make_dirs(self, _id, dir_name='data', structure='default'):
        '''Create directories for storing data'''

        if structure == 'default':
            paths = [f'{dir_name}', f'{dir_name}/%s' %
                     _id, f'{dir_name}/%s/models' % _id]
        else:
            paths = [f'{dir_name}', f'{dir_name}/%s' % _id]

        for p in paths:
            if not os.path.isdir(p):
                os.mkdir(p)


def setup_logging():
    '''Configure logging object to track parameter settings, training, and evaluation'''

    logger = logging.getLogger('py-ann')
    logger.setLevel(logging.INFO)

    stdout = logging.StreamHandler(sys.stdout)
    stdout.setLevel(logging.INFO)
    logger.addHandler(stdout)
    logger.propagate = False

    return logger
