import os
import shutil

from mlProject import logger
from mlProject.utils.common import get_size
from mlProject.entity.config_entity import DataIngestionConfig


from pathlib import Path

class DataIngestion:
    def __init__(self, config:DataIngestionConfig):
        self.config = config

    def retrieve_datas(self):
        if not os.path.exists(self.config.root_dir + self.config.save_file):
            shutil.copyfile(self.config.source_file, self.config.root_dir + "/" + self.config.save_file)
            logger.info(f'data file retrieved!')
        else:
            logger.info(f'File already exists of size : {get_size(Path(self.config.save_dir + self.config.save_file))}')