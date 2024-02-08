from mlProject.config.configuration import ConfigurationManager
from mlProject.components.custom_data_ingestion import DataIngestion
from mlProject import logger




class DataIngestionTrainingPipeline:
    def __init__(self):
        pass
    
    def main(self):
        config = ConfigurationManager() 
        data_ingestion_config = config.get_data_ingestion_config()
        data_ingestion= DataIngestion(config=data_ingestion_config)
        data_ingestion.retrieve_datas()