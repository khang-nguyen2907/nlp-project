from textSummerizer.config.config import ConfigurationManager
from textSummerizer.conponents.data_validation import DataValidation
from textSummerizer.logging import logger

class DataValidationTrainingPipeline: 
    def __init__(self) -> None:
        pass
    
    def main(self):
        config = ConfigurationManager()
        data_validation_config = config.get_data_validation_config()
        data_validation = DataValidation(config=data_validation_config)
        data_validation.validation_all_files_exist()
