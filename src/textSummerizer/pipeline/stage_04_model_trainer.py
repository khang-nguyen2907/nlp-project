from textSummerizer.config.config import ConfigurationManager
from textSummerizer.conponents.model_trainer import ModelTrainer
from textSummerizer.logging import logger

class ModelTrainerPipeline: 
    def __init__(self) -> None:
        pass
    
    def main(self):
        config = ConfigurationManager()
        model_trainer_config = config.get_model_trainer_config()
        model_trainer = ModelTrainer(config=model_trainer_config)
        model_trainer.train()