import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer
@dataclass
class DataIngestionConfig:
    def __init__(self, category):
        self.train_data_path = os.path.join('artifacts', f"{category}_train.csv")
        self.test_data_path = os.path.join('artifacts', f"{category}_test.csv")
        self.raw_data_path = os.path.join('artifacts', f"{category}_data.csv")

class DataIngestion:
    def __init__(self, category):
        self.ingestion_config = DataIngestionConfig(category)
        self.category = category

    def initiate_data_ingestion(self):
        logging.info(f"{self.category} için veri alma işlemi başlatıldı.")
        try:
            df = pd.read_csv(f'notebook/data/{self.category}.csv')
            logging.info(f'{self.category} veri seti okundu.')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info("Train test split başlatıldı.")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info(f"{self.category} için veri alma işlemi tamamlandı.")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e, sys)
        
if __name__ == "__main__":
    categories = ["tablet", "telefon", "monitor", "headset", "akilli_saat", "laptop"]
    for category in categories:
        print(f"{category} için işlemler başlatılıyor...")
        obj = DataIngestion(category)
        train_data, test_data = obj.initiate_data_ingestion()

        data_transformation = DataTransformation(category)
        train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data, test_data)

        modeltrainer = ModelTrainer(category)
        print(f"{category} için skor:", modeltrainer.initiate_model_trainer(train_arr, test_arr))