from src.component import *
from src.Config.configuration import *
import pandas as pd
import numpy as np
import os,sys
from src.logger import logging
from src.exception import CustomException
import pickle
from src.utils import load_model
from sklearn.pipeline import Pipeline

PREDICTION_FOLDER = "batch_prediction"
PREDICTION_CSV = "prediction_csv"
PREDICTION_FILE = "prediction.csv"
FEATURE_ENGINEERING_FOLDER = "feature_engg"

ROOT_DIR = os.getcwd()
BATCH_PREDICTION = os.path.join(ROOT_DIR,PREDICTION_FOLDER,PREDICTION_CSV)
FEATURE_ENGG = os.path.join(ROOT_DIR,PREDICTION_FOLDER,FEATURE_ENGINEERING_FOLDER)

class batch_prediction:
    def __init__(self, input_file_path,
                 model_file_path,
                 transformer_file_path,
                 feature_engineering_file_path) -> None:
        
        self.input_file_path = input_file_path
        self.model_file_path = model_file_path
        self.transformer_file_path = transformer_file_path
        self.feature_engineering_file_path = feature_engineering_file_path


    def start_batch_prediction(self):
        try:
            # Load the Feature engineering pipeline path
            with open(self.feature_engineering_file_path, 'rb') as f:
                featur_pipeline = pickle.load(f)

            # Load data transformation pipeline path
            with open(self.transformer_file_path, 'rb') as f:
                processor = pickle.load(f)

            # Load the model
            model = load_model(file_path=self.model_file_path)
            logging.info(f"Logistic Regression Model loaded successfully.")
            # create a feature engineering pipeline
            feature_engineering_pipeline = Pipeline([
                ("feature_engineering", featur_pipeline)
            ])

            df = pd.read_csv(self.input_file_path)
            logging.info(f"Original DataFrame:\n{df.head()}")

            # Apply Feature engineering Pipeline Steps
            df = feature_engineering_pipeline.transform(df)
            logging.info(f"DataFrame after feature engineering:\n{df.head()}")

            FEATURE_ENGINEERING_PATH = FEATURE_ENGG
            os.makedirs(FEATURE_ENGINEERING_PATH, exist_ok=True)
            file_path = os.path.join(FEATURE_ENGINEERING_PATH, "batch_feature_engg.csv")
            df.to_csv(file_path, index=False)

            # Attrition
            if 'Attrition' in df.columns:
                df = df.drop('Attrition', axis=1)
            else:
                logging.warning("'Attrition' column not found in DataFrame.")

            df.to_csv("Attrition_droped.csv")
            transform_data = processor.transform(df)

            file_path = os.path.join(FEATURE_ENGINEERING_PATH, 'processor.csv')

            # Print or log the data types of each column in transform_data
            logging.info(f"Data types after transformation:\n{transform_data.dtypes}")

            

            predictions = model.predict(transform_data)
            df_prediction = pd.DataFrame(predictions, columns=['prediction'])

            BATCH_PREDICTION_PATH = BATCH_PREDICTION
            os.makedirs(BATCH_PREDICTION_PATH, exist_ok=True)
            csv_path = os.path.join(BATCH_PREDICTION_PATH, 'output.csv')


            df_prediction.to_csv(csv_path, index=False)
            logging.info(f"Batch Prediction Done")

        except Exception as e:
            # Print the full traceback for debugging
            import traceback
            traceback.print_exc()
            CustomException(e, sys)
