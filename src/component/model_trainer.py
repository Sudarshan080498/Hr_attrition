import os
import sys
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from dataclasses import dataclass
from src.logger import logging
from src.Config.configuration import *
from src.exception import CustomException
from src.utils import save_obj
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score,recall_score, f1_score, confusion_matrix
from src.utils import evaluate_model
class ModelTrainerConfig:
    trained_model_file_path = MODEL_FILE_PATH

class ModelTrainer():

    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig

    def initiate_model_training(self, train_array, test_array):

        try:
            X_train, y_train,X_test, y_test = (train_array[:, :-1], train_array[:, -1],
                                                test_array[:, :-1], test_array[:, -1])
            

            models = {
                    "Logistic Regression": LogisticRegression(),
                    "Artificil Neural Network": MLPClassifier(max_iter=1000, random_state=42),
                    "Random Forest Classifier": RandomForestClassifier(random_state=42),
            }


            model_report: dict = evaluate_model(X_train,y_train,X_test, y_test,models)
            print (model_report)

            best_model_name, best_model_score = max(model_report.items(), key=lambda x: x[1])
            
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            print(f"Best Model found , Model  Name:{best_model_name}, Accuracy_score:{best_model_score}")
            logging.info(f"Best Model found , Model  Name:{best_model_name}, Accuracy_score:{best_model_score}")



            save_obj(file_path=self.model_trainer_config.trained_model_file_path,
                     obj = best_model)
        except Exception as e:
            raise CustomException(e, sys)