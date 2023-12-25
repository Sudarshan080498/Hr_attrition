import os
import sys
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException
from src.utils import save_obj
from src.Config.configuration import (
    PREPROCESSED_OBJ_FILE,
    TRANSFORM_TRAIN_FILE_PATH,
    TRANSFORM_TEST_FILE_PATH,
    FE_OBJ_FILE_PATH,
    DATASET_PATH,
)

df = pd.read_csv(DATASET_PATH)


class Feature_Engineering():
    def __init__(self):
        logging.info("**********Feature Engineering started ************")

    def transform_data(self,df):
        try:
            
        # Drop unnecessary columns
            df.drop(['Over18', 'EmployeeCount', 'StandardHours', 'EmployeeNumber'],  axis = 1, inplace = True)
            df['Attrition'] = df['Attrition'].apply(lambda x: 1 if x == 'Yes' else 0)
            df['OverTime'] = df['OverTime'].apply(lambda x: 1 if x == 'Yes' else 0)


            logging.info("Dropped unnecessary columns")
            return df
        
        except Exception as e:
            raise CustomException( e, sys)
        
    def fit(self,X,y=None):
        return self
    
    def transform(self,X:pd.DataFrame, y=None):
        try:    
            transformed_df=self.transform_data(X)
                
            return transformed_df
        except Exception as e:
            raise CustomException(e,sys) from e
        

@dataclass
class DataTransformationConfig:
    processed_obj_file_path = PREPROCESSED_OBJ_FILE
    transform_train_path = TRANSFORM_TRAIN_FILE_PATH
    transform_test_path = TRANSFORM_TEST_FILE_PATH
    feature_engg_obj_path = FE_OBJ_FILE_PATH


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()


    def get_data_transformation_obj(self,df):
        try: 
            if isinstance(df, pd.DataFrame):
                df = pd.DataFrame(df, columns=df.columns)
            elif isinstance(df, np.ndarray):
                df = pd.DataFrame(df)
            numerical_feature = list(df.select_dtypes(include=['number']).columns)
            categorical_features = list(df.select_dtypes(exclude=['int64', 'float64']).columns)

            # Numerical pipeline
            numerical_pipeline = Pipeline(steps = [
                ('impute', SimpleImputer(strategy = 'constant', fill_value=0)),
                ('scaler', StandardScaler(with_mean=False))
            ])

                # Categorical Pipeline
            categorical_pipeline = Pipeline(steps = [
                ('impute', SimpleImputer(strategy = 'most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown = 'ignore')),
                ('scaler', StandardScaler(with_mean=False))
            ])

            preprocessor = ColumnTransformer(transformers=[
                ('numerical_pipeline', numerical_pipeline, numerical_feature),
                ('categorical_pipeline', categorical_pipeline, categorical_features)
            ])

            logging.info("Pipeline Steps Completed")
            return preprocessor
        except Exception as e:
            raise CustomException( e,sys)
    

    def get_feature_engineering_object(self):
        try:
            feature_engineering = Pipeline(steps = [("fe",Feature_Engineering())])

            return feature_engineering

        except Exception as e:
            raise CustomException( e,sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
        

            logging.info("Obtaining FE steps object")
            fe_obj = self.get_feature_engineering_object()

            train_df = fe_obj.fit_transform(train_df)
            test_df = fe_obj.transform(test_df)

            train_df.to_csv("train_data.csv")
            test_df.to_csv("test_data.csv")
            
            # print("train df columns:",train_df.columns)
            # print("test df columns:",test_df.columns)
        
            # processing_obj = self.get_data_transformation_obj(train_df)

            target_columns_name = "Attrition"

            X_train = train_df.drop(columns=target_columns_name, axis=1)
            y_train = train_df[target_columns_name]
            X_test = test_df.drop(columns= target_columns_name,  axis =1)
            y_test = test_df[target_columns_name]
            X_train = pd.DataFrame(X_train, columns=X_train.columns)
            X_test = pd.DataFrame(X_test, columns=test_df.columns)
           

            processing_obj = self.get_data_transformation_obj(X_train)


            X_train = processing_obj.fit_transform(X_train)

            # fe_obj.fit_transform(X_train)


            # processing_obj = self.get_data_transformation_obj(X_train)

            X_test = processing_obj.transform(X_test)
            train_arr = np.c_[X_train, np.array(y_train)]
            test_arr = np.c_[X_test, np.array(y_test).reshape(-1, 1)]


            df_train = pd.DataFrame(train_arr)
            df_test = pd.DataFrame(test_arr)


            os.makedirs(os.path.dirname(self.data_transformation_config.transform_train_path), exist_ok=True)
            df_train.to_csv(self.data_transformation_config.transform_train_path, index = False, header = True)

            os.makedirs(os.path.dirname(self.data_transformation_config.transform_test_path), exist_ok=True)
            df_test.to_csv(self.data_transformation_config.transform_test_path, index = False, header = True)


            save_obj(file_path = self.data_transformation_config.processed_obj_file_path,
                     obj = fe_obj)

            save_obj(file_path = self.data_transformation_config.feature_engg_obj_path,
                     obj = fe_obj)
            return(train_arr,
                   test_arr,
                self.data_transformation_config.processed_obj_file_path
            )
        

        except Exception as e:
            raise CustomException(e,sys)
