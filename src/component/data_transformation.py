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
from sklearn.preprocessing import MinMaxScaler
from src.utils import save_obj
from src.Config.configuration import (
    PREPROCESSED_OBJ_FILE,
    TRANSFORM_TRAIN_FILE_PATH,
    TRANSFORM_TEST_FILE_PATH,
    FE_OBJ_FILE_PATH,
    DATASET_PATH,
)

folder_path = r'D:\Sudarshan\HR\Hr_attrition\batch_prediction\UPLOADED_CSV_FILE'

# List all files in the folder
all_files = os.listdir(folder_path)

# Filter files to include only CSV files
csv_files = [file for file in all_files if file.endswith('.csv')]

# Sort CSV files by modification time (most recent first)
sorted_csv_files = sorted(csv_files, key=lambda x: os.path.getmtime(os.path.join(folder_path, x)), reverse=True)

# Choose the most recent CSV file
if sorted_csv_files:
    most_recent_csv = sorted_csv_files[0]

    # Create the full file path
    file_path = os.path.join(folder_path, most_recent_csv)

    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path)

    # Display the DataFrame
    print(df.head())
else:
    print("No CSV files found in the specified folder.")


class Feature_Engineering():

    def __init__(self):
        logging.info("**********Feature Engineering started ************")

    def transform_data(self, df):
        try:
            # Check if columns exist before dropping
            columns_to_drop = ['Over18', 'EmployeeCount', 'StandardHours', 'EmployeeNumber']
            existing_columns = df.columns
            columns_to_drop = [col for col in columns_to_drop if col in existing_columns]

            logging.info(f"Columns before one-hot encoding: {df.columns}")

            # Check if one-hot encoding has already been performed
            one_hot_columns = ['BusinessTravel_Non-Travel', 'Department_Human Resources', 'EducationField_Human Resources', 
                               'Gender_Female', 'JobRole_Healthcare Representative', 'MaritalStatus_Divorced']

            if any(col in df.columns for col in one_hot_columns):
                logging.warning("One-hot encoding already applied. Skipping the one-hot encoding step.")
                return df

            required_columns = ['BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                print("Missing columns in DataFrame X:", missing_columns)
                # Optionally, raise an exception or handle the missing columns here
            else:
                print("All required columns are present in DataFrame X.")

            X = pd.get_dummies(df, columns=required_columns)
            logging.info(f"DataFrame after one-hot encoding: {X}")

            # Drop columns if they exist
            if columns_to_drop:
                df.drop(columns=columns_to_drop, axis=1, inplace=True)
                logging.info(f"Dropped columns: {columns_to_drop}")
            else:
                logging.warning("Columns to drop not found in DataFrame.")

            return X

        except Exception as e:
            raise CustomException(e, sys)
        
    def fit(self,df,y=None):
        return self
    
    def transform(self,df:pd.DataFrame, y=None):
        try:    
            transformed_df=self.transform_data(df)
                
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
            numerical_feature = ['Age','DailyRate','DistanceFromHome','Education','EnvironmentSatisfaction','HourlyRate',
                                'JobInvolvement','JobLevel','JobSatisfaction','MonthlyIncome','MonthlyRate','NumCompaniesWorked','PercentSalaryHike','PerformanceRating','RelationshipSatisfaction',
                                'StockOptionLevel','TotalWorkingYears','TrainingTimesLastYear','WorkLifeBalance','YearsAtCompany','OverTime','YearsInCurrentRole','YearsSinceLastPromotion','YearsWithCurrManager']
            categorical_features = ['BusinessTravel','Department','EducationField','Gender','JobRole','MaritalStatus']

            # Numerical pipeline
            numerical_pipeline = Pipeline(steps = [
                ('impute', SimpleImputer(strategy = 'constant', fill_value=0)),
                ('scaler', StandardScaler(with_mean=False)),
                ('minmax_scaler', MinMaxScaler())
            ])

                # Categorical Pipeline
            categorical_pipeline = Pipeline(steps = [
                ('impute', SimpleImputer(strategy = 'most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown = 'ignore')),
                # ('scaler', StandardScaler(with_mean=False))
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

            logging.info("Transforming training data")
            train_df = fe_obj.fit_transform(train_df)
            logging.info(f"Transformed training data columns: {train_df.columns}")

            logging.info("Transforming test data")
            test_df = fe_obj.transform(test_df)
            logging.info(f"Transformed test data columns: {test_df.columns}")

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
            y_train = y_train.values
            y_test = y_test.values

            processing_obj = self.get_data_transformation_obj(X_train)


            X_train = processing_obj.fit_transform(X_train)
            

            # fe_obj.fit_transform(X_train)


            # processing_obj = self.get_data_transformation_obj(X_train)

            X_test = processing_obj.transform(X_test)
            train_arr = np.c_[X_train, np.array(y_train)]
            test_arr = np.c_[X_test, np.array(y_test).reshape(-1, 1)]

            # print("type of X_train is:", type(X_train))
            # print("type of y_train is:", type(y_train))
            # print("type of y_test is:", type(y_test))
            # print("type of X_test is:", type(X_test))


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
