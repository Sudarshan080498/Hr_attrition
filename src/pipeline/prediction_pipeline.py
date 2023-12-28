from src.component import *
from src.logger import logging
from src.exception import CustomException
import os, sys
from src.Config.configuration import *
from src.utils import load_model
import pandas as pd

class PredictionPipeline():
    def __init__(self):
        pass

    def predict(self, features):
        try:
            preprocessor_path =  PREPROCESSED_OBJ_FILE
            model_path = MODEL_FILE_PATH

            preprocessor = load_model(preprocessor_path)
            model = load_model(model_path)

            data_scaled = preprocessor.transform(features)

            pred = model.predict(data_scaled)
            return pred
        


        except Exception as e:
            logging.info("Error occured in processing prediction pipeline")
            CustomException(e,sys)


class CustomData:
    def __init__(self,
                    Age: int,
                    DailyRate: int,
                    DistanceFromHome: int,
                    Education: int,
                    EnvironmentSatisfaction: int,
                    HourlyRate: int,
                    JobInvolvement: int,
                    JobLevel: int,
                    JobSatisfaction: int,
                    MonthlyIncome: int,
                    MonthlyRate: int,
                    NumCompaniesWorked: int,
                    OverTime: str,
                    PercentSalaryHike: int,
                    PerformanceRating: int,
                    RelationshipSatisfaction: int,
                    StockOptionLevel: int,
                    TotalWorkingYears: int,
                    TrainingTimesLastYear: int,
                    WorkLifeBalance: int,
                    YearsAtCompany: int,
                    YearsInCurrentRole: int,
                    YearsSinceLastPromotion: int,
                    YearsWithCurrManager: int,
                    BusinessTravel: str,
                    Department: str,
                    EducationField: str,
                    Gender: str,
                    JobRole: str,
                    MaritalStatus: str):

        self.Age = Age
        self.DailyRate = DailyRate
        self.DistanceFromHome = DistanceFromHome
        self.Education = Education
        self.EnvironmentSatisfaction = EnvironmentSatisfaction
        self.HourlyRate = HourlyRate
        self.JobInvolvement = JobInvolvement
        self.JobLevel = JobLevel
        self.JobSatisfaction = JobSatisfaction
        self.MonthlyIncome = MonthlyIncome
        self.MonthlyRate = MonthlyRate
        self.NumCompaniesWorked = NumCompaniesWorked
        self.OverTime = OverTime
        self.PercentSalaryHike = PercentSalaryHike
        self.PerformanceRating = PerformanceRating
        self.RelationshipSatisfaction = RelationshipSatisfaction
        self.StockOptionLevel = StockOptionLevel
        self.TotalWorkingYears = TotalWorkingYears
        self.TrainingTimesLastYear = TrainingTimesLastYear
        self.WorkLifeBalance = WorkLifeBalance
        self.YearsAtCompany = YearsAtCompany
        self.YearsInCurrentRole = YearsInCurrentRole
        self.YearsSinceLastPromotion = YearsSinceLastPromotion
        self.YearsWithCurrManager = YearsWithCurrManager
        self.BusinessTravel = BusinessTravel
        self.Department = Department
        self.EducationField = EducationField
        self.Gender = Gender
        self.JobRole = JobRole
        self.MaritalStatus = MaritalStatus


    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                'Age ':[self.Age],
                'DailyRate': [self.DailyRate],
                'DistanceFromHome': [self.DistanceFromHome],
                'Education': [self.Education],
                'EnvironmentSatisfaction': [self.EnvironmentSatisfaction],
                'HourlyRate': [self.HourlyRate],
                'JobInvolvement': [self.JobInvolvement],
                'JobLevel': [self.JobLevel],
                'JobSatisfaction': [self.JobSatisfaction],
                'MonthlyIncome': [self.MonthlyIncome],
                'MonthlyRate': [self.MonthlyRate],
                'NumCompaniesWorked': [self.NumCompaniesWorked],
                'OverTime': [self.OverTime],
                'PercentSalaryHike': [self.PercentSalaryHike],
                'PerformanceRating': [self.PerformanceRating],
                'RelationshipSatisfaction': [self.RelationshipSatisfaction],
                'StockOptionLevel': [self.StockOptionLevel],
                'TotalWorkingYears': [self.TotalWorkingYears],
                'TrainingTimesLastYear': [self.TrainingTimesLastYear],
                'WorkLifeBalance': [self.WorkLifeBalance],
                'YearsAtCompany': [self.YearsAtCompany],
                'YearsInCurrentRole': [self.YearsInCurrentRole],
                'YearsSinceLastPromotion': [self.YearsSinceLastPromotion],
                'YearsWithCurrManager': [self.YearsWithCurrManager],
                'BusinessTravel': [self.BusinessTravel],
                'Department': [self.Department],
                'EducationField': [self.EducationField],
                'Gender': [self.Gender],
                'JobRole': [self.JobRole],
                'MaritalStatus': [self.MaritalStatus],
            }

            df = pd.DataFrame(custom_data_input_dict)
            logging.info(f"DataFrame for prediction: {df}")

            return df
        except Exception as e:
            logging.error("Error occurred in Custom Pipeline DataFrame")
            raise CustomException(e, sys)