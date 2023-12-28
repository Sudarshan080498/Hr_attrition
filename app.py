from flask import Flask, render_template, request, redirect, url_for, session, make_response, send_file
from werkzeug.utils import secure_filename
import io
import os
from src.pipeline.prediction_pipeline import PredictionPipeline
from src.pipeline.prediction_pipeline import CustomData
from src.logger import logging
from src.Batch_Prediction.batch import batch_prediction
from src.pipeline.training_pipeline import Train
import pandas as pd
from src.Config.configuration import FE_OBJ_FILE_PATH, PREPROCESSED_OBJ_FILE, MODEL_FILE_PATH

UPLOAD_FOLDER = "batch_prediction/UPLOADED_CSV_FILE"

app = Flask(__name__, template_folder="templates")
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'

ALLOWED_EXTENSIONS = {'csv'}

@app.route('/')
def home_page():
    return render_template('index.html')

@app.route('/single_prediction_form', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        app.logger.info(f"Form Data Received: {request.form}")
        return render_template('single_prediction_form.html')
    else:
        data = CustomData(
            Age=int(request.form.get('Age')),
            DailyRate=int(request.form.get('DailyRate')),
            DistanceFromHome=int(request.form.get('DistanceFromHome')),
            Education=int(request.form.get('Education')),
            EnvironmentSatisfaction=int(request.form.get('EnvironmentSatisfaction')),
            HourlyRate=int(request.form.get('HourlyRate')),
            JobInvolvement=int(request.form.get('JobInvolvement')),
            JobLevel=int(request.form.get('JobLevel')),
            JobSatisfaction=int(request.form.get('JobSatisfaction')),
            MonthlyIncome=int(request.form.get('MonthlyIncome')),
            MonthlyRate=int(request.form.get('MonthlyRate')),
            NumCompaniesWorked=int(request.form.get('NumCompaniesWorked')),
            OverTime=request.form.get('OverTime'),
            PercentSalaryHike=int(request.form.get('PercentSalaryHike')),
            PerformanceRating=int(request.form.get('PerformanceRating')),
            RelationshipSatisfaction=int(request.form.get('RelationshipSatisfaction')),
            StockOptionLevel=int(request.form.get('StockOptionLevel')),
            TotalWorkingYears=int(request.form.get('TotalWorkingYears')),
            TrainingTimesLastYear=int(request.form.get('TrainingTimesLastYear')),
            WorkLifeBalance=int(request.form.get('WorkLifeBalance')),
            YearsAtCompany=int(request.form.get('YearsAtCompany')),
            YearsInCurrentRole=int(request.form.get('YearsInCurrentRole')),
            YearsSinceLastPromotion=int(request.form.get('YearsSinceLastPromotion')),
            YearsWithCurrManager=int(request.form.get('YearsWithCurrManager')),
            BusinessTravel=request.form.get('BusinessTravel'),
            Department=request.form.get('Department'),
            EducationField=request.form.get('EducationField'),
            Gender=request.form.get('Gender'),
            JobRole=request.form.get('JobRole'),
            MaritalStatus=request.form.get('MaritalStatus')
        )

        final_new_data = data.get_data_as_dataframe()
        predict_pipeline = PredictionPipeline()
        pred = predict_pipeline.predict(final_new_data)

        if pred is not None and len(pred) > 0:
            result = int(pred[0])
            return render_template('single_prediction_result.html', prediction_result=result)
        else:
            return render_template('single_prediction_result.html', error='Prediction result is None')

@app.route("/batch_prediction_form", methods=["GET", "POST"])
def batch_form():
    return render_template("batch_prediction_form.html")

@app.route('/perform_batch_prediction', methods=["POST"])
def perform_batch_prediction():
    if request.method == "GET":
        return render_template("batch_prediction.html")
    else:
        file = request.files['csvFile']
        directory_path = UPLOAD_FOLDER

        os.makedirs(directory_path, exist_ok=True)

        if file and "." in file.filename and file.filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS:
            for filename in os.listdir(os.path.join(UPLOAD_FOLDER)):
                file_path = os.path.join(UPLOAD_FOLDER, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)

            filename = secure_filename(file.filename)
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(file_path)
            logging.info("CSV received and uploaded")

            # Add logging statement to check file_path
            logging.info(f"File saved to: {file_path}")

            batch = batch_prediction(file_path, MODEL_FILE_PATH, PREPROCESSED_OBJ_FILE, FE_OBJ_FILE_PATH)
            batch_predictions = batch.start_batch_prediction()

            # Add logging statement to check batch_predictions
            logging.info(f"Batch predictions: {batch_predictions}")

            session['batch_predictions'] = batch_predictions

            output = "Batch prediction done"
            return render_template("batch_prediction_result.html", prediction_result=output)
        else:
            return render_template("batch_prediction_form.html", error='Invalid file format')


@app.route('/download_prediction', methods=["GET"])
def download_prediction():
    batch_predictions = session.get('batch_predictions', None)

    if batch_predictions is not None and isinstance(batch_predictions, pd.DataFrame):
        print("Excel file created successfully")
        excel_file = io.BytesIO()
        batch_predictions.to_excel(excel_file, index=False)
        excel_file.seek(0)

        response = make_response(send_file(excel_file, as_attachment=True, download_name="predictions.xlsx"))
        response.headers["Content-Disposition"] = "attachment; filename=predictions.xlsx"
        response.headers["Content-Type"] = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

        return response
    else:
        return render_template("batch_prediction_result.html", error='No predictions available')

@app.route('/train_data_form', methods=['GET', 'POST'])
def train():
    if request.method == 'GET':
        return render_template('train_data_form.html')
    else:
        try:
            pipeline = Train()
            pipeline.main()
            return render_template('train_data_result.html', message='Training completed')
        except Exception as e:
            logging.error(f"{e}")
            error_message = str(e)
            return render_template('index.html', error=error_message)

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port='8888')