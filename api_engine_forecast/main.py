from flask import Flask, request, render_template, jsonify
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField
from werkzeug.utils import secure_filename
import os
from wtforms.validators import InputRequired
import pandas as pd
from forecasting import H2OModel
import logging
import h2o
from datetime import timedelta
from flask_wtf import CSRFProtect
from wtforms import FileField, SubmitField, StringField
from wtforms.validators import DataRequired
from flask_wtf.csrf import generate_csrf


app = Flask(__name__)
app.config['SECRET_KEY'] = 'superb'
app.config['UPLOAD_FOLDER'] = 'static/files'

# Add CSRF token
csrf = CSRFProtect(app)


# Configure logging
logging.basicConfig(level=logging.DEBUG)

class UploadFileForm(FlaskForm):
    file = FileField('CSV File', validators=[DataRequired()])
    y_target = StringField('Target Column', validators=[DataRequired()])
    date_col = StringField('Date Column', validators=[DataRequired()])
    submit = SubmitField('Upload')

@app.route('/get_csrf_token', methods=['GET'])
def get_csrf_token():
    token = generate_csrf()
    return jsonify({"csrf_token": token})

@app.route('/', methods=['GET', 'POST'])
def home():
    return 'hello world'

@app.route('/forecasting/train', methods=['GET', 'POST'])
def training_forecasting():
    form = UploadFileForm()
    if form.validate_on_submit():
        # Grab the file
        file = form.file.data
        # Save the file
        file_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), app.config['UPLOAD_FOLDER'], 'dataset.csv')
        file.save(file_path)
        
        # Read the uploaded file into a DataFrame
        df = pd.read_csv(file_path)
        
        
        # Get y_target from the form input
        y_target = request.form.get('y_target')  

        if y_target is None or y_target not in df.columns:
            return "Error: y_target is not provided or not found in the uploaded file columns.", 400
        
        # Get date column from the form input
        date_col = request.form.get('date_col')  

        if date_col is None or date_col not in df.columns:
            return "Error: date_col is not provided or not found in the uploaded file columns.", 400
        
        
        # Initialize and use the H2O model
        automl = H2OModel(df, y_target, date_col)
        automl.run_modelling()
        automl.get_prediction_result()
        mae = automl.get_mae()
        
        data = automl.modified_data
        data.to_csv('static/files/subdataset.csv')
        
        model = automl.model
        
        model_directory = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'model')
        if not os.path.exists(model_directory):
            os.makedirs(model_directory)
        model_path = h2o.save_model(model=model, path=model_directory, filename='forecasting_model', force=True)
        
        response = {"message": f"Model successfully trained with target {y_target} gives error mae: {mae}"}
                
        return jsonify(response)
            
    # Check errors
    errors = form.errors
    return jsonify({"error": "Form is not valid", "details": errors}), 400



@app.route('/forecasting/predict', methods=['GET', 'POST'])
def predict_forecasting():
    h2o.init()

    # Get the JSON data from the request
    days = request.form.get('days')

    if days is None:
        return "Error: days is not provided", 400
    
    try:
        days = int(days)
    except ValueError:
        return "Error: days must be an integer", 400
    
    df = pd.read_csv('static/files/subdataset.csv')
    date_col = df.columns[0]
    y_target = df.columns[1]
    
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col)
    
    df[y_target] = pd.to_numeric(df[y_target], errors='coerce')
    df = df.fillna(method='ffill').dropna()
    
    if df.empty:
        return "Error: The dataset is empty after preprocessing", 400

    # Load model using model_path
    model_directory = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'model', 'forecasting_model')
    saved_model = h2o.load_model(model_directory)

    if len(df) < 7:
        return "Error: Not enough data points for rolling window calculations", 400

    # Make future predictions
    last_row = df.iloc[-1].copy()
    last_date = last_row.name  # a datetime object

    if isinstance(last_date, str):
        last_date = pd.to_datetime(last_date)

    future_dates = [last_date + timedelta(days=i) for i in range(1, days + 1)]
    future_predictions = []

    for date in future_dates:
        last_row[date_col] = date

        for lag in [1, 2, 3]:
            last_row[f'lag_{lag}'] = last_row[y_target] if lag == 1 else df[f'lag_{lag-1}'].iloc[-1]

        # Recalculate rolling features
        last_row['rolling_mean_7'] = df[y_target].rolling(window=7).mean().iloc[-1]
        last_row['rolling_std_7'] = df[y_target].rolling(window=7).std().iloc[-1]

        # Prepare data for prediction
        last_row_df = pd.DataFrame(last_row).T
        h2o_last_row = h2o.H2OFrame(last_row_df)
        prediction = saved_model.predict(h2o_last_row)
        predicted_value = prediction.as_data_frame().iloc[0, 0]

        future_predictions.append(predicted_value)
        last_row[y_target] = predicted_value

    # Create a DataFrame with the future dates and their predictions
    future_df = pd.DataFrame({'date': future_dates, 'predicted_humidity': future_predictions})


    # Return the received JSON data and predictions for verification
    return jsonify({
        'last date': last_date,
        'date': future_dates,
        'predicted_humidity': future_predictions
    })


if __name__ == '__main__':
    app.run(debug=True)
