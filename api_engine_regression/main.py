from flask import Flask, request, render_template, jsonify
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField
from werkzeug.utils import secure_filename
import os
from wtforms.validators import InputRequired
import pandas as pd
from regression import H2OModel
import logging
import h2o
# from h2o.automl import H2OAutoML

app = Flask(__name__)
app.config['SECRET_KEY'] = 'supersecretkey'
app.config['UPLOAD_FOLDER'] = 'static/files'


# Configure logging
logging.basicConfig(level=logging.DEBUG)

class UploadFileForm(FlaskForm):
    file = FileField("File", validators=[InputRequired()])
    submit = SubmitField("Upload File")

@app.route('/', methods=['GET', 'POST'])
def home():
    return 'hello world'

@app.route('/regression/train', methods=['GET', 'POST'])
def training_regression():
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
        
        # Initialize and use the H2O model
        automl = H2OModel(df, y_target)
        automl.run_modelling()
        automl.get_mae()
        
        model = automl.model
        
        model_directory = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'model')
        if not os.path.exists(model_directory):
            os.makedirs(model_directory)
        model_path = h2o.save_model(model=model, path=model_directory, filename='regression_model', force=True)
        
        
        return f"Model successfully trained with target {y_target} gives error mae:{automl.get_mae()}"
    
    return render_template('index.html', form=form)


@app.route('/regression/predict', methods=['GET', 'POST'])
def predict_regression():    
        h2o.init()
        
        # Get the JSON data from the request
        json_data = request.get_json(force=True)
        if not json_data:
            return jsonify({"error": "No data provided"}), 400
        
        # Extract columns (keys) and values from JSON data
        columns = list(json_data.keys())
        values = [list(json_data.values())]

        # Convert to Pandas DataFrame to H2OFrame
        df = pd.DataFrame(values, columns=columns)
        hf_var = h2o.H2OFrame(df)

        # Load model using model_path
        model_directory = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'model', 'regression_model')
        saved_model = h2o.load_model(model_directory)

        # Make predictions
        predictions = saved_model.predict(hf_var)
        # Convert predictions to a list or a dictionary to send as JSON response
        predictions_list = predictions.as_data_frame().values.flatten().tolist()
        
        # Return the received JSON data and predictions for verification
        return jsonify({
            "prediction": predictions_list
        })


if __name__ == '__main__':
    app.run(debug=True)
