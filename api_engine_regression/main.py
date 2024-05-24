import os
import logging
import pandas as pd
from flask import Flask, request, render_template, jsonify
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField
from wtforms.validators import InputRequired
import h2o
from regression import H2OModel

# Initialize the Flask application
app = Flask(__name__)
app.config['SECRET_KEY'] = 'supersecretkey'
app.config['UPLOAD_FOLDER'] = 'static/files'

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Define a file upload form
class UploadFileForm(FlaskForm):
    file = FileField("File", validators=[InputRequired()])
    submit = SubmitField("Submit")

@app.route('/', methods=['GET', 'POST'])
def home():
    return 'Hello, World!'

@app.route('/regression/train', methods=['GET', 'POST'])
def training_regression():
    form = UploadFileForm()
    if form.validate_on_submit():
        file = form.file.data
        y_target = request.form.get('y_target')
        name = request.form.get('name', '').lower()

        if not name:
            return "Error: Model name is not provided.", 400
        
        file_path = os.path.join(os.path.abspath(os.path.dirname(__file__)),app.config['UPLOAD_FOLDER'], f'dataset_{name}.csv')
        file.save(file_path)
        
        df = pd.read_csv(file_path)
        if y_target is None or y_target not in df.columns:
            return "Error: y_target is not provided or not found in the uploaded file columns.", 400
        
        automl = H2OModel(df, y_target)
        automl.run_modelling()
        mae = automl.get_mae()

        model_directory = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'model')
        os.makedirs(model_directory, exist_ok=True)
        model_path = h2o.save_model(model=automl.model, path=model_directory, filename=f'regression_{name}', force=True)
        
        return f"Model {name} successfully trained with target {y_target}. MAE: {mae}"
    
    return render_template('index.html', form=form)

@app.route('/regression/predict', methods=['POST'])
def predict_regression():
    h2o.init()
    
    json_data = request.get_json(force=True)
    if not json_data or 'Name' not in json_data or 'Data' not in json_data:
        return jsonify({"error": "Invalid request format. Use 'Name' for model name and 'Data' for the data."}), 400
    
    model_name = json_data['Name']
    model_file = f'regression_{model_name}'
    df_data = json_data['Data']
    
    df = pd.DataFrame(df_data, index=[0])
    hf_var = h2o.H2OFrame(df)
    
    model_directory = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'model')
    model_path = os.path.join(model_directory, model_file)
    if not os.path.exists(model_path):
        return jsonify({"error": "Model does not exist"}), 400
    
    saved_model = h2o.load_model(model_path)
    predictions = saved_model.predict(hf_var)
    predictions_list = predictions.as_data_frame().values.flatten().tolist()
    
    return jsonify({
        "model_name": model_name,
        "prediction": predictions_list
    })

if __name__ == '__main__':
    app.run(debug=True)
