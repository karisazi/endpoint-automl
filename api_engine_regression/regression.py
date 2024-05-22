import h2o
from h2o.automl import H2OAutoML
import pandas as pd
# from h2o.estimators import H2OOneHotEncoderEstimator

class H2OModel:
    def __init__(self, df, y_target):
        self.df = df
        self.y_target = y_target
        self.model_path = None
        self.data_train = None
        self.data_test = None
        self.data_valid = None
        self.data_pred = None
        self.x_features = None
        self.result = None
        self.aml = None
        self.model = None
        self.mae = None
        self.shap = None
        self.predictvar = None
        self.custompredict = None
        self.error_df = None
        self.importantvar_df = None

    def run_modelling(self):
        h2o.init()
        self.hf = h2o.H2OFrame(self.df)
        self.data_train, self.data_test, self.data_valid = self.hf.split_frame(ratios=[.8, .1])
        self.x_features = self.df.columns.tolist()
        self.x_features = [x for x in self.x_features if x != self.y_target]

        self.aml = H2OAutoML(max_models=10, seed=10, verbosity="info", nfolds=0, max_runtime_secs=30)
        self.aml.train(x=self.x_features, y=self.y_target, training_frame=self.data_train, validation_frame=self.data_valid)

        self.model = self.aml.leader
        self.result = self.model.predict(self.data_test)

    def get_model(self):
        return self.model

    def get_mae(self):
        # mae = self.model.mae(valid=True)
        # mse = self.model.mse(valid=True)
        # rmse = self.model.rmse(valid=True)
        # rmsle = self.model.rmsle(valid=True)
        mae = self.model.mae(valid=True)
        mae = format(mae, '10.2f')
        
        return mae

    def get_shap(self):
        return self.model.shap_summary_plot(self.data_test)
    
    def get_prediction_result(self):
        # data_pred_hf = h2o.H2OFrame(self.result)
        data_pred = self.data_test[self.y_target].concat(self.result, axis=1)
        data_pred['Difference'] = data_pred[self.y_target] - data_pred['predict']
        data_pred = data_pred.as_data_frame()
        return data_pred

    def get_important_features(self):
        varimp = self.model.varimp(use_pandas=True)['variable']
        self.predictvar = varimp.tolist()
        self.importantvar_df = pd.DataFrame({'Important Feature': self.predictvar})
        
        return varimp[:5]
    
    def get_custompredict(self, value_to_predict):
        hf_var = h2o.H2OFrame(value_to_predict, column_names=self.predictvar)
        
        return self.model.predict(hf_var)
    
    # def encode_df(self, from_df, toencode_df):
    #     encoder = H2OOneHotEncoderEstimator()

    #     encoder.train(training_frame=from_df)

    #     encoded_df = encoder.transform(toencode_df)
    #     return encoded_df


