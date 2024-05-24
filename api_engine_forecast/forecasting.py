import h2o
from h2o.automl import H2OAutoML
import pandas as pd
from sklearn.metrics import mean_absolute_error

from datetime import timedelta

class H2OModel:
    def __init__(self, df, y_target, date_col):
        self.df = df
        self.y_target = y_target
        self.date_col = date_col
        self.data_train = None
        self.data_test = None
        self.data_pred = None
        self.x_features = None
        self.result = None
        self.model = None
        self.leaderboard = None
        self.modified_data = None

    def run_modelling(self):
        h2o.init(nthreads=-1)
        df = self.df[[self.date_col, self.y_target]]


        # Specify the date column
        df[self.date_col] = pd.to_datetime(df[self.date_col])
        df = df.set_index(self.date_col)
        
        df = self.create_lagged_features(df, lags=[1, 2, 3])
        df = self.create_rolling_features(df, window=7)
        df.dropna(inplace=True)
        self.modified_data = df

        
        X = df.columns.drop(self.y_target).tolist()
        y = self.y_target
        
        # Convert to H2O Frame
        hf = h2o.H2OFrame(df)
        
        train, test = hf.split_frame(ratios=[0.8], seed=1234)


        aml = H2OAutoML(max_runtime_secs = 30, seed = 42)
        aml.train(x=X, y=y, training_frame=train)
        self.model = aml.leader
        self.leaderboard = aml.leaderboard
        self.data_test=test
        self.result = self.model.predict(test)
        
    def get_model(self):
        return self.model
    
    def get_leaderboard(self):
        return self.leaderboard
    
    def get_prediction_result(self):
        # data_pred_hf = h2o.H2OFrame(self.result)
        data_pred = self.data_test[self.y_target].concat(self.result, axis=1)
        data_pred['Difference'] = data_pred[self.y_target] - data_pred['predict']
        data_pred.columns = ['ground_truth', 'prediction', 'difference']
        self.data_pred = data_pred.as_data_frame()
        return self.data_pred

    def get_mae(self):
        # Calculate MAE
        mae = mean_absolute_error(self.data_pred['ground_truth'], self.data_pred['prediction'])
        return mae
    
    def get_important_features(self):
        varimp = self.model.varimp(use_pandas=True)['variable']
        return varimp[:5]

    def generate_future_dates(self, num_days):
        last_date = self.df[self.date_col].max()
        future_dates = [last_date + timedelta(days=i) for i in range(1, num_days + 1)]
        future_df = pd.DataFrame({self.date_col: future_dates})
        return future_df
    
    def create_lagged_features(self, data, lags):
        for lag in lags:
            data[f'lag_{lag}'] = data[self.y_target].shift(lag)
        return data

    def create_rolling_features(self, data, window):
        data[f'rolling_mean_{window}'] = data[self.y_target].rolling(window=window).mean()
        data[f'rolling_std_{window}'] = data[self.y_target].rolling(window=window).std()
        return data
