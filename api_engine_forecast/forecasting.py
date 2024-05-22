import h2o
from h2o.automl import H2OAutoML
import pandas as pd
from sklearn.metrics import mean_absolute_error
# import warnings
# import locale
# locale.setlocale(locale.LC_ALL, 'en_GB.UTF-8')
# warnings.filterwarnings('ignore')

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

    def run_modelling(self):
        h2o.init(nthreads=-1)
        df = self.df[[self.date_col, self.y_target]]
        # Convert to H2O Frame
        hf = h2o.H2OFrame(df)

        # Specify the date column
        hf[self.date_col] = hf[self.date_col].as_date('%Y-%m-%d')
        
        df_processed = process_data(df, self.y_target)

        df_processed.reset_index(drop=True,inplace=True)
        self.data_train = h2o.H2OFrame(df_processed.loc[:int(df_processed.shape[0]*0.8),:])
        self.data_test = h2o.H2OFrame(df_processed.loc[int(df_processed.shape[0]*0.8):,:])
        
        self.x_features = df_processed.columns.tolist()
        self.x_features = [x for x in self.x_features if x != self.y_target]

        aml = H2OAutoML(max_runtime_secs = 30, seed = 42)
        aml.train(x=self.x_features, y=self.y_target, training_frame=self.data_train, leaderboard_frame = self.data_test)
        self.model = aml.leader
        self.leaderboard = aml.leaderboard
        self.result = self.model.predict(self.data_test)
        
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


def process_data(df, target):
    numerical_df = df.select_dtypes(include=['number'])
    df2 = numerical_df.copy()
    num_lags = 3 # number of lags and window lenghts for mean aggregation
    delay = 1 # predict target one step ahead
    for column in df2:
        for lag in range(1,num_lags+1):
            df2[column + '_lag' + str(lag)] = df2[column].shift(lag*-1-(delay-1))
            if column != 'wnd_dir':
                df2[column + '_avg_window_length' + str(lag+1)] = df2[column].shift(-1-(delay-1)).rolling(window=lag+1,center=False).mean().shift(1-(lag+1))

    df2.dropna(inplace=True)
    
    mask = (df2.columns.str.contains(target) | df2.columns.str.contains('lag') | df2.columns.str.contains('window'))
    df_processed = df2[df2.columns[mask]]

    return df_processed