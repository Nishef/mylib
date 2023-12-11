# this file contains of all LSTM methods for preprocessing right now
# my pipeline for LSTM

import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from statsmodels.tsa.stattools import adfuller, kpss
from tensorflow.keras import optimizers
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l1, l2


class Preprocessing(object):
    def __init__(self, dataset=None):
        self.dataset = dataset

    def read_file(self, file_path, *args, **kwargs):
        file_name, file_extension = os.path.splitext(file_path)
        read_functions = {
            ".csv": pd.read_csv,
            ".xlsx": pd.read_excel,
            ".txt": pd.read_csv,
            ".json": pd.read_json
        }

        if file_extension not in read_functions:
            return "File type is not supported."
        else:
            return read_functions[file_extension](file_path)

        '''
        def read_file(self, filename, *args, **kwargs):
        # df_data_1 ['DCOILBRENTEU'] = pd.to_numeric(df_data_1['DCOILBRENTEU'], errors='coerce')
        # df_data_1 = df_data_1.replace(np.nan, 0, regex=True)
        self.dataset = pd.read_table(filename, *args, **kwargs)
        self.length  = len(self.dataset)
        # fix random seed for reproducibility
        np.random.seed(7)
        tf.random.set_seed(20)

        # parse_dates: This specifies the column which contains the date-time information. As we say above, the column name is ‘Month’.
        # index_col: A key idea behind using Pandas for TS data is that the index has to be the variable depicting date-time information. So this argument tells pandas to use the ‘Month’ column as index.
        # date_parser: This specifies a function which converts an input string into datetime variable. Be default Pandas reads data in format ‘YYYY-MM-DD HH:MM:SS’. If the data is not in this format, the format has to be manually defined. Something similar to the dataparse function defined here can be used for this purpose.
        print('\ndataset info:',self.dataset.info())
        print('--------------------------------------')
        print('\ndataset described:\n',self.dataset.describe())

        return self.dataset'''

    def is_stationary(self, dataset):
        alpha = 0.05

        dftest = adfuller(df['column_name'], autolag='AIC')
        kpsstest = kpss(df['column_name'], regression="c", nlags="auto")

        adf_output = pd.Series(dftest[0:4], index=[
                               "Test Statistic", "p-value", "Num Of Lags", "Num Of Observations Used"])
        kpss_output = pd.Series(kpsstest[0:3], index=[
                                "Test Statistic", "p-value", "Lags Used"])

        if dftest[0] <= dftest[4]['5%'] and kpsstest[0] >= kpsstest[3]['5%']:
            print('Dataframe is stationary')
            return True
        else:
            print('Dataframe is not stationary')
            return False

    '''def fuller_kpss_test(self,dataset):
        alpha = 0.05

        dftest = adfuller(dataset, autolag = 'AIC')
        print("1. ADF : ",dftest[0])
        print("2. P-Value : ", dftest[1])
        print("3. Num Of Lags : ", dftest[2])
        print(
            "4. Num Of Observations Used For ADF Regression and Critical Values Calculation :", dftest[3])
        print("5. Critical Values :")
        for key, val in dftest[4].items():
            print("\t",key, ": ", val)
        # P-Value > 0.05 - This implies that time-series is non-stationary.
        # P-Value <=0.05 - This implies that time-series is is stationary.
        print('\n---------------------------')
        print("Results of KPSS Test:")
        kpsstest = kpss(dataset, regression="c", nlags="auto")
        kpss_output = pd.Series(
            kpsstest[0:3], index=["Test Statistic", "p-value", "Lags Used"]
        )
        for key, value in kpsstest[3].items():
            kpss_output["Critical Value (%s)" % key] = value
        print(kpss_output)

        # Note Kpss is opposite of ADF
        if dftest[1] > alpha and kpsstest[1] < alpha:
            print('*Data does not look stationary*')

        elif dftest[1] < alpha and kpsstest[1] > alpha:
            print('*Data looks stationary*')
        # case 3:
        elif dftest[1] > alpha and kpsstest[1] > alpha:
            print('*Trend needs to be removed to make series strict stationary*')
        # case 4:
        elif dftest[1] < alpha and kpsstest[1] < alpha:
            print('*Differencing is to be used to make series stationary*')'''

      # Case 3: KPSS indicates stationarity and ADF indicates non-stationarity - The series is trend stationary. Trend needs to be removed to make series strict stationary. The detrended series is checked for stationarity.
      # Case 4: KPSS indicates non-stationarity and ADF indicates stationarity - The series is difference stationary. Differencing is to be used to make series stationary. The differenced series is checked for stationarity.
    '''def make_differenced_series(series):
        return [series[i] - series[i-1] for i in range(1, len(series))]


    def inverse_differenced_series(differenced_series, initial_value):
        inverse_series = [initial_value]
        for value in differenced_series:
            inverse_series.append(inverse_series[-1] + value)
        return inverse_series'''

    def difference(self, dataset, interval=1):
        # Use the pandas shift function to create the differenced series
        diff = dataset.diff(periods=interval)
        # Remove the first row with NaN values
        diff.dropna(inplace=True)
        return diff

    def inverse_difference(self, last_ob, value):
        return value + last_ob

    # define a dataset with a linear trend
    # data = [i+1 for i in range(20)]
    # print(data)
    # difference the dataset
    # diff = difference(data)
    # print(diff)
    # invert the difference
    # inverted = [inverse_difference(data[i], diff[i]) for i in range(len(diff))]
    # print(inverted)
    def get_train_length(self, dataset, batch_size, test_percent, timesteps, i_loc=0):
        """
        This function calculates the length of the training data for a given dataset.

        Parameters:
        self (object): The object of the class.
        dataset (dataframe): The dataset from which the length of the training data is calculated.
        batch_size (int): The batch size of the dataset.
        test_percent (float): The percentage of the data that is used for testing.
        timesteps (int): The number of time steps in the dataset.
        i_loc (int): The index location of the dataset.

        Returns:
        self.dataset_last (float64): The length of the training data.
        """

        self.length = len(dataset)
        self.length *= 1 - test_percent
        train_length_value = [x for x in range(
            int(self.length) - 100, int(self.length)) if x % batch_size == 0]

        upper_train = int(self.length) + timesteps * 2
        dataset = dataset[0:upper_train].reset_index(drop=True)
        self.dataset_last = np.float64(dataset.iloc[:, i_loc].values)
        print(max(train_length_value), len(self.dataset_last))

        return self.dataset_last

    '''def get_train_length(self, dataset, batch_size, test_percent, timesteps, i_loc=0):
        # substract test_percent to be excluded from training,
        # reserved for testset
        self.length = len(dataset)
        self.length *= 1 - test_percent
        train_length_value = []
        for x in range(int(self.length) - 100, int(self.length)):
            modulo = x % batch_size
            if (modulo == 0):
                train_length_value.append(x)

        # adding timesteps * 2
        upper_train = int(self.length) + timesteps * 2

        dataset = dataset[0:upper_train].reset_index(drop=True)
        self.dataset_last = dataset.iloc[:, i_loc].values
        self.dataset_last = np.float64(self.dataset_last)
        print(max(train_length_value), len(self.dataset_last))
        return self.dataset_last'''

    def split_dataset(self, percent=0.67):
        # if you want test set from dataset first you need to run this function
        # and after that normalizing the dataset
        # data splitting before normalizing and train_x and train_y creation
        x = len(self.dataset_last)
        self.train_per = int(x * percent)
        self.train_percent = self.dataset_last[:self.train_per]
        self.test_percent = self.dataset_last[self.train_per:]

        print('train_data =', len(self.train_percent),
              '\ntest_data =', len(self.test_percent))
        return self.train_percent, self.test_percent

    def norm_Data(self, full_range=None):
        """Normalize data

        This function normalizes the data in the range of 0 to 1 or -1 to 1. 
        The default range is 0 to 1.

        Parameters
        ----------
        self : Object
            The object containing the data
        full_range : int, optional
            The range of normalization. The default is None.

        Returns
        -------
        train_X : array
            Normalized training data
        test_X : array
            Normalized testing data

        """
        # Normalization across instances should be done
        # after splitting the data between training and test set,
        # using only the data from the training set.
        if full_range is None:
            self.sc = MinMaxScaler(feature_range=(0, 1))
        else:
            self.sc = MinMaxScaler(feature_range=(-1, 1))
        # scaler = StandardScaler()

        # change dataframe type to 64bit array and
        # reshape it to a matrix with one column
        # self.dataset_last = np.array(np.float64(self.dataset_last))
        self.train_percent_reshaped = self.train_percent.reshape(-1, 1)
        self.test_percent_reshaped = self.test_percent.reshape(-1, 1)

        train_X = self.sc.fit_transform(self.train_percent_reshaped)

        if self.test_percent is not None:
            test_X = self.sc.transform(self.test_percent_reshaped)
            return train_X, test_X

        return train_X
    
    def inverse_norm(self, dataset):
        dataset = dataset.reshape(-1, 1)
        self.inverted = self.sc.inverse_transform(dataset)

    def creat_same_dataset(self, dataset, look_back=1, features=1, target_column_index=0):
        # output and input lenght is same
        # dataset = dataset.astype('float64')
        x_train = []  # input data which are lists
        y_train = []  # output data

        # creating a data structure with n lookback(timestep)
        for i in range(look_back,  len(dataset) - look_back):
            if features == 1:
                a = dataset[i-look_back:i, 0]
            elif features > 1:
                a = dataset[i-look_back:i, features]
            x_train.append(a)
            y_train.append(dataset[i:i + look_back, target_column_index])

        return np.array(x_train), np.array(y_train)


    def create_dataset(self, dataset, look_back=1, features=1, target_column_index=0):
        # output is only one
        # lstm input(input_data_shape,lookback(timesteps),features)
        dataX, dataY = [], []
        for i in range(len(dataset) - look_back):
            # if you have one column feature use this
            if features == 1:
                a = dataset[i:(i + look_back), 0]
            elif features > 1:
                a = dataset[i:(i + look_back), features]
            dataX.append(a)
            dataY.append(dataset[i + look_back, target_column_index])
        return np.array(dataX), np.array(dataY)



        return self.inverted
    
    def test_stationarity(self, timeseries, windows=15):
        rolmean = timeseries.rolling(window=windows).mean()
        rolstd = timeseries.rolling(window=windows).std()

        fig, ax = plt.subplots(figsize=(10, 6))
        orig = ax.plot(timeseries, color='blue', label='Original')
        mean = ax.plot(rolmean, color='red', label='Rolling Mean')
        std = ax.plot(rolstd, color='black', label='Rolling Std')

        ax.legend(loc='best')
        ax.set_title('Rolling Mean & Standard Deviation')
        sns.despine(left=True, ax=ax)
        plt.show()

        print('<Results of Dickey-Fuller Test>')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4],
                         index=['Test Statistic', 'p-value', '#Lags Used','Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    print(dfoutput)
    
    #  Also, the test statistic is smaller than the 5% critical values so we can say with 95% confidence that this is a stationary series.

    # def decompose(self, dataset):

    # Additive combination
    # If the seasonal and noise components change the trend by an amount that is independent of the value of trend, the trend, seasonal and noise components are said to behave in an additive way. One can represent this situation as follows:

    # y_i = t_i + s_i + n_i

    # where y_i = the value of the time series at the ith time step.
    # t_i = the trend component at the ith time step.
    # s_i = the seasonal component at the ith time step.
    # n_i = the noise component at the ith time step.

    # Multiplicative combination
    # If the seasonal and noise components change the trend by an amount that depends on the value of trend, the three components are said to behave in a multiplicative way as follows:

    # y_i = t_i * s_i * n_i

    # acc = accuracy_score(_testLabels, np.round(preds))*100
    # cm = confusion_matrix(_testLabels, np.round(preds))
    # tn, fp, fn, tp = cm.ravel()

    # print('\nCONFUSION MATRIX FORMAT ------------------\n')
    # print("[true positives    false positives]")
    # print("[false negatives    true negatives]\n\n")

    # print('CONFUSION MATRIX ------------------')
    # print(cm)

    # print('\nTEST METRICS ----------------------')
    # precision = tp/(tp+fp)*100
    # recall = tp/(tp+fn)*100
    # specificity = tn/(tn+fp)*100 #Jordan_note: added specificity calculation
    # print('Accuracy: {}%'.format(acc))
    # print('Precision: {}%'.format(precision))
    # print('Recall/Sensitivity: {}%'.format(recall)) #Jordan_note: added sensitivity label
    # print('Specificity {}%'.format(specificity)) #Jordan_note: added specificity calculation
    # print('F1-score: {}'.format(2*precision*recall/(precision+recall)))

    # if enableTraining:
    #     checkpoint = ModelCheckpoint(filepath=hdf5_testSaveFileName, save_best_only=True, save_weights_only=True)
    #     lr_reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=2, verbose=2, mode='max')
    #     early_stop = EarlyStopping(monitor='val_loss', min_delta=0.1, patience=1, mode='min')

    #     hist = ___model.fit_generator(
    #                ___train_gen, steps_per_epoch=___test_gen.samples // __batch_size,
    #                epochs=__epochs, validation_data=___test_gen,
    #                validation_steps=___test_gen.samples // __batch_size, callbacks=[checkpoint, lr_reduce])

    #     print('\nTRAIN METRIC ----------------------')
    #     print('Covid19 Train acc: {}'.format(np.round((hist.history['accuracy'][-1])*100, 2)))
