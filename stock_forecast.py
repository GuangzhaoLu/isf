import akshare as ak
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import sklearn as sk
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
from sklearn.metrics import r2_score

import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras.layers import LSTM

class stock_forecast:

    def __init__(self, stock_code = "sh000001", useage_days = 30, split_date = '2018-01-01'):
        df_history = ak.stock_zh_index_daily_em(symbol=stock_code) #history data get from akshare
        df_history['date'] = pd.to_datetime(df_history['date'])    #transfer date column into pandas datetime type
        print('Data up to date is: ',df_history['date'].iloc[-1])  #print newest data update time
        df_history = df_history.set_index(['date'], drop=True)

        list_head_name = list(df_history)
        self.dict_head_name = {}
        for i in range(len(list_head_name)):
            self.dict_head_name[list_head_name[i]] = i

        self.df_history = df_history
        self.useage_days = useage_days

        split_date = pd.Timestamp(split_date)   #set split date (defaule 2018.01.01) to partition data into train set and test set
        df_train = df_history.loc[:split_date]  #split out train data set
        df_test = df_history.loc[split_date:]   #split out test data set

        list_train = np.array(df_train)     #transfer train dataframe into np.array type
        list_test = np.array(df_test)       #transfer test dataframe into np.array type

        self.list_train = list_train
        self.list_test = list_test

        self.__model_set_flag = False  #whether the model have set
        self.__trained_flag = False    #show whether have a trained model

    def data_smooth(self, smooth_days = 5):  #in > self.list_train out > self.list_train
        list_train = self.list_train.tolist()
        list_test = self.list_test.tolist()
        train_size = len(list_train)
        list_all = list_train + list_test
        def get_single_list_column(list_name,index):
            list_temp = []
            for i in range(len(list_name)):
                list_temp.append(list_name[i][index])
            return list_temp

        #print(len(list_train),len(list_train[0]))
        #print(len(list_test),len(list_test[0]))
        #print(len(list_all),len(list_all[0]))

        list_head_padding = []

        for i in range(smooth_days - 1):
            list_head_padding.append(list_all[0])
        #    list_end_padding.append(list_all[-1])

        #list_all = list_head_padding + list_all
        #print(len(list_all),len(list_all[0]))

        #list_close_temp = get_single_list_column(list_all,1)

        #plt.figure(figsize=(10, 6))
        #plt.plot(list_close_temp, label='real')
        #plt.title("5 day average")
        #plt.xlabel('days')
        #plt.ylabel('Close')
        #plt.legend()
        #plt.savefig('5_day_average_padded')


        list_all_new = []
        for i in range(smooth_days - 1,len(list_all)):
            temp_list = []
            for j in range(len(list_all[0])):
                sum = 0
                for k in range(smooth_days):
                    sum = sum + list_all[i - k][j]
                temp_list.append((sum/smooth_days))
            list_all_new.append(temp_list)

        #print(len(list_all_new),len(list_all_new[0]))

        self.list_train = np.array(list_all_new[:train_size])
        self.list_test = np.array(list_all_new[train_size:])
        #print(self.list_train.shape[0],self.list_train.shape[1])
        #print(self.list_test.shape[0],self.list_test.shape[1])

        #list_close_temp = get_single_list_column(list_all_new,1)

        #plt.figure(figsize=(10, 6))
        #plt.plot(list_close_temp, label='real')
        #plt.title("5 day average")
        #plt.xlabel('days')
        #plt.ylabel('Close')
        #plt.legend()
        #plt.savefig('5_day_average2')


    def data_normalization(self, min = -1, max = 1):   #do normalization //in > self.list_train out > self.scaled_list_train
        scaler = preprocessing.MinMaxScaler(feature_range=(min, max))
        self.scaled_list_train = scaler.fit_transform(self.list_train)
        self.scaled_list_test = scaler.transform(self.list_test)
        self.scaler = scaler    
    
    def generate_ann_input_data(self):

        def get_merged_X(in_scaled_list, in_train_days):  #merge data in useage train days in to one array
            target_train = []
            width = len(in_scaled_list[0]) 
            for i in range (in_train_days - 1, len(in_scaled_list)-1):
                temp_tuple = []
                for j in range(in_train_days):
                    for k in range(width):
                        temp_tuple.append(in_scaled_list[i - in_train_days + 1 + j][k])
                target_train.append(temp_tuple)
            return np.array(target_train)

        self.X_train = get_merged_X(self.scaled_list_train,self.useage_days)
        self.Y_train = np.array(self.scaled_list_train[self.useage_days:])
        self.X_test = get_merged_X(self.scaled_list_test,self.useage_days)
        self.Y_test = np.array(self.scaled_list_test[self.useage_days:])

    def generate_lstm_input_data(self):

        def get_merged_X(in_scaled_list, in_train_days):  #merge data in useage train days in to one array
            target_train = []
            for i in range (in_train_days - 1, len(in_scaled_list)-1):
                temp_tuple = []
                for j in range(in_train_days):
                        temp_tuple.append(in_scaled_list[j])
                target_train.append(temp_tuple)
            return np.array(target_train)
        
        self.X_train = get_merged_X(self.scaled_list_train,self.useage_days)
        self.Y_train = np.array(self.scaled_list_train[self.useage_days:])
        self.X_test = get_merged_X(self.scaled_list_test,self.useage_days)
        self.Y_test = np.array(self.scaled_list_test[self.useage_days:])
    
    def model_ann_set(self):
        self.sf_model=Sequential()
        self.sf_model.add(Dense(3 * 6 * self.useage_days,input_dim = 6 * self.useage_days, activation='sigmoid'))
        #self.sf_model.add(LSTM(7,activation='relu', kernel_initializer='lecun_uniform', return_sequences=False)
        #self.sf_model.add(LSTM(units = 3 * 6 * self.useage_days, input_dim=6 * self.useage_days, activation='relu')) #Full connection layer
        self.sf_model.add(Dense(3 * 6 * self.useage_days, activation='sigmoid'))
        self.sf_model.add(Dense(3 * 6 * self.useage_days, activation='sigmoid'))
        self.sf_model.add(Dropout(0.25))  #dropout
        self.sf_model.add(Dense(2 * 6 * self.useage_days, activation='sigmoid'))
        self.sf_model.add(Dropout(0.25))  #dropout
        #self.sf_model.add(Dense(1 * 6 * self.useage_days, activation='sigmoid'))
        self.sf_model.add(Dropout(0.25))  #dropout
        #self.sf_model.add(Dense(10, activation='relu'))
        self.sf_model.add(Dense(6))
        self.sf_model.compile(loss='mean_squared_error', optimizer='Adam')
        self.early_stop = EarlyStopping(monitor='loss', patience=10, verbose=1)
        
        self.__model_set_flag = True

    def model_lstm_set(self):
        self.sf_model=Sequential()
        self.sf_model.add(LSTM(3 * 6 * self.useage_days,activation='sigmoid', kernel_initializer='lecun_uniform', return_sequences=True))
        self.sf_model.add(LSTM(3 * 6 * self.useage_days,activation='sigmoid', return_sequences=True))
        self.sf_model.add(LSTM(3 * 6 * self.useage_days,activation='sigmoid', return_sequences=True))
        self.sf_model.add(LSTM(3 * 6 * self.useage_days,activation='sigmoid', return_sequences=True))
        self.sf_model.add(Dense(6))
        self.early_stop = EarlyStopping(monitor='loss', patience=10, verbose=1)
        self.sf_model.compile(loss='mean_squared_error', optimizer='Adam')

        self.__model_set_flag = True

    def model_train(self, iepoch = 2000, ibatch_size=1):
        #self.history = self.sf_model.fit(self.X_train, self.Y_train, epochs=iepoch, batch_size=ibatch_size, verbose=1, callbacks=[self.early_stop], shuffle=False)
        self.history = self.sf_model.fit(self.X_train, self.Y_train, epochs=iepoch, batch_size=ibatch_size, verbose=1, shuffle=False)
        self.__trained_flag = True
    
    def model_save(self, file = 'model_saved'):
        if(not(self.__model_set_flag)):
            print('Error: No set model.')
        elif(not(self.__trained_flag)):
            print('Warning: No trained model.')
        else:
            self.sf_model.save(file)
            print('#####\nModel saved.\n#####')

    def model_load(self, file = 'model_saved'):
        self.sf_model = load_model(file)  #keras.models.load_model(filename)

    def verification(self, column = 'close', SaveFig = True):
        Y_train_pred = self.sf_model.predict(self.X_train)  #predict Y train 
        Y_test_pred = self.sf_model.predict(self.X_test)    #predict Y test

        print("The R2 score on the Train set is:\t{:0.3f}".format(r2_score(self.Y_train, Y_train_pred)))
        print("The R2 score on the Test set is:\t{:0.3f}".format(r2_score(self.Y_test, Y_test_pred)))

        #recover normalized data
        Y_test_final = self.scaler.inverse_transform(self.Y_test)
        Y_test_pred_final = self.scaler.inverse_transform(Y_test_pred)

        def get_single_column_data(in_data_set, in_column):
            index = self.dict_head_name[in_column]
            target_column = []
            for i in range(len(in_data_set)):
                target_column.append(in_data_set[i][index])
            return target_column

        Y_test_final_close = get_single_column_data(Y_test_final, column)
        Y_test_pred_final_close = get_single_column_data(Y_test_pred_final, column)

        print('data in 6 days:')
        print('real close value   :'+'%s'%Y_test_final_close[-6:])
        print('pridict close value:'+'%s'%Y_test_pred_final_close[-6:])

        if(SaveFig):
            #show compare of real data and predict data
            plt.figure(figsize=(10, 6))
            plt.plot(Y_test_final_close, label='real')
            plt.plot(Y_test_pred_final_close, label='predict')
            plt.title("Neural Network's Prediction")
            plt.xlabel('Observation')
            plt.ylabel('Close')
            plt.legend()
            plt.savefig('verification')


if __name__ == "__main__":
    def main():
        test = stock_forecast("sh000001",50,'2018-01-01')
        test.data_smooth(20)
        test.data_normalization()
        test.generate_ann_input_data()
        test.model_ann_set()
        test.model_train(500)
        test.model_save()
        #test.model_load()
        test.verification()
    
    main()

