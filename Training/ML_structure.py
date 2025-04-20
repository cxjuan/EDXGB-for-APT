import numpy as np
import pandas as pd
import os

# from MLStructure.Training.ML_model import memory

np.set_printoptions(threshold=10000)
from numpy import where

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Dropout, Input, Flatten, Conv1D, MaxPooling1D
from sklearn.ensemble import IsolationForest
from scipy import stats

import keras.utils
import keras.callbacks
from keras.models import Model, load_model
import os
from keras import optimizers
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler,MaxAbsScaler, StandardScaler, RobustScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.metrics import auc, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from collections import Counter
import time
import datetime
import matplotlib.pyplot as plt
import MLStructure.PathList as Pathlists
from numba import jit, cuda
from memory_profiler import profile
import GPUtil
import gpumonitor

import warnings
warnings.simplefilter('ignore')

path_src = Pathlists.folder_check(Pathlists.ML_DATA_DIR)
result_graph = Pathlists.folder_check(Pathlists.RESULT_train_GRAPH_res_DIR)
result_dir = Pathlists.folder_check(Pathlists.RESULT_train_HISTORY_res_DIR)

class superviced_ML():
    def __init__(self, x_nodes, y_nodes, x_train, y_train, x_vali, y_vali, dataset_name, new_dataset_name, item_key):
        self.x_nodes = x_nodes
        self.y_nodes = y_nodes
        self.x_train = x_train
        self.y_train = y_train
        self.x_vali = x_vali
        self.y_vali = y_vali
        self.dataset_name = dataset_name
        self.new_dataset_name = new_dataset_name
        self.item_key = item_key


    @jit(target_backend='cuda')
    def cnn_structur(self, stru_index):
        if stru_index == 1:
            inputs1 = Input(shape=(1, self.x_nodes))
            print(inputs1)
            x1 = Conv1D(128, kernel_size=3, padding="same", activation='relu')(inputs1)
            x1 = MaxPooling1D(pool_size=1, strides=1)(x1)
            x1 = Dropout(.3)(x1)
            x1 = Flatten()(x1)
            x1 = Dense(128, activation='relu')(x1)
            x1 = Dropout(.2)(x1)
            # x1 = Dense(, activation='relu', name='layer_two')(x1)
        elif stru_index == 2:
            inputs1 = Input(shape=(1, self.x_nodes))
            x1 = Conv1D(128, kernel_size=3, padding="same", activation='relu')(inputs1)
            x1 = MaxPooling1D(pool_size=1)(x1)
            x1 = Dropout(.2)(x1)
            x1 = Conv1D(64, kernel_size=3, padding="same", activation='relu', name='layer_two')(x1)
            x1 = MaxPooling1D(pool_size=1, strides=1)(x1)
            x1 = Dropout(.3)(x1)
            x1 = Flatten()(x1)
            x1 = Dense(128, activation='relu')(x1)
            x1 = Dropout(.2)(x1)
        elif stru_index == 3:
            inputs1 = Input(shape=(1, self.x_nodes))
            x1 = Conv1D(128, kernel_size=3, padding="same", activation='relu')(inputs1)
            x1 = MaxPooling1D(pool_size=1, strides=1)(x1)
            x1 = Dropout(.3)(x1)
            x1 = Conv1D(32, kernel_size=3, padding="same", activation='relu', name='layer_two')(x1)
            x1 = MaxPooling1D(pool_size=1, strides=1)(x1)
            x1 = Dropout(.3)(x1)
            x1 = Flatten()(x1)
            x1 = Dense(256, activation='relu')(x1)
            x1 = Dropout(.2)(x1)
        elif stru_index == 4:
            inputs1 = Input(shape=(1, self.x_nodes))
            x1 = Conv1D(256, kernel_size=4, padding="same", activation='relu')(inputs1)
            x1 = MaxPooling1D(pool_size=1, strides=1, padding='same')(x1)
            x1 = Dropout(.3)(x1)
            x1 = Conv1D(128, kernel_size=3, padding="same", activation='relu', name='layer_two')(x1)
            x1 = MaxPooling1D(pool_size=1, strides=1, padding='same')(x1)
            x1 = Dropout(.3)(x1)
            x1 = Conv1D(64, kernel_size=3, padding="same", activation='relu', name='layer_three')(x1)
            x1 = MaxPooling1D(pool_size=1, strides=1, padding='same')(x1)
            x1 = Dropout(.2)(x1)
            x1 = Flatten()(x1)
            x1 = Dense(256, activation='relu')(x1)
            x1 = Dropout(.1)(x1)
        else:
            return -1
        return inputs1, x1

    # For design model structure
    @jit(target_backend='cuda')
    def dnn_structur(self, stru_index):
        if stru_index == 1:
            inputs1 = Input(shape=(self.x_nodes,))
            x1 = Dense(128, activation='relu', name='layer_one')(inputs1)
            x1 = Dropout(.1)(x1)
            x1 = Dense(64, activation='relu', name='layer_two')(x1)
            x1 = Dropout(.1)(x1)
        elif stru_index == 2:
            inputs1 = Input(shape=(self.x_nodes,))
            x1 = Dense(128, activation='relu', name='layer_one')(inputs1)
            x1 = Dropout(.1)(x1)
            x1 = Dense(32, activation='relu', name='layer_two')(x1)
            x1 = Dropout(.1)(x1)
        elif stru_index == 3:
            inputs1 = Input(shape=(self.x_nodes,))
            x1 = Dense(128, activation='relu', name='layer_one')(inputs1)
            x1 = Dropout(.1)(x1)
            x1 = Dense(32, activation='relu', name='layer_two')(x1)
            x1 = Dropout(.1)(x1)
            x1 = Dense(64, activation='relu', name='layer_three')(x1)
            x1 = Dropout(.1)(x1)
        elif stru_index == 4:
            inputs1 = Input(shape=(self.x_nodes,))
            x1 = Dense(128, activation='relu', name='layer_one')(inputs1)
            x1 = Dropout(.1)(x1)
            x1 = Dense(64, activation='relu', name='layer_two')(x1)
            x1 = Dropout(.1)(x1)
            x1 = Dense(32, activation='relu', name='layer_three')(x1)
            x1 = Dropout(.1)(x1)
        else:
            return -1
        return inputs1, x1

    # Model
    @jit(target_backend='cuda')
    def createModel(self, inputs1, x1, epoch, batch_size, stru_index, algorithm_name):
        predicitons1 = Dense(self.y_nodes, activation='softmax', name='layer_output')(x1)
        model = Model(inputs=inputs1, outputs=predicitons1)
        time_one_start = time.time()

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        print("-----------------{}{} is training----------------".format(algorithm_name, stru_index))
        print(model.summary())

        '''
        Next, we setup training logs for tensorboard as well as some tensorboard callbacks.
    
        tensorboard - callback that logs training data.
        EarlyStopping - callback that monitors 'loss (function)' metric and if the loss function does not get better in tne hext 10 iterations, 
        callback stops the training and restores the network with best weights up untill that iteration.
        '''
        log_dir = os.path.join("logs","{}Train_logs_{}_{}".format(algorithm_name, stru_index, self.new_dataset_name),
                               datetime.datetime.now().strftime("%Y%m%d-%H%M%S"), )
        # TF callback that sets up TensorBoard with training logs.
        tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

        # TF callback that stops training when best value of accuracy function is reached.
        # It also restores weights from the best training iteration.
        # monitor: Available metrics are: loss,accuracy,val_loss,val_accuracy
        early_stop_callback = keras.callbacks.EarlyStopping(monitor='loss', mode='min', patience=17,
                                                            restore_best_weights=True)

        history = model.fit(self.x_train, self.y_train, epochs=epoch, verbose=2, batch_size=batch_size, validation_data=(self.x_vali, self.y_vali),
                            callbacks=[tensorboard_callback, early_stop_callback])

        time_one_finish = time.time()
        time_costing = time_one_finish - time_one_start
        print("time consumed: ", time_costing, "s")

        # Saving the model
        path_model_save = '{}_models_{}/_{}{}_{}.h5'.format(self.item_key, self.dataset_name, algorithm_name, stru_index, self.new_dataset_name)
        print('saved model to {}'.format(path_model_save))
        model.save(path_model_save)

        # to save history as csv
        dataFrm = pd.DataFrame({'acc1': history.history['accuracy'], 'loss1': history.history['loss'], 'val_acc1': history.history['val_accuracy'], 'val_loss1': history.history['val_loss']})
        folder_save = result_dir + "train_history_" + algorithm_name + '_' + str(stru_index) + '_' + self.new_dataset_name +'.csv'
        dataFrm.to_csv(folder_save)

        return model


    # @profile
    @jit(target_backend='cuda')
    def dnn_model(self, epoch, batch_size, stru_index, algorithm_name):

        time_one_start = time.time()
        monitor = gpumonitor.monitor.GPUStatMonitor(delay=1)

        inputs1, x1 = superviced_ML.dnn_structur(self, stru_index)
        dnn_model = superviced_ML.createModel(self, inputs1, x1, epoch, batch_size, stru_index, algorithm_name)

        monitor.stop()
        monitor.display_average_stats_per_gpu()
        time_one_finish = time.time()
        time_costing = time_one_finish - time_one_start


        print('DNN model {} Training is finished.'.format(stru_index))
        return dnn_model, time_costing

    @jit(target_backend='cuda')
    def cnn_model(self, epoch, batch_size, stru_index, algorithm_name):
        time_one_start = time.time()
        monitor = gpumonitor.monitor.GPUStatMonitor(delay=1)

        inputs1, x1 = superviced_ML.cnn_structur(self, stru_index)
        cnn_model = superviced_ML.createModel(self, inputs1, x1, epoch, batch_size, stru_index, algorithm_name)

        monitor.stop()
        monitor.display_average_stats_per_gpu()
        time_one_finish = time.time()
        time_costing = time_one_finish - time_one_start

        print('CNN model {} Training is finished. Time_costing = {}'.format(stru_index, time_costing))

        return cnn_model, time_costing


if __name__ == '__main__':
    print('------ML_structure_implements--------')