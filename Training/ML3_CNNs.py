from MLStructure.Training import ML_structure
from MLStructure.Training import ML_model

import pandas as pd
import numpy as np
import keras

from sklearn.preprocessing import MinMaxScaler,MaxAbsScaler, StandardScaler, RobustScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.pyplot import style
import os
import csv
import MLStructure.PathList as Pathlists
from numba import jit, cuda
import warnings
from timeit import default_timer

warnings.simplefilter('ignore')

def save_computing_costs(costs, save_path_):
    print(costs)
    with open(save_path_, mode='w', newline='', encoding="utf-8") as file:
        # Create a DictWriter object
        writer = csv.DictWriter(file, fieldnames=costs.keys())
        # Write the header
        writer.writeheader()
        # Write the data rows
        writer.writerows(costs)

@jit(target_backend='cuda')
def main_training(train_path, dataset_name, new_dataset_name, model_name_index, file_name, tr_model_index=-1):
    algorithm_names_cnn = ['CNN-experi', 'CNN-experi-cii', 'CNN-experi-ciii', 'CNN-experi-civ-c2',
                           'CNN-experi-civ-size']

    # ------------------------------ < Main > -------------------------
    x, x_vali, y, y_vali = ML_model.exam_data_feature_DVolatility(dataset_name=dataset_name, path=train_path,
                                                                  file_name=file_name, model_index=model_name_index)

    input_nodes = x.shape[1]
    output_nodes = len(y.unique())#6/2
    print('input_nodes = {}, output_nodes = {}'.format(input_nodes, output_nodes))

    #X: features, y: labels
    #Y: Converts a class vector (integers) to binary class matrix.
    y_train = keras.utils.to_categorical(y, num_classes=output_nodes)

    # reshape input to be [samples, time steps, features]
    x_train = np.reshape(x, (x.shape[0], 1, x.shape[1]))
    #num_input = np.reshape(num_input, (num_input, 1))
    #print(num_input.shape)

    y_val = keras.utils.to_categorical(y_vali, num_classes=output_nodes)
    X_val = np.reshape(x_vali, (x_vali.shape[0], 1, x_vali.shape[1]))

    epoch = 32
    batch_size = 64

    keys = ['Experi_Saved']
    item_key = keys[0]

    supervised_model_ = ML_structure.superviced_ML(input_nodes, output_nodes, x_train, y_train, X_val, y_val, dataset_name,
                                                   new_dataset_name, item_key=item_key)

    algorithm_name = algorithm_names_cnn[model_name_index]
    time_costs = {}

    for i in range(1, tr_model_index):
        print('Model {}_{} Start to Training...')
        start_gup = default_timer()
        model, compute_cost = supervised_model_.cnn_model(epoch=epoch, batch_size=batch_size, stru_index=i, algorithm_name=algorithm_name)
        compute_costing = default_timer() - start_gup

        time_costs[algorithm_name + '_' + str(i)] = compute_cost
        print('Model_{}: {} ...'.format(i, compute_costing))
        print('Lets Go to the Next Model...')

    save_path_ = "***" + file_name + "_training_time_cost_CNNs.csv"
    save_computing_costs(time_costs, save_path_)

if __name__ == '__main__':
    # ---------------------<main>-----------------------------
    path_src = Pathlists.folder_check(Pathlists.ML_DATA_DIR)
    res_data_dir = Pathlists.folder_check(Pathlists.RES_DATA_DIR)

    repetition = 1

    dataname = [
        'Unraveled',
        'SCVIC-APT-2021'
    ]

    samplers = [
        'Original',
        'ROSampler',
        'SMOTE',
        'ADASYN',
    ]

    data_files = {
        'Original_entire': 0,  # entire data without case separation
        'Original_cii': 1,  # data for case c-ii
        'Original_ciii': 2,  # data for case c-iii
        'Original_civ_c2': 3,  # data for case c-iv-C2
        'Original_civ_size': 4,  # data for case c-iv-Size
    }

    words = ['train', 'test']
    word = words[0]

    for dataset_name in dataname:
        dataWeNeed_path = path_src + 'res_datasets/' + dataset_name
        for sampler in samplers:
                for file_name, model_name_index in data_files.items():
                    print('----------- start [ dataset={}, file_name={}, model_index={} ]-------------'.format(dataset_name, file_name, model_name_index))
                    path_tr = dataWeNeed_path + "/" + word + '_final_' + file_name + "_Re_" +sampler+ ".csv"
                    print('training set path = ', path_tr)

                    new_dataset_name = dataset_name + '_' + file_name +'_'+ sampler
                    for i in range(repetition):
                        main_training(train_path=path_tr, dataset_name=dataset_name, new_dataset_name=new_dataset_name,
                                      model_name_index=model_name_index, file_name=file_name)








