

from MLStructure.Training import ML_structure
from MLStructure.Training import ML_model
from collections import Counter
from imblearn.over_sampling import SMOTE

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
# import MLStructure.dataResample as dataResample

from numba import jit, cuda, njit
import warnings

from timeit import default_timer

from MLStructure.Training.ML_model import memory

warnings.simplefilter('ignore')




@jit(target_backend='cuda')
def main_training(train_path, dataset_name, new_dataset_name, model_name_index, file_name):
    algorithm_names_dnn = ['DNN-experi', 'DNN-experi-cii', 'DNN-experi-ciii', 'DNN-experi-civ-c2',
                           'DNN-experi-civ-size']

    # ------------------------------ < Main > -------------------------
    x, x_vali, y, y_vali = ML_model.exam_data_feature_DVolatility(dataset_name=dataset_name, path=train_path, file_name=file_name, model_index=model_name_index)

    list_label = list(y.unique())
    print(len(list_label), list_label)
    print('-----------------------')
    input_nodes = x.shape[1]
    output_nodes = len(list_label)#6
    print('input nodes = ', input_nodes)
    print('output nodes = ', output_nodes)

    print('training set y:', y.value_counts())
    print('vali set y_vali:', y_vali.value_counts())


    y = keras.utils.to_categorical(y, num_classes=output_nodes)
    y_val = keras.utils.to_categorical(y_vali, num_classes=output_nodes)

    X = x
    X_val = x_vali

    epoch = 32
    batch_size = 64

    keys = ['Experi_Saved']
    item_key = keys[0]

    supervised_model_ = ML_structure.superviced_ML(input_nodes, output_nodes, X, y, X_val, y_val, dataset_name, new_dataset_name, item_key=item_key)


    algorithm_name = algorithm_names_dnn[model_name_index]
    time_costs={}
    gpu_costs={}

    for i in range(1,5):
        print('Model {}_{} Start to Training...')
        # start_gup = default_timer()
        model, compute_cost = supervised_model_.dnn_model(epoch=epoch, batch_size=batch_size, stru_index=i, algorithm_name=algorithm_name)
        # compute_cost = default_timer() - start_gup

        print('Model_{}: time = {} ...'.format(i, compute_cost))
        time_costs[algorithm_name+'_'+str(i)] = compute_cost
        # gpu_costs[algorithm_name+'_'+str(i)] = memory_costs

        print('Lets Go to the Next Model...')

    save_path_ = "***" + str(model_name_index) + "_training_time_cost_DNNs.csv"
    ML_model.save_computing_costs(time_costs, save_path_)


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
        if dataset_name == 'SCVIC-APT-2021' or dataset_name == 'Unraveled':
            for sampler in samplers:
                for file_name_, model_name_index in data_files.items():
                    if file_name_ == 'Original_cii' or file_name_ == 'Original_ciii':
                        file_name = 'Original_cii_ciii'
                    else:
                        file_name = file_name_

                    print('----------- start [ dataset={}, file_name={}, model_index={} ]-------------'.format(dataset_name, file_name_, model_name_index))
                    path_tr = dataWeNeed_path + "/" + word + '_final_' + file_name + "_Re_" + sampler + ".csv"

                    new_dataset_name = dataset_name + '_' + file_name +'_'+ sampler

                    for i in range(repetition):
                        main_training(train_path=path_tr, dataset_name=dataset_name, new_dataset_name=new_dataset_name, model_name_index=model_name_index, file_name=file_name)








