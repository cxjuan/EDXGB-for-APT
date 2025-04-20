import numpy as np
from sklearn.metrics import accuracy_score, classification_report, multilabel_confusion_matrix, confusion_matrix

np.set_printoptions(threshold=10000)
import pandas as pd
import os

import keras.utils
import keras.callbacks
from MLStructure.Training import ML_structure
from MLStructure.Training import ML_model
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

import xgboost as xgb
from xgboost import XGBClassifier

from collections import Counter
from timeit import default_timer
import MLStructure.PathList as Pathlists
from numba import jit, cuda
import gpumonitor
import warnings
warnings.simplefilter('ignore')

@jit(target_backend='cuda')
def get_intermediate_layer(model, data):
    total_layers = len(model.layers)
    print('total layers of model = ', total_layers)

    fl_index = total_layers - 1

    intermediate_layer_model = keras.Model(
        inputs=model.input,
        outputs=model.get_layer(index=fl_index-1).output)
    intermediate_output = intermediate_layer_model.predict(data)

    return intermediate_output

@jit(target_backend='cuda')
def xgb_classifier(X_train, y_train,depth, LRate, estimator):
    model = XGBClassifier(max_depth=depth, learning_rate=LRate, n_estimators=estimator)
    model.fit(X_train, y_train)
    return model


@jit(target_backend='cuda')
def main_ensembel_xgb(x_train, y_train, algorithm_name,index,path_model, xgbpath, depth, Learning_Rate, estimator):
    print('Loading Model {}_{}...'.format(algorithm_name, index))

    """
    # To get nodes' outputs from the last hidden-layer: intermediate_output = intermediate_layer_model.predict(data)
    """
    nn_model = keras.models.load_model(path_model)
    intermediate_output = get_intermediate_layer(model=nn_model, data=x_train)

    start_time = default_timer()
    monitor = gpumonitor.monitor.GPUStatMonitor(delay=1)

    # using nodes' outputs to train trees
    print('Training XGBoost Model...')
    ensembel_xgb_model = xgb_classifier(intermediate_output, y_train, depth=depth, LRate=Learning_Rate,
                                        estimator=estimator)

    monitor.stop()
    if monitor.average_stats is None:
        print('average_stats is None')
        pass
    else:
        monitor.display_average_stats_per_gpu()

    elapsed_time = default_timer() - start_time

    ensembel_xgb_model.save_model(xgbpath)
    return elapsed_time

@jit(target_backend='cuda')
def main_xgb(X_train, y_train, X_test, y_test, labels, xgbpath):
    xgb = xgb_classifier(X_train, y_train)
    xgb.save_model(xgbpath)

    y_pred = xgb.predict(X_test)
    print(y_pred)
    print(Counter(y_pred))
    print(np.unique(y_pred))

    # Run Classification Metrics
    print("Accuracy Score :")
    print(accuracy_score(y_test, y_pred))
    print("Classification Report :")
    print(classification_report(y_test, y_pred, labels=labels))

    print("multilabel_confusion_matrix Report :")
    print(multilabel_confusion_matrix(y_test, y_pred, labels=labels))
    matrixes = confusion_matrix(y_test, y_pred, labels=labels)
    print('--------confusion_matrix-------------')
    print(matrixes)

@jit(target_backend='cuda')
def main_(algorithm_name, dataset_name, new_dataset_name, new_path_train, flag, count, path_src_model, item_key,model_index,file_name):

    # ------------------------------ < Main > -------------------------
    x, x_vali, y, y_vali = ML_model.exam_data_feature_DVolatility(dataset_name=dataset_name, path=new_path_train,
                                                                  file_name=file_name, model_index=model_index)

    list_label = list(y.unique())
    print(len(list_label), list_label)

    input_nodes = x.shape[1]
    output_nodes = len(list_label)  # 6

    # X: features, y: labels
    # Y: Converts a class vector (integers) to binary class matrix.
    # y_train = keras.utils.to_categorical(y, num_classes=output_nodes)
    y_train = y
    y_val = y_vali
    if flag == 'dnn':
        #DNN input
        x_train = x
        X_val = x_vali
    elif flag == 'cnn':
        #CNN input
        x_train = np.reshape(x, (x.shape[0], 1, x.shape[1]))
        X_val = np.reshape(x_vali, (x_vali.shape[0], 1, x_vali.shape[1]))

    xgb_name = 'xgb_'
    depths = [
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        20,
        30,
    ]
    estimators = [
        20,
        30,
        40,
        50,
        100,
        150,
        200,
    ]
    lrates = [
        0.1,
        0.01,
        0.001,
    ]
    edxgb_tr_time = {}


    for i in range(1, count):
        print(algorithm_name)
        path_model = path_src_model+'{}_models_{}/_{}{}_{}.h5'.format(item_key, dataset_name, algorithm_name, i, new_dataset_name)
        xgbpath = path_src_model+'{}_models_{}/_{}{}_{}_{}.h5'.format(item_key, dataset_name, algorithm_name, i, new_dataset_name, xgb_name)
        xgbpath_single = path_src_model+'{}_xgb_{}/_{}.h5'.format(item_key, dataset_name, xgb_name)

        for depth in depths:
            for estimator in estimators:
                for Learning_Rate in lrates:
                    #training & save ensemble xgb models
                    print('----------depth={}, Learning_Rate={}, estimator={}----------'.format(depth, Learning_Rate, estimator))
                    new_xgb_name = 'xgb_d{}_l{}_e{}'.format(depth, Learning_Rate, estimator)
                    new_xgbpath = path_src_model+'{}_models_{}/_{}{}_{}_{}.h5'.format(item_key, dataset_name, algorithm_name, i, new_dataset_name, new_xgb_name)

                    elapsed_time = main_ensembel_xgb(x_train, y_train,algorithm_name,i,path_model, new_xgbpath, Learning_Rate=Learning_Rate, depth=depth, estimator=estimator)
                    edxgb_tr_time[str(i)+'_'+new_xgb_name] = elapsed_time
                    print('Time Price: ', elapsed_time)

        #training & save xgb models
        # main_xgb(x_train, y_train, X_val, y_val, list_label, xgbpath_single)

        print('Lets Go to the Next Model...')
    return edxgb_tr_time

def main_execute(dataset_name, path_train, algorithm_name_dnn, algorithm_name_cnn, new_dataset_name, path_src_model, item_key, model_index, file_name):

    flag_un_dnn = 'dnn'
    count_un_dnn = 5
    edxgb_tr_time_d = main_(algorithm_name=algorithm_name_dnn, dataset_name=dataset_name, new_path_train=path_train, new_dataset_name=new_dataset_name,
         flag=flag_un_dnn, count=count_un_dnn, path_src_model=path_src_model, item_key=item_key, model_index=model_index, file_name=file_name)
    print(edxgb_tr_time_d)

    flag_un_cnn = 'cnn'
    count_un_cnn = 5
    edxgb_tr_time_c = main_(algorithm_name=algorithm_name_cnn, dataset_name=dataset_name, new_path_train=path_train, new_dataset_name=new_dataset_name,
         flag=flag_un_cnn, count=count_un_cnn, path_src_model=path_src_model, item_key=item_key, model_index=model_index, file_name=file_name)
    print(edxgb_tr_time_c)


if __name__ == '__main__':
    path_src = Pathlists.folder_check(Pathlists.ML_DATA_DIR)
    path_src_model = Pathlists.folder_check(Pathlists.MODEL_DIR)
    res_data_dir = Pathlists.folder_check(Pathlists.RES_DATA_DIR)

    algorithm_names_dnn = ['DNN-experi', 'DNN-experi-cii', 'DNN-experi-ciii', 'DNN-experi-civ-c2',
                           'DNN-experi-civ-size']

    algorithm_names_cnn = ['CNN-experi', 'CNN-experi-cii', 'CNN-experi-ciiiI', 'CNN-experi-civ-c2',
                           'CNN-experi-civ-size']

    keys = ['Experi_Saved']
    item_key = keys[0]

    dataname = [
        'SCVIC-APT-2021',
        'Unraveled'
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
        for file_name, model_name_index in data_files.items():
                algorithm_name_dnn = algorithm_names_dnn[model_name_index]
                algorithm_name_cnn = algorithm_names_cnn[model_name_index]

                for sampler in samplers:
                    path = dataWeNeed_path + "/" + word + '_final_' + file_name + "_Re_" +sampler+ ".csv"
                    new_dataset_name = dataset_name + '_' + file_name + '_' + sampler
                    main_execute(dataset_name=dataset_name, path_train=path,
                                 algorithm_name_dnn=algorithm_name_dnn, algorithm_name_cnn=algorithm_name_cnn, new_dataset_name=new_dataset_name,
                                 path_src_model=path_src_model, item_key=item_key, model_index=model_name_index, file_name=file_name)





