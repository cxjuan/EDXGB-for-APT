import warnings

# from Demos.win32cred_demo import target
from sklearn.model_selection import train_test_split
from MLStructure.Training import ML_structure
import pandas as pd
import numpy as np
from collections import Counter
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import MinMaxScaler,MaxAbsScaler, StandardScaler, RobustScaler
np.set_printoptions(threshold=10000)
import os, sys, humanize, psutil, GPUtil
from numba import jit, cuda
import csv

warnings.simplefilter('ignore')
@jit(target_backend='cuda')
def load_data_path(path):
    dataset = pd.read_csv(path)
    return dataset

@jit(target_backend='cuda')
def load_data(dataset_name, path):
    if dataset_name != None:
        dataset = pd.read_csv(path)
    else:
        print('Please Input A Dataset Name!')
        return -1
    return dataset


# turn data from str to float
@jit(target_backend='cuda')
def dataToFloat(data_array):
    return data_array.astype(float)


# turn data from str to float
@jit(target_backend='cuda')
def dataToInt(data_array):
    return data_array.astype(int)


# data minmax normalization
@jit(target_backend='cuda')
def dataNormalization(array):
    scaler = MinMaxScaler()
    scaler.fit(array)
    print('data shape', array.shape)
    return scaler.transform(array)

def folder_check(f_name):
    try:
        if not os.path.exists(f_name):
            os.makedirs(f_name)
    except OSError:
        print("The folder could not be created!")

# @jit(target_backend='cuda')
def exam_data_test(dataset_name, Data, model_index):
    df = dataToFloat(Data)

    # X: features, y: labels
    labels = df['Label']
    features = df.iloc[:, df.columns != 'Label'].astype('float64')
    x_train = dataNormalization(features)

    return x_train, labels

@jit(target_backend='cuda')
def labelChange_onlyDexfil(Data, dataset_name):
    # print(Data.columns.tolist())
    case = [4, 8, 9]
    Data = Data[Data['Label'].isin(case)]

    if dataset_name == 'SCVIC-APT-2021':
        item = 4
        label_8 = 5
        label_9 = label_8 + 1
    elif dataset_name == 'Unraveled':
        item = 4
        label_4 = 0
        label_8 = 1
        label_9 = 2


    print('............ label 8 = {}, label 9 = {}'.format(label_8, label_9))

    item_list_8 = [8]
    item_list_9 = [9]
    item_list_4 = [4]

    for i in range(0, len(Data['Label'])):
        # print('-------', i)
        # print('----', Data['Label'].iloc[i])
        x = Data['Label'].iloc[i]
        # print(x)
        if x in item_list_4:
            Data['Label'].iloc[i] = label_4
        elif x in item_list_8:
            Data['Label'].iloc[i] = label_8
        elif x in item_list_9:
            Data['Label'].iloc[i] = label_9
    print('after label decend, Counter(y): ', Data['Label'].value_counts())

    # df_c2 = [1,2,4]
    # df = Data[Data['Label'].isin(df_c2)]

    return Data


@jit(target_backend='cuda')
def labelChange_no_re(Data, dataset_name):
    if dataset_name == 'SCVIC-APT-2021':
        item = 4
        label_8 = 5
        label_9 = label_8 + 1
    elif dataset_name == 'Unraveled':
        item = 4
        label_8 = 1
        label_9 = item + 1

    print('............ label 8 = {}, label 9 = {}'.format(label_8, label_9))

    item_list_8 = [8]
    item_list_9 = [9]
    # item_list_4 = [4]

    for i in range(0, len(Data['Label'])):
        # print('-------', i)
        # print('----', Data['Label'].iloc[i])
        x = Data['Label'].iloc[i]
        # print(x)
        if x in item_list_8:
            Data['Label'].iloc[i] = label_8
        elif x in item_list_9:
            Data['Label'].iloc[i] = label_9
    print('after label decend, Counter(y): ', Data['Label'].value_counts())
    return Data


@jit(target_backend='cuda')
def labelChange(Data, dataset_name, file_name, model_index):

    # '-Entire', '-experi', '-experi-cii', '-experi-ciii', '-experi-civ-c2', '-experi-civ-size',
    if dataset_name == 'SCVIC-APT-2021':
        if model_index == 2 or model_index == 4:
        # if file_name == 'Original_cii' or file_name == 'Original_civ_c2':
            case = [0, 1, 2, 3, 4, 8]
            Data = Data[Data['Label'].isin(case)]
            label_8 = 5
            label_9 = -1
        elif model_index == 3 or model_index == 5:
        # elif file_name == 'Original_ciii' or file_name == 'Original_civ_size':
            case = [0, 1, 2, 3, 4, 9]
            Data = Data[Data['Label'].isin(case)]
            label_8 = -1
            label_9 = 5

        else:
            item = 5
            label_8 = 5
            label_9 = item + 1
    elif dataset_name == 'Unraveled':
        if model_index == 2 or model_index == 4:
        # if file_name == 'Original_cii' or file_name == 'Original_civ_c2':
            case = [0, 2, 3, 4, 8]
            Data = Data[Data['Label'].isin(case)]
            label_8 = 1
            label_9 = -1
        elif model_index == 3 or model_index == 5:
        # elif file_name == 'Original_ciii' or file_name == 'Original_civ_size':
            case = [0, 2, 3, 4, 9]
            Data = Data[Data['Label'].isin(case)]
            label_8 = -1
            label_9 = 1
        else:
            item = 4
            label_8 = 1
            label_9 = item + 1

    print('............ label 8 = {}, label 9 = {}'.format(label_8, label_9))

    item_list_8 = [8]
    item_list_9 = [9]
    for i in range(0, len(Data['Label'])):
        x = Data['Label'].iloc[i]
        if x in item_list_8:
            Data['Label'].iloc[i] = label_8
        elif x in item_list_9:
            Data['Label'].iloc[i] = label_9
    print('after label decend, Counter(y): ', Data['Label'].value_counts())

    return Data


#after data volatility
@jit(target_backend='cuda')
def exam_data_feature_DVolatility(dataset_name, path, file_name, model_index):
    Data = load_data(dataset_name, path)

    #To change label "8" or "9" into "1"
    if model_index in [2,3,4,5]:
        Data_labelChanged = labelChange(Data=Data, dataset_name=dataset_name, file_name=file_name, model_index=model_index)
    else:
        Data_labelChanged = Data

    train_data = dataToFloat(Data_labelChanged)

    list_label = list(train_data['Label'].unique())
    print(len(list_label), list_label)

    lables = train_data['Label']
    features = train_data.iloc[:, train_data.columns != 'Label'].astype('float64')

    #Real data we're gonna using
    features_train, features_vali, labels_train, labels_vali = train_test_split(features, lables, test_size=.2)

    x_train = dataNormalization(features_train)
    x_vali = dataNormalization(features_vali)

    return x_train, x_vali, labels_train, labels_vali


def save_computing_costs(costs, save_path_):
    print(costs)
    print(type(costs))
    with open(save_path_, mode='w', newline='', encoding="utf-8") as file:
        # Create a DictWriter object
        writer = csv.DictWriter(file, fieldnames=costs.keys())
        # Write the header
        writer.writeheader()
        # Write the data rows
        writer.writerow(costs)

def memory():
    import os
    from wmi import WMI
    w = WMI('.')
    result = w.query("SELECT WorkingSet FROM Win32_PerfRawData_PerfProc_Process WHERE IDProcess=%d" % os.getpid())
    return int(result[0].WorkingSet)

def process_memory():
    import psutil
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss

def mem_report():
    print("CPU RAM Free: " + humanize.naturalsize(psutil.virtual_memory().available))

    GPUs = GPUtil.getGPUs()
    for i, gpu in enumerate(GPUs):
        print('GPU {:d} ... Mem Free: {:.0f}MiB / {:.0f}MiB | Utilization {:3.0f}%'.format(i, gpu.memoryFree,
                                                                                         gpu.memoryTotal,
                                                                                         gpu.memoryUtil * 100))
if __name__ == '__main__':
    print("******************** ML_model_functions ********************")

