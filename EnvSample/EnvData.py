from collections import Counter
import MLStructure.src_path as src_path

import pandas as pd
import numpy as np
import datetime
from sklearn.model_selection import train_test_split
from imblearn import FunctionSampler  # to use a idendity sampler
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from imblearn.pipeline import make_pipeline


from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.pyplot import style
import seaborn as sns
from numba import jit, cuda
import warnings
import MLStructure.PathList as Pathlists

warnings.simplefilter('ignore')
np.seterr(divide='ignore', invalid='ignore')

path_src = Pathlists.MODEL_DIR

def load_data(dataset_name, path):
    if dataset_name != None:
        dataset = pd.read_csv(path)
    else:
        print('Which dataset do you want to use? Please comfirm...')
        return -1
    return dataset

def datachecking(dataset):

    print(np.all(np.isfinite(dataset)))
    print(np.any(np.isnan(dataset)))

    #To drop infinites and na
    if np.all(np.isfinite(dataset)) ==  False:
        print('***isfinite: ')
        infinity_col = [col for col in dataset if not np.all(np.isfinite(dataset[col]))]
        print(infinity_col)
        dataset.replace([np.inf, -np.inf], np.nan, inplace=True)
        dataset.dropna(inplace=True)
        dataset.reset_index(drop=True, inplace=True)

    if np.any(np.isnan(dataset)) ==  True:
        print('***NA: ')
        na_col = [col for col in dataset if not np.all(np.isnan(dataset[col]))]
        print(na_col)
        dataset.dropna(inplace=True)
        dataset.reset_index(drop=True, inplace=True)

def plot_resampling(X, y, sampler, ax, title=None):
    # X_res, y_res = sampler.fit_resample(X, y)
    X_res = X
    y_res = y
    ax.scatter(X_res[:, 0], X_res[:, 1], c=y_res, alpha=0.8, edgecolor="k")
    if title is None:
        title = f"Resampling with {sampler.__class__.__name__}"
    ax.set_title(title)
    sns.despine(ax=ax, offset=10)


def Samplers(strategy):
    samplers = [
        FunctionSampler(),
        RandomOverSampler(sampling_strategy=strategy),
        SMOTE(sampling_strategy=strategy),
        ADASYN(sampling_strategy=strategy),
    ]
    return samplers

def resampling(X, y, sampler):
    # print('Before reampling: ', y.value_counts())
    print(f"Resampling using {sampler.__class__.__name__}")
    X_res, y_res = sampler.fit_resample(X, y)
    df_X = pd.DataFrame(X_res)
    df_y = pd.DataFrame(y_res)
    df = df_X.merge(df_y, how='outer', left_index=True, right_index=True)
    print('After reampling: ', df_y.value_counts())
    return df, X_res, y_res

@jit(target_backend='cuda')
def illustration_no_decisionBoundary(X, y, samplers, word, file_name, dataWeNeed_path,set_name_final):
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(15, 15))

    samplers_name = {
        '_Re_FunctionSampler': '_Re_Original',
        '_Re_RandomOverSampler': '_Re_ROSampler',
        '_Re_SMOTE': '_Re_SMOTE',
        '_Re_ADASYN': '_Re_ADASYN',
    }

    for ax, sampler in zip(axs.ravel(), samplers):
        title = "Original dataset" if isinstance(sampler, FunctionSampler) else None
        # X_res, y_res = sampler.fit_resample(X, y)
        samplers_name_fun = f"_Re_{sampler.__class__.__name__}"
        df, X_res, y_res = resampling(X, y, sampler)
        plot_resampling(np.array(X_res), np.array(y_res), sampler, ax, title=title)
        print('------to save as {} -----------'.format(dataWeNeed_path + "/" + word + set_name_final + file_name + samplers_name[samplers_name_fun] + ".csv"))
        df.to_csv(dataWeNeed_path + "/" + word + set_name_final + file_name + samplers_name[samplers_name_fun] + ".csv")
    fig.tight_layout()
    fig.savefig(dataWeNeed_path+"/after_res_final_"+word+".pdf")
    plt.show()

    return df

def label_recovery(dataset, dataset_name):
    if dataset_name == 'SCVIC-APT-2021':
        label_ = 1
    elif dataset_name == 'Unraveled':
        label_ = 2
    elif dataset_name == 'dapt2020':
        label_ = 1

    item_list = [8, 9]
    for i in range(0, len(dataset['Label'])):
        x = dataset.loc[i, :]
        if x['Label'] in item_list:
            dataset.loc[i, 'Label'] = label_
    return dataset

@jit(target_backend='cuda')
def data_volatility_cii(dataset, dataset_name, DExfil_label):
    counter_c2 = 0
    counter_size = 0
    '''
    exfil_c2: 8
    exfil_sizelimit: 9
    '''
    train_data = dataset
    extracted_by_CICFlowMeter = ['SCVIC-APT-2021']
    extracted_by_NFStream = ['Unraveled']

    label = DExfil_label
    for i in range(0, len(train_data['Label'])):
        x = train_data.loc[i, :]

        if x['Label'] == label:
            if dataset_name in extracted_by_CICFlowMeter:
                Total_Bwd_packets = x['Total Bwd packets']
                Total_Fwd_packets = x['Total Fwd Packet']
                Total_Length_of_Bwd_Packet = x['Total Length of Bwd Packet']
                Total_Length_of_Fwd_Packet = x['Total Length of Fwd Packet']
                Flow_Duration = x['Flow Duration'] # 1 microsecond = 0.000001 second
                t_first = x['Timestamp']
            elif dataset_name in extracted_by_NFStream:
                Total_Bwd_packets = x['dst2src_packets']
                Total_Fwd_packets = x['src2dst_packets']
                Total_Length_of_Bwd_Packet = x['dst2src_bytes']
                Total_Length_of_Fwd_Packet = x['src2dst_bytes']
                Flow_Duration = x['bidirectional_duration_ms']
                t_first = x['bidirectional_first_seen_ms']

            # --------------PTR------------------------
            PTR_I = Total_Bwd_packets / Flow_Duration
            PTR_O = Total_Fwd_packets / Flow_Duration

            # --------------BTR------------------------
            BTR_I = Total_Length_of_Bwd_Packet / Flow_Duration
            BTR_O = Total_Length_of_Fwd_Packet / Flow_Duration

            # exfil_c2
            if PTR_I <= PTR_O and BTR_I > BTR_O:
                counter_c2 += 1
                train_data.loc[i, 'Label'] = 8
            # exfil_size
            elif PTR_I > PTR_O and BTR_I <= BTR_O:
                counter_size += 1
                train_data.loc[i, 'Label'] = 9
            # exfil_c2_size
            elif PTR_I > PTR_O and BTR_I > BTR_O:
                counter_size += 1
                train_data.loc[i, 'Label'] = 89

    return train_data

def main_scv(dataset_name):
    new_path_train = path_src + dataset_name + '/_Encoder_Train' + '+.csv'
    new_path_test = path_src + dataset_name + '/_Encoder_Test' + '+.csv'

    old_path_train = path_src + dataset_name + '/Training.csv'
    old_path_test = path_src + dataset_name + '/Testing.csv'

    TrainData = load_data(dataset_name, old_path_train)
    TestData = load_data(dataset_name, old_path_test)

    print(TrainData['Label'].value_counts())
    print(TestData['Label'].value_counts())
    return TrainData,TestData


def main_un(dataset_name):
    #----------Using Activity as Label--------
    path_tr = path_src + dataset_name + '/exfil_lab/exil_data_train.csv'
    path_te = path_src + dataset_name + '/exfil_lab/exil_data_test.csv'

    data_tr = load_data(dataset_name, path_tr)
    data_te = load_data(dataset_name, path_te)

    print(data_tr['Label'].value_counts())
    print(data_te['Label'].value_counts())
    return data_tr, data_te

def resampling_execu(aim, dataset, dataWeNeed_path, file_name, strategy, set_name_final):
    samplers_ = Samplers(strategy)
    # to draw to plt
    X_ = dataset.loc[:, dataset.columns != 'Label']
    y_ = dataset['Label']
    df = illustration_no_decisionBoundary(X_, y_, samplers=samplers_,dataWeNeed_path=dataWeNeed_path, word=aim,
                                          file_name=file_name, set_name_final=set_name_final)

    return df


if __name__ == '__main__':
    result_graph = path_src + 'img/'
    '''
    Label Setting:
        exfil_c2: 8
        exfil_sizelimit: 9
    '''

    # ------------------------------ < Main > ------------------------
    print(datetime.datetime.now())
    words = [
        'train',
        'test',
    ]
    dataname = [
        'SCVIC-APT-2021',
        'Unraveled'
    ]

    samplers = [
        'Original_entire',#no 8 and 9
        'Original_cii_ciii',#both 8 and 9
        'Original_civ_c2',#either 8 or 9
        'Original_civ_size',  # either 8 or 9
        # 'ROSampler',
        # 'SMOTE',
        # 'ADASYN',
    ]

    dataset_name = dataname[1]
    dataWeNeed_path = path_src + 'res_datasets/' + dataset_name
    paths = [
        # un-train
        path_src + dataset_name + '/exfil_lab/activity_2_label_exil_data_train.csv',
        # un-test
        path_src + dataset_name + '/exfil_lab/activity_2_label_exil_data_test.csv'
    ]

    for aim in words:
        dataset_name = dataname[0]
        dataWeNeed_path = path_src + 'res_datasets/' + dataset_name

        print('Let\'s Check: dataset_name = {}, aim = {}'.format(dataset_name, aim))
        #Data loading
        if dataset_name == 'SCVIC-APT-2021':
            df_ = main_scv(dataset_name)
            DExfil_ = 5
        elif dataset_name == 'Unraveled':
            df_ = main_un(dataset_name)
            DExfil_ = 1
        word = aim

        print('--------For Original_cii_ciii-------------')
        print(df_['Label'].value_counts())
        data_new_cii = data_volatility_cii(df_, dataset_name=dataset_name, DExfil_label=DExfil_)
        print(data_new_cii['Label'].value_counts())

        data_new_cii.to_csv(dataWeNeed_path + "/" + word + "_" + samplers[1] + ".csv")
        print('saved into ', dataWeNeed_path + "/" + word + "_" + samplers[1] + ".csv")

    print(datetime.datetime.now())


