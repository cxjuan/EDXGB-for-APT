import sys
import os
from idlelib.iomenu import errors

# from Data.ML_data.Unraveled.tools.machine_learning.OCSVM.SVMOneClass import threshold

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import keras
import tensorflow as tf
import xgboost as xgb
import os
import seaborn as sn

from sklearn.metrics import matthews_corrcoef, accuracy_score, classification_report, multilabel_confusion_matrix, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay, roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn import metrics

from MLStructure.Training import ML_model
from MLStructure.Training.ML_hybid import get_intermediate_layer
from MLStructure.dataResample import recoveried_dataset, label_recovery
import MLStructure.PathList as Pathlists
import MLStructure.dataResample as dataResample
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from collections import Counter

from imblearn import FunctionSampler  # to use a idendity sampler
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN

# Test with GPU
from numba import jit, cuda
# To measure exec time
from timeit import default_timer
from xgboost import XGBClassifier

import csv

path_src = Pathlists.MODEL_DIR
path_src_model = Pathlists.folder_check(Pathlists.MODEL_DIR)
path_src_result = Pathlists.RESULT_SAVE_DIR

# @jit(target_backend='cuda')
def evluation(y_pred, y_true, labels, algorithem_name, title_plt, labels_matrix_dict, tour_guide):
    result_graph = os.path.join(path_src_result, "test_result_confustionMatrix", dataset_name)
    Pathlists.folder_check(result_graph)

    if len(y_pred.shape) == 2:
        y_pred = np.argmax(y_pred, axis=1)
    elif len(y_pred.shape) == 1:
        y_pred = y_pred

    print('---------------------')
    acc = accuracy_score(y_true, y_pred)
    f_1 = f1_score(y_true, y_pred, average='macro')
    pr = precision_score(y_true, y_pred, average='macro')
    rc = recall_score(y_true, y_pred, average='macro')

    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred, pos_label=int(float(tour_guide)))
    auc_score = metrics.auc(fpr, tpr)
    mcc_score = matthews_corrcoef(y_true, y_pred)
    print(f'For label {tour_guide}, auc_mcc = {auc_score}_{mcc_score}')

    print("Classification Report :")
    report = classification_report(y_true, y_pred, labels=labels, digits=4, output_dict=True)
    print(report[tour_guide])


    print("multilabel_confusion_matrix Report :")
    matrixes = multilabel_confusion_matrix(y_true, y_pred, labels=labels)
    matrixes_all = confusion_matrix(y_true, y_pred, labels=labels)
    print(matrixes)
    print('---------------------')

    matrixes_df = pd.DataFrame.from_dict(matrixes_all)
    m_path = result_graph + '/ConMatrix_report_{}'.format(algorithem_name) + ".csv"
    matrixes_df.to_csv(m_path, index=False, header=True)

    return pr, rc, f_1, acc, report, matrixes, auc_score, mcc_score


def test_single_xgb(xgbpath_single, x_test, y_true, algorithm_name, labels):
    print('Loading XGBoost Model...')
    my_xgb_model = xgb.XGBClassifier()
    my_xgb_model.load_model(xgbpath_single)

    pred_results = my_xgb_model.predict(x_test)

    y_pred = pred_results
    print('--------Model {} Evluation-------------'.format(algorithm_name))
    TP, FP, FN, TN = evluation(y_pred=y_pred, y_true=y_true, labels=labels, algorithem_name=algorithm_name)
    print('TP, FP, FN, TN = ', TP, FP, FN, TN)
    acc = (TP + TN) / (TP + TN + FP + FN)
    print(acc)


def test_single_nn(path_nn_model, x_test, y_true, labels, algorithm_name, index, tour_guide):
    print('Loading Model {}_{}...'.format(algorithm_name, index))
    nn_model = keras.models.load_model(path_nn_model)
    print('----------------Baseline Model {}{}----------------'.format(algorithm_name, index))
    y_pred_nn = nn_model.predict(x_test)
    print(y_pred_nn)

    intermediate_test_output = get_intermediate_layer(model=nn_model, data=x_test)
    print(intermediate_test_output)

    print('--------Model Evluation-------------')
    y_pred = y_pred_nn
    TP, FP, FN, TN, report = evluation(y_pred=y_pred, y_true=y_true, labels=labels, algorithem_name=algorithm_name, tour_guide=tour_guide)
    print('TP, FP, FN, TN = ', TP, FP, FN, TN)
    acc = (TP + TN) / (TP + TN + FP + FN)
    print(acc)

def csv_writer(folder_save,new_algorithem_name, precision,accuracy,recall,f1):
    with open(folder_save, "a", newline="",
              encoding="utf-8") as f:  # all the values found are saved in the opened file.
        wrt = csv.writer(f)
        wrt.writerow(['', "accuracy", "precision", "recall", "f1"])
        for i in range(0, len(precision)):
            wrt.writerow([new_algorithem_name, accuracy[i], precision[i], recall[i], f1[i]])


@jit(target_backend='cuda')
def test_edxgbs(path_xgb_model, intermediate_test_output, algorithm_name, index, setName,
                    dislabel_ed, y_true, labels, precision, recall, f1, accuracy,
                    folder_save_dir, new_algorithem_name, tour_guide):

    edxgb_test_start = default_timer()

    print('Loading XGBoost Model...')
    my_xgb_model = xgb.XGBClassifier()
    my_xgb_model.load_model(path_xgb_model)

    print('--------Ensemble Model-------------')
    pred_results = my_xgb_model.predict(intermediate_test_output)

    y_pred = pred_results

    algorithem_name = 'EDeepXGB based on {}_{}({})'.format(algorithm_name, index, setName)
    dislabel_ed = dislabel_ed[index]
    print('dislabel_base = ', dislabel_ed)
    precision_em, recall_em, f1_em, accuracy_em, report_edeepxgb, matrixes_edxgb, auc_score, mcc_score = evluation(y_pred=y_pred,
                                                                                             y_true=y_true,
                                                                                             labels=labels,
                                                                                             algorithem_name=algorithem_name,
                                                                                             title_plt=dislabel_ed,
                                                                                             labels_matrix_dict=labels_matrix_dict,
                                                                                             tour_guide=tour_guide)

    precision.append(float(precision_em))
    recall.append(float(recall_em))
    f1.append(float(f1_em))
    accuracy.append(float(accuracy_em))

    print("************* overall performance (Ensemble)*******************")
    print(precision_em)
    print(recall_em)
    print(f1_em)
    print(accuracy_em)


    dataFrm2 = pd.DataFrame.from_dict(report_edeepxgb)
    folder_save2 = folder_save_dir + "/_EDxgb_result_" + new_algorithem_name + '.csv'
    dataFrm2.to_csv(folder_save2, index=False, header=True)
    key_report = report_edeepxgb.keys()

    folder_save3_dir = os.path.join(path_src_result, "_acc_precision", dataset_name)
    Pathlists.folder_check(folder_save3_dir)
    folder_save3 = folder_save3_dir + "/_metrix_accResult_" + new_algorithem_name + '.csv'

    csv_writer(folder_save3, new_algorithem_name, precision, accuracy, recall, f1)
    edxgb_test_end = default_timer()-edxgb_test_start

    return matrixes_edxgb, key_report, edxgb_test_end, auc_score, mcc_score


@jit(target_backend='cuda')
def test_ensemble_xgb(path_nn_model, path_xgb_model, x_test, y_true, labels, algorithm_name, index, dislabel_ba,
                      dislabel_ed, setName, labels_matrix_dict, base_auc, edxgb_auc, base_mcc, edxgb_mcc,tour_guide):
    precision = []
    recall = []
    f1 = []
    accuracy = []

    result_tree = os.path.join(path_src_result, 'result_graph', 'dexgb', dataset_name)
    Pathlists.folder_check(result_tree)
    print('Loading Model {}_{}...'.format(algorithm_name, index))
    print(path_nn_model)

    nn_model = keras.models.load_model(path_nn_model)
    intermediate_test_output = get_intermediate_layer(model=nn_model, data=x_test)

    print('----------------Baseline Model {}{}----------------'.format(algorithm_name, index))

    total_layers = len(nn_model.layers)
    fl_index = total_layers - 2
    final_index = total_layers - 1
    intermediate_layer_model = keras.Model(
        inputs=nn_model.get_layer(index=fl_index).output,
        outputs=nn_model.get_layer(index=final_index).output)
    y_pred_nn = intermediate_layer_model.predict(intermediate_test_output)

    new_algorithem_name = algorithm_name + str(index) + "_" + setName
    dislabel_base = dislabel_ba[index]
    print('dislabel_base = ', dislabel_base)
    precision_base, recall_base, f1_base, accuracy_base, report_base, matrixes_base, auc_score_base, mcc_score_base = evluation(y_pred=y_pred_nn,
                                                                                                y_true=y_true,
                                                                                                labels=labels,
                                                                                                algorithem_name=new_algorithem_name,
                                                                                                title_plt=dislabel_base,
                                                                                                labels_matrix_dict=labels_matrix_dict,
                                                                                                tour_guide=tour_guide)

    base_auc[str(index)] = auc_score_base
    base_mcc[str(index)] = mcc_score_base


    precision.append(float(precision_base))
    recall.append(float(recall_base))
    f1.append(float(f1_base))
    accuracy.append(float(accuracy_base))


    print("************* overall performance (Base)*******************")
    dataFrm = pd.DataFrame.from_dict(report_base)
    folder_save_dir = os.path.join(path_src_result, "reports", dataset_name, "Classification_Report")
    Pathlists.folder_check(folder_save_dir)
    folder_save = folder_save_dir + "/_base_result_" + new_algorithem_name + '.csv'
    dataFrm.to_csv(folder_save, index=False, header=True)

    print('----------------EDXGB Model {}{}----------------'.format(algorithm_name, index))
    matrixes_edxgb, key_report, time_cost, aucs_edxgb, mccs_edxgb = test_edxgbs(path_xgb_model=path_xgb_model, intermediate_test_output=intermediate_test_output,
                                             algorithm_name=algorithm_name, index=index, setName=setName,
                dislabel_ed=dislabel_ed, y_true=y_true, labels=labels, precision=precision, recall=recall, f1=f1, accuracy=accuracy,
                folder_save_dir=folder_save_dir, new_algorithem_name=new_algorithem_name, tour_guide=tour_guide)

    edxgb_auc['edxgb_' + str(index)] = aucs_edxgb
    edxgb_mcc['edxgb_' + str(index)] = mccs_edxgb

    return matrixes_base, matrixes_edxgb, key_report, base_auc, base_mcc, edxgb_auc, edxgb_mcc


# @jit(target_backend='cuda')
def main_(flag, count, dataset_name, dataset, setName, algorithm_name, new_dataset_name, item_key, dis_label,
          labels_matrix_dict, tour_guide):
    # ------------------------------ < Main > -------------------------
    """
    :param flag: dnn/cnn
    """

    x, y = ML_model.exam_res_data_test(Data=dataset)

    if flag == 'dnn':
        x_test = x
        y_test = y

        dis_label_dict = {
            1: 'DNN\N{SUBSCRIPT one}',
            2: 'DNN\N{SUBSCRIPT two}',
            3: 'DNN\N{SUBSCRIPT three}',
            4: 'DNN\N{SUBSCRIPT four}',
        }

        dis_label_dict_ed = {
            1: 'EDXGB_D\N{SUBSCRIPT one}',
            2: 'EDXGB_D\N{SUBSCRIPT two}',
            3: 'EDXGB_D\N{SUBSCRIPT three}',
            4: 'EDXGB_D\N{SUBSCRIPT four}',
        }

        # if dataset_name == 'SCVIC-APT-2021':
        #     dis_label = ['Normal', 'Pivoting', 'Reconn', 'L-Move', 'Initial-Com', 'DExfil']
        # elif dataset_name == 'Unraveled':
        #     dis_label = ['Normal', 'DExfil', 'E-foothold', 'L-Move', 'C-up']

    elif flag == 'cnn':
        x_test = np.reshape(x, (x.shape[0], 1, x.shape[1]))
        y_test = y
        dis_label_dict = {
            1: 'CNN\N{SUBSCRIPT one}',
            2: 'CNN\N{SUBSCRIPT two}',
            3: 'CNN\N{SUBSCRIPT three}',
            4: 'CNN\N{SUBSCRIPT four}',
        }
        dis_label_dict_ed = {
            1: 'EDXGB_C\N{SUBSCRIPT one}',
            2: 'EDXGB_C\N{SUBSCRIPT two}',
            3: 'EDXGB_C\N{SUBSCRIPT three}',
            4: 'EDXGB_C\N{SUBSCRIPT four}',
        }

    labels = y.unique()
    print('labels = ', labels)

    # Test XGB
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

    tree_results = {}

    for depth in depths:
        for estimator in estimators:
            for Learning_Rate in lrates:
                # training & save ensemble xgb models

                print('----------depth={}, Learning_Rate={}, estimator={}----------'.format(depth, Learning_Rate,
                                                                                            estimator))
                tps = []
                fps = []
                tns = []
                fns = []
                tps_ = []
                fps_ = []
                tns_ = []
                fns_ = []

                base_auc_dic = {}
                edxgb_auc_dic = {}
                base_mcc_dic = {}
                edxgb_mcc_dic = {}

                for i in range(1, count):
                    tree_id = '{}_depth={}_Learning_Rate={}_estimator={}_'.format(i, depth, Learning_Rate,
                                                                                estimator)
                    print('algorith name = ', algorithm_name)
                    path_model = path_src_model + '{}_models_{}/_{}{}_{}.h5'.format(item_key, dataset_name,
                                                                                    algorithm_name, i, new_dataset_name)
                    # xgb_path = path_src_model + '{}_models_{}/_{}{}{}{}.h5'.format(item_key, dataset_name,

                    new_xgb_name = 'xgb_d{}_l{}_e{}'.format(depth, Learning_Rate, estimator)
                    new_xgbpath = path_src_model + '{}_models_{}/_{}{}_{}_{}.h5'.format(item_key, dataset_name,
                                                                                      algorithm_name, i,
                                                                                      new_dataset_name, new_xgb_name)


                    # setName = 'ADASYN' #'ADASYN'
                    # setName = 'ADASYN'
                    print(path_model)
                    print(new_xgbpath)

                    new_setName = setName + '_{}_{}_{}'.format(depth, Learning_Rate, estimator)

                    (matrix_base, matrix_edeepxgb, keys_report,
                     base_auc_dic,base_mcc_dic,edxgb_auc_dic,edxgb_mcc_dic) = test_ensemble_xgb(path_nn_model=path_model,
                                                                                  path_xgb_model=new_xgbpath,
                                                                                  x_test=x_test, y_true=y_test,
                                                                                  labels=labels,
                                                                                  algorithm_name=algorithm_name,
                                                                                  index=i,
                                                                                  dislabel_ba=dis_label_dict,
                                                                                  dislabel_ed=dis_label_dict_ed,
                                                                                  setName=new_setName,
                                                                                  labels_matrix_dict=labels_matrix_dict,
                                                                                  base_auc=base_auc_dic,
                                                                                  base_mcc=base_mcc_dic,
                                                                                  edxgb_auc=edxgb_auc_dic,
                                                                                  edxgb_mcc=edxgb_mcc_dic,
                                                                                  tour_guide=tour_guide)


                    print('keys of report_edxgb', keys_report)
                    try:
                        print(matrix_base[tour_guide])
                    except Exception as e:
                        print(e)
                        print(matrix_base)
                        pass
                    for index, item in enumerate(keys_report):
                        if item == tour_guide:
                            i_exfil = index
                            TN = matrix_base[i_exfil][0, 0]
                            FP = matrix_base[i_exfil][0, 1]
                            FN = matrix_base[i_exfil][1, 0]
                            TP = matrix_base[i_exfil][1, 1]

                            # ----------For all classes
                            # FP = matrix_base.sum(axis=0) - np.diag(matrix_base)
                            # FN = matrix_base.sum(axis=1) - np.diag(matrix_base)
                            # TP = np.diag(matrix_base)
                            # TN = matrix_base.sum() - (FP + FN + TP)
                            tps.append(TP)
                            fps.append(FP)
                            tns.append(TN)
                            fns.append(FN)

                            TN_ = matrix_edeepxgb[i_exfil][0, 0]
                            FP_ = matrix_edeepxgb[i_exfil][0, 1]
                            FN_ = matrix_edeepxgb[i_exfil][1, 0]
                            TP_ = matrix_edeepxgb[i_exfil][1, 1]

                            # ----------For all classes
                            tps_.append(TP_)
                            fps_.append(FP_)
                            tns_.append(TN_)
                            fns_.append(FN_)
                    print('---------- tps, fps, tns, fns --------------')
                    print(tps, fps, tns, fns)
                    print('---------- tps_, fps_, tns_, fns_ --------------')
                    print(tps_, fps_, tns_, fns_)
                    print('------------ auc, mcc ---------')
                    print(edxgb_auc_dic, edxgb_mcc_dic)

                metrix_base = {
                    'TP': tps,
                    'FP': fps,
                    'TN': tns,
                    'FN': fns,
                }

                metrix_edxgb = {
                    'TP': tps_,
                    'FP': fps_,
                    'TN': tns_,
                    'FN': fns_,
                }
                df_metrix_base = pd.DataFrame.from_dict(metrix_base)
                df_metrix_edxgb = pd.DataFrame.from_dict(metrix_edxgb)

                tree_results[tree_id] = df_metrix_edxgb


                folder_save_dir = os.path.join(path_src_result, "reports", dataset_name, "tp_fp_tn_fn")
                Pathlists.folder_check(folder_save_dir)

                new_algorithem_name = algorithm_name + "_" + new_setName
                base_folder_save = folder_save_dir + "/_base_result_" + new_algorithem_name + '.csv'
                df_metrix_base.to_csv(base_folder_save, index=False, header=True)

                edxgb_folder_save = folder_save_dir + "/_edxgb_result_" + new_algorithem_name + '.csv'
                df_metrix_edxgb.to_csv(edxgb_folder_save, index=False, header=True)

                DF_base_acc = pd.DataFrame.from_dict(base_auc_dic, orient='index', columns=['Base_AUC'])
                DF_base_mcc = pd.DataFrame.from_dict(base_mcc_dic, orient='index', columns=['Base_MCC'])
                DF_edxgb_acc = pd.DataFrame.from_dict(edxgb_auc_dic, orient='index', columns=['EDXGB_AUC'])
                DF_edxgb_mcc = pd.DataFrame.from_dict(edxgb_mcc_dic, orient='index', columns=['EDXGB_MCC'])

                print(DF_base_acc)
                # exit()

                folder_save_dir_auc_mcc = os.path.join(path_src_result, "reports", dataset_name, "auc_mcc")
                Pathlists.folder_check(folder_save_dir_auc_mcc)
                folder_save_base_a = folder_save_dir_auc_mcc + "/" + flag + "_auc_base_" + str(model_name_index) + new_setName + ".csv"
                folder_save_base_m = folder_save_dir_auc_mcc + "/" + flag + "_mcc_base_" + str(model_name_index) + new_setName + ".csv"
                folder_save_edxgb_a = folder_save_dir_auc_mcc + "/" + flag + "_auc_edxgb_" + str(model_name_index) + new_setName + ".csv"
                folder_save_edxgb_m = folder_save_dir_auc_mcc + "/" + flag + "_mcc_edxgb_" + str(model_name_index) + new_setName + ".csv"

                DF_base_acc.to_csv(folder_save_base_a, index=True, header=True)
                DF_base_mcc.to_csv(folder_save_base_m, index=True, header=True)
                DF_edxgb_acc.to_csv(folder_save_edxgb_a, index=True, header=True)
                DF_edxgb_mcc.to_csv(folder_save_edxgb_m, index=True, header=True)

    print(tree_results)

    print('-------------tree_results---------------')
    print(len(tree_results))
    # max_key = max(tree_results, key=tree_results.get)
    for tree_key, tree_value in tree_results.items():
        if tree_value['TP'].all() != 0:
            print('None-zero value in dict: ', tree_key)
            print(tree_value)
        else:
            # print("No tree_value meets the requirement!")
            print(tree_key)
            print(tree_value.iloc[2:3])


if __name__ == '__main__':
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
    word = words[1]

    exfil_files = ['Original_cii', 'Original_ciii', 'Original_civ_c2', 'Original_civ_size', 'Original_cii_ciii']

    for file_name, model_name_index in data_files.items():
            algorithm_name_dnn = algorithm_names_dnn[model_name_index]
            algorithm_name_cnn = algorithm_names_cnn[model_name_index]

            print('********************** {}: {} ***************************'.format(file_name_, model_name_index))
            for dataset_name in dataname:

                # sampler = ''
                dataWeNeed_path = path_src + 'res_datasets/' + dataset_name
                if dataset_name == 'SCVIC-APT-2021':
                    print('------------------------------ < test/SCVIC-APT > -------------------------')
                    dataWeNeed_path = path_src + 'res_datasets/' + dataset_name
                    dis_label = ['Normal', 'Pivoting', 'Reconn', 'L-Move', 'Initial-Com', 'DExfil']
                    labels_matrix_dict = {'2.0': 'Normal',
                                              '3.0': 'Pivoting',
                                              '4.0': 'Reconn',
                                              '0.0': 'I-Compro',
                                              '1.0': 'L_Move',
                                              '5.0': 'D_Exfil',
                                              }
                    # For results' plt of Data Exfiltration
                    tour_guide = '5.0'
                    new_dataset_name = dataset_name + '_' + file_name

                elif dataset_name == 'Unraveled':
                    print('------------------------------ < test/Unraveled > -------------------------')
                    # For label distribution of sample sets in environments
                    labels_matrix_dict = {'4.0': 'Normal',
                                              '3.0': 'L_Move',
                                              '2.0': 'E_Foot',
                                              '1.0': 'D_Exfil',
                                              '0.0': 'C_up'}
                    tour_guide = '1.0'
                    dataWeNeed_path = path_src + 'res_datasets/' + dataset_name
                    # new_dataset_name = dataset_name + '_' + file_name+ '_' + sampler

                    new_dataset_name = dataset_name + '_' + file_name
                    # plt_title = file_name_

                # ============================================================
                plt_title = model_name_index
                for sampler in samplers:
                    for test_file in samplers:
                        path_for_data = dataWeNeed_path + "/" + word + '_final_' + file_name + "_Re_" + test_file + ".csv"
                        df = pd.read_csv(path_for_data)
                        print('------------', path_for_data)
                        print(df['Label'].value_counts())

                        # converting label 8 & 9 to 1
                        dataset = ML_model.labelChange(Data=df, dataset_name=dataset_name, file_name=file_name, model_index=model_name_index)

                        # ========== dnn-based ==============
                        start_gup = default_timer()
                        flag1 = 'dnn'
                        count1 = 5
                        main_(flag=flag1, count=count1, dataset_name=dataset_name, dataset=dataset, setName=test_file,
                                          algorithm_name=algorithm_name_dnn, new_dataset_name=new_dataset_name+ '_' + sampler, item_key=item_key,
                                  dis_label=plt_title, labels_matrix_dict=labels_matrix_dict,
                                  tour_guide=tour_guide)
                        print('With GPU:', default_timer()-start_gup)

                        # ========== cnn-based ==============
                        start_gup = default_timer()
                        flag2 = 'cnn'
                        count2 = 5
                        main_(flag=flag2, count=count2, dataset_name=dataset_name, dataset=dataset, setName=test_file,
                                  algorithm_name=algorithm_name_cnn, new_dataset_name=new_dataset_name + '_' + sampler, item_key=item_key,
                                  dis_label=plt_title, labels_matrix_dict=labels_matrix_dict,
                                  tour_guide=tour_guide)
                        print('With GPU:', default_timer() - start_gup)









