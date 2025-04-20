'''Refer: https://github.com/kahramankostas/Anomaly-Detection-in-Networks-Using-Machine-Learning'''

import numpy as np
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout,TimeDistributed
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from tensorflow.python.layers.core import Flatten
from sklearn.metrics import matthews_corrcoef, classification_report, accuracy_score
import keras
from imblearn.under_sampling import RandomUnderSampler

from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

from MLStructure.Training import ML_model
import gpumonitor
import csv
import time
import warnings
import MLStructure.PathList as Pathlists
from numba import cuda, jit

from MLStructure.Training.ML_model import load_data

label_un = 5
label_sc = 6
label_ = label_un

model_lstm = Sequential()
model_lstm.add(LSTM(units=32, return_sequences=True))
model_lstm.add(Dropout(0.3))

model_lstm.add(Flatten())
model_lstm.add(Dense(units=label_un, activation='softmax'))
model_lstm.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model_gru = Sequential()
model_gru.add(GRU(units=32, return_sequences=True))
model_lstm.add(Dropout(0.4))


model_gru.add(Flatten())
model_gru.add(Dense(units=label_un, activation='softmax'))
# model_gru.add(TimeDistributed(Dense(label_sc)))
model_gru.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

def test_seperate(test_data_files,ml_algorithm, clf,precision, recall, f1, accuracy,t_time, result ):
    for file_name_custom, name_index_custom in test_data_files.items():

        if file_name_custom == 'Original_cii' or file_name_custom == 'Original_ciii':
            file_name_custom = 'Original_cii_ciii'

        path_te_custom = dataWeNeed_path + "/" + words[
            1] + '_final_' + file_name_custom + "_Re_" + 'ROSampler' + ".csv"
        data_te_custom = load_data(dataset_name, path_te_custom)
        dataset_te = ML_model.labelChange(Data=data_te_custom, dataset_name=dataset_name,
                                          file_name=file_name_custom,
                                          model_index=name_index_custom)
        X_test_custom, y_test_custom = ML_model.exam_data_test(dataset_name, dataset_te, model_index)

        if ml_algorithm in ["LSTM", "GRU"]:
            X_test_custom = np.reshape(X_test_custom, (X_test_custom.shape[0], 1, X_test_custom.shape[1]))
            predict = clf.predict(X_test_custom)
            predict = np.argmax(predict, axis=1)
        else:
            predict = clf.predict(X_test_custom)

        # makes "classification report" and assigns the precision, f-measure, and recall values.s.
        f_1 = f1_score(y_test_custom, predict, average='macro')  # 'macro'
        pr = precision_score(y_test_custom, predict, average='macro')
        rc = recall_score(y_test_custom, predict, average='macro')
        acc = accuracy_score(y_test_custom, predict)

        precision.append(float(pr))
        recall.append(float(rc))
        f1.append(float(f_1))
        accuracy.append(float(acc))

        # The round(x) method returns the rounded value of the floating point number x.
        print('%-17s  %-15s %-15s %-15s %-15s %-15s' % (
            file_name_custom, str(round(np.mean(accuracy), 2)), str(round(np.mean(precision), 2)),
            str(round(np.mean(recall), 2)), str(round(np.mean(f1), 2)),
            str(round(np.mean(t_time), 6))))  # the result of the ten repetitions is printed on the screen.

        with open(result, "a", newline="", encoding="utf-8") as f:  # all the values found are saved in the opened file.
            wrt = csv.writer(f)
            for i in range(0, len(t_time)):
                wrt.writerow([file_name_custom, accuracy[i], precision[i], recall[i], f1[i], t_time[
                    i]])  # file name, algorithm name, precision, recall and f-measure are writed in CSV file
            # a.append(f1)

def mlgroup(result, X_train, y_train, X_test, y_test, repetition, folder_name, path_model):
    print('------------ mlgroup')
    # The machine learning algorithms dictionary.
    ml_list = {
        "Naive Bayes": GaussianNB(),
        "QDA": QDA(),
        "Random Forest": RandomForestClassifier(max_depth=10, n_estimators=20, max_features=1),
        "AdaBoost": AdaBoostClassifier(),
        "LSTM": model_lstm,
        'GRU': model_gru,
        }

    seconds = time.time()  # time stamp for all processing time

    with open(result, "w", newline="", encoding="utf-8") as f:  # a CSV file is created to save the results obtained.
        wrt = csv.writer(f)
        wrt.writerow(["File", "ML algorithm", "accuracy", "Precision", "Recall", "F1-score", "AUC", "MCC", "Time"])

    print('%-17s  %-15s %-15s %-15s %-15s %-15s %-15s %-15s' % (
    "ML algorithm", "accuracy", "Precision", "Recall", "F1-score", "AUC", "MCC", "Time"))  # print output header
    a = []

    for ml_algorithm in ml_list:  # this loop runs on the list containing the machine learning algorithm names. Operations are repeated for all the 7 algorithm
            precision = []
            recall = []
            f1 = []
            accuracy = []
            t_time = []

            for i in range(repetition):  # This loop allows cross-validation and machine learning algorithm to be repeated 10 times
                second = time.time()  # time stamp for processing time

                # training
                monitor = gpumonitor.monitor.GPUStatMonitor(delay=1)

                clf = ml_list[ml_algorithm]  # choose algorithm from ml_list dictionary
                if ml_algorithm in ["LSTM"]:
                    x_train_l = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
                    y_train_l = keras.utils.to_categorical(y_train, num_classes=label_)
                    # y_test = keras.utils.to_categorical(y_test, num_classes=label_sc)
                    clf.fit(x_train_l, y_train_l, epochs=5, batch_size=128, verbose=2)
                elif ml_algorithm in ["GRU"]:
                    x_train_g = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
                    y_train_g = keras.utils.to_categorical(y_train, num_classes=label_)
                    y_test = keras.utils.to_categorical(y_test, num_classes=label_)
                    print(x_train_g.shape)
                    clf.fit(x_train_g, y_train_g, epochs=5, batch_size=512, verbose=2)
                else:
                    clf.fit(X_train, y_train)

                monitor.stop()
                monitor.display_average_stats_per_gpu()
                t_time.append(float((time.time() - second)))

                predict = [-1]

                #test
                if ml_algorithm in ["LSTM"]:
                    print(X_test.shape)
                    X_test_l = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))
                    print(X_test_l.shape)
                    predict_l = clf.predict(X_test_l)
                    predict = np.argmax(predict_l, axis=1)
                    # y_test = np.argmax(y_test, axis=1)
                elif ml_algorithm in ["GRU"]:
                    print(X_test.shape)
                    X_test_g = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))
                    print(X_test_g.shape)
                    predict_g = clf.predict(X_test_g)
                    predict = np.argmax(predict_g, axis=1)
                    y_test = np.argmax(y_test, axis=1)
                else:
                    predict = clf.predict(X_test)

                # makes "classification report" and assigns the precision, f-measure, and recall values.s.
                f_1 = f1_score(y_test, predict, average='macro')  # 'macro'
                pr = precision_score(y_test, predict, average='macro')
                rc = recall_score(y_test, predict, average='macro')
                acc = accuracy_score(y_test, predict)
                reports = classification_report(y_test, predict)

                precision.append(float(pr))
                recall.append(float(rc))
                f1.append(float(f_1))
                accuracy.append(float(acc))
                print(reports)

            # The round(x) method returns the rounded value of the floating point number x.
            print('%-17s  %-15s %-15s %-15s %-15s %-15s' % (
            ml_algorithm, str(round(np.mean(accuracy), 2)), str(round(np.mean(precision), 2)),
            str(round(np.mean(recall), 2)), str(round(np.mean(f1), 2)),
            str(round(np.mean(t_time), 4))))  # the result of the ten repetitions is printed on the screen.

            with open(result, "a", newline="", encoding="utf-8") as f:  # all the values found are saved in the opened file.
                wrt = csv.writer(f)
                for i in range(0, len(t_time)):
                    wrt.writerow([ml_algorithm, accuracy[i], precision[i], recall[i], f1[i], t_time[
                        i]])  # file name, algorithm name, precision, recall and f-measure are writed in CSV file
            a.append(f1)

    print("mission accomplished!")
    print("Total operation time: = ", time.time() - seconds, "seconds")


def main(result, repetition, folder_name, dataset_name, path_tr, path_te, model_index, path_model):
    print('------------ main')
    # X: features, y: labels
    data_tr = load_data(dataset_name,path_tr)
    data_te = load_data(dataset_name,path_te)

    dataset_tr = ML_model.labelChange(Data=data_tr, dataset_name=dataset_name, file_name=file_name,
                                   model_index=model_index)
    dataset_te = ML_model.labelChange(Data=data_te, dataset_name=dataset_name, file_name=file_name,
                                   model_index=model_index)

    if dataset_name == 'SCVIC-APT-2021':
        # For the imbalance problem:
        y_4under = data_tr['Label']
        X_4under = data_tr.iloc[:, data_tr.columns != 'Label']
        target_num = len(data_tr[data_tr['Label'] == 5])

        # Define the undersampling strategy: specify the number of samples for label 2:normal traffic (sc)
        undersample = RandomUnderSampler(sampling_strategy={2: target_num}, random_state=42)

        # Apply the undersampling
        x_unders, y_unders = undersample.fit_resample(X_4under, y_4under)
        data_tr = x_unders.merge(y_unders, how='outer', left_index=True, right_index=True)

        print('after dump, Counter(y): ', data_tr['Label'].value_counts())


    X_train, y_train = ML_model.exam_data_test(dataset_name, dataset_tr, model_index)
    X_test, y_test = ML_model.exam_data_test(dataset_name, dataset_te, model_index)

    mlgroup(result=result, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
            repetition=repetition, folder_name=folder_name, path_model=path_model)


warnings.filterwarnings("ignore")

def SCVIC_main(result, repetition, path_tr, path_te, dataset_name, folder_name,model_index, path_model):
    print('------------------------------ < ML1/SCVIC-APT > -------------------------')
    main(result=result, repetition=repetition, dataset_name=dataset_name,
         path_tr=path_tr, path_te=path_te, folder_name=folder_name, model_index=model_index, path_model=path_model)


def Unraveled_main(result, repetition, path_tr, path_te, dataset_name, folder_name, model_index, path_model):
    print('------------------------------ < ML1/Unraveled-multi > -------------------------')
    main(result=result, repetition=repetition, dataset_name=dataset_name,
         path_tr=path_tr, path_te=path_te, folder_name=folder_name, model_index=model_index, path_model=path_model)


if __name__ == '__main__':
    # ---------------------<main>-----------------------------
    group_results = Pathlists.folder_check(Pathlists.GROUP_result_experi2_DIR)
    print('group_results', group_results)

    path_src = Pathlists.folder_check(Pathlists.ML_DATA_DIR)
    print('path_src', path_src)
    res_data_dir = Pathlists.folder_check(Pathlists.RES_DATA_DIR)
    print('res_data_dir', res_data_dir)


    repetition = 1
    dataname = [
        'Unraveled',
        'SCVIC-APT-2021'
    ]

    #Algorithms for imbalance handling
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
    # word = words[0]

    for file_name, model_index in data_files.items():
        print('********************** {}: {} ***************************'.format(file_name, model_index))
        print('-----------start-------------')
        for dataset_name in dataname:
            dataWeNeed_path = path_src + 'res_datasets/' + dataset_name
            for sampler in samplers:
                    print('file_name = {}, model_index = {}'.format(file_name, model_index))

                    path_tr = dataWeNeed_path + "/" + words[0] + '_final_' + file_name + "_Re_" +sampler+ ".csv"
                    path_te = dataWeNeed_path + "/" + words[1] + '_final_' + file_name + "_Re_" + sampler + ".csv"
                    print('training set path = ', path_tr)

                    new_dataset_name = dataset_name + '_' + file_name + str(model_index) +'_'+ sampler
                    result_un = group_results + new_dataset_name +"_Group_results_weight_un.csv"
                    result_sc = group_results + new_dataset_name + "_Group_results_weight_sc.csv"
                    path_model = group_results + new_dataset_name
                    print('result_un = ', result_un)
                    foldername = group_results + new_dataset_name + str(model_index) +'group_pdf_'
                    print('foldername = ', foldername)

                    SCVIC_main(result=result_sc, repetition=repetition, dataset_name=dataset_name, path_tr=path_tr, path_te=path_te,
                         folder_name=foldername, model_index=model_index, path_model=path_model)

                    Unraveled_main(result=result_un, repetition=repetition, dataset_name=dataset_name, path_tr=path_tr, path_te=path_te,
                         folder_name=foldername, model_index=model_index, path_model=path_model)
