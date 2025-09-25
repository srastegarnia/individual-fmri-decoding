import numpy as np
import pandas as pd
import glob
import os
import sys
import warnings
import math
import csv
from time import time
import logging
from pathlib import Path

import matplotlib.pyplot as plt
from nilearn.maskers import NiftiLabelsMasker, NiftiMasker, NiftiMapsMasker
from nilearn.plotting import plot_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut, train_test_split
from sklearn.model_selection import RepeatedStratifiedKFold, KFold, cross_val_score, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB, GaussianNB, ComplementNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
# from keras.models import Sequential
# from keras.layers import Dense
from termcolor import colored

# sys.path.append(os.path.join(".."))
# works both as package and script:
try:
    from . import visualization
except ImportError as e:
    if getattr(e, "name", "") == "visualization":
        import visualization  # fallback if run as a script
    else:
        raise

# ---------- Paths, logging, constants ----------
logging.basicConfig(
    level=os.getenv("LOGLEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("benchmark_decoders")

REPO_ROOT = Path(__file__).resolve().parent.parent           # .../individual-fmri-decoding
DATA_DIR = REPO_ROOT / "data"                                # .../individual-fmri-decoding/data
OUTPUTS_DIR = REPO_ROOT / "outputs"                          # .../individual-fmri-decoding/outputs
RESULTS_ROOT = OUTPUTS_DIR / "results"                       # .../individual-fmri-decoding/outputs/results

SEED = 0
TEST_SIZE = 0.2
np.random.seed(SEED)

"""
Script for running benchmark models.
This script inputs are outputs files of hcptrt_data_prep.py.
"""


def _generate_all_modality_files(subject, modalities, region_approach,
                                  HRFlag_process, proc_data_path, resolution):

    all_modality_concat_bold = []
    all_modality_concat_labels = []
    parcels_no = []

    for modality in modalities:

        final_bold_outpath = glob.glob(
            proc_data_path + 'processed_data/proc_fMRI/{}/{}/{}/{}*{}*{}*.npy'.format(
                region_approach, resolution, subject, subject, modality, HRFlag_process))

        final_labels_outpath = glob.glob(
            proc_data_path + 'processed_data/proc_events/{}/{}/{}/{}*{}*{}*.csv'.format(
                region_approach, resolution, subject, subject, modality, HRFlag_process))

        for b_outpath in final_bold_outpath:
            bold_file = np.load(b_outpath)
            all_modality_concat_bold.append(bold_file)

        for l_outpath in final_labels_outpath:
            labels_file = pd.read_csv(l_outpath, sep='\t', encoding="utf8", header=None)
            lable_arr = np.array(labels_file[0], dtype=object)
            labels_file = lable_arr
            all_modality_concat_labels.append(labels_file)

    flat_bold = [val for sublist in all_modality_concat_bold for val in sublist]
    all_modality_concat_bold = flat_bold

    flat_labels = [item for sublist in all_modality_concat_labels for item in sublist]
    all_modality_concat_labels = flat_labels

    return (all_modality_concat_bold, all_modality_concat_labels)


def _grid_svm_decoder(all_modality_concat_bold, all_modality_concat_labels,
                      subject, decoder, region_approach, HRFlag_process,
                      results_outpath, cm_results_outpath, resolution):
    """
    Support Vector Machine classifier with GridSearchCV.
    """
    title = '{} Support Vector Machine using {}{}, {} HRFlag'.format(subject, region_approach, resolution, HRFlag_process)
    output_file_name = '{}_SVM_{}{}_{}_HRFlag'.format(subject, region_approach, resolution, HRFlag_process)

    X = all_modality_concat_bold
    y = all_modality_concat_labels

    categories = np.unique(y)
    unique_conditions, order = np.unique(categories, return_index=True)
    unique_conditions = unique_conditions[np.argsort(order)]

    labelencoder_y = LabelEncoder()
    y = labelencoder_y.fit_transform(y).ravel()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=SEED)

    param_grid = {'C': [10], 'gamma': [0.001], 'kernel': ['rbf']}
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=SEED)

    t0 = time()
    grid = GridSearchCV(SVC(random_state=SEED), param_grid, refit=True, n_jobs=-1, cv=cv, verbose=3)
    grid.fit(X_train, y_train)

    log.info(f"Best params: {grid.best_params_}")

    grid_predictions = grid.predict(X_test)
    print(classification_report(y_test, grid_predictions))
    log.info(f"SVM decoding time: {round(time()-t0, 3)} s")

    cm_svm = confusion_matrix(y_test, grid_predictions)
    model_cm = np.round(cm_svm.astype('float') / cm_svm.sum(axis=1)[:, np.newaxis], 2)

    visualization.conf_matrix(model_cm, unique_conditions, title, cm_results_outpath,
                              output_file_name, decoder, subject, region_approach,
                              resolution, HRFlag_process)


def _svm_decoder(all_modality_concat_bold, all_modality_concat_labels,
                 subject, decoder, region_approach, HRFlag_process,
                 results_outpath, cm_results_outpath, resolution):
    """
    Support Vector Machine classifier.
    """
    title = '{} Support Vector Machine using {}{}, {} HRFlag'.format(subject, region_approach, resolution, HRFlag_process)
    output_file_name = '{}_SVM_{}{}_{}_HRFlag'.format(subject, region_approach, resolution, HRFlag_process)

    X = all_modality_concat_bold
    y = all_modality_concat_labels

    categories = np.unique(y)
    unique_conditions, order = np.unique(categories, return_index=True)
    unique_conditions = unique_conditions[np.argsort(order)]

    labelencoder_y = LabelEncoder()
    y = labelencoder_y.fit_transform(y).ravel()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=SEED)

    model_svm = SVC(kernel='rbf', random_state=SEED, C=1)
    model_svm.fit(X_train, y_train)

    print('Accuracy of the prediction on the test set:')
    y_test_pred = model_svm.predict(X_test)
    print(classification_report(y_test, y_test_pred))

    cm_svm = confusion_matrix(y_test, y_test_pred)
    model_cm = cm_svm.astype('float') / cm_svm.sum(axis=1)[:, np.newaxis]

    visualization.conf_matrix(model_cm, unique_conditions, title, cm_results_outpath,
                              output_file_name, decoder, subject, region_approach,
                              resolution, HRFlag_process)


def _grid_mlp_decoder(all_modality_concat_bold, all_modality_concat_labels,
                      subject, decoder, region_approach, HRFlag_process,
                      results_outpath, cm_results_outpath, resolution):
    """
    Scikit-Learn MLP with grid search.
    """
    title = '{} Scikit-Learn’s MLPClassifier using {}{}, {} HRFlag'.format(subject, region_approach, resolution, HRFlag_process)
    output_file_name = '{}_skl_mlp_{}{}_{}_HRFlag'.format(subject, region_approach, resolution, HRFlag_process)

    X = all_modality_concat_bold
    y = all_modality_concat_labels

    categories = np.unique(y)
    num_cond = len(set(categories))
    unique_conditions, order = np.unique(categories, return_index=True)
    unique_conditions = unique_conditions[np.argsort(order)]

    labelencoder_y = LabelEncoder()
    y = labelencoder_y.fit_transform(y)
    y = np.reshape(y, (len(all_modality_concat_labels), 1))

    enc = OneHotEncoder(handle_unknown='ignore')
    y_onehot = enc.fit_transform(np.array(y).reshape(-1, 1))
    y = pd.DataFrame(y_onehot.toarray())

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=SEED)

    param_grid = {
        'hidden_layer_sizes': [(int(resolution/(math.pow(2,1))), int(resolution/(math.pow(2,2))))],
        'activation': ['relu'],
        'solver': ['adam'],
        'alpha': [0.05],
        'learning_rate': ['constant']
    }
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=SEED)
    grid = GridSearchCV(MLPClassifier(random_state=SEED), param_grid, refit=True, n_jobs=-1, cv=cv, verbose=3)

    grid.fit(X_train, y_train)
    log.info(f"Best params: {grid.best_params_}")
    log.info(f"Best estimator: {grid.best_estimator_}")

    for mean, std, params in zip(grid.cv_results_['mean_test_score'],
                                 grid.cv_results_['std_test_score'],
                                 grid.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

    grid_predictions = grid.predict(X_test)
    print('Results on the test set:')
    print(classification_report(y_test, grid_predictions))

    cm_ann = confusion_matrix(y_test.values.argmax(axis=1), grid_predictions.argmax(axis=1))
    model_cm = cm_ann.astype('float') / cm_ann.sum(axis=1)[:, np.newaxis]

    visualization.conf_matrix(model_cm, unique_conditions, title, cm_results_outpath,
                              output_file_name, decoder, subject, region_approach,
                              resolution, HRFlag_process)


def _mlp_decoder(all_modality_concat_bold, all_modality_concat_labels,
                 subject, decoder, region_approach, HRFlag_process,
                 results_outpath, cm_results_outpath, resolution):
    """
    Keras MLP, two dense layers.
    """
    title = '{} Multi Layer Perceptron Neural Networks using {}{}, {} HRFlag'.format(subject, region_approach, resolution, HRFlag_process)
    output_file_name = '{}_mlp_{}{}_{}_HRFlag'.format(subject, region_approach, resolution, HRFlag_process)

    X = all_modality_concat_bold
    y = all_modality_concat_labels

    categories = np.unique(y)
    unique_conditions, order = np.unique(categories, return_index=True)
    num_cond = len(set(categories))
    unique_conditions = unique_conditions[np.argsort(order)]

    labelencoder_y = LabelEncoder()
    y = labelencoder_y.fit_transform(y)
    y = np.reshape(y, (len(all_modality_concat_labels), 1))

    enc = OneHotEncoder(handle_unknown='ignore')
    y_onehot = enc.fit_transform(np.array(y).reshape(-1, 1))
    y = pd.DataFrame(y_onehot.toarray())

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=SEED)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    warnings.filterwarnings('ignore')

    import importlib, warnings

    # -------- Optional dependency gate: TF/Keras only if installed --------
    if importlib.util.find_spec("tensorflow") is not None:
        from tensorflow.keras import Sequential
        from tensorflow.keras.layers import Dense
    elif importlib.util.find_spec("keras") is not None:
        from keras.models import Sequential
        from keras.layers import Dense
    else:
        warnings.warn("[mlp_decoder] Keras backend not available; skipping Keras MLP. "
                  "Install tensorflow or keras to enable.", RuntimeWarning)
        return
    # ----------------------------------------------------------------------

    t0 = time()
    model_mlp = Sequential()
    model_mlp.add(Dense(int(resolution/(math.pow(2,1))), input_dim=resolution,
                        kernel_initializer='uniform', activation='relu', use_bias=True,
                        bias_initializer='zeros'))
    model_mlp.add(Dense(int(resolution/(math.pow(2,2))), kernel_initializer='uniform',
                        activation='relu', use_bias=True, bias_initializer='zeros'))
    model_mlp.add(Dense(num_cond, activation='softmax'))

    model_mlp.compile(optimizer='adamax', loss='categorical_crossentropy', metrics=['accuracy'])
    history = model_mlp.fit(X_train, y_train, batch_size=10, epochs=10, validation_split=0.1)

    visualization.classifier_history(history, title, results_outpath, output_file_name)

    y_test_pred = model_mlp.predict(X_test)
    print(classification_report(y_test.values.argmax(axis=1), y_test_pred.argmax(axis=1)))
    print('mean accuracy score:', np.round(accuracy_score(y_test.values.argmax(axis=1),
                                                          y_test_pred.argmax(axis=1),
                                                          normalize=True, sample_weight=None), 2))
    log.info(f"MLP decoding time: {round(time()-t0, 3)} s")

    cm_ann = confusion_matrix(y_test.values.argmax(axis=1), y_test_pred.argmax(axis=1))
    model_cm = np.round(cm_ann.astype('float') / cm_ann.sum(axis=1)[:, np.newaxis], 2)

    visualization.conf_matrix(model_cm, unique_conditions, title, cm_results_outpath,
                              output_file_name, decoder, subject, region_approach,
                              resolution, HRFlag_process)


def _grid_knn_decoder(all_modality_concat_bold, all_modality_concat_labels,
                      subject, decoder, region_approach, HRFlag_process,
                      results_outpath, cm_results_outpath, resolution):
    """
    k-nearest neighbors classifier with GridSearchCV.
    """
    title = '{} KNN using {}{}, {} HRFlag'.format(subject, region_approach, resolution, HRFlag_process)
    output_file_name = '{}_KNN_{}{}_{}_HRFlag'.format(subject, region_approach, resolution, HRFlag_process)

    X = all_modality_concat_bold
    y = all_modality_concat_labels

    categories = np.unique(y)
    unique_conditions, order = np.unique(categories, return_index=True)
    unique_conditions = unique_conditions[np.argsort(order)]

    labelencoder_y = LabelEncoder()
    y = labelencoder_y.fit_transform(y).ravel()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=SEED)

    param_grid = {'n_neighbors': [4], 'leaf_size': [1], 'p': [1], 'weights': ['distance'],
                  'metric': ['minkowski'], 'algorithm': ['auto']}
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=SEED)

    t0 = time()
    grid = GridSearchCV(KNeighborsClassifier(), param_grid, refit=True, n_jobs=-1, cv=cv, verbose=3)
    grid.fit(X_train, y_train)

    log.info(f"Best params: {grid.best_params_}")

    grid_predictions = grid.predict(X_test)
    print(classification_report(y_test, grid_predictions))
    log.info(f"KNN decoding time: {round(time()-t0, 3)} s")

    cm_knn = confusion_matrix(y_test, grid_predictions)
    model_cm = np.round(cm_knn.astype('float') / cm_knn.sum(axis=1)[:, np.newaxis], 2)

    visualization.conf_matrix(model_cm, unique_conditions, title, cm_results_outpath,
                              output_file_name, decoder, subject, region_approach,
                              resolution, HRFlag_process)


def _knn_decoder(all_modality_concat_bold, all_modality_concat_labels,
                 subject, decoder, region_approach, HRFlag_process,
                 results_outpath, cm_results_outpath, resolution):
    """
    k-nearest neighbors classifier. (k=8)
    """
    title = '{} KNN using {}{}, {} HRFlag'.format(subject, region_approach, resolution, HRFlag_process)
    output_file_name = '{}_KNN_{}{}_{}_HRFlag'.format(subject, region_approach, resolution, HRFlag_process)

    X = all_modality_concat_bold
    y = all_modality_concat_labels

    categories = np.unique(y)
    unique_conditions, order = np.unique(categories, return_index=True)
    unique_conditions = unique_conditions[np.argsort(order)]

    labelencoder_y = LabelEncoder()
    y = labelencoder_y.fit_transform(y).ravel()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=SEED)

    cv = KFold(n_splits=10, random_state=SEED, shuffle=True)
    model_knn = KNeighborsClassifier(n_neighbors=8, algorithm='kd_tree')

    scores = cross_val_score(model_knn, X_test, y_test, scoring='accuracy', cv=cv, n_jobs=-1)
    y_pred = cross_val_predict(model_knn, X_test, y_test, cv=cv)
    report = classification_report(y_test, y_pred)

    print(report)
    print(scores)
    print('mean accuracy:%.4f' % np.mean(scores))

    cm_knn = confusion_matrix(y_test, y_pred)  # fixed undefined var
    model_cm = np.round(cm_knn.astype('float') / cm_knn.sum(axis=1)[:, np.newaxis], 2)

    visualization.conf_matrix(model_cm, unique_conditions, title, cm_results_outpath,
                              output_file_name, decoder, subject, region_approach,
                              resolution, HRFlag_process)


def _grid_random_forest_decoder(all_modality_concat_bold, all_modality_concat_labels,
                                subject, decoder, region_approach, HRFlag_process,
                                results_outpath, cm_results_outpath, resolution):
    """
    Random Forest classifier with GridSearchCV.
    """
    title = '{} Random Forest using {}{}, {} HRFlag'.format(subject, region_approach, resolution, HRFlag_process)
    output_file_name = '{}_RandomForest_{}{}_{}_HRFlag'.format(subject, region_approach, resolution, HRFlag_process)

    X = all_modality_concat_bold
    y = all_modality_concat_labels

    categories = np.unique(y)
    unique_conditions, order = np.unique(categories, return_index=True)
    unique_conditions = unique_conditions[np.argsort(order)]

    labelencoder_y = LabelEncoder()
    y = labelencoder_y.fit_transform(y).ravel()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=SEED)

    param_grid = {
        'n_estimators': [200],
        'max_features': [21],
        'max_depth': [20],
        'min_samples_split': [2],
        'min_samples_leaf': [1],
        'bootstrap': [False]
    }
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=SEED)

    t0 = time()
    grid = GridSearchCV(RandomForestClassifier(), param_grid, refit=True, n_jobs=-1, cv=cv, verbose=3)
    grid.fit(X_train, y_train)

    log.info(f"Best params: {grid.best_params_}")

    grid_predictions = grid.predict(X_test)
    print(classification_report(y_test, grid_predictions))
    log.info(f"RandomForest decoding time: {round(time()-t0, 3)} s")

    cm_rf = confusion_matrix(y_test, grid_predictions)
    model_cm = np.round(cm_rf.astype('float') / cm_rf.sum(axis=1)[:, np.newaxis], 2)

    visualization.conf_matrix(model_cm, unique_conditions, title, cm_results_outpath,
                              output_file_name, decoder, subject, region_approach,
                              resolution, HRFlag_process)


def _random_forest_decoder(all_modality_concat_bold, all_modality_concat_labels,
                           subject, decoder, region_approach, HRFlag_process,
                           results_outpath, cm_results_outpath, resolution):

    title = '{} Random Forest using {}{}, {} HRFlag'.format(subject, region_approach, resolution, HRFlag_process)
    output_file_name = '{}_RF_{}_{}_HRFlag'.format(subject, region_approach, HRFlag_process)

    X = all_modality_concat_bold
    y = all_modality_concat_labels

    categories = np.unique(y)
    unique_conditions, order = np.unique(categories, return_index=True)
    unique_conditions = unique_conditions[np.argsort(order)]

    labelencoder_y = LabelEncoder()
    y = labelencoder_y.fit_transform(y).ravel()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=SEED)

    cv = KFold(n_splits=10, random_state=SEED, shuffle=True)
    model_rfc = RandomForestClassifier(max_depth=2, random_state=SEED)

    scores = cross_val_score(model_rfc, X_test, y_test, scoring='accuracy', cv=cv, n_jobs=-1)
    y_pred = cross_val_predict(model_rfc, X_test, y_test, cv=cv)
    report = classification_report(y_test, y_pred)

    print(report)
    print(scores)
    print('mean accuracy:%.4f' % np.mean(scores))

    cm_rfc = confusion_matrix(y_test, y_pred)
    model_cm = cm_rfc.astype('float') / cm_rfc.sum(axis=1)[:, np.newaxis]

    visualization.conf_matrix(model_cm, unique_conditions, title, cm_results_outpath,
                              output_file_name, decoder, subject, region_approach,
                              resolution, HRFlag_process)


def _grid_logistic_regression_decoder(all_modality_concat_bold, all_modality_concat_labels,
                                      subject, decoder, region_approach, HRFlag_process,
                                      results_outpath, cm_results_outpath, resolution):
    """
    Logistic Regression classifier with GridSearchCV.
    """
    title = '{} Logistic Regression using {}{}, {} HRFlag'.format(subject, region_approach, resolution, HRFlag_process)
    output_file_name = '{}_LogisticRegression_{}{}_{}_HRFlag'.format(subject, region_approach, resolution, HRFlag_process)

    X = all_modality_concat_bold
    y = all_modality_concat_labels

    categories = np.unique(y)
    unique_conditions, order = np.unique(categories, return_index=True)
    unique_conditions = unique_conditions[np.argsort(order)]

    labelencoder_y = LabelEncoder()
    y = labelencoder_y.fit_transform(y).ravel()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=SEED)

    param_grid = {'solver': ['liblinear'],
                  'penalty': ['l1'],
                  'C': [0.1],
                  'max_iter': [20],
                  'class_weight': [None]}
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=SEED)

    t0 = time()
    grid = GridSearchCV(LogisticRegression(), param_grid, refit=True, n_jobs=-1, cv=cv, verbose=3)
    grid.fit(X_train, y_train)

    log.info(f"Best params: {grid.best_params_}")

    grid_predictions = grid.predict(X_test)
    print(classification_report(y_test, grid_predictions))
    log.info(f"Logistic regression decoding time: {round(time()-t0, 3)} s")

    cm_lr = confusion_matrix(y_test, grid_predictions)
    model_cm = np.round(cm_lr.astype('float') / cm_lr.sum(axis=1)[:, np.newaxis], 2)

    visualization.conf_matrix(model_cm, unique_conditions, title, cm_results_outpath,
                              output_file_name, decoder, subject, region_approach,
                              resolution, HRFlag_process)


def _grid_ridge_decoder(all_modality_concat_bold, all_modality_concat_labels,
                        subject, decoder, region_approach, HRFlag_process,
                        results_outpath, cm_results_outpath, resolution):
    """
    Ridge Regression classifier with GridSearchCV.
    """
    title = '{} Ridge Regression using {}{}, {} HRFlag'.format(subject, region_approach, resolution, HRFlag_process)
    output_file_name = '{}_RidgeRegression_{}{}_{}_HRFlag'.format(subject, region_approach, resolution, HRFlag_process)

    X = all_modality_concat_bold
    y = all_modality_concat_labels

    categories = np.unique(y)
    unique_conditions, order = np.unique(categories, return_index=True)
    unique_conditions = unique_conditions[np.argsort(order)]

    labelencoder_y = LabelEncoder()
    y = labelencoder_y.fit_transform(y).ravel()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=SEED)

    # removed deprecated 'normalize' parameter
    param_grid = {'alpha': [0.1],
                  'solver': ['lsqr']}
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=SEED)

    t0 = time()
    grid = GridSearchCV(RidgeClassifier(random_state=SEED), param_grid, refit=True, n_jobs=-1, cv=cv, verbose=3)
    grid.fit(X_train, y_train)

    log.info(f"Best params: {grid.best_params_}")

    grid_predictions = grid.predict(X_test)
    print(classification_report(y_test, grid_predictions))
    log.info(f"Ridge decoding time: {round(time()-t0, 3)} s")

    cm_ridge = confusion_matrix(y_test, grid_predictions)
    model_cm = np.round(cm_ridge.astype('float') / cm_ridge.sum(axis=1)[:, np.newaxis], 2)

    visualization.conf_matrix(model_cm, unique_conditions, title, cm_results_outpath,
                              output_file_name, decoder, subject, region_approach,
                              resolution, HRFlag_process)


def _grid_bagging_decoder(all_modality_concat_bold, all_modality_concat_labels,
                          subject, decoder, region_approach, HRFlag_process,
                          results_outpath, cm_results_outpath, resolution):
    """
    Bagged Decision Trees (Bagging) classifier with GridSearchCV.
    """
    title = '{} Bagging using {}{}, {} HRFlag'.format(subject, region_approach, resolution, HRFlag_process)
    output_file_name = '{}_Bagging_{}{}_{}_HRFlag'.format(subject, region_approach, resolution, HRFlag_process)

    X = all_modality_concat_bold
    y = all_modality_concat_labels

    categories = np.unique(y)
    unique_conditions, order = np.unique(categories, return_index=True)
    unique_conditions = unique_conditions[np.argsort(order)]

    labelencoder_y = LabelEncoder()
    y = labelencoder_y.fit_transform(y).ravel()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=SEED)

    param_grid = {'n_estimators': [1000],
                  'bootstrap': [False],
                  'max_features': [100],
                  'bootstrap_features': [False],
                  'oob_score': [False],
                  'warm_start': [True]}
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=SEED)

    t0 = time()
    grid = GridSearchCV(BaggingClassifier(random_state=SEED), param_grid, refit=True, n_jobs=-1, cv=cv, verbose=3)
    grid.fit(X_train, y_train)

    log.info(f"Best params: {grid.best_params_}")

    grid_predictions = grid.predict(X_test)
    print(classification_report(y_test, grid_predictions))
    log.info(f"Bagging decoding time: {round(time()-t0, 3)} s")

    cm_bagging = confusion_matrix(y_test, grid_predictions)
    model_cm = np.round(cm_bagging.astype('float') / cm_bagging.sum(axis=1)[:, np.newaxis], 2)

    visualization.conf_matrix(model_cm, unique_conditions, title, cm_results_outpath,
                              output_file_name, decoder, subject, region_approach,
                              resolution, HRFlag_process)


def _grid_gaussian_nb_decoder(all_modality_concat_bold, all_modality_concat_labels,
                               subject, decoder, region_approach, HRFlag_process,
                               results_outpath, cm_results_outpath, resolution):
    """
    Gaussian Naive Bayes classifier with GridSearchCV.
    """
    title = '{} Gaussian Naive Bayes using {}{}, {} HRFlag'.format(subject, region_approach, resolution, HRFlag_process)
    output_file_name = '{}_GaussianNB_{}{}_{}_HRFlag'.format(subject, region_approach, resolution, HRFlag_process)

    X = all_modality_concat_bold
    y = all_modality_concat_labels

    categories = np.unique(y)
    unique_conditions, order = np.unique(categories, return_index=True)
    unique_conditions = unique_conditions[np.argsort(order)]

    labelencoder_y = LabelEncoder()
    y = labelencoder_y.fit_transform(y).ravel()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=SEED)

    param_grid = {'var_smoothing': [0.0001]}
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=SEED)

    t0 = time()
    grid = GridSearchCV(GaussianNB(), param_grid, refit=True, n_jobs=-1, cv=cv, verbose=3)
    grid.fit(X_train, y_train)

    log.info(f"Best params: {grid.best_params_}")

    grid_predictions = grid.predict(X_test)
    print(classification_report(y_test, grid_predictions))
    log.info(f"GaussianNB decoding time: {round(time()-t0, 3)} s")

    cm_nb = confusion_matrix(y_test, grid_predictions)
    model_cm = np.round(cm_nb.astype('float') / cm_nb.sum(axis=1)[:, np.newaxis], 2)

    visualization.conf_matrix(model_cm, unique_conditions, title, cm_results_outpath,
                              output_file_name, decoder, subject, region_approach,
                              resolution, HRFlag_process)


def postproc_benchmark_decoder(subjects, modalities, decoders, region_approach,
                               HRFlag_process, resolution):

    home_dir = str(REPO_ROOT) + "/"
    proc_data_path = str(DATA_DIR) + "/"

    for subject in subjects:

        results_outpath = str(RESULTS_ROOT / f"{region_approach}" / f"{resolution}" / f"{subject}" / f"{HRFlag_process}") + "/"
        cm_results_outpath = results_outpath + 'cm_results/'

        # remove previous content
        if os.path.exists(cm_results_outpath):
            files = glob.glob(os.path.join(cm_results_outpath, "*"))
            for f in files:
                os.remove(f)

        if not os.path.exists(results_outpath):
            os.makedirs(results_outpath)

        if not os.path.exists(cm_results_outpath):
            os.makedirs(cm_results_outpath)

        # create a .csv file of decoders results summary
        header = ['subject','decoder','region_approach', 'resolution', 'HRFlag_process',
                  'body0b', 'body2b', 'face0b', 'face2b', 'fear', 'footL', 'footR', 'handL',
                  'handR', 'match', 'math', 'mental', 'place0b', 'place2b', 'random',
                  'relational', 'shape', 'story', 'tongue', 'tool0b', 'tool2b']

        pd.DataFrame().to_csv(os.path.join(cm_results_outpath, 'results_summary.csv'), index=False)
        with open(os.path.join(cm_results_outpath, 'results_summary.csv'), 'w', encoding='UTF8') as rslt_smry:
            writer = csv.writer(rslt_smry)
            writer.writerow(header)

        results_summary_file = os.path.join(cm_results_outpath, 'results_summary.csv')

        log.info(f"{subject} | {region_approach}{resolution} | HRFlag={HRFlag_process}")

        all_modality_concat_bold, all_modality_concat_labels = _generate_all_modality_files(
            subject, modalities, region_approach, HRFlag_process, proc_data_path, resolution)

        # number of parcels (for soft parcellation like dypac)
        df_path = proc_data_path + 'medial_data/fMRI2/{}/{}/{}/{}_wm_fMRI2.npy'.format(
            region_approach, resolution, subject, subject)
        df = np.load(df_path)
        parcel_no = int(len(df[0][:][1]))

        print('all_modality_concat_bold shape', np.shape(all_modality_concat_bold))
        print('all_modality_concat_labels shape', np.shape(all_modality_concat_labels), '\n')

        for decoder in decoders:

            if decoder == 'svm_nogrid':
                log.info('Support Vector Machine classifier')
                _svm_decoder(all_modality_concat_bold=all_modality_concat_bold,
                             all_modality_concat_labels=all_modality_concat_labels,
                             subject=subject, decoder=decoder,
                             region_approach=region_approach,
                             HRFlag_process=HRFlag_process,
                             results_outpath=results_outpath,
                             cm_results_outpath=cm_results_outpath,
                             resolution=resolution)

            elif decoder == 'skl_mlp':
                log.info('MLPClassifier with grid search')
                _grid_mlp_decoder(all_modality_concat_bold=all_modality_concat_bold,
                                  all_modality_concat_labels=all_modality_concat_labels,
                                  subject=subject, decoder=decoder,
                                  region_approach=region_approach,
                                  HRFlag_process=HRFlag_process,
                                  results_outpath=results_outpath,
                                  cm_results_outpath=cm_results_outpath,
                                  resolution=resolution)

            elif decoder == 'knn_nogrid':
                log.info('K-Nearest Neighbours classifier')
                _knn_decoder(all_modality_concat_bold=all_modality_concat_bold,
                             all_modality_concat_labels=all_modality_concat_labels,
                             subject=subject, decoder=decoder,
                             region_approach=region_approach,
                             HRFlag_process=HRFlag_process,
                             results_outpath=results_outpath,
                             cm_results_outpath=cm_results_outpath,
                             resolution=resolution)

            elif decoder == 'random_forest_nogrid':
                log.info('Random Forest classifier')
                _random_forest_decoder(all_modality_concat_bold=all_modality_concat_bold,
                                       all_modality_concat_labels=all_modality_concat_labels,
                                       subject=subject, decoder=decoder,
                                       region_approach=region_approach,
                                       HRFlag_process=HRFlag_process,
                                       results_outpath=results_outpath,
                                       cm_results_outpath=cm_results_outpath,
                                       resolution=resolution)

            elif decoder == 'svm':
                log.info('SVM with grid search')
                _grid_svm_decoder(all_modality_concat_bold=all_modality_concat_bold,
                                  all_modality_concat_labels=all_modality_concat_labels,
                                  subject=subject, decoder=decoder,
                                  region_approach=region_approach,
                                  HRFlag_process=HRFlag_process,
                                  results_outpath=results_outpath,
                                  cm_results_outpath=cm_results_outpath,
                                  resolution=resolution)

            elif decoder == 'mlp':
                log.info('Keras MLP (two dense layers)')
                _mlp_decoder(all_modality_concat_bold=all_modality_concat_bold,
                             all_modality_concat_labels=all_modality_concat_labels,
                             subject=subject, decoder=decoder,
                             region_approach=region_approach,
                             HRFlag_process=HRFlag_process,
                             results_outpath=results_outpath,
                             cm_results_outpath=cm_results_outpath,
                             resolution=resolution)

            elif decoder == 'knn':
                log.info('KNN with grid search')
                _grid_knn_decoder(all_modality_concat_bold=all_modality_concat_bold,
                                  all_modality_concat_labels=all_modality_concat_labels,
                                  subject=subject, decoder=decoder,
                                  region_approach=region_approach,
                                  HRFlag_process=HRFlag_process,
                                  results_outpath=results_outpath,
                                  cm_results_outpath=cm_results_outpath,
                                  resolution=resolution)

            elif decoder == 'random_forest':
                log.info('Random Forest with grid search')
                _grid_random_forest_decoder(all_modality_concat_bold=all_modality_concat_bold,
                                            all_modality_concat_labels=all_modality_concat_labels,
                                            subject=subject, decoder=decoder,
                                            region_approach=region_approach,
                                            HRFlag_process=HRFlag_process,
                                            results_outpath=results_outpath,
                                            cm_results_outpath=cm_results_outpath,
                                            resolution=resolution)

            elif decoder == 'logistic_regression':
                log.info('Logistic Regression with grid search')
                _grid_logistic_regression_decoder(all_modality_concat_bold=all_modality_concat_bold,
                                                  all_modality_concat_labels=all_modality_concat_labels,
                                                  subject=subject, decoder=decoder,
                                                  region_approach=region_approach,
                                                  HRFlag_process=HRFlag_process,
                                                  results_outpath=results_outpath,
                                                  cm_results_outpath=cm_results_outpath,
                                                  resolution=resolution)

            elif decoder == 'ridge':
                log.info('Ridge Regression with grid search')
                _grid_ridge_decoder(all_modality_concat_bold=all_modality_concat_bold,
                                    all_modality_concat_labels=all_modality_concat_labels,
                                    subject=subject, decoder=decoder,
                                    region_approach=region_approach,
                                    HRFlag_process=HRFlag_process,
                                    results_outpath=results_outpath,
                                    cm_results_outpath=cm_results_outpath,
                                    resolution=resolution)

            elif decoder == 'bagging':
                log.info('Bagged Decision Trees with grid search')
                _grid_bagging_decoder(all_modality_concat_bold=all_modality_concat_bold,
                                      all_modality_concat_labels=all_modality_concat_labels,
                                      subject=subject, decoder=decoder,
                                      region_approach=region_approach,
                                      HRFlag_process=HRFlag_process,
                                      results_outpath=results_outpath,
                                      cm_results_outpath=cm_results_outpath,
                                      resolution=resolution)

            elif decoder == 'gaussian_nb':
                log.info('Gaussian Naive Bayes with grid search')
                _grid_gaussian_nb_decoder(all_modality_concat_bold=all_modality_concat_bold,
                                           all_modality_concat_labels=all_modality_concat_labels,
                                           subject=subject, decoder=decoder,
                                           region_approach=region_approach,
                                           HRFlag_process=HRFlag_process,
                                           results_outpath=results_outpath,
                                           cm_results_outpath=cm_results_outpath,
                                           resolution=resolution)

            else:
                print('The model is not defined')

                      
