import numpy as np
import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler
from multiprocessing import set_start_method, Pool
from hardware_language_library_extractor.logger import Logger

logger = Logger('allenai')
logger = logger.logger

# base_path = '/mnt/c/Users/kkkoo/OneDrive/Documents/VirginiaTech/DACResearch/data/outputjson/outputjson
# /extracted_sentences/'
base_path = '/home/group/cset/extracted_sentences/'
file_names = [['spacyhardware_sub_sent_embeddings.txt', 'spacyhardware_prev_sent_embeddings.txt'],
              ['spacylibrary_sub_sent_embeddings.txt', 'spacylibrary_prev_sent_embeddings.txt'],
              ['spacylanguage_sub_sent_embeddings.txt', 'spacylanguage_prev_sent_embeddings.txt']]
model_filenames = ['finalized_hardware_model.sav',
                   'finalized_library_model.sav',
                   'finalized_language_model.sav']


def load_sentence_embeddings(file1, file2):
    df1 = pd.DataFrame.from_records(np.loadtxt(os.path.join(base_path, file1)))
    df2 = pd.DataFrame.from_records(np.loadtxt(os.path.join(base_path, file2)))
    df = df1.append(df2)
    labels = [1] * df1.shape[0]
    labels2 = [0] * df2.shape[0]
    labels.extend(labels2)
    train_features, test_features, train_labels, test_labels = train_test_split(df, labels)
    return train_features, test_features, train_labels, test_labels


def search_parameters(train_features, train_labels):
    parameters = {'C': np.linspace(0.0001, 100, 20)}
    grid_search = GridSearchCV(LogisticRegressionCV(max_iter=5000), parameters)
    grid_search.fit(train_features, train_labels)
    return grid_search


def train_classifier(train_features, train_labels, test_features, test_labels, c_param, feature):
    lr_clf = LogisticRegression(C=c_param, max_iter=10000)
    lr_clf.fit(train_features, train_labels)
    print('For {} feature {}'.format(feature, lr_clf.score(test_features, test_labels)))
    return lr_clf


def comparision_to_random_classifier(train_features, train_labels):
    clf = DummyClassifier()
    scores = cross_val_score(clf, train_features, train_labels)
    print("Dummy classifier score: %0.3f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


def driver(index):
    try:
        train_features, test_features, train_labels, test_labels = load_sentence_embeddings(file_names[index][0],
                                                                                            file_names[index][1])
        logger.info(
            "training data size: {} and testing size is {}".format(train_features.shape[0], test_features.shape[0]))
        scaler = StandardScaler()
        train_features = scaler.fit_transform(train_features)
        test_features = scaler.fit_transform(test_features)
        grid_search = search_parameters(train_features, train_labels)
        feature_name = model_filenames[index].split('_')[1]
        lr_clf = train_classifier(train_features, train_labels, test_features, test_labels,
                                  grid_search.best_params_['C'], feature_name)
        predictions = lr_clf.predict(test_features)
        pickle.dump(lr_clf, open(model_filenames[index], 'wb'))
        logger.info('{} model saved'.format(model_filenames[index]))
        # loaded_model = pickle.load(open(model_filename, 'rb'))
        print(predictions)
        comparision_to_random_classifier(train_features, train_labels)
    except Exception as e:
        logger.error("Error received for filename: {} and the error is: {}".format(file_names[index][0], e))


def main():
    indexes = [0, 1, 2]
    try:
        set_start_method('spawn')
        with Pool(processes=3) as pool:
            pool.map(driver, indexes)
    except RuntimeError:
        pass


if __name__ == '__main__':
    main()
