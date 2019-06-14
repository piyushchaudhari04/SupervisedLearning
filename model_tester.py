import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
import learners.DecisionTreeClassification as dt
import learners.NeuralNetworkClassification as mlp
import learners.KNeighborsClassification as knn
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
import learners.SVMClassification as svmc


def read_data(file_name):
    return pd.read_csv(file_name)


def remove_unwanted_features(dataset):
    X = dataset.iloc[:, 3:13].values
    y = dataset.iloc[:, 13].values
    return X, y


def label_encoder(X):
    label_encoder_X = LabelEncoder()
    X[:, 1] = label_encoder_X.fit_transform(X[:, 1])
    X[:, 2] = label_encoder_X.fit_transform(X[:, 2])
    return X


def hot_encoder(X):
    one_hot_encoder = OneHotEncoder(categorical_features=[1])
    X = one_hot_encoder.fit_transform(X).toarray()
    return X


def feature_scaling(dataset):
    scaler = StandardScaler()
    cols = ["CreditScore", "Age", "Tenure", "Balance", "EstimatedSalary"]
    dataset[cols] = scaler.fit_transform(dataset[cols])
    return dataset


def run_decision_tree(X_train, y_train, X_test, y_test):
    args = {"criterion": 'entropy', "splitter": "random", "max_depth": 7, "min_samples_split": 300,
            "min_samples_leaf": 50}
    model = dt.DecisionTreeClassification(args)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    cm = metrics.confusion_matrix(y_test, y_pred)
    print(metrics.accuracy_score(y_test, y_pred))


def run_decision_tree_cross_validation(X, y):
    args = {"criterion": 'entropy', "splitter": "random", "max_depth": 7, "min_samples_split": 300,
            "min_samples_leaf": 50}
    model = dt.DecisionTreeClassification(args)
    classifier = model.get_skl_learner()
    all_accuracies = cross_val_score(estimator=classifier, X=X, y=y, cv=10)
    print(all_accuracies.mean())


def run_neural_network(X_train, y_train, X_test, y_test):
    layers = (11,) * 10
    args = {"activation": "tanh", "max_iter": 1000, "hidden_layer_sizes": layers}
    model = mlp.NeuralNetworkClassification(args)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    cm = metrics.confusion_matrix(y_test, y_pred)
    print(metrics.accuracy_score(y_test, y_pred))


def run_neural_network_cross_validation(X, y):
    layers = (11,) * 10
    args = {"activation": "tanh", "max_iter": 1000, "hidden_layer_sizes": layers}
    model = mlp.NeuralNetworkClassification(args)
    all_accuracies = cross_val_score(estimator=model.get_learner(), X=X, y=y, cv=10)
    print(all_accuracies.mean())


def run_k_neighbors(X_train, y_train, X_test, y_test):
    args = {"algorithm": 'brute', "leaf_size": 30, "metric": 'cityblock', "n_jobs": -1, "n_neighbors": 9,
            "weights": 'uniform'}
    model = knn.KNeighborsClassification(args)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    cm = metrics.confusion_matrix(y_test, y_pred)
    print(metrics.accuracy_score(y_test, y_pred))


def run_svm(X_train, y_train, X_test, y_test):
    args={"gamma" : 0.9}
    classifier = svmc.SVMClassification(args)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    print(metrics.accuracy_score(y_test, y_pred))


if __name__ == '__main__':
    np.set_printoptions(threshold=np.inf)
    dataset = read_data("Churn_Modelling.csv")
    dataset = feature_scaling(dataset)
    X, y = remove_unwanted_features(dataset)
    X = label_encoder(X)
    X = hot_encoder(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    run_decision_tree(X_train, y_train, X_test, y_test)
    run_decision_tree_cross_validation(X, y)
    run_neural_network(X_train, y_train, X_test, y_test)
    run_neural_network_cross_validation(X, y)
    run_k_neighbors(X_train, y_train, X_test, y_test)
    run_svm(X_train, y_train, X_test, y_test)
