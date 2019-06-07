from sklearn.neural_network import MLPClassifier


class NeuralNetworkClassification(object):

    def __init__(self, args=None):
        self.model = MLPClassifier(**args)

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def get_learner(self):
        return self.model
