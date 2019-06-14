from sklearn.neighbors import KNeighborsClassifier


class KNeighborsClassification(object):

    def __init__(self, args=None):
        self.model = KNeighborsClassifier(**args)

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def get_learner(self):
        return self.model


