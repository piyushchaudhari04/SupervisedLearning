from sklearn.tree import DecisionTreeClassifier


class DecisionTreeClassification(object):

    def __init__(self, args=None):
        self.model = DecisionTreeClassifier(**args)

    def fit(self, x_train, y_train):
        self.model.fit(x_train, y_train)

    def predict(self, x_test):
        return self.model.predict(x_test)

    def get_skl_learner(self):
        return self.model
