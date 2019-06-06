from sklearn.tree import DecisionTreeClassifier


class DecisionTreeClassification(object):

    def __init__(self, args={}):
        self.model = DecisionTreeClassifier(**args)

    def fit(self, x_train, y_train):
        self.fit(x_train, y_train)

    def predict(self, x_test):
        return self.model.predict(x_test)
