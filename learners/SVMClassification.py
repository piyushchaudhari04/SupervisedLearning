from sklearn import svm
class SVMClassification(object):

    def __init__(self , args= None):
        self.svmc = svm.SVC(**args)

    def fit(self,train_x, train_y):
        self.svmc.fit(train_x, train_y)

    def predict(self, test_x):
        return self.svmc.predict(test_x)

    def get_classifier(self):
        return self.svmc
