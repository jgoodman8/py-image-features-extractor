from sklearn.linear_model import LogisticRegression

class Model:

    def __init__(self, train, test):
        self.train = train
        self.test = test
        self.model = None

    def build_logistic_regression_model(self):
        logistic_regression = LogisticRegression()
        self.model = logistic_regression.fit(self.train['features'], self.train['label'])

    def predict(self):
        return self.model.predict(self.test[0].reshape(1, -1))