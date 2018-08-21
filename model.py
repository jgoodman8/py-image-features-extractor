import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


class Model:

    def __init__(self, train, test):
        self.train_df = pd.DataFrame.from_dict(train)
        self.test_df = pd.DataFrame.from_dict(test)
        self.model = None

        self.X_train = np.array(map(lambda item: item['features'], self.train_df))
        self.y_train = np.array(map(lambda item: item['label'], self.train_df))

        self.X_test = np.array(map(lambda item: item['features'], self.test_df))
        self.y_test = np.array(map(lambda item: item['label'], self.test_df))

    def build_logistic_regression_model(self):
        logistic_regression = LogisticRegression()
        self.model = logistic_regression.fit(self.X_train, self.y_train)

    def predict(self):
        return self.model.predict(self.X_test)
