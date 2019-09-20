import sys
import numpy as np
import math
class LinearRegression:
    def __init__(self, training,degree, lambda_, test):
        self.train_rows = None
        self.train_cols = None
        self.test_rows = None
        self.test_cols = None
        self.count = 0
        self.training_path = training
        self.test_path = test
        self.degree = int(degree)
        self.lambda_ = int(lambda_)
        #self.classes_train = {}
        self.test_data = None
        self.training_data = None
        self.test_labels = None
        self.training_labels = None
        self.w = None
        #self.idx_mapping = {}
        self.init_classes()
    def init_classes(self):
    	self.load_classes(self.training_path)
    def load_data(self, path):
        return np.loadtxt(path)
    def load_classes(self, data_path):
        data = self.load_data(data_path)
        self.training_data = self.load_data(self.training_path)
        self.test_data = self.load_data(self.test_path)

        self.train_rows = self.training_data.shape[0]
        self.train_cols = self.training_data.shape[1]
        self.test_rows = self.test_data.shape[0]
        self.test_cols = self.test_data.shape[1]



        self.training_labels = self.training_data[:,[-1]]
        self.test_labels = self.test_data[:,[-1]]
        self.training_data = self.training_data[:,[x for x in range(self.train_cols - 1)]]
        self.test_data = self.test_data[:,[x for x in range(self.test_cols - 1)]]
    def train(self):
        ones_arr = np.ones((self.train_rows,1))
        phi = []
        for x in self.training_data:
            row = []
            for y in x:
                for j in range(1, self.degree + 1):
                    row.append(np.power(y, j))
            phi.append(row)
            row = []
        phi = np.array(phi)
        phi = np.hstack((ones_arr, phi))
        w = np.dot(phi.T, phi)
        w = np.linalg.pinv(w)
        w = np.dot(w, phi.T)
        w = np.dot(w, self.training_labels)
        self.w = w
    def predict(self, x):
        phi = []
        for y in x:
            for i in range(1, self.degree + 1):
                phi.append(np.power(y, i))
        phi = np.array([1] + phi)
        res = np.dot(self.w.T,phi)
        return res
    def run_predictions(self):
        id = 1
        for i, x in enumerate(self.test_data):
            prediction = self.predict(x)
            print('ID=%5d, output=%14.4f, target value = %10.4f, squared error = %.4f' % (id, prediction, self.test_labels[i], (prediction-self.test_labels[i]) ** 2))
            id += 1
            
def main():
    np.set_printoptions(threshold=sys.maxsize)
    np.set_printoptions(suppress=True)
    if len(sys.argv) < 5:
        print('Usage: [Path to training file] degree lambda [path to test file]')
    classifier = LinearRegression(*sys.argv[1:6])
    classifier.train()
    classifier.run_predictions()
if __name__ == '__main__':
    main()
