import unittest
import numpy as np
from main_runtime import train_model, download_and_prepare_data

class TestModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.X, cls.y = download_and_prepare_data()
        cls.model = train_model(cls.X[:60000], cls.y[:60000])
    
    def test_predict_accuracy(self):
        y_pred = self.model.predict(self.X[60000:])
        accuracy = np.mean(y_pred == self.y[60000:])
        self.assertGreater(accuracy, 0.70, "Accuracy should be greater than 70%")
    
    def test_training_time(self):
        import time
        start_time = time.time()
        train_model(self.X[:60000], self.y[:60000])
        end_time = time.time()
        self.assertLess(end_time - start_time, 30, "Training time should be less than 30 seconds")

if __name__ == "__main__":
    unittest.main()
