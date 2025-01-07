import sys
import time
import unittest
import numpy as np
from main_runtime import train_model, download_and_prepare_data

class TestModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        print("Setting up class for runtime tests.\n")
        cls.X, cls.y = download_and_prepare_data()
    
    def test_predict_accuracy(self):
        model = train_model(self.X[:60000], self.y[:60000])
        y_pred = model.predict(self.X[60000:])
        accuracy = np.mean(y_pred == self.y[60000:])
        print(f"Test Accuracy: {accuracy * 100:.2f}%\n")
        self.assertGreater(accuracy, 0.70, "Accuracy should be greater than 70%")

    def test_training_time(self):
        start_time = time.time()
        train_model(self.X[:60000], self.y[:60000])
        end_time = time.time()
        print(f"Training time: {end_time - start_time:.5f} sec\n")
        self.assertLess(end_time - start_time, 274, "Training time should be less than 274 seconds")

if __name__ == "__main__":
    # Umleitung der Ausgaben in die Datei "ausgabe2.txt"
    with open("ausgabe2.txt", "w") as f:
        sys.stdout = f
        unittest.main()

