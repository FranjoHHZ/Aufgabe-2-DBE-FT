import time
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression

# Logger-Decorator
def my_logger(orig_func):
    import logging
    logging.basicConfig(filename=f"{orig_func.__name__}.log", level=logging.INFO)
    
    def wrapper(*args, **kwargs):
        logging.info(f"Ran with args: {args}, kwargs: {kwargs}")
        return orig_func(*args, **kwargs)
    return wrapper

# Timer-Decorator
def my_timer(orig_func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = orig_func(*args, **kwargs)
        end_time = time.time()
        print(f"{orig_func.__name__} ran in: {end_time - start_time:.5f} sec")
        return result
    return wrapper

@my_logger
@my_timer
def download_and_prepare_data():
    mnist = fetch_openml('mnist_784', version=1)
    X, y = mnist.data.astype('float32'), mnist.target.astype('int64')
    X /= 255.0
    return X, y

@my_logger
@my_timer
def train_model(X, y):
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    
    model = LogisticRegression(solver='saga', multi_class='multinomial', max_iter=100, penalty='l2')
    model.fit(X, y)
    return model

if __name__ == "__main__":
    X, y = download_and_prepare_data()
    model = train_model(X[:60000], y[:60000])
    print("Training completed.")
