from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

def small_dataset(test_size=0.2, random_state=42):
    X, y = load_iris(return_X_y=True)
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
