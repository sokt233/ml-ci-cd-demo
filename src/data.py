from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

def load_data(test_size=0.2, random_state=42):
    # Ambil dataset iris dari sklearn (sudah built-in)
    iris = load_iris(as_frame=True)
    X = iris.data         # fitur (sepal_length, sepal_width, dll)
    y = iris.target       # label (Setosa, Versicolor, Virginica)

    # Split jadi train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test
