import numpy as np
from matplotlib import pyplot as plt


class LinearRegModel(object):
    def __init__(self):
        self.m = 0.0
        self.b = 0.0

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        n = float(x.shape[0])
        self.m = (n*np.sum(x*y) - np.sum(x)*np.sum(y))/(n*np.sum(x**2) - np.sum(x)**2)
        self.b = (np.sum(y) - self.m*np.sum(x))/n

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.m*x + self.b


def main():
    x = np.arange(0, 10, 0.1)
    y = x*5 + np.random.uniform(low=-5.0, high=5.0, size=(100,))
    plt.plot(x, y, "ro")

    model = LinearRegModel()
    model.fit(x, y)

    y_pred = model.predict(x)

    plt.plot(x, y_pred)

    plt.show()


if __name__ == "__main__":
    main()
