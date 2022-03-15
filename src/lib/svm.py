from typing import Text, Union, List, Any
import numpy as np
import warnings
from numbers import Number
import multiprocessing
from joblib import Parallel, delayed
from lib.base import _Base

cpu = multiprocessing.cpu_count() - 1

class _SVM(_Base):

    d_kernels = {
        "linear": super().kernel_linear,
        "poly": super().kernel_poly,
        "rbf": super().kernel_rbf
    }

    def __init__(self,
                 C: float = 1.0,
                 kernel: Text = "linear",
                 degree: int = 3,
                 gamma: Union[Text, float] = "scale",
                 tol: float = 1e-3,
                 max_iter: int = 100,
                 lr: float = 0.001,
                 random_state: int = None):
        
        super().__init__()
        self.C = C
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.tol = tol
        self.max_iter = max_iter
        self.lr = lr
        self.random_state = random_state
        self.kernels = None

    def distances(self, 
                  X: np.ndarray,
                  y: np.arrange,
                  with_lagrange: bool=True):
        distances = y * np.dot(self.W, X) - 1

        if with_lagrange:
            # if distance is more than 0, sample is not on the support vector
            # lagrange multiplier will be 0
            distances[distances > 0] = 0
        
        return distances

    def sgd(self,
            iter: int,
            X: np.ndarray,
            y: np.array):

        losses = float("inf")
        early_stop = 0

        # distance
        distances = self.distances(X, y)

        for index, d in enumerate(distances):

            if d == 0:
                pass
            else:
                dW, db = super().gradients(X[index], y[index])

            self.W[index] -= self.lr * dW
            self.b[index] -= self.lr * db

        if (iter % 2) or (iter == self.max_iter - 1):
            loss = super().loss(X, y)
            print(f"loss in {iter} is: {loss}")

            if (early_stop == 3) or (iter == self.max_iter - 1):
                early_stop = 0
                return 
            
            if loss > losses:
                early_stop += 1

            losses = loss

    def fit(self,
            X: np.ndarray,
            y: np.array) -> np.ndarray:

        if not isinstance(self.C, Number):
            raise ValueError(
                "C must be a number; got (C=%r" % self.C
            )

        if self.kernel not in ("linear", "poly", "rbf", "sigmoid"):
            raise ValueError(
                "kernel must be in 'linear', 'poly', 'rbf' or 'sigmoid'; "
                "got (kernel=%s" % self.kernel
            ) 

        if self.kernel is "poly":
            if not isinstance(self.degree, Number):
                raise ValueError(
                    "Degree must be a number; got (degree=%r" % self.degree
                )

        if self.kernel in ("rbf", "poly", "sigmoid"):
            if isinstance(self.gamma, str) and self.gamma not in ("scale", "auto"):
                raise ValueError(
                    "gamma should be in 'scale', 'auto' or float; got (gamma=%r" % self.gamma
                )

        # set vairables
        d_id_class = {}
        m, n = X.shape
        self.classes_ = np.unique(y)
        y_classes = len(self.classes_)
        self.kernels = self.d_kernels[self.kernel]
        self.alpha = np.zeros((m))

        for index, y_ in enumerate(self.classes_):
            d_id_class[index] = y_
        self.id_2_class = d_id_class

        # initializing weights and bias to zeros
        if self.multi_class:
            self.W = np.zeros((y_classes, n))
            self.b = np.zeros(y_classes)
        else:
            self.W = np.zeros((n))
            self.b = 0
                
        # normalize the inputs
        X = super().normalize(X)

        Parallel(
            n_jobs=cpu,
            backend="threading"
        )(delayed(super().sgd)(
            iter_, X, y
        )
        for iter_ in range(self.max_iter))

    def predict(self, X: np.ndarray) -> np.array:
        
        pred = np.ndarray(X.shape[0])

        preds = self.predict_proba(X)

        max_prop = np.argmax(preds.T, axis=0)

        for index in self.id_2_class:
            pred[max_prop == index] = self.id_2_class[index]

        return pred

    def predict_proba(self, X: np.ndarray) -> np.array:

        # normalizing
        X = super().normalize(X)

        arr_prop = np.zeros((len(self.classes_), X.shape[0]))
        
        pred = super().sigmoid(np.dot(self.W, X) + self.b)

        arr_prop[0] = 1 - pred
        arr_prop[1] = pred

        return arr_prop.T