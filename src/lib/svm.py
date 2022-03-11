from typing import Text, Union, List, Any
import numpy as np
import warnings
from numbers import Number
import multiprocessing
from joblib import Parallel, delayed
from lib.base import _Base

cpu = multiprocessing.cpu_count() - 1

class _SVM(_Base):

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

    def sgd(self,
            iter: int,
            X: np.ndarray,
            y: np.array):

        losses, weigths, bias = float("inf"), None, None
        early_stop = 0

        for index, x in enumerate(len(X)):

            dW, db = super().gradients(x, y[index])

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
        self.lagrange = np.zeros(y_classes)
                
        # normalize the inputs
        X = super().normalize(X)

        if self.multi_class is True:

        else:
            Parallel(
                n_jobs=cpu,
                backend="threading"
            )(delayed(super().sgd)(
                iter_, X, y
            )
            for iter_ in range(self.max_iter))

    def predict(self, X: np.ndarray) -> np.array:

        # normalizing
        X = super().normalize(X)

        return np.sign(np.dot(self.W, X) + self.b)

    def predict_proba(self, X: np.ndarray) -> np.array:

        # normalizing
        X = super().normalize(X)

        if self.multi_class:
            arr_prop = np.zeros((len(self.classes_), X.shape[0]))



        return arr_prop