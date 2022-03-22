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
                 multi_class: bool = False,
                 random_state: int = None):
        
        super().__init__()
        self.C = C
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.tol = tol
        self.max_iter = max_iter
        self.lr = lr
        self.multi_class = multi_class
        self.random_state = random_state
        self.d_kernels = {
                "linear": super().kernel_linear,
                "poly": super().kernel_poly,
                "rbf": super().kernel_rbf
            }

        np.random.seed(self.random_state)

    def sgd(self,
            iter: int,
            X: np.ndarray,
            y: np.array,
            alpha: np.array,
            func):

        losses = float("inf")
        early_stop = 0

        dW, db = super().gradients(X, y, alpha)

        self.W -= self.lr * dW
        self.b -= self.lr * db

        if (iter % 2) or (iter == self.max_iter):
            loss = func(X, y, alpha)
            print(f"loss in {iter} is: {loss}")

            if (early_stop == 3) or (iter == self.max_iter):
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
        alpha = np.random.rand(m)

        for index, y_ in enumerate(self.classes_):
            d_id_class[index] = y_
        d_id_class[-1] = d_id_class.pop(0)
        self.id_2_class = d_id_class

        # change y
        y[y == self.id_2_class[-1]] = -1

        # initializing weights and bias to zeros
        if self.multi_class:
            self.W = np.ones((y_classes, n))
            self.b = np.zeros(y_classes)
        else:
            self.W = np.ones((n))
            self.b = 0

        # set gamma
        if self.gamma == "scale":
            self.gamma = 1 / (n * X.var())
        elif self.gamma == "auto":
            self.gamma = 1 / n

        # set kernel function
        if self.kernel == "linear":
            loss_func = super().linear_loss
        else:
            loss_func = super().kernel_loss
                
        # normalize the inputs
        X = super().normalize(X)

        print(f"shape of X: {X.shape}")
        print(f"shape of y: {y.shape}")
        print(f"shape of alpha: {alpha.shape}")

        Parallel(
            n_jobs=cpu,
            backend="threading"
        )(delayed(self.sgd)(
            iter_, X, y, alpha, loss_func
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
        
        pred = super().sigmoid(np.dot(X, self.W) + self.b)

        arr_prop[0] = 1 - pred
        arr_prop[1] = pred

        return arr_prop.T