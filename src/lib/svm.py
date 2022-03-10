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
                 random_state: int = None):
        
        super().__init__()
        self.C = C
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.tol = tol
        self.max_iter = max_iter
        self.random_state = random_state

    def sgd(self,
            iter: int,
            X: np.ndarray,
            y: np.array,
            losses: List[float]):

        

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

        # losses and weights
        losses = []

        Parallel(
            n_jobs=cpu,
            backend="threading"
        )(delayed(super().sgd)(
            iter_, X, y, losses
        )
        for iter_ in range(self.max_iter))

