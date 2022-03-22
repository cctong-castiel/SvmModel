import numpy as np
from sklearn.datasets import load_breast_cancer 
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from lib.svm import _SVM
from dotenv import load_dotenv
import os

load_dotenv()
seed = int(os.getenv("seed"))

def main():

    # load X and y
    X, y = load_breast_cancer(return_X_y=True)

    # train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

    # create model objects
    clf_skl = SVC(probability=True, max_iter=500, random_state=seed)
    clf_ctm = _SVM(max_iter=500, random_state=seed, kernel="linear")

    # fit
    print("model fitting")
    clf_skl.fit(X_train, y_train)
    clf_ctm.fit(X_train, y_train)

    # predict
    print("model prediction")
    y_pred_skl = clf_skl.predict(X_test)
    y_pred_ctm = clf_ctm.predict(X_test)

    # classification report
    print(f"classification report of sklearn SVC: \n{classification_report(y_test, y_pred_skl)}\n")
    print(f"classification report of custom SVC: \n{classification_report(y_test, y_pred_ctm)}\n")

if __name__ == "__main__":
    main()