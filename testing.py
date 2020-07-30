from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.datasets import load_iris
from naive_bayes_classifier import *
import numpy as np



if __name__ == "__main__":
    # loading iris dataset
    X, y = load_iris(return_X_y=True)
    # train test splitting the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)


    ## SCRATCH:
    # instantiate model
    nb = Naive_Bayes()
    # fit
    nb.fit(X_train, y_train)
    # predict
    y_pred_scratch = nb.predict(X_test)    
    # accuracy
    print(f"\nSCRATCH Accuracy: {nb.accuracy(y_test, y_pred_scratch)}")


    ## SKLEARN:
    # GNB: instantiate model
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    y_pred_sklearn = gnb.predict(X_test)
    
    # # MNB: instantiate model
    # mnb = MultinomialNB()
    # mnb.fit(X_train, y_train)
    # y_pred_sklearn = mnb.predict(X_test)
    # accuracy
    print(f"\nSKLEARN Accuracy: {nb.accuracy(y_test, y_pred_sklearn)}\n")
