from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.model_selection import train_test_split
from sklearn import datasets
import numpy as np


class Naive_Bayes:
    
    def calculate_priors(self, X, y):
        ## Calculate Priors
        # class_types, count_per_class per each classification outcome type
        class_types, count_per_class = np.unique(y, return_counts=True)
        # zip the class_types and n_class_types together so it's easier to
        # reference later when I calculate probilities
        class_type_dist = list(zip(class_types, count_per_class))
        # find the total number of datapoints, and sum them together
        total_ = [c[1] for c in class_type_dist]
        total = sum(total_)
        # find the percentage distribution of each class_type -- set the priors 
        # parameter equal to the percentage of prior occurrences
        self.priors = np.array([c[1] / total for c in class_type_dist])

    def calculate_likelihood(self, X, y):
        # identify the unique number of classification types (class_types)
        # set parameter class_types equal to them
        self.class_types = np.unique(y)
        # identify the number_rows, number_features in the dataset from X
        n_rows, n_features = X.shape
        # set parameters mean, variance equal to numpy arrays with zeros
        self.mean = np.zeros((n_rows, n_features), dtype=np.float64)
        self.variance = np.zeros((n_rows, n_features), dtype=np.float64)
        # loop over the different idx and class_type in the class_types parameter
        for idx_class_type, class_type in enumerate(self.class_types):
            # set the X datapoints equal to the respective class_type
            X_classes = X[y == class_type]
            # calculate the mean of the X datapoints for each feature and set them
            # equal to the self.mean parameter for the specified idx
            self.mean[idx_class_type, :] = X_classes.mean(axis=0)
            # calculate the variance of the X datapoints for each feature and set them
            # equal to the self.mean parameter for the specified idx
            self.variance[idx_class_type, :] = X_classes.var(axis=0)

    ## FIT
    def fit(self, X, y):
        # calling calculate_priors function
        self.calculate_priors(X, y)
        # calling calculate_likelihood function
        self.calculate_likelihood(X, y)
        
    def prob_density_function(self, mean, variance, x):
        # exponenet in the pdf
        exponent = np.exp((-(x - mean) ** 2) / (2 * variance))
        # entire probability
        p_x_given_y = 1 / np.sqrt(2 * np.pi * variance) * exponent
        return p_x_given_y


    def posterior_prob(self, x):
        # instantiate an open list to hold all the posterior_probs
        posterior_probs = []
        # loop over the class_types
        for idx_class_type, class_type in enumerate(self.class_types):
            # instantiate mean, variance, priors for each respective idx_class_type
            mean = self.mean[idx_class_type]
            variance = self.variance[idx_class_type]
            prior = self.priors[idx_class_type]

            # posteriors are equal to the sum, of the log, of the pdf
            posterior = np.sum(np.log(self.prob_density_function(mean, variance, x)))
            # add that new posterior prob to the prior prob to get the new
            # posterior_prob
            posterior += prior
            # append the posterior_prob to the posterior_probs empty list
            posterior_probs.append(posterior)
        
        # pull out the max_posterior_prob for a given set of class_types
        max_posterior_prob = self.class_types[np.argmax(posterior_probs)]
        # return max
        return max_posterior_prob
    
    ## PREDICT
    def predict(self, X):
        # instantiate an empty prediction list
        y_pred = []
        # for x datapoints in X
        for x in X:
            # calculate the posterior_prob and append it to the list
            y_pred.append(self.posterior_prob(x))
        # return the list as a numpy array
        return np.array(y_pred)

    def accuracy(self, y_true, y_pred):
        # calculate, and return the accuracy via the sum of the correct predictions, over 
        # the total number of predictions
        return np.sum(y_true == y_pred) / len(y_true)
  
        



if __name__ == "__main__":
    
    ## TESTING:
    X, y = datasets.make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=123)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

    
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
    # GNB: instantiate model, fit, predict
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    y_pred_sklearn = gnb.predict(X_test)
    
    # # MNB: instantiate model
    # mnb = MultinomialNB()
    # mnb.fit(X_train, y_train)
    # y_pred_sklearn = mnb.predict(X_test)
    # accuracy
    print(f"\nSKLEARN Accuracy: {nb.accuracy(y_test, y_pred_sklearn)}\n")
