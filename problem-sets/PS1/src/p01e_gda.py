import numpy as np
import util

from linear_model import LinearModel

import time
def main(train_path, eval_path, pred_path):
    """Problem 1(e): Gaussian discriminant analysis (GDA)

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    # Load dataset
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)

    # *** START CODE HERE ***
    # print(x_train.shape, x_train[0])

    # Train GDA
    model = GDA()
    model.fit(x_train, y_train)

    # Plot
    util.plot(x_train, y_train, model.theta, 'output/p01e_{}.png'.format(pred_path[-5]))

    x_val, y_val = util.load_dataset(eval_path, add_intercept=True)
    y_pred = model.predict(x_val)

    val_accuracy = 1- np.abs(y_val - y_pred).sum()/len(y_val)
    print("Validation accuracy: {}".format(val_accuracy))

    # *** END CODE HERE ***


class GDA(LinearModel):
    """Gaussian Discriminant Analysis.

    Example usage:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Fit a GDA model to training set given by x and y.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).

        Returns:
            theta: GDA model parameters.
        """
        # *** START CODE HERE ***
        phi, mu_0, mu_1, sigma = None, None, None, None
        m, n = x.shape
        self.theta = np.zeros(n+1)

        phi = np.sum(y)/len(y)
        mu_0 = x[np.where(y==0)].mean(axis=0) # avg over training examples, resulitng in per-feature average
        mu_1 = x[np.where(y==1)].mean(axis=0)

        # simple unvectorized version of sigma

        # sigma_unvec = np.zeros((n, n))
        # start = time.time()

        # for i in range(m):
        #     mu = mu_0 if (y[i] == 0) else mu_1
        #     sigma_unvec += np.outer((x[i]-mu), (x[i]-mu))
        # sigma_unvec /= m
        # end = time.time()
        # print("First implementation time:", end - start)
        # print(sigma_unvec)
        #        array([[9.53653342e-01, 5.87339678e+01],
        #               [5.87339678e+01, 1.15818332e+04]])

        sigma = np.zeros((n, n))
        start = time.time()
        mu_conditional = np.where((y[:,None]==1), mu_1, mu_0)
        x_variance = x - mu_conditional # m,n (-> n,n)
        sigma = (x_variance.T @ x_variance) / m # Compute outer product over all training examples. Result is same as sigma_unvec
        end = time.time()
        # print("Second implementation time:", end - start)

        sig_inv = np.linalg.inv(sigma)
        self.theta[1:] = sig_inv @ (mu_1 - mu_0) # (n,n) @ (n,) -> (n,)

        self.theta[0] = 1/2 * (mu_0+mu_1).T @ sig_inv @ (mu_0-mu_1) - np.log( (1-phi)/(phi) ) # -> scalar


        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        linear = x@self.theta# (m,n) @ (n,) -> m,1
        y_pred = 1/(1+np.exp(-1*linear))
        # print(y_pred)

        return (y_pred >= 0.5)

        # *** END CODE HERE
