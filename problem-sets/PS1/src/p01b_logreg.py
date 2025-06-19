import numpy as np
import util
import os

import time
from datetime import datetime

from linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """Problem 1(b): Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***

    # Train logistic regression
    model = LogisticRegression()
    model.fit(x=x_train, y=y_train)

    # Plot
    util.plot(x_train, y_train, model.theta, 'output/p01b_{}.png'.format(pred_path[-5]))


    # Evaluate
    x_val, y_val = util.load_dataset(eval_path, add_intercept=True)
    y_val_pred = model.predict(x_val)
    val_error = np.sum(np.abs(y_val_pred - y_val)) / len(y_val)
    acc = 1 - val_error
    print("Validation accuracy: ", acc)
    print("Validation error: ", val_error)

    # Save
    os.makedirs(os.path.dirname(pred_path), exist_ok=True)
    np.savetxt(pred_path, y_val_pred.astype(int), fmt='%d')







    # *** END CODE HERE ***


class LogisticRegression(LinearModel):
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def g(self, z): # sigmoid
            return 1 / (1 + np.exp(-z))

    def fit(self, x, y):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        # *** START CODE HERE ***


        m,n = x.shape
        print(f"m: {m}, n: {n}")

        # print(x)

        # initialize
        self.theta = np.zeros((n))

        eps = 1e-4 #or 10e-5

        H = np.zeros((n,n))
        grad = np.zeros((n))
        step = 1
        start = time.time()
        print(datetime.now().time())

        weight_arr = np.zeros((30000, 3))
        theta_history = []

        while True:

            # form hessian and grad
            for i in range(len(x)):
                # Hessian of i-th example
                g_z = self.g(x[i]@self.theta)
                H += -1* g_z*(1-g_z) * np.outer(x[i], x[i])

                grad += (y[i]-g_z) * x[i]

            H_inv = np.linalg.inv(H)

            # do 1 update of newton's method- hessian
            update = H_inv @ grad
            self.theta = self.theta - (update)
            # weight_arr[step] = self.theta
            theta_history.append(self.theta.copy())


            update_norm = np.linalg.norm(update, ord=1)
            if update_norm < eps:
                print(f"Final 1-norm: {update_norm}\n Num steps: {step}")
                end = time.time()
                print(datetime.now().time())
                print(f"Total time: {end - start}")
                theta_history = np.array(theta_history)
                return theta_history
            else:
                # print(f"Update 1-norm: {update_norm}")
                step += 1

            # normal update (t): 252116 (21m)
            # halved update (t): (no result yet)
            # normal update (y): (30341) (5m)
            # halved update (y): (120826) (10:30s)
            # - might be 2x as many steps. Or might be less than that if
            #   my theory is right and newton method continuously 'overshoots'





        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        confidence_arr = self.g(x@self.theta)
        y_pred = (confidence_arr >= 0.5)

        return y_pred

        # *** END CODE HERE ***
