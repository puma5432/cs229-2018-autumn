import numpy as np
import util

from p01b_logreg import LogisticRegression

# Character to replace with sub-problem letter in plot_path/pred_path
WILDCARD = 'X'


def main(train_path, valid_path, test_path, pred_path):
    """Problem 2: Logistic regression for incomplete, positive-only labels.

    Run under the following conditions:
        1. on t-labels, (true label)
        2. on y-labels, (labeled-ness)
        3. on y-labels with correction factor alpha. (labeled-ness)

    Args:
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        pred_path: Path to save predictions.
    """
    pred_path_c = pred_path.replace(WILDCARD, 'c')
    pred_path_d = pred_path.replace(WILDCARD, 'd')
    pred_path_e = pred_path.replace(WILDCARD, 'e')

    # *** START CODE HERE ***

    # Part (c): Train and test on true labels
    # Make sure to save outputs to pred_path_c
    x_train, t_train = util.load_dataset(train_path, label_col='t', add_intercept=True)
    x_test, t_test = util.load_dataset(test_path, label_col='t', add_intercept=True)


    # Train
    model_t = LogisticRegression()
    model_t.fit(x=x_train, y=t_train)

    # Test
    t_pred_a = model_t.predict(x_test)

    # Eval
    t_test_acc = np.sum(t_pred_a == t_test) / len(t_test)
    print("Test accuracy (t labels): ", t_test_acc)


    # Part (d): Train on y-labels and test on true labels
    # Make sure to save outputs to pred_path_d
    x_train, y_train = util.load_dataset(train_path, label_col='y', add_intercept=True)
    x_test, y_test = util.load_dataset(test_path, label_col='y', add_intercept=True)

    # Train
    model_y = LogisticRegression()
    model_y.fit(x=x_train, y=y_train)

    # Test
    y_pred_a = model_y.predict(x_test)

    # Eval
    y_test_acc = np.sum(y_pred_a == y_test) / len(y_test)
    print("Test accuracy (y): ", y_test_acc)


    # Part (e): Apply correction factor using validation set and test on true labels
    # Plot and use np.savetxt to save outputs to pred_path_e
    # *** END CODER HERE
