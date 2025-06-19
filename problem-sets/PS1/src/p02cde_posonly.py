import numpy as np
import util
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib import cm

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
    # x_train, t_train = util.load_dataset(train_path, label_col='t', add_intercept=True)
    # x_test, t_test = util.load_dataset(test_path, label_col='t', add_intercept=True)


    # # Train
    # model_t = LogisticRegression()
    # model_t.fit(x=x_train, y=t_train)

    # # Test
    # t_pred_c = model_t.predict(x_test)

    # # Eval
    # t_test_acc = np.sum(t_pred_c == t_test) / len(t_test)
    # print("Test accuracy (t labels): ", t_test_acc)


    # Part (d): Train on y-labels and test on true labels
    # Make sure to save outputs to pred_path_d
    x_train, y_train = util.load_dataset(train_path, label_col='y', add_intercept=True)
    x_test, y_test = util.load_dataset(test_path, label_col='y', add_intercept=True)

    # Train
    model_y = LogisticRegression()
    theta_history = model_y.fit(x=x_train, y=y_train)
    # print(theta_history)

    # Test
    y_pred_d = model_y.predict(x_test)

    # Eval
    y_test_acc = np.sum(y_pred_d == y_test) / len(y_test)
    print("Test accuracy (y): ", y_test_acc)

    ## Plot, static colors
    # plot_with_static_color(theta_history)

    ## Plot, dynamic colors
    plot_with_dynamic_color(theta_history)



    # Part (e): Apply correction factor using validation set and test on true labels
    # Plot and use np.savetxt to save outputs to pred_path_e
    # *** END CODER HERE


def plot_with_static_color(theta_history):
    ax = plt.figure().add_subplot(projection='3d')
    ax.plot(theta_history[:, 0], theta_history[:, 1], theta_history[:, 2], label='parametric curve')
    ax.legend()
    plt.show()

def plot_with_dynamic_color(theta_history):
    # Prepare data
    x = theta_history[:,0]
    y = theta_history[:,1]
    z = theta_history[:,2]

    # Create the plot
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    # Create segments along the curve
    points = np.array([x, y, z]).T
    segments = [points[i:i+2] for i in range(len(points)-1)]

    # Create a LineCollection and apply color based on the theta value
    norm = plt.Normalize(0, len(theta_history)+1)  # Normalize theta to [0,1]
    lc = Line3DCollection(segments, cmap='viridis', norm=norm, linewidth=2)

    color_arr = np.arange(len(theta_history))
    color_arr = (color_arr // 100) * 100
    lc.set_array(color_arr[:-1])  # Color by the theta values

    # Add the collection to the plot
    ax.add_collection(lc)

    # Add colorbar
    fig.colorbar(lc, ax=ax, label='Num Steps')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    margin = 0.1
    ax.set_xlim(np.min(x) - margin, np.max(x) + margin)
    ax.set_ylim(np.min(y) - margin, np.max(y) + margin)
    ax.set_zlim(np.min(z) - margin, np.max(z) + margin)

    plt.show()
    # print(f"X: {(x.min(), x.max())}\nY: {(y.min(), y.max())}\nZ: {(z.min(), z.max())}")