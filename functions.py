import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


# Returns indices of k data points in x_data which are closest to x_s.
# Default is k = 1 (NN).
def knn(x_s, x_data, k=1):
    rows = x_s.size

    x_norm_diff = np.linalg.norm(x_data - x_s.reshape(rows, 1), axis=0)

    indices = np.argsort(x_norm_diff)[0:k]

    return indices


# Partitions a data set into a training set and a testing set.
# n_p is the number of classes. Default is 52.
# n_pp is the number of data points in each class. Default is 10.
# n_pp_train is the number of unique data points in each class assigned to training set. Default is 7.
def data_partition_train_test(data, n_pp_train=7, n_pp=10, n_p=52):
    rows, cols = data.shape

    n_pp_test = n_pp - n_pp_train

    # Allocate memory space for training and testing sets (greater efficiency).
    data_train = np.zeros((rows, int(cols * n_pp_train / n_pp)), dtype=np.uint8)
    data_test = np.zeros((rows, int(cols * n_pp_test / n_pp)), dtype=np.uint8)

    array = np.arange(n_pp)     # Index the data points in a class from 0 to n_pp - 1.

    # Go through each class in the data set.
    for j in range(0, n_p):
        p_array = np.random.permutation(array)  # Permute the indices randomly.

        # Indices of data points in data set to be allocated to training set.
        i_set = p_array[0:n_pp_train] + j * n_pp
        # Indices of data points in training set to be defined.
        k_set = np.arange(j * n_pp_train, (j + 1) * n_pp_train)
        # Allocate to training set for current class.
        data_train[:, k_set] = data[:, i_set]

        # Indices of data points in data set to be allocated to testing set.
        i_set = p_array[n_pp_train:n_pp] + j * n_pp
        # Indices of data points in testing set to be defined.
        k_set = np.arange(j * n_pp_test, (j + 1) * n_pp_test)
        # Allocate to testing set for current class.
        data_test[:, k_set] = data[:, i_set]

    return data_train, data_test


# Creates a sub training set from
def data_train_subspace(data_train, n_pp_train, n_t=None):
    rows, cols = data_train.shape

    if n_t is None:
        n_t = cols

    data_train_sub = np.zeros([rows, n_t], dtype=np.uint8)

    i_set = np.sort(np.random.randint(0, cols, n_t))
    data_train_sub[:, :] = data_train[:, i_set]

    data_id_memory = i_set // n_pp_train

    return data_train_sub, data_id_memory


def eigen_order(s, m=None):
    lc, vc = np.linalg.eig(s)

    indices = np.argsort(np.abs(lc))[::-1]

    if type(m) is np.ndarray:
        v = np.real_if_close(vc[:, indices], tol=0.001)[:, m]
    else:
        v = np.real_if_close(vc[:, indices], tol=0.001)[:, 0:m]

    return v


def conf_mat(y_actu, y_pred):
    cm = confusion_matrix(y_actu, y_pred)

    plt.figure()
    plt.matshow(cm, cmap='Blues')
    plt.colorbar()
    plt.ylabel('Actual id')
    plt.xlabel('Predicted id')
    plt.show()
