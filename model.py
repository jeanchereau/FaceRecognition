import numpy as np
from functions import eigen_order


# Model
def pca(data_train, m_pca=None):
    rows, cols = data_train.shape

    mu = np.mean(data_train, axis=1)

    st = (data_train - mu[:, None]).T.dot(data_train - mu[:, None])

    if m_pca is None:
        m_pca = int(cols * 6 / 10)

    u = (data_train - mu[:, None]).dot(eigen_order(st, m=m_pca))

    length = u.shape[1]
    for i in range(length):
        norm = np.linalg.norm(u[:, i])
        u[:, i] = u[:, i] / norm

    data_train_proj = u.T.dot(data_train - mu[:, None])

    return data_train_proj, u, mu


def pca_lda(data_train, data_id_memory, m_lda=None, m_pca=None, n_p=52):
    rows, cols = data_train.shape

    mu = np.mean(data_train, axis=1)

    sw = np.zeros((rows, rows))
    mu_cluster = np.zeros((rows, n_p))
    for k in range(0, n_p):
        k_set = np.transpose(np.argwhere(data_id_memory == k))[0]

        if k_set.size == 0:
            mu_cluster[:, k] = np.zeros(rows)
        else:
            mu_cluster[:, k] = np.mean(data_train[:, k_set], axis=1)

        if k_set.size > 1:
            mu_w = mu_cluster[:, k]
            sw = sw + (data_train[:, k_set] - mu_w[:, None]).dot((data_train[:, k_set] - mu_w[:, None]).T)

    sb = (mu_cluster - mu[:, None]).dot((mu_cluster - mu[:, None]).T)

    if m_pca is None:
        m_pca = n_p - 1
        if m_lda is None:
            m_lda = int(m_pca * 2 / 3)
    elif m_lda is None and type(m_pca) is not np.ndarray:
        if m_pca > n_p - 1:
            m_lda = n_p - 1
        else:
            m_lda = m_pca

    if type(m_pca) is np.ndarray:
        i_set = np.transpose(np.argwhere(m_pca > cols - 1 - n_p))[0]
        m_pca[i_set] = (cols - 1 - n_p) * np.ones(i_set.size)
        if m_lda is None:
            m_lda = n_p - 1

    st = (data_train - mu[:, None]).T.dot(data_train - mu[:, None])
    u = (data_train - mu[:, None]).dot(eigen_order(st, m=m_pca))

    slda = np.linalg.inv(u.T.dot(sw.dot(u))).dot(u.T.dot(sb.dot(u)))
    w = u.dot(eigen_order(slda, m=m_lda))

    data_train_proj = w.T.dot(data_train - mu[:, None])

    return data_train_proj, w, mu


class CommSubmod:
    def __init__(self, model_id, data_train, data_id_memory, n_p):
        self.model_id = model_id
        self.data_train = data_train
        self.data_id_memory = data_id_memory
        self.n_p = n_p
        self.data_train_proj = None
        self.w = None
        self.mu = None

    def setup(self):
        print('Building Committee Machine sub-model', self.model_id, '...')
        self.data_train_proj, self.w, self.mu = pca_lda(self.data_train, self.data_id_memory, n_p=self.n_p)
        print('sub-model', self.model_id, 'done!')


class RandsmpSubmod:
    def __init__(self, model_id, data_train, data_id_memory, n_p, m0, m1):
        self.model_id = model_id
        self.data_train = data_train
        self.data_id_memory = data_id_memory
        self.n_p = n_p
        self.m0 = m0
        self.m1 = m1
        self.data_train_proj = None
        self.w = None
        self.mu = None

    def setup(self):
        print('Building Random Feature Sampling sub-model', self.model_id, '...')
        array = np.random.permutation(np.arange(self.m0, self.n_p - 1))
        m_ar = np.concatenate((np.arange(self.m0), array[0:self.m1]), axis=None)
        self.data_train_proj, self.w, self.mu = pca_lda(self.data_train, self.data_id_memory, m_pca=m_ar, n_p=self.n_p)
        print('sub-model', self.model_id, 'done!')
