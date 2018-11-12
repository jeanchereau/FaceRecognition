import numpy as np
from functions import eigen_order


def pca():
    pass


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
            sw = sw + (k_set.size - 1) * np.cov(data_train[:, k_set])

    sb = (n_p - 1) * np.cov(mu_cluster)

    if m_pca is None:
        m_pca = np.linalg.matrix_rank(sb, tol=0.001)
        m_lda = int(m_pca * 2 / 3)
    elif m_lda is None:
        m_lda = m_pca

    st = (cols - 1) * np.cov(data_train, rowvar=False)
    u = (data_train - mu.reshape(rows, 1)).dot(eigen_order(st, m=m_pca))

    slda = np.linalg.inv(u.T.dot(sw.dot(u))).dot(u.T.dot(sb.dot(u)))
    w = u.dot(eigen_order(slda, m=m_lda))

    data_train_proj = w.T.dot(data_train - mu.reshape(rows, 1))

    return data_train_proj, w, mu


class comm_submod (th.Thread):
    def __init__(self, modelID, face_data_training, n_t, n_p, id_memory):
        th.Thread.__init__(self)
        self.modelID = modelID
        self.face_data_training = face_data_training
        self.n_t = n_t
        self.n_p = n_p
        self.id_memory = id_memory
        self.W = None
        self.mu = None
    def run(self):
        print('Starting Thread', self.modelID, '...')
        self.W, self.mu = pca_lda(self.face_data_training, self.n_t, self.n_p, self.id_memory)
        print('Thread', self.modelID, 'done!')
