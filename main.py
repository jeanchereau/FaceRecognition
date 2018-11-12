import scipy.io as sio
import numpy as np
from functions import data_partition_train_test, data_train_subspace
from model import pca_lda
from test import test_pca_lda
from yaml import load


print('Setting things up...')
with open('cfgs/conf.yml', 'r') as ymlfile:
    cfg = load(ymlfile)
for section in cfg:
    for attr in section.items():
        if attr[0] == 'SETUP':
            n_p = attr[1].get('n_p')
            n_fpp = attr[1].get('n_fpp')
            n_fpp_train = attr[1].get('n_fpp_train')
            n_t = attr[1].get('n_t')
            tt = attr[1].get('T')
print('Done!!')

print('Downloading data...')
mat_content = sio.loadmat('assets/face.mat')
face_data = mat_content['X']
print('Done!!')

print('Splitting data into training and test sets...')
face_data_training, face_data_testing = data_partition_train_test(face_data, n_pp_train=n_fpp_train)
id_memory = np.zeros(n_p * n_fpp_train, dtype=int)
for i in range(0, n_p):
    id_memory[i*n_fpp_train:(i+1)*n_fpp_train] = np.ones(n_fpp_train, dtype=int) * i
print('Done!!')

print('Building PCA...')
# face_data_training_proj_pca, W_pca, mu_pca = pca()
print('Done!!')
print('Testing PCA...')
# test_pca_lda(face_data_testing, id_memory, W_pca, mu_pca)...
print('Done!')

face_data_training_proj_pca_lda, w_pca_lda, mu_pca_lda = pca_lda(face_data_training, id_memory)
test_pca_lda(face_data_testing, id_memory, face_data_training_proj_pca_lda, w_pca_lda, mu_pca_lda)
