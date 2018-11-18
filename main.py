import scipy.io as sio
import numpy as np
from functions import data_partition_train_test, data_train_subspace
from model import pca, pca_lda, CommSubmod, RandsmpSubmod
from test import test_pca_lda, test_machine, test_mmachine
from yaml import load
from timeit import default_timer as timer


# Get information from conf.yml in "cfgs" folder to assign parameters.
print('Setting things up...')
with open('cfgs/conf.yml', 'r') as ymlfile:
    cfg = load(ymlfile)
for section in cfg:
    for attr in section.items():
        if attr[0] == 'BASE':
            n_p = attr[1].get('n_p')
            n_fpp = attr[1].get('n_fpp')
            n_fpp_train = attr[1].get('n_fpp_train')
            m_pca = attr[1].get('m_pca')
            m_lda = attr[1].get('m_lda')
        elif attr[0] == 'COMM':
            n_t = attr[1].get('n_fpp_t') * n_p
            t_t = attr[1].get('T')
        elif attr[0] == 'RANDSMP':
            m0 = attr[1].get('m0')
            m1 = attr[1].get('m1')
            t_r = attr[1].get('R')
        elif attr[0] == 'MMACH':
            fusion = attr[1].get('fusion')
print('Done!!')
# Load face data from face.mat in "assets" folder
print('Downloading data...')
mat_content = sio.loadmat('assets/face.mat')
face_data = mat_content['X']
print('Done!!')

# Split the face data into training and testing sets based on conf.yml.
print('Splitting data into training and test sets...')
face_data_training, face_data_testing = data_partition_train_test(face_data, n_pp_train=n_fpp_train)
id_memory = np.zeros(n_p * n_fpp_train, dtype=int)  # Array to remember the identity of each face in face_data_training.
for i in range(0, n_p):
    id_memory[i*n_fpp_train:(i+1)*n_fpp_train] = np.ones(n_fpp_train, dtype=int) * i
print('Done!!')

# Build PCA based model with training data.
# Measure time to build.
# Then test model with testing data.
print('Building PCA...')
start = timer()
face_data_training_pj_pca, u_pca, mu_pca = pca(face_data_training, m_pca=m_pca)
end = timer()
print('Time: %.2f seconds' % float(end - start))
print('Done!')
print('Testing PCA...')
test_pca_lda(face_data_testing, id_memory, face_data_training_pj_pca, u_pca, mu_pca)
print('Done!')

# Build PCA-LDA based model with training data.
# Measure time to build.
# Then test model with testing data.
print('Building PCA-LDA Ensemble...')
start = timer()
face_data_training_pj_pca_lda, w_pca_lda, mu_pca_lda = pca_lda(face_data_training, id_memory, m_pca=m_pca, m_lda=m_lda)
end = timer()
print('Time: %.2f seconds' % float(end - start))
print('Done!')
print('Testing PCA-LDA Ensemble...')
test_pca_lda(face_data_testing, id_memory, face_data_training_pj_pca_lda, w_pca_lda, mu_pca_lda)
print('Done!')

# Build Committee Machine with "bagging" technique on training data.
# Measure time to build.
# Then test model with testing data.
print('Building Committee Machine...')
start = timer()
comm = []
for i in range(0, t_t):
    face_data_training_sub, id_memory_sub = data_train_subspace(face_data_training, n_fpp_train, n_t)
    submod = CommSubmod(i, face_data_training_sub, id_memory_sub, n_p)
    submod.setup()
    comm.append(submod)
end = timer()
print('Time: %.2f seconds' % float(end - start))
print('Done!')
print('Testing Committee Machine...')
test_machine(face_data_testing, comm, n_p)
print('Done!')

# Build Random Feature Sampling Machine with randomisation in feature space.
# Use training data.
# Measure time to build.
# Then test model with testing data.
print('Building Random Feature Sampling Machine...')
start = timer()
randsmp = []
for i in range(0, t_r):
    submod = RandsmpSubmod(i, face_data_training, id_memory, n_p, m0, m1)
    submod.setup()
    randsmp.append(submod)
end = timer()
print('Time: %.2f seconds' % float(end - start))
print('Done!')
print('Testing Random Feature Sampling Machine...')
test_machine(face_data_testing, randsmp, n_p)
print('Done!')

# Combine Committee Machine and Random Feature Sampling Machine.
# Measure time to build (fast because both already built).
# Then test model with testing data.
print('Building Master Machine...')
start = timer()
master = (comm, randsmp)
end = timer()
print('Time: %.2f seconds' % float(end - start))
print('Done!')
print('Testing Master Machine...')
test_mmachine(face_data_testing, *master, n_p=n_p, fusion=fusion)
print('Done!')

print('All Done!!!')
