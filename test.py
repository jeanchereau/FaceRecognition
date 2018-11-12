import numpy as np
from scipy.stats import mode
from functions import conf_mat, knn
from model import comm_submod


def test_pca_lda(data_test, data_id_memory, data_train_proj, w, mu, n_p=52):
    rows, cols = data_test.shape
    n_pp_test = int(cols / n_p)

    prec = 0
    y_actu, y_pred = np.zeros(cols), np.zeros(cols)

    for j in range(0, cols):
        test = data_test[:, j]

        test_proj = w.T.dot((test - mu).reshape(rows, 1))

        index = knn(test_proj, data_train_proj)

        y_pred[j] = p_id_predicted = data_id_memory[index]
        y_actu[j] = p_id = j // n_pp_test

        if p_id == p_id_predicted:
            prec += 1

    prec /= cols
    print('Precision is %.2f' % prec)

    conf_mat(y_actu, y_pred)


# TODO
#def test_comm(data_test, sub_models, n_p=52):
#    rows, cols = data_test.shape
#    n_pp_test = int(cols / n_p)
#
#    p_id_predicted_ar = np.zeros(T, dtype=int)
#    prec = 0
#    eps = np.zeros((T, cols))
#    y_actu, y_pred = np.zeros(n_p * n_pp_test, dtype=int), np.zeros(cols)
#
#    for j in range(0, cols):
#        test = data_test[:, j]
#
#        p_id = j // n_pp_test
#
#        for model in sub_models:
#            test_proj = model.w.T, (test - model.mu).reshape(rows, 1))
#            indices = knn(test_proj, face_data_training_sub_proj[t])
#            p_id_predicted_ar[t] = mode(id_memory[t, indices])[0]
#
#            if p_id != p_id_predicted_ar[t]:
#                eps[t, j] += 1
#
#        p_id_predicted = mode(p_id_predicted_ar)[0]
#
#        if p_id == p_id_predicted:
#            prec += 1
#
#        y_actu[j] = p_id
#        y_pred[j] = p_id_predicted
#
#    prec /= cols
#    print('Precision is %.2f' % prec)
#
#    Eav = np.mean(np.mean(np.square(eps), axis=1))
#    Ecom = np.mean(np.square(np.mean(eps, axis=0)))
#
#    print('The average error by acting individually is %.2f' % Eav)
#    print('The expected error of the committe machine is %.2f' % Ecom)
#    if Ecom <= Eav:
#        print('We have Ecom <= Eav \n -- Success!!')
#    else:
#        print('We have Ecom > Eav \n -- Failure...')
#
#    conf_mat(y_actu, y_pred)
