import numpy as np
from scipy.stats import mode
from functions import conf_mat, knn


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


def test_machine(data_test, machine, n_p=52):
    rows, cols = data_test.shape
    n_pp_test = int(cols / n_p)
    machine_size = len(machine)

    p_id_predicted_ar = np.zeros(machine_size, dtype=int)
    prec = 0
    eps = np.zeros((machine_size, cols))
    y_actu, y_pred = np.zeros(n_p * n_pp_test, dtype=int), np.zeros(cols)

    for j in range(0, cols):
        test = data_test[:, j]

        y_actu[j] = p_id = j // n_pp_test

        for t in range(0, machine_size):
            test_proj = machine[t].w.T.dot((test - machine[t].mu)[:, None])
            indices = knn(test_proj, machine[t].data_train_proj)
            p_id_predicted_ar[t] = mode(machine[t].data_id_memory[indices])[0]

            if p_id != p_id_predicted_ar[t]:
                eps[t, j] += 1

        y_pred[j] = p_id_predicted = mode(p_id_predicted_ar)[0]

        if p_id == p_id_predicted:
            prec += 1

    prec /= cols
    print('Precision is %.2f' % prec)

    e_av = np.mean(np.mean(np.square(eps), axis=1))
    e_com = np.mean(np.square(np.mean(eps, axis=0)))

    print('The average error by acting individually is Eav = %.2f' % e_av)
    print('The expected error of the committe machine is Ecom = %.2f' % e_com)
    if e_com <= e_av:
        print('We have Ecom <= Eav \n -- Success!!')
    else:
        print('We have Ecom > Eav \n -- Failure...')

    conf_mat(y_actu, y_pred)
