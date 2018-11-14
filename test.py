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
    print('Precision is %.2f%%' % (100 * prec))

    conf_mat(y_actu, y_pred)


def test_machine(data_test, machine, n_p=52):
    rows, cols = data_test.shape
    n_pp_test = int(cols / n_p)
    mach_size = len(machine)

    p_id_pred_ar = np.zeros(mach_size, dtype=int)
    prec = 0
    eps = np.zeros((mach_size, cols))
    y_actu, y_pred = np.zeros(cols, dtype=int), np.zeros(cols)

    for j in range(0, cols):
        test = data_test[:, j]

        y_actu[j] = p_id = j // n_pp_test

        for t in range(0, mach_size):
            test_proj = machine[t].w.T.dot((test - machine[t].mu)[:, None])
            indices = knn(test_proj, machine[t].data_train_proj)
            p_id_pred_ar[t] = mode(machine[t].data_id_memory[indices])[0]

            if p_id != p_id_pred_ar[t]:
                eps[t, j] += 1

        y_pred[j] = p_id_predicted = mode(p_id_pred_ar)[0]

        if p_id == p_id_predicted:
            prec += 1

    prec /= cols
    print('Precision is %.2f%%' % (100 * prec))

    e_av = np.mean(np.mean(np.square(eps), axis=1))
    e_com = np.mean(np.square(np.mean(eps, axis=0)))

    print('The average error of each machine member by acting individually is Eav = %.2f' % e_av)
    print('The expected error of the whole machine is Ecom = %.2f' % e_com)
    if e_com <= e_av:
        print('We have Ecom <= Eav \n -- Success!! Machine performs better than individual members. Good teamwork!')
    else:
        print('We have Ecom > Eav \n -- Failure... You need to review your teamwork.')

    conf_mat(y_actu, y_pred)


def test_mmachine(data_test, *mmachs, n_p=52, fusion='vote'):
    rows, cols = data_test.shape
    n_pp_test = int(cols / n_p)
    mmach_size = len(mmachs)
    p_id_pred_mtrx = np.zeros((mmach_size, n_p), dtype=int)

    prec = 0
    eps = np.zeros((mmach_size, cols))
    y_actu, y_pred = np.zeros(cols, dtype=int), np.zeros(cols)

    print('Fusion scheme is \'' + str(fusion) + '\'')

    for j in range(0, cols):
        test = data_test[:, j]

        y_actu[j] = j // n_pp_test

        for mach in mmachs:
            mach_size = len(mach)

            p_id_pred_ar = np.zeros(mach_size, dtype=int)

            for t in range(0, mach_size):
                test_proj = mach[t].w.T.dot((test - mach[t].mu)[:, None])
                indices = knn(test_proj, mach[t].data_train_proj)
                p_id_pred_ar[t] = mode(mach[t].data_id_memory[indices])[0]

            i = 0
            for k in range(0, n_p):
                k_set = np.transpose(np.argwhere(p_id_pred_ar == k))[0]
                p_id_pred_mtrx[i, k] = k_set.size
            i += 1

        if fusion == 'vote':
            y_pred[j] = mode(p_id_pred_mtrx)[0]
        elif fusion == 'prod':
            for i in range(0, mmach_size):
                pp = p_id_pred_mtrx[:()]

            y_pred[j] = p_id_pred_mtrx.max()
        elif fusion == 'sum':
            pass
        else:
            pass

        if y_actu[j] == y_pred[j]:
            prec += 1

    prec /= cols
    print('Precision is %.2f%%' % (100 * prec))

    e_mach_avg = np.mean(np.mean(np.square(eps), axis=1))
    e_mmach = np.mean(np.square(np.mean(eps, axis=0)))

    print('The average error of each machine member by acting individually is Emach = %.2f' % e_mach_avg)
    print('The expected error of the whole master machine is Emach+ = %.2f' % e_mmach)
    if e_mmach <= e_mach_avg:
        print('We have Emach+ <= Emach \n -- Success!! Master Machine performs better than individual machines. '
              'Great teamwork!')
    else:
        print('We have Emach+ > Emach \n -- Failure... You need to review your teamwork.')

    conf_mat(y_actu, y_pred)