# @Time     : Jan. 10, 2019 15:15
# @Author   : Veritas YIN
# @FileName : math_utils.py
# @Version  : 1.0
# @IDE      : PyCharm
# @Github   : https://github.com/VeritasYin/Project_Orion

import numpy as np

def z_inverse(x, mean, std):
    '''
    The inverse of function z_score().
    :param x: np.ndarray, input to be recovered.
    :param mean: float, the value of mean.
    :param std: float, the value of standard deviation.
    :return: np.ndarray, z-score inverse array.
    '''
    return x * std + mean


def MAPE(v, v_):
    '''
    Mean absolute percentage error.
    :param v: np.ndarray or int, ground truth.
    :param v_: np.ndarray or int, prediction.
    :return: int, MAPE averages on all elements of input.
    '''
    return np.mean(np.abs(v_ - v) / (v + 1e-5))


def RMSE(v, v_):
    '''
    Mean squared error.
    :param v: np.ndarray or int, ground truth.
    :param v_: np.ndarray or int, prediction.
    :return: int, RMSE averages on all elements of input.
    '''
    return np.sqrt(np.mean((v_ - v) ** 2))


def MAE(v, v_):
    '''
    Mean absolute error.
    :param v: np.ndarray or int, ground truth.
    :param v_: np.ndarray or int, prediction.
    :return: int, MAE averages on all elements of input.
    '''
    return np.mean(np.abs(v_ - v))


def evaluation(y, y_, data_stats):
    '''
    Evaluation function: interface to calculate MAPE, MAE and RMSE between ground truth and prediction.
    Extended version: multi-step prediction can be calculated by self-calling.
    :param y: np.ndarray or int, ground truth.
    :param y_: np.ndarray or int, prediction.
    :param x_stats: dict, paras of z-scores (mean & std).
    :return: np.ndarray, averaged metric values.
    '''
    dim = len(y_.shape)

    if dim == 3:
        # single_step case
        v = z_inverse(y, data_stats['mean'], data_stats['std'])
        v_ = z_inverse(y_, data_stats['mean'], data_stats['std'])
        return np.array([MAPE(v, v_), MAE(v, v_), RMSE(v, v_)])
    else:
        # multi_step case
        tmp_list = []
        # y -> [time_step, batch_size, n_route, 1]

        y = np.swapaxes(y, 0, 1)
        y_ = np.swapaxes(y_, 0,1)

        # recursively call
        for i in range(0, y_.shape[0], 3):
            tmp_res = evaluation(y[i], y_[i], data_stats)
            # tmp_list.append(tmp_res)
            print(f'Time Step {i + 1}: MAPE {tmp_res[0]:7.3%}; MAE  {tmp_res[1]:4.3f}; RMSE {tmp_res[2]:6.3f}.')
        # return np.concatenate(tmp_list, axis=-1)
