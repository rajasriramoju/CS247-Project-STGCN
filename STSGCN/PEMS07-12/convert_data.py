import numpy as np 
import pandas as pd

def weight_matrix(W, sigma2=0.1, epsilon=0.5, scaling=True):
    if set(np.unique(W)) == {0, 1}:
        print('The input graph is a 0/1 matrix; set "scaling" to False.')
        scaling = False

    if scaling:
        n = W.shape[0]
        W_scale = W / 10000.
        W2, W_mask = W_scale * W_scale, np.ones([n, n]) - np.identity(n)
        # refer to Eq.10
        return np.exp(-W2 / sigma2) * (np.exp(-W2 / sigma2) >= epsilon) * W_mask
    else:
        return W

v = pd.read_csv('V_228_new.csv', index_col=False, header=None).to_numpy()
w = pd.read_csv('W_228_new.csv', index_col=False, header=None).to_numpy()

w = weight_matrix(w)
v = np.expand_dims(v,2)

graph_signal_matrix = np.savez_compressed('PEMS07-12', data=v)

tmp = []
for from_node in range(w.shape[0]):
    for to_node in range(w.shape[1]):
        if w[from_node][to_node] != 0.0:
            tmp.append([int(from_node), int(to_node), w[from_node][to_node]])

res = pd.DataFrame(np.array(tmp))
res.columns = ['from', 'to', 'distance']
res = res.astype({'from': int, 'to': int})
res.to_csv('PEMS07-12.csv', header=['from', 'to', 'distance'], index=False)

indices = np.arange(w.shape[0])
indices = pd.DataFrame(indices)
indices.to_csv('PEMS07-12.txt', header=None, index=False)