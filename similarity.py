import json

import numpy as np
import torch
from scipy.cluster.hierarchy import linkage, dendrogram
from tqdm import tqdm
import matplotlib.pyplot as plt

from alphagen.utils.correlation import batch_spearman
from alphagen_qlib.stock_data import StockData
from assets import ZZ300_2019
from validation import build_expr_tree


def corr(factor1: str, factor2: str, data: StockData) -> float:
    expr1, expr2 = build_expr_tree(factor1), build_expr_tree(factor2)
    data1, data2 = expr1.evaluate(data), expr2.evaluate(data)
    corrs = batch_spearman(data1, data2)
    ret = corrs.mean().item()
    return ret


FILENAME = '/DATA/xuehy/preload/zz300_static_20190101_20211231.json'

if __name__ == '__main__':
    INSTRUMENT = ZZ300_2019
    START_TIME = '2019-01-01'
    END_TIME = '2021-12-31'
    data = StockData(INSTRUMENT, START_TIME, END_TIME, device=torch.device('cuda:0'))

    with open(FILENAME, encoding='utf-8') as f:
        cache = json.load(f)

    sorted_values = sorted(cache['valid'].values())
    threshold2 = 1.
    threshold1 = 0.0535
    tops = [key for key in cache['valid'] if threshold1 <= cache['valid'][key] < threshold2]
    datas = [build_expr_tree(factor).evaluate(data) for factor in tops]
    n = len(tops)

    corr_mat = np.zeros((n, n))
    for i in tqdm(range(n)):
        for j in range(i+1):
            corr_mat[i][j] = corr_mat[j][i] = batch_spearman(datas[i], datas[j]).mean().item()


    def dist(i, j):
        return 1. - corr_mat[int(i)][int(j)]


    linkage_data = linkage(np.array([[i] for i in range(n)]), method='complete', metric=dist)
    dendrogram(linkage_data)

    plt.xlabel('alpha index')
    plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], [1.0, 0.8, 0.6, 0.4, 0.2, 0.0])
    plt.ylabel('clustering threshold(spearman corr)')
    plt.show()
