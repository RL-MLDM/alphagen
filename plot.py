from matplotlib import pyplot as plt
import json

FILENAME_TRAIN = '/DATA/xuehy/preload/zz300_static_20160101_20181231.json'
FILENAME_VALIDATION = '/DATA/xuehy/preload/zz300_static_20190101_20211231.json'

if __name__ == '__main__':
    with open(FILENAME_TRAIN, encoding='utf-8') as f:
        cache_train = json.load(f)['valid'].values()
    with open(FILENAME_VALIDATION, encoding='utf-8') as f:
        cache_validation = json.load(f)['valid'].values()

    plt.hist(cache_train, 20000, range=[-1, 1],
             # density=True,
             cumulative=-1,
             color='green',
             histtype='step',
             label='training set(20160101-20181231)',
             alpha=1.)
    plt.hist(cache_validation, 20000, range=[-1, 1],
             # density=True,
             cumulative=-1,
             color='blue',
             histtype='step',
             label='test set(20190101-20211231)',
             alpha=1.)

    plt.xlabel('threshold(20D spearman corr)')
    plt.ylabel('#(corr>threshold)')
    plt.xlim(0.0, 0.095)
    plt.ylim(1, 100000)
    plt.yscale('log')
    plt.legend(loc='lower left')
    plt.show()
