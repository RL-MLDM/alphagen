import numpy as np
import scipy
from scipy import linalg
import time

import sys
args = sys.argv
if len(args) == 4:
    nparams = int(args[1])
    nsamples = int(args[2])
    ntries = int(args[3])
else:
    nparams = 100
    nsamples = 1000
    ntries = 100

def mat_mult_benchmark(nparams, nsamples):
    # create data
    X = 100.0 * np.random.rand(nsamples, nparams) - 50.0
    X[:,0] = 1.0
    X_pinv = scipy.linalg.pinv(X)
    X_pinv_32 = X_pinv.astype(np.float32, copy=True)
    y = 100.0 * np.random.rand(nsamples) - 50.0
    # do least squares
    tls = time.time()
    beta = scipy.linalg.lstsq(X, y)
    tls = time.time() - tls
    # use pinv with dot
    tdot = time.time()
    beta = np.dot(X_pinv, y)
    tdot = time.time() - tdot
    # use pinv with matmul
    tmatmul = time.time()
    beta = np.matmul(X_pinv, y)
    tmatmul = time.time() - tmatmul
    # use pinv with matmul and float32
    tmatmul32 = time.time()
    y_32 = y.astype(np.float32, copy=True)
    beta = np.matmul(X_pinv_32, y_32)
    tmatmul32 = time.time() - tmatmul32
    # print results
    print("pinv-dot: ", tls/tdot, "x; pinv-matmul: ", tls/tmatmul,
          "x; pinv-matmul32:", tls/tmatmul32, "x")

for i in range(ntries):
    mat_mult_benchmark(nparams, nsamples)
