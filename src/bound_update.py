"""
"""
import os.path as osp
import numpy as np
import math
import matplotlib
matplotlib.use('Agg')
import numexpr as ne
from matplotlib import pyplot as plt
import multiprocessing

SHARED_VARS = {}
SHARED_array = {}
### supporting functions to make parallel updates of clusters

def n2m(a):
   """
   Return a multiprocessing.Array COPY of a numpy.array, together
   with shape, typecode and matrix flag.
   """
   if not isinstance(a, np.ndarray): a = np.array(a)
   return multiprocessing.Array(a.dtype.char, a.flat, lock=False), tuple(a.shape), a.dtype.char, isinstance(a, np.matrix)

def m2n(buf, shape, typecode, ismatrix=False):
   """
   Return a numpy.array VIEW of a multiprocessing.Array given a
   handle to the array, the shape, the data typecode, and a boolean
   flag indicating whether the result should be cast as a matrix.
   """
   a = np.frombuffer(buf, dtype=typecode).reshape(shape)
   if ismatrix: a = np.asmatrix(a)
   return a


def new_shared_array(shape, typecode='d', ismatrix=False):
   """
   Allocate a new shared array and return all the details required
   to reinterpret it as a numpy array or matrix (same order of
   output arguments as n2m)
   """
   typecode = np.dtype(typecode).char
   return multiprocessing.Array(typecode, int(np.prod(shape)), lock=False), tuple(shape), typecode, ismatrix

def get_shared_arrays(*names):
   return [m2n(*SHARED_VARS[name]) for name in names]

def init(*pargs, **kwargs):
   SHARED_VARS.update(pargs, **kwargs)

def normalize(Y_in):
    maxcol = np.max(Y_in, axis=1)
    Y_in = Y_in - maxcol[:, np.newaxis]
    N = Y_in.shape[0]
    size_limit = 150000
    if N > size_limit:
        batch_size = 1280
        Y_out = []
        num_batch = int(math.ceil(1.0 * N / batch_size))
        for batch_idx in range(num_batch):
            start = batch_idx * batch_size
            end = min((batch_idx + 1) * batch_size, N)
            tmp = np.exp(Y_in[start:end, :])
            tmp = tmp / (np.sum(tmp, axis=1)[:, None])
            Y_out.append(tmp)
        del Y_in
        Y_out = np.vstack(Y_out)
    else:
        Y_out = np.exp(Y_in)
        Y_out = Y_out / (np.sum(Y_out, axis=1)[:, None])

    return Y_out

def normalize_2(Y_in):
    Y_in_sum = Y_in.sum(1)[:, np.newaxis]
    Y_in = np.divide(Y_in, Y_in_sum)
    return Y_in


def entropy_energy(Y, unary, kernel, bound_lambda, batch=False):
    tot_size = Y.shape[0]
    pairwise = kernel.dot(Y)
    if batch == False:
        temp = (unary * Y) + (-bound_lambda * pairwise * Y)
        E = (Y * np.log(np.maximum(Y, 1e-20)) + temp).sum()
    else:
        batch_size = 1024
        num_batch = int(math.ceil(1.0 * tot_size / batch_size))
        E = 0
        for batch_idx in range(num_batch):
            start = batch_idx * batch_size
            end = min((batch_idx + 1) * batch_size, tot_size)
            temp = (unary[start:end] * Y[start:end]) + (-bound_lambda * pairwise[start:end] * Y[start:end])
            E = E + (Y[start:end] * np.log(np.maximum(Y[start:end], 1e-20)) + temp).sum()

    return E

def get_S_discrete(l,N,K):
    x = range(N)
    temp =  np.zeros((N,K),dtype=float)
    temp[(x,l)]=1
    return temp

def normalize_2(S_in):
    S_in_sum = S_in.sum(1)[:,np.newaxis]
    # S_in = np.divide(S_in,S_in_sum)
    S_in = ne.evaluate('S_in/S_in_sum')
    return S_in

def plot_convergence(filename, E_list):
    # Plot the convergence of bound updates

    E_list = np.asarray(E_list)
    length = len(E_list)
    iter_range = np.asarray(list(range(length)))
    plt.figure(1, figsize=(6.4, 4.8))
    plt.ion()
    plt.clf()
    ylabel = r'$\mathcal{B}_i(\mathbf{Y})$'
    plt.plot(iter_range, E_list, 'b-', linewidth=2.2)
    plt.xticks(iter_range[1::2],(iter_range[1::2]+1))
    plt.xlabel('iterations')
    plt.ylabel(ylabel)
    plt.savefig(filename, format='png', dpi=800, bbox_inches='tight')
    plt.show()
    plt.close('all')


def bound_update(unary, Y, X, kernel, bound_lambda, bound_iteration=20, batch=False):
    """
    """
    oldE = float('inf')
    # Y = normalize(-unary)
    E_list = []
    for i in range(bound_iteration):
        additive = -unary
        mul_kernel = kernel.dot(Y)
        Y = - bound_lambda * mul_kernel
        additive = additive - Y
        Y = normalize(additive)
        E = entropy_energy(Y, unary, kernel, bound_lambda, batch)
        E_list.append(E)
        # print('entropy_energy is ' +repr(E) + ' at iteration ',i)
        if (i > 1 and (abs(E - oldE) <= 1e-7 * abs(oldE))):
            # print('Converged')
            break

        else:
            oldE = E.copy()

    # if args.plot_converge:
    #   filename = osp.join(args.save_path,'convergence_{}.png'.format(args.arch))
    #   plot_convergence(filename,E_list)

    l = np.argmax(Y, axis=1)
    ind = np.argmax(Y, axis=0)
    C= X[ind,:]
    return l,C,Y