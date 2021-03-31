#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Created on Tue May 30 18:09:40 2017

@author: ziko
"""
from __future__ import print_function,division
import sys
import numpy as np
import math
from sklearn.metrics.pairwise import euclidean_distances as ecdist
import multiprocessing
from sklearn.neighbors import NearestNeighbors

import bound_update as bound
import timeit

def normalizefea(X):
    """
    Normalize each row
    """

    feanorm = np.maximum(1e-14,np.sum(X**2,axis=1))
    X_out = X/(feanorm[:,None]**0.5)
    return X_out

def kmeans_update(tmp):
    """
    """
    # print("ID of process running worker: {}".format(os.getpid()))
    X = bound.SHARED_VARS['X_']
    X_tmp = X[tmp, :]
    c1 = X_tmp.mean(axis = 0)

    return c1

def kmeans_update_soft(S_k):
    """
    """
    # print("ID of process running worker: {}".format(os.getpid()))
    X = bound.SHARED_VARS['X_']
    # X_tmp = X[tmp, :]
    # c1 = X_tmp.mean(axis = 0)
    c1 = np.dot(X.transpose(),S_k)
    c1 = c1/S_k.sum()
    return c1

def MS(X,s,tmp,c0,tol,maxit):
    """
    Mean-shift iteration until convergence
    """
    # print ('inside meanshift iterations.')
    for i in range(maxit):
        Y = ecdist(c0,X[tmp,:],squared=True)
        W = np.exp((-Y)/(2 * s ** 2))
        c1 = np.dot(W,X[tmp,:])/np.sum(W)
        if np.amax(np.absolute(c1-c0))<tol*np.amax(np.absolute(c0)):
            break
        else:
            c0 = c1.copy()
    return c1

def MS_par(slices):
    """
    K-modes in parallel

    """
    s,k = slices
    l,C_s,C_out = bound.get_shared_arrays('l_s','C_s','C_out')
    X = bound.SHARED_VARS['X_']
    tmp=np.asarray(np.where(l==k))
    if tmp.size !=1:
        tmp = tmp.squeeze()
    else:
        tmp = tmp[0]
    C_out[[k],:] = MS(X,s,tmp,C_s[[k],:],1e-5, int(1e3))

def KM_par(slices):

    """
    Mode using definition m_l = \max(x_p in X) \sum_q k(x_p,x_q) in parallel for each cluster

    """

    s,k = slices
    # print('Inside parallel wth ' +repr(k) + 'and sigma '+repr(s))
    l,C_out = bound.get_shared_arrays('l_s','C_out')
#    X =np.memmap('X_MNIST_gan.dat',dtype='float32',mode='c',shape=(70000,256))
    X = bound.SHARED_VARS['X_']
    tmp=np.asarray(np.where(l== k))
    tmp_size = tmp.size
    if tmp_size !=1:
        tmp = tmp.squeeze()
    else:
        tmp = tmp[0]
    # Using Gaussian Filtering
#    s =0.5
    size_limit = 25000
    if tmp_size>size_limit:
        batch_size = 1024
        Deg = []
        num_batch = int(math.ceil(1.0*tmp_size/batch_size))
        for batch_idx in range(num_batch):
            start = batch_idx*batch_size
            end = min((batch_idx+1)*batch_size, tmp_size)
            pairwise_dists = ecdist(X[tmp[start:end]],squared =True)
            W = np.exp(-pairwise_dists/(2* (s ** 2)))
            np.fill_diagonal(W,0)
            Deg_batch = np.sum(W,axis=1).tolist()
            Deg.append(Deg_batch)
        m = max(Deg)
        ind = Deg.index(m)
        mode_index = tmp[ind]
        C_out[[k],:] = X[[tmp[ind]],:]
    else:
        pairwise_dists = ecdist(X[tmp,:],squared =True)
        W = np.exp(-pairwise_dists/(2* (s ** 2)))
        np.fill_diagonal(W,0)
        Deg = np.sum(W,axis=1)
        ind = np.argmax(Deg)
        mode_index = tmp[ind]
        C_out[[k],:] = X[[tmp[ind]],:]

    return mode_index

def km_le(X,M,assign,sigma):

    """
    Discretize the assignments based on center

    """
    e_dist = ecdist(X,M)
    if assign == 'gp':
        g_dist =  np.exp(-e_dist**2/(2*sigma**2))
        l = g_dist.argmax(axis=1)
        energy = compute_km_energy(l,g_dist.T)
        # print('Energy of Kmode = ' + repr(energy))
    else:
        l = e_dist.argmin(axis=1)

    return l

def compute_km_energy(l,dist_c):
    """
    compute K-modes energy
    """
#    dist_c_sum = np.sum(dist_c,axis=1)
    E = 0.0;
    for k in range(len(dist_c)):
        tmp = np.asarray(np.where(l== k)).squeeze()
        E -= np.sum(dist_c[k,tmp])
    return E

def Laplacian_term(W,Z_k):
    Lap_term = np.dot(Z_k, W.dot(Z_k))
    return Lap_term

def compute_energy_lapkmode_cont(X, C, Z, l, W, sigma, bound_lambda, method='kmeans'):

    """
    compute Laplacian K-modes energy

    """
    e_dist = ecdist(X,C,squared =True)
    K = C.shape[0]

    if method =='kmeans':
        clustering_E = (Z*e_dist).sum()

    elif method in ['MS','BO']:
        g_dist =  np.exp(-e_dist/(2*sigma**2))
        clustering_E = (-(Z*g_dist)).sum()

    E_lap = [Laplacian_term(W,Z[:,k]) for k in range(K)]
    E_lap = (bound_lambda*sum(E_lap)).sum()

    E = clustering_E - E_lap

    return E

def estimate_median_sigma(sample,knn):
    # compute distances of the nearest neighbors
    nbrs = NearestNeighbors(n_neighbors=knn).fit(sample)
    distances, _ = nbrs.kneighbors(sample)
    # return the median distance
    return np.median(distances[:,knn-1])

def estimate_sigma(X,W,knn,N):
    if N>70000:
        batch_size = 4560
        num_batch = int(math.ceil(1.0*X.shape[0]/batch_size))
        sigma_square = 0
        for batch_A in range(num_batch):
            start1 = batch_A*batch_size
            end1 = min((batch_A+1)*batch_size, N)
            for batch_B in range(num_batch):
                start2 = batch_B*batch_size
                end2 = min((batch_B+1)*batch_size, N)
                # print("start1 = %d|start2 = %d"%(start1,start2))
                pairwise_dists = ecdist(X[start1:end1],X[start2:end2],squared =True)
                W_temp = W[start1:end1,:][:,start2:end2]
                sigma_square = sigma_square+(W_temp.multiply(pairwise_dists)).sum()
                # print (sigma_square)
        sigma_square = sigma_square/(knn*N)
        sigma = np.sqrt(sigma_square)
    else:
        pairwise_dists = ecdist(X,squared =True)
        sigma_square = W.multiply(pairwise_dists).sum()
        sigma_square = sigma_square/(knn*N)
        sigma = np.sqrt(sigma_square)
    return sigma

def SLK(X, W, C, shot_size, lmd, method = 'BO', lap_bound = True, sigma = None):

    """
    Proposed SLK clustering for few-shot learning

    """
    if lap_bound==False:
        lmd=0.0

    support_labels = []
    for i, count in enumerate(shot_size):
        temp = [i]*count
        support_labels.extend(temp)

    K = C.shape[0]
    N,D = X.shape
    start_time = timeit.default_timer()
    if sigma is None:
        sigma = estimate_sigma(X,W,5,N)

    l = km_le(X,C,None,None)
    l[:len(support_labels)] = support_labels
    assert len(np.unique(l)) == K
    krange = list(range(K))
    srange = [sigma] * K
    bound.init(X_ = X)
    bound.init(C_out=bound.new_shared_array([K,D], C.dtype))
    for i in range(100):
        oldC = C.copy()
        oldl = l.copy()
        bound.init(C_s = bound.n2m(C))
        bound.init(l_s = bound.n2m(l))

        if K<5:
            pool = multiprocessing.Pool(processes=K)
        else:
            pool = multiprocessing.Pool(processes=5)

        if method == 'MS':
            # print('Inside meanshift update . ... ........................')
            pool.map(MS_par,zip(srange,krange))
            _,C = bound.get_shared_arrays('l_s','C_out')
            sqdist = ecdist(X,C,squared=True)
            unary = np.exp((-sqdist)/(2 * sigma ** 2))
            a_p = -unary

        elif method == 'kmeans':
            # print ('Inside mean update')
            tmp_list = [np.where(l==k)[0] for k in range(K)]

            C_list = pool.map(kmeans_update, tmp_list)
            C = np.asarray(np.vstack(C_list))
            a_p = ecdist(X,C,squared=True)

        elif method not in ['BO','MS','KM','kmeans','kernelcut']:
            print(' Error: Give appropriate method from MS/BO/KM/kmeans')
            sys.exit(1)

        pool.close()
        pool.join()
        pool.terminate()
        # Bound Update
        if lap_bound == True or lmd!=0.0:
            bound_iterations = 600
            Y = bound.normalize(-a_p)
            # Y[:len(support_labels), :] = 0
            # Y[np.arange(len(support_labels)), support_labels] = 1
            batch = True if X.shape[0]>100000 else False
            if method == 'BO':
                l,C,Z= bound.bound_update(a_p, Y, X, W, lmd, bound_iterations, batch)
            else:
                l,_,Z= bound.bound_update(a_p, Y, X, W, lmd, bound_iterations, batch)

        else:
            if method == 'MS':
                l = km_le(X,C,str('gp'),sigma)
                Z = bound.get_S_discrete(l,N,K)
            else:
                l = km_le(X,C,None,None)
                Z = bound.get_S_discrete(l,N,K)


        # Laplacian K-prototypes Energy

        currentE = compute_energy_lapkmode_cont(X, C, Z, l, W, sigma, lmd, method=method)
        # print('Laplacian K-prototype Energy is = {:.5f}'.format(currentE))

        # Convergence based on Laplacian K-prototypes Energy
        if (i>1 and (abs(currentE-oldE)<= 1e-5*abs(oldE))):
            # print('......Job  done......')
            break

        else:
            oldE = currentE.copy()


    elapsed = timeit.default_timer() - start_time
    # print(elapsed)
    l = l[len(support_labels):]
    return C,l
