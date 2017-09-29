"""Trains the model, given data."""

import numpy as np
from numpy import matmul as mx
import pca
import sys
import os

out_dir = sys.argv[2] + '/'
def train(Y, Yt, d = 10, h = 5, max_iter = 100):
    
    S = {}
    for u in Yt:
        S[u] = np.random.randn(h,1) #random normal h x 1

    F = {}
    for i in Y:
        F[i] = np.random.randn(d,h)

    for ep in range(max_iter):
        print('ep ' + str(ep) + '\tSSD = ' + str(ss_err(Y, F, S)[0]))
        nextF = solveF(Y, S)
        F = nextF

        nextS = solveS(Yt, F)
        S = nextS

    return (F,S)

def solveF(Y, S):
    F = {}
    
    for i in Y:
        Sit = []
        Yit = []
        for u in Y[i]:
            Sit.append(S[u])
            Yit.append(Y[i][u])

        Sit = np.concatenate(Sit, axis=1)
        Yit = np.concatenate(Yit, axis=1)
            
        Si = Sit
        Yi = Yit

        #print(i)
        #print(Si.shape)
        #print(Yi.shape)
        Fi = mx(mx( Yi, np.transpose(Si)), np.linalg.inv(mx(Si, np.transpose(Si)))) #needs to use matmul i believe

        F[i] = Fi

    return F

def solveS(Yt, F):
    h = F[F.keys()[0]].shape[1]
    #print('d = ' + str(d))
    
    S = {}
    
    for u in Yt:
        Su = np.zeros((h,1))
        Isize = 0

        for i in Yt[u]:
            d = F[i].shape[0]

            if h <= d:
                tv = mx(np.transpose(F[i]), F[i])
                tv = np.linalg.inv(tv)
                tv = mx(tv, np.transpose(F[i]))
                tv = mx(tv, Yt[u][i])
                Su = Su + tv
            else:
                tv = np.transpose(F[i])
                tv = mx(tv, np.linalg(mx(np.transpose(F[i]), F[i])))
                tv = mx(tv, Yt[u][i])
                Su = Su + tv
                #Su = Su + mx(mx(np.linalg.inv(mx(np.transpose(F[i]), F[i])), np.transpose(F[i])), Yt[u][i])
            Isize += 1

        Su = Su / Isize

        S[u] = Su

    return S
        

def randomize_data(Y, Yt):
    try:
        Z, Zt = yread(out_dir+'pickled/tdat/random1')
    except:
        sigma = calc_std(Y)
    
        Z = {}
        Zt = {}
        for i in Y:
            for u in Y[i]:
                if i not in Z:
                    Z[i] = {}
                if u not in Zt:
                    Zt[u] = {}

                d = Y[i][u].shape[0]
                Z[i][u] = sigma*np.random.randn(d,1)
                Zt[u][i] = Z[i][u]
        ywrite(Z, Zt, out_dir+'pickled/tdat/random1')
    return (Z, Zt)

def train_basic(Y, Yt, d=10, h=5, max_iter = 1000):
    
    F = {}
    S = {}
    
    for i in Y:
        F[i] = np.random.randn(d, h)
        F[i] = np.zeros((d, h))
    for u in Yt:
        S[u] = np.random.randn(h, 1)
        S[u] = np.zeros((h,1))

    return (F,S)
            
    scale = 1.0
    dscale = 0.1
    momentum = 0
    SSD_old = ss_err(Y, F, S)

    for ep in range(max_iter):
        tscale = scale + dscale

        SSD = ss_err(Y, F, S, tscale)
        if SSD < SSD_old:
            momentum += 0.1*dscale
            dscale += momentum
        else:
            dscale = -0.1*dscale
            momentum = 0

        SSD_old = SSD

    for i in F:
        F[i] = scale * F[i]
    for u in S:
        S[u] = scale * S[u]

    return (F,S)


def ss_err(Y, F, S, scale = 1.):
    err = 0.

    for i in Y:
        for u in Y[i]:
            R = Y[i][u] - scale*mx(F[i], S[u])
            err += sum(R**2)

    return err

def calc_std(Y):
    X = []
    for i in Y:
        for u in Y[i]:
            for k in range(len(Y[i][u])):
                X.append(Y[i][u][k])
    return np.std(X)

def main():
    if len(sys.argv) <= 2:
        print('not enough cmd args')
        exit()
        
    Y, Yt = pca.run_pca()
    #print('got data')
    Z, Zt = randomize_data(Y, Yt)

    '''h = 5
    F, S = train(Y, Yt, 10, h)
    FZ, SZ = train(Z, Zt, 10, h)

    pwrite(F, S, sys.argv[2]+'total_model_10_'+str(h))
    pwrite(FZ, SZ, sys.argv[2]+'total_randomized_10_'+str(h))

    F_basic, S_basic = train_basic(Y, Yt, 10, h)
    FZ_basic, SZ_basic = train_basic(Z, Zt, 10, h)

    pwrite(F_basic, S_basic, sys.argv[2]+'basic_model_10_'+str(h))
    pwrite(FZ_basic, SZ_basic, sys.argv[2]+'basic_randomized_10_'+str(h))'''

     
    x = range(0,10)
    for h in x:
        try:
            a = pread(out_dir+'pickled/models/total_model/total_model_10_' + str(h))
            #b = pread( out_dir+'pickled/models/total_randomized/total_randomized_10_' + str(h))
            c = pread(out_dir+'pickled/models/basic_model/basic_model_10_'+str(h))
            #d = pread( out_dir+'pickled/models/basic_randomized/basic_randomized_10_'+str(h))
        except:
            try:
                F, S = train(Y, Yt, 10, h)
                #FZ, SZ = train(Z, Zt, 10, h)

                pwrite(F, S, out_dir+'pickled/models/total_model/total_model_10_' + str(h))
                #pwrite(FZ, SZ, out_dir+'pickled/models/total_randomized/total_randomized_10_' + str(h))

                F_basic, S_basic = train_basic(Y, Yt, 10, h)
                #FZ_basic, SZ_basic = train_basic(Z, Zt, 10, h)

                pwrite(F_basic, S_basic, out_dir+'pickled/models/basic_model/basic_model_10_'+str(h))
                #pwrite(FZ_basic, SZ_basic, out_dir+'pickled/models/basic_randomized/basic_randomized_10_'+str(h)) 
            except:
                print('Singular matrix: ' + str(h))

    train_random()
if __name__ == '__main__':
    main()
