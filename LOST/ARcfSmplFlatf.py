import numpy.polynomial.polynomial as _Npp
import scipy.stats as _ss
import LOST.kdist as _kd
import LOST.ARlib as _arl
import warnings
#import logerfc as _lfc
import LOST.commdefs as _cd
import numpy as _N
#from ARcfSmplFuncs import ampAngRep, randomF, dcmpcff, betterProposal
from LOST.ARcfSmplFuncs import ampAngRep, dcmpcff
#import ARcfSmplFuncsCy as ac
import matplotlib.pyplot as _plt
import time as _tm


def ARcfSmpl(N, k, AR2lims, smpxU, smpxW, q2, R, Cs, Cn, alpR, alpC, TR, accepts=1, prior=_cd.__COMP_REF__, aro=_cd.__NF__, sig_ph0L=-1, sig_ph0H=0):
    ttt1 = _tm.time()

    C = Cs + Cn

    #  I return F and Eigvals

    ujs     = _N.zeros((TR, R, N + 1, 1))
    wjs     = _N.zeros((TR, C, N + 2, 1))

    #  CONVENTIONS, DATA FORMAT
    #  x1...xN  (observations)   size N-1
    #  x{-p}...x{0}  (initial values, will be guessed)
    #  smpx
    #  ujt      depends on x_t ... x_{t-p-1}  (p-1 backstep operators)
    #  1 backstep operator operates on ujt
    #  wjt      depends on x_t ... x_{t-p-2}  (p-2 backstep operators)
    #  2 backstep operator operates on wjt
    #  
    #  smpx is N x p.  The first element has x_0...x_{-p} in it
    #  For real filter
    #  prod_{i neq j} (1 - alpi B) x_t    operates on x_t...x_{t-p+1}
    #  For imag filter
    #  prod_{i neq j,j+1} (1 - alpi B) x_t    operates on x_t...x_{t-p+2}

    ######  COMPLEX ROOTS.  Cannot directly sample the conditional posterior

    Ff  = _N.zeros((1, k-1))
    F0  = _N.zeros(2)
    F1  = _N.zeros(2)
    A   = _N.zeros(2)

    #Xs     = _N.zeros((TR, N-2, 2))
    Xs     = _N.zeros((N-2, 2))
    Ys     = _N.zeros((N-2, 1))
    H      = _N.zeros((TR, 2, 2))
    iH     = _N.zeros((TR, 2, 2))
    mu     = _N.zeros((TR, 2, 1))
    J      = _N.zeros((2, 2))
    Mj     = _N.zeros(TR)
    Mji    = _N.zeros(TR)
    mj     = _N.zeros(TR)

    #  r = sqrt(-1*phi_1)   0.25 = -1*phi_1   -->  phi_1 >= -0.25   gives r >= 0.5 for signal components   
    if aro == _cd.__SF__:    #  signal first
        arInd = range(C)
    else:                    #  noise first
        arInd = range(C-1, -1, -1)

    ttt2 = _tm.time()
    ttt2a = 0
    ttt2b = 0

    for c in arInd:   #  Filtering signal root first drastically alters the strength of the signal root upon update sometimes.  
        # rprior = prior
        # if c >= Cs:
        #    rprior = _cd.__COMP_REF__
        tttA = _tm.time()
        if c >= Cs:
            ph0L = -1
            ph0H = 0
        else:
            ph0L = sig_ph0L   # 
            ph0H = sig_ph0H #  R=0.97, reasonably oscillatory
            
        j = 2*c + 1
        p1a =  AR2lims[c, 0]
        p1b =  AR2lims[c, 1]

        # given all other roots except the jth.  This is PHI0
        jth_r1 = alpC.pop(j)    #  negative root   #  nothing is done with these
        jth_r2 = alpC.pop(j-1)  #  positive root

        #  exp(-(Y - FX)^2 / 2q^2)
        Frmj = _Npp.polyfromroots(alpR + alpC).real
        #print "ampAngRep"
        #print ampAngRep(alpC)

        Ff[0, :]   = Frmj[::-1]
        #  Ff first element k-delay,  Prod{i neq j} (1 - alfa_i B)

        ##########  CREATE FILTERED TIMESERIES   ###########
        ##########  Frmj is k x k, smpxW is (N+2) x k ######

        for m in range(TR):
            _N.dot(smpxW[m], Ff.T, out=wjs[m, c])
            #_N.savetxt("%(2)d-%(1)d" % {"2" : aro, "1" : c}, wjs[m, c])
            #_plt.figure(figsize=(15, 4))
            #_plt.plot(wjs[m, c, 1000:1300])

            ####   Needed for proposal density calculation
            #Ys[:, 0]    = wj[2:N, 0]
            Ys[:]    = wjs[m, c, 2:N]
            #Ys          = Ys.reshape(N-2, 1)
            #Xs[m, :, 0] = wj[1:N-1, 0]   # new old
            #Xs[m, :, 0] = wjs[m, c, 1:N-1, 0]   # new old
            Xs[:, 0] = wjs[m, c, 1:N-1, 0]   # new old
            #Xs[m, :, 1] = wj[0:N-2, 0]
            #Xs[m, :, 1] = wjs[m, c, 0:N-2, 0]
            Xs[:, 1] = wjs[m, c, 0:N-2, 0]
            #iH[m]       = _N.dot(Xs[m].T, Xs[m]) / q2[m]
            iH[m]       = _N.dot(Xs.T, Xs) / q2[m]
            #H[m]        = _N.linalg.inv(iH[m])   #  aka A
            H[m,0,0]=iH[m,1,1]; H[m,1,1]=iH[m,0,0];
            H[m,1,0]=-iH[m,1,0];H[m,0,1]=-iH[m,0,1];
            H[m]         /= (iH[m,0,0]*iH[m,1,1]-iH[m,0,1]*iH[m,1,0])
            #mu[m]        = _N.dot(H[m], _N.dot(Xs[m].T, Ys))/q2[m]
            mu[m]        = _N.dot(H[m], _N.dot(Xs.T, Ys))/q2[m]

        #  
        Ji  = _N.sum(iH, axis=0)
        #J   = _N.linalg.inv(Ji)
        J[0,0]=Ji[1,1]; J[1,1]=Ji[0,0];
        J[1,0]=-Ji[1,0];J[0,1]=-Ji[0,1];
        J         /= (Ji[0,0]*Ji[1,1]-Ji[0,1]*Ji[1,0])

        U   = _N.dot(J, _N.einsum("tij,tjk->ik", iH, mu))


        ##########  Sample *PROPOSED* parameters 

        # #  If posterior valid Gaussian    q2 x H - oNZ * prH00

        ###  This case is still fairly inexpensive.  
        vPr1  = J[0, 0] - (J[0, 1]*J[0, 1])   / J[1, 1]   # known as Mj
        vPr2  = J[1, 1]
        svPr2 = _N.sqrt(vPr2)     ####  VECTORIZE
        svPr1 = _N.sqrt(vPr1)

        #b2Real = (U[1, 0] + 0.25*U[0, 0]*U[0, 0] > 0)
        ######  Initialize F0

        mu1prp = U[1, 0]
        mu0prp = U[0, 0]

        tttB = _tm.time()

        #ph0j2 = _kd.truncnormC(a=ph0L, b=ph0H, u=mu1prp, std=svPr2)
        ph0j2 = _kd.truncnormC(ph0L, ph0H, mu1prp, svPr2)
        r1    = _N.sqrt(-1*ph0j2)
        mj0   = mu0prp + (J[0, 1] * (ph0j2 - mu1prp)) / J[1, 1]
        #ph0j1 = _kd.truncnormC(a=p1a*r1, b=p1b*r1, u=mj0, std=svPr1)
        #print("%(u1).4e   %(s1).4e    %(u2).4e   %(s2).4e" % {"u1" : mu1prp, "s1" : svPr2, "u2" : mj0, "s2" : svPr1})

        #ph0j1 = _kd.truncnormC(p1a*r1, p1b*r1, mj0, svPr1)
        #  p1a, p1b  are close to -2, 2
        ph0j1 = convolve_f_margin_with_normal(p1a*r1, p1b*r1, mj0, svPr1, ph0j2)
        #print("%(nm).4f   %(f).4f" % {"nm" : ph0j1, "f" : ph0j1_f})
        tttC = _tm.time()
        A[0] = ph0j1; A[1] = ph0j2

        #F0[0] = ph0j1; F0[1] = ph0j2

        #  F1 +/- sqrt(F1^2 + 4F1) / 
        img        = _N.sqrt(-(A[0]*A[0] + 4*A[1]))*1j
            
        #alpC.insert(j-1, (A[0] + img)*0.5)
        #alpC.insert(j,   (A[0] - img)*0.5)       #  vals[1] now at end

        #  the positive root comes first
        alpC.insert(j-1, (A[0] - img)*0.5)     #alpC.insert(j - 1, jth_r1)
        alpC.insert(j-1, (A[0] + img)*0.5)     #alpC.insert(j - 1, jth_r2)
        tttC = _tm.time()
        ttt2a += tttB - tttA
        ttt2b += tttC - tttB


    Ff  = _N.zeros((1, k))
    ######     REAL ROOTS.  Directly sample the conditional posterior
    for j in range(R - 1, -1, -1):
        # given all other roots except the jth
        jth_r = alpR.pop(j)

        Frmj = _Npp.polyfromroots(alpR + alpC).real #  Ff first element k-delay
        Ff[0, :] = Frmj[::-1]   #  Prod{i neq j} (1 - alfa_i B)

        ##########  CREATE FILTERED TIMESERIES   ###########
        ##########  Frmj is k x k, smpxU is (N+1) x k ######

        for m in range(TR):
            #uj  = _N.dot(Ff, smpxU[m].T).T
            #_N.dot(Ff, smpxU[m].T, out=ujs[m, j])
            _N.dot(smpxU[m], Ff.T, out=ujs[m, j])
            #ujs.append(uj)

            ####   Needed for proposal density calculation
            
            Mji[m] = _N.dot(ujs[m, j, 0:N, 0], ujs[m, j, 0:N, 0]) / q2[m]
            Mj[m] = 1 / Mji[m]
            mj[m] = _N.dot(ujs[m, j, 1:, 0], ujs[m, j, 0:N, 0]) / (q2[m]*Mji[m])

        #  truncated Gaussian to [-1, 1]
        Ji = _N.sum(Mji)
        J  = 1 / Ji
        U  = J * _N.dot(Mji, mj)

        #  we only want 
        rj = _kd.truncnorm(a=-1, b=1, u=U, std=_N.sqrt(J))

        alpR.insert(j, rj)
    ttt3 = _tm.time()

    # print("--------------------------------")
    # print("t2-t1  %.4f" % (ttt2-ttt1))
    # print("t3-t2  %.4f" % (ttt3-ttt2))
    # print("ttt2a  %.4f" % ttt2a)
    # print("ttt2b  %.4f" % ttt2b)


    return ujs, wjs#, lsmpld


def FilteredTimeseries(N, k, smpxU, smpxW, q2, R, Cs, Cn, alpR, alpC, TR):
    C = Cs + Cn

    #  I return F and Eigvals

    ujs     = _N.zeros((TR, R, N + 1, 1))
    wjs     = _N.zeros((TR, C, N + 2, 1))

    #  CONVENTIONS, DATA FORMAT
    #  x1...xN  (observations)   size N-1
    #  x{-p}...x{0}  (initial values, will be guessed)
    #  smpx
    #  ujt      depends on x_t ... x_{t-p-1}  (p-1 backstep operators)
    #  1 backstep operator operates on ujt
    #  wjt      depends on x_t ... x_{t-p-2}  (p-2 backstep operators)
    #  2 backstep operator operates on wjt
    #  
    #  smpx is N x p.  The first element has x_0...x_{-p} in it
    #  For real filter
    #  prod_{i neq j} (1 - alpi B) x_t    operates on x_t...x_{t-p+1}
    #  For imag filter
    #  prod_{i neq j,j+1} (1 - alpi B) x_t    operates on x_t...x_{t-p+2}

    ######  COMPLEX ROOTS.  Cannot directly sample the conditional posterior

    Ff  = _N.zeros((1, k-1))

    for c in range(C-1, -1, -1):
        j = 2*c + 1

        # given all other roots except the jth.  This is PHI0
        jth_r1 = alpC.pop(j)
        jth_r2 = alpC.pop(j-1)

        #  print jth_r1
        #  exp(-(Y - FX)^2 / 2q^2)
        Frmj = _Npp.polyfromroots(alpR + alpC).real

        Ff[0, :]   = Frmj[::-1]
        #  Ff first element k-delay,  Prod{i neq j} (1 - alfa_i B)

        ##########  CREATE FILTERED TIMESERIES   ###########
        ##########  Frmj is k x k, smpxW is (N+2) x k ######

        for m in range(TR):
            _N.dot(smpxW[m], Ff.T, out=wjs[m, c])
            
        alpC.insert(j-1, jth_r1)
        alpC.insert(j,   jth_r2)

    Ff  = _N.zeros((1, k))
    ######     REAL ROOTS.  Directly sample the conditional posterior
    for j in range(R - 1, -1, -1):
        # given all other roots except the jth
        jth_r = alpR.pop(j)

        Frmj = _Npp.polyfromroots(alpR + alpC).real #  Ff first element k-delay
        Ff[0, :] = Frmj[::-1]   #  Prod{i neq j} (1 - alfa_i B)

        ##########  CREATE FILTERED TIMESERIES   ###########
        ##########  Frmj is k x k, smpxU is (N+1) x k ######

        for m in range(TR):
            _N.dot(smpxU[m], Ff.T, out=ujs[m, j])

        alpR.insert(j, jth_r)

    return ujs, wjs#, lsmpld




def convolve_f_margin_with_normal(phi1Lo, phi1Hi, mj0, svPr1, ph0j2):
    """
    """
    phi2 = ph0j2
    #print("%(L).4f  %(H).4f" % {"L" : phi1Lo, "H" : phi1Hi})
    phi1s = _N.linspace(phi1Lo, phi1Hi, 10001, endpoint=True)
    dx   = phi1s[1] - phi1s[0]
    phi1s_smler = _N.linspace(phi1Lo+0.2*dx, phi1Hi-0.2*dx, 10001, endpoint=True)
    lpr_phi1 = _N.zeros(phi1s.shape[0])
    lpr_phi1[1:-1] = -0.5*(_N.log(-phi2) + _N.log(-phi1s[1:-1]**2 - 4*phi2))
    lpr_phi1[0] = lpr_phi1[1]
    lpr_phi1[-1] = lpr_phi1[-2]
    
    llklhd = -0.5*(phi1s - mj0)*(phi1s - mj0) / (svPr1*svPr1) - 0.5*_N.log(2*_N.pi*svPr1*svPr1)

    #fig = _plt.figure()

    lpostprob = llklhd + lpr_phi1
    maxat = _N.where(lpostprob == _N.max(lpostprob))[0][0]
    lpostprob -= lpostprob[maxat]
    postprob = _N.exp(lpostprob)
    #postprob = pr_phi1

    # fig = _plt.figure()
    # _plt.plot(llklhd)
    # fig = _plt.figure()
    # _plt.plot(lpr_phi1)
    cdf = _N.zeros(lpr_phi1.shape[0]+1)
    # _plt.plot(cdf)

    cdf[1:] = _N.cumsum(postprob)
    #  
    if cdf[-1] == 0:   #  far from mean
        print("cdf[-1] == 0")
        return mj0
    cdf /= cdf[-1]

    rnd = _N.random.rand()
    #print(cdf[0:5])
    #print(cdf[995:])
    #print(lklhd[0:5])
    #print(lklhd[995:])
    #

    #print("%(rnd).4e    %(cdf).4e" % {"rnd" : rnd, "cdf" : cdf[0]})
    nAt = _N.where((rnd >= cdf[0:-1]) & (rnd < cdf[1:]))[0]
    if len(nAt) == 0:
        _plt.plot(cdf)
        print("len(nAT) == 0,  %(mj0).4e   %(s).4e" % {"mj0" : mj0, "s" : svPr1})
        print(cdf)

    return phi1s_smler[nAt[0]]

