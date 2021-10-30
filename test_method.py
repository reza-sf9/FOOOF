# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 22:54:25 2021

@author: rsaadatifard
"""

 # # define the model/function to be fitted.
 # import pymc3 as pm

 method_solving = input_model['method_solving']
 
 import theano.tensor as tt
 import theano
 with pm.Model() as model3:
     # test for making matrix of variable

     test_=0
     if test_:
         aa = pm.Normal('aa', mu=0, sd=10)
         bb = pm.Normal('bb', mu=0, sd=10)
         cc = pm.Normal('cc', mu=0, sd=10)


         RR = tt.stack([aa, bb, bb, cc]).reshape((2, 2))

         R = theano.shared(np.identity(4))
         R = tt.set_subtensor(R[1, 1], aa)
         R = tt.set_subtensor(R[1, 2], bb)
         R = tt.set_subtensor(R[2, 1], bb)
         R = tt.set_subtensor(R[2, 2], cc)


         # mat_ = tt.cmatrix('mat_')
         # mat_[1, 1] = aa


     #  SPECIFY prior distributions for parameters
     #  weigts

     a = pm.HalfNormal('a', 20)
     b = pm.Normal('b', 0.13, 40)
     c = pm.Normal('c', mu=0., sd=4)
     
     

     # meth_0 = 0
     # if meth_0:
     #     count = 0

         # Mat_cov = theano.shared(np.identity(l_freq*l_freq))
         # # Mat_cov = tt.cmatrix('Mat_cov')
         # for ii in range(l_freq):
         #     for jj in range(l_freq):
         #         if ii==jj:
         #             temp = s_d
         #             Mat_cov = tt.set_subtensor(Mat_cov[ii, jj], temp)
         #             # a = tt.stack([a, temp]).reshape((count,))
         #             # a += temp
         #         else:
         #             # temp = error * (ro**np.abs(ii-jj))
         #             temp = s_d
         #             Mat_cov = tt.set_subtensor(Mat_cov[ii, jj], temp)
         #             # a = tt.stack([a, temp]).reshape((count, ))
         #                 # a += temp



     # a.reshape((l_freq, l_freq))
     # K=1

     # cov_error = tt.matrix('cov_error', )


     # LKJ choleskey decomposition
     # packed_L = pm.LKJCholeskyCov("packed_L", n=2, eta=2.0, sd_dist=pm.Exponential.dist(1.0))

     # eta_dist = pm.Uniform('eta_dist', lower=0, upper=1)
     # chol, corr, stds = pm.LKJCholeskyCov(
     #     "chol", n=f_at_obs.shape[0], eta= eta_dist, sd_dist=pm.Exponential.dist(1.0)
     # )
     # cov = pm.Deterministic("cov", chol.dot(chol.T))


     num_bumps=input_model['num_bumps']
     
     
     mu_1 = input_model['mu_1']
     sd_1 = input_model['sd_1']
     
     w1 = pm.HalfNormal('w1', sd=1)  # reza: changing all sigma to sd in the code
     sig1 = pm.HalfNormal('sig1', sd=2)
     mu1 = pm.Normal('mu1', mu=mu_1, sd=sd_1)  # initialize w/ diff values
     
     if num_bumps == 1:

         if one_d_cov:
             ##  Gaussian noise assumed on top of a/(f+b) shape
             # prior of paramteried model > is the mean of likelihood function
             gauss = pm.Deterministic('gauss', w1 * np.exp(-0.5 * (f_at_obs - mu1) ** 2 / (sig1 ** 2)) + a / (f_at_obs + b) + c)
         
         
         elif multi_d_cov:
             sd_manu = pm.HalfNormal("s_d", sd=sd_1)
             ro_manu = pm.Uniform('ro', lower=0, upper=1)

             l_freq = f_at_obs.shape[0]
             mu_ = calc_mu_vec(w1, mu1, sig1, a, b, c, f_at_obs)
             cov_ = calc_cov_mat(sd_manu, ro_manu, l_freq, mode_cov)
             
             
             
             mu_ = pm.Deterministic('mu_', mu_)
             cov_ = pm.Deterministic('cov_', cov_)
         
         # elif meth_1:
         #     k=1
             # Vec_mu = theano.shared(np.identity(l_freq ))
             # for ii in range(l_freq):
             #     f_temp = f_at_obs[ii]
             #     temp = w1 * np.exp(-0.5 * (f_temp - mu1) ** 2 / (sig1 ** 2)) + a / (f_temp + b) + c
             #     Vec_mu = tt.set_subtensor(Vec_mu[ii , ], temp)
                 
         

     elif num_bumps == 2:
         mu_1 = input_model['mu_1']
         sd_1 = input_model['sd_1']
         mu_2 = input_model['mu_2']
         sd_2 = input_model['sd_2']
         
         w1 = pm.HalfNormal('w1', sd=1)  # reza: changing all sigma to sd in the code
         w2 = pm.HalfNormal('w2', sd=1)
         sig1 = pm.HalfNormal('sig1', sd=2)
         sig2 = pm.HalfNormal('sig2', sd=2)
         mu1 = pm.Normal('mu1', mu=mu_1, sd=sd_1)  # initialize w/ diff values
         mu2 = pm.Normal('mu2', mu=mu_2, sd=sd_2)
     
         gauss = pm.Deterministic('gauss', w1 * np.exp(-0.5 * (f_at_obs - mu1) ** 2 / (sig1 ** 2)) + w2 * np.exp(
             -0.5 * (f_at_obs - mu2) ** 2 / (sig2 ** 2)) + a / (f_at_obs + b) + c)
         
     elif num_bumps == 3:
         w1 = pm.HalfNormal('w1', sd=1)  # reza: changing all sigma to sd in the code
         w2 = pm.HalfNormal('w2', sd=1)
         w3 = pm.HalfNormal('w3', sd=1)
         sig1 = pm.HalfNormal('sig1', sd=2)
         sig2 = pm.HalfNormal('sig2', sd=2)
         sig3 = pm.HalfNormal('sig3', sd=2)
         mu1 = pm.Normal('mu1', mu=10, sd=2)  # initialize w/ diff values
         mu2 = pm.Normal('mu2', mu=20, sd=3)
         mu3 = pm.Normal('mu3', mu=35, sd=4)


         ##  Gaussian noise assumed on top of a/(f+b) shape
         # prior of paramteried model > is the mean of likelihood function
         gauss = pm.Deterministic('gauss', w1 * np.exp(-0.5 * (f_at_obs - mu1) ** 2 / (sig1 ** 2)) + w2 * np.exp(
             -0.5 * (f_at_obs - mu2) ** 2 / (sig2 ** 2)) + w3 * np.exp(-0.5 * (f_at_obs - mu3) ** 2 / (sig3 ** 2)) + a / (f_at_obs + b) + c)

     # generate sample from dist
     # gauss_sampel = mu1(size=(10,))

     # run full likelihood p(observation|prior)
     

     
     if one_d_cov:
         error = pm.HalfNormal("error", sd=0.3)
         y = pm.Normal('y', mu=gauss, sd=error, observed=log_psd)
             
     elif multi_d_cov:
         # this is likelihood
         y = pm.MvNormal('y', mu=mu_, cov=cov_, observed=log_psd)

     # map_estimate=pm.find_MAP()    #  can also do point MAP estimate (we can use it as starting point)
     step = pm.NUTS()

     
     # this piece run Marcov Chain Mont Carlo> sample of posterior
     # we are going to take 2000 samples from posterior distribution for our parameters > w1, w2, ..., error
     trace = pm.sample(num_samp, cores=1, chains=1)  # reza: adding cores here

     # pm.traceplot(trace, ['gauss', 'error'])
     # plt.figure()
     # # pm.traceplot(trace, ['mu1', 'sig1', 'w1', 'ro', 's_d', 'c', 'b', 'a'])
     # pm.traceplot(trace)
     # plt.show()
     k = 1