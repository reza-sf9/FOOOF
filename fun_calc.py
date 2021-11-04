# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 12:49:41 2021

@author: rsaadatifard

"""
import pymc3 as pm 
import numpy as np 
import matplotlib.pyplot as plt
from spectrum import *
    

### 4- Fit the spectrum of simulated EEG

##  fit to PYMC3 model using posterior sampling.
##  model assumes up to 3 bumps in power spectrum
def do_fit_psd(log_psd, f_at_obs, input_model, num_sample_post):
    # # define the model/function to be fitted.
    # import pymc3 as pm
   
    method_solving = input_model['method_solving']
    if method_solving=='M':
        cov_mode = input_model['cov_mode']
    
    import theano.tensor as tt
    import theano
    with pm.Model() as model3:

        #  SPECIFY prior distributions for parameters
        #  weigts

        a = pm.Uniform('a', lower=5, upper=10)
        b = pm.Normal('b', 0.13, 40)
        c = pm.Normal('c', mu=0., sd=4)


        num_bumps=input_model['num_bumps']
        
        
        mu_1 = input_model['mu_1']
        sd_1 = input_model['sd_1']
        
        w1 = pm.Exponential('w1', lam=1)  # reza: changing all sigma to sd in the code
        sig1 = pm.Exponential('sig1', lam=1) # when put Exponantial > error 
        mu1 = pm.Normal('mu1', mu=mu_1, sd=sd_1)  # initialize w/ diff values
        
        if num_bumps == 1:

            if method_solving == '1':
                ##  Gaussian noise assumed on top of a/(f+b) shape
                # prior of paramteried model > is the mean of likelihood function
                gauss = pm.Deterministic('gauss', w1 * np.exp(-0.5 * (f_at_obs - mu1) ** 2 / (sig1 ** 2)) + a / (f_at_obs + b) + c)
            
            
            elif method_solving == 'M':
                # sig_cov = pm.HalfNormal("s_d", sd=sd_1) # REZA: get error if put normal 
                sig_cov = 1
                ro_cov = pm.Uniform('ro', lower=-1, upper=1)

                l_freq = f_at_obs.shape[0]
                mu_ = calc_mu_vec(w1, mu1, sig1, a, b, c, f_at_obs)
                cov_ = calc_cov_mat(sig_cov, ro_cov, l_freq, cov_mode)

                mu_ = pm.Deterministic('mu_', mu_)
                cov_ = pm.Deterministic('cov_', cov_)
   

        elif num_bumps == 2:
            
            
            mu_2 = input_model['mu_2']
            sd_2 = input_model['sd_2']
            
            w2 = pm.HalfNormal('w2', sd=1)
            sig2 = pm.HalfNormal('sig2', sd=2)
            
            mu1 = pm.Normal('mu1', mu=mu_1, sd=sd_1)  # initialize w/ diff values
            mu2 = pm.Normal('mu2', mu=mu_2, sd=sd_2)
        
            gauss = pm.Deterministic('gauss', w1 * np.exp(-0.5 * (f_at_obs - mu1) ** 2 / (sig1 ** 2)) + w2 * np.exp(
                -0.5 * (f_at_obs - mu2) ** 2 / (sig2 ** 2)) + a / (f_at_obs + b) + c)
            
        elif num_bumps == 3:
            w2 = pm.HalfNormal('w2', sd=1)
            w3 = pm.HalfNormal('w3', sd=1)
            sig2 = pm.HalfNormal('sig2', sd=2)
            sig3 = pm.HalfNormal('sig3', sd=2)
            mu2 = pm.Normal('mu2', mu=20, sd=3)
            mu3 = pm.Normal('mu3', mu=35, sd=4)


            ##  Gaussian noise assumed on top of a/(f+b) shape
            # prior of paramteried model > is the mean of likelihood function
            gauss = pm.Deterministic('gauss', w1 * np.exp(-0.5 * (f_at_obs - mu1) ** 2 / (sig1 ** 2)) + w2 * np.exp(
                -0.5 * (f_at_obs - mu2) ** 2 / (sig2 ** 2)) + w3 * np.exp(-0.5 * (f_at_obs - mu3) ** 2 / (sig3 ** 2)) + a / (f_at_obs + b) + c)


        # run full likelihood p(observation|prior)

        if method_solving == '1':
            error = pm.HalfNormal("error", sd=0.3)
            y = pm.Normal('y', mu=gauss, sd=error, observed=log_psd)
                
        elif method_solving == 'M':
            # this is likelihood
            y = pm.MvNormal('y', mu=mu_, cov=cov_, observed=log_psd)

        # map_estimate=pm.find_MAP()    #  can also do point MAP estimate (we can use it as starting point)
        step = pm.NUTS()

        
        # this piece run Marcov Chain Mont Carlo> sample of posterior
        # we are going to take 2000 samples from posterior distribution for our parameters > w1, w2, ..., error
        trace = pm.sample(num_sample_post, cores=1, chains=1)  # reza: adding cores here

        # pm.traceplot(trace, ['gauss', 'error'])
        # plt.figure()
        # # pm.traceplot(trace, ['mu1', 'sig1', 'w1', 'ro', 's_d', 'c', 'b', 'a'])
        # pm.traceplot(trace)
        # plt.show()
        k = 1

    return trace, model3


def synthetic_data(input_synthetic):
    
    
    
    # async config 
    a= input_synthetic['a']
    b= input_synthetic['b']
    c = input_synthetic['c']
    
    # 1st bump 
    mu_1 = input_synthetic['mu_1']
    sigma_1 = input_synthetic['sigma_1']
    w_1 = input_synthetic['w_1']
    
    # 2nd bump 
    mu_2 = input_synthetic['mu_2']
    sigma_2 = input_synthetic['sigma_2']
    w_2 = input_synthetic['w_2']
    
    # noise config 
    noise_coef = input_synthetic['noise_coef']
    mean_noise = input_synthetic['mean_noise']
    var_noise = input_synthetic['var_noise']
    ro_noise = input_synthetic['ro_noise']
    
    # general config 
    num_bumps = input_synthetic['num_bumps']
    num_taper = input_synthetic['num_taper']
    low_fr  = input_synthetic['low_fr']
    high_fr = input_synthetic['high_fr']
    step_fr = input_synthetic['step_fr']
    

    
    
    
    
    freq_range = np.arange(low_fr, high_fr, step_fr)
    datanp = np.zeros((num_taper, freq_range.shape[0]))
    num_samples_noise = freq_range.shape[0]
    
    for tpr in range(num_taper):
        
        async_ = a/(freq_range+b) + c
        
        
        if num_bumps==1:
            sync_ = w_1*np.exp(-(freq_range-mu_1)**2/(2*sigma_1**2))
        elif num_bumps==2:
            sync_ = w_1 * np.exp(-(freq_range - mu_1) ** 2 / (2 * sigma_1 ** 2)) + \
                    w_2 * np.exp(-(freq_range - mu_2) ** 2 / (2 * sigma_2 ** 2))
    
    
        data = async_ + sync_

                
        # generating white noise
        # noise_ = noise_coef*np.random.normal(mean_noise, std_noise, size=num_samples_noise)
        
        datanp[tpr, :] = data 
    
    ## create covariate of noise 
    cov_mat_noise = np.zeros((num_taper, num_taper))
    for ii in range(num_taper):
        for jj in range(num_taper):
            cov_mat_noise[ii, jj] = var_noise**2 * ro_noise**np.abs(ii-jj) 
            
    mu_vec_noise = mean_noise*np.ones((num_taper,))
    
    noise_multi = np.transpose(np.random.multivariate_normal(mu_vec_noise, cov_mat_noise, size=num_samples_noise))
    
    # estimated_cov = (np.cov(noise_multi, bias= True))
    # plt.figure()
    # plt.imshow(estimated_cov)
    # ro_str = "{:.2f}".format(input_synthetic['ro_noise'])
    # plt.title('estimated cov matrix - ro = ' + ro_str)
    # plt.colorbar()
    # plt.colormaps()
    # # plt.clim(0, 8) 
    # plt.show()

    
    datanp = datanp + noise_multi
        
    if num_taper==1:
        plt.figure()
        plt.plot(freq_range, datanp[0, :])
        plt.title('synthetic PSD for 1 trial')
        plt.xlabel('freq (hz)')
        plt.show()
    
    elif num_taper==10:
        
        fig, axs = plt.subplots(5,2)
        ro_str = "{:.2f}".format(ro_noise)
        fig.suptitle('Multiple trials of synthetic PSD - ro=' +  ro_str)
        
        axs[0,0].plot(freq_range, datanp[0, :])
        axs[0,0].set_title('1')
        axs[0,1].plot(freq_range, datanp[1, :])
        axs[0,1].set_title('2')
        
        axs[1,0].plot(freq_range, datanp[2, :])
        axs[1,0].set_title('3')
        axs[1,1].plot(freq_range, datanp[3, :])
        axs[1,1].set_title('4')
        
        axs[2,0].plot(freq_range, datanp[4, :])
        axs[2,0].set_title('5')
        axs[2,1].plot(freq_range, datanp[5, :])
        axs[2,1].set_title('6')
        
        axs[3,0].plot(freq_range, datanp[6, :])
        axs[3,0].set_title('7')
        axs[3,1].plot(freq_range, datanp[7, :])
        axs[3,1].set_title('8')
        
        axs[4,0].plot(freq_range, datanp[8, :])
        axs[4,0].set_title('9')
        axs[4,0].set_xlabel('freq (Hz)')
        axs[4,1].plot(freq_range, datanp[9, :])
        axs[4,1].set_title('10')
        axs[4,1].set_xlabel('freq (Hz)')
    
    
    
    
    # plt.show()
    
    k=1
    
    return datanp, freq_range


def calc_mu_vec(w1, mu1, sig1, a, b, c, f_at_obs):
    import theano.tensor as tt

    if f_at_obs.shape[0] == 9:
        gauss_0 = pm.Deterministic('gauss_0', w1 * np.exp(-0.5 * (f_at_obs[0] - mu1) ** 2 / (sig1 ** 2)) + a / (f_at_obs[0] + b) + c)
        gauss_1 = pm.Deterministic('gauss_1', w1 * np.exp(-0.5 * (f_at_obs[1] - mu1) ** 2 / (sig1 ** 2)) + a / (f_at_obs[1] + b) + c)
        gauss_2 = pm.Deterministic('gauss_2', w1 * np.exp(-0.5 * (f_at_obs[2] - mu1) ** 2 / (sig1 ** 2)) + a / (f_at_obs[2] + b) + c)
        gauss_3 = pm.Deterministic('gauss_3', w1 * np.exp(-0.5 * (f_at_obs[3] - mu1) ** 2 / (sig1 ** 2)) + a / (f_at_obs[3] + b) + c)
        gauss_4 = pm.Deterministic('gauss_4', w1 * np.exp(-0.5 * (f_at_obs[4] - mu1) ** 2 / (sig1 ** 2)) + a / (f_at_obs[4] + b) + c)
        gauss_5 = pm.Deterministic('gauss_5', w1 * np.exp(-0.5 * (f_at_obs[5] - mu1) ** 2 / (sig1 ** 2)) + a / (f_at_obs[5] + b) + c)
        gauss_6 = pm.Deterministic('gauss_6', w1 * np.exp(-0.5 * (f_at_obs[6] - mu1) ** 2 / (sig1 ** 2)) + a / (f_at_obs[6] + b) + c)
        gauss_7 = pm.Deterministic('gauss_7', w1 * np.exp(-0.5 * (f_at_obs[7] - mu1) ** 2 / (sig1 ** 2)) + a / (f_at_obs[7] + b) + c)
        gauss_8 = pm.Deterministic('gauss_8', w1 * np.exp(-0.5 * (f_at_obs[8] - mu1) ** 2 / (sig1 ** 2)) + a / (f_at_obs[8] + b) + c)


        mu_  = tt.stack([gauss_0, gauss_1, gauss_2, gauss_3, gauss_4, gauss_5, gauss_6, gauss_7, gauss_8]).reshape((9, ))
    
    if f_at_obs.shape[0] == 18:
        gauss_0 = pm.Deterministic('gauss_0', w1 * np.exp(-0.5 * (f_at_obs[0] - mu1) ** 2 / (sig1 ** 2)) + a / (f_at_obs[0] + b) + c)
        gauss_1 = pm.Deterministic('gauss_1', w1 * np.exp(-0.5 * (f_at_obs[1] - mu1) ** 2 / (sig1 ** 2)) + a / (f_at_obs[1] + b) + c)
        gauss_2 = pm.Deterministic('gauss_2', w1 * np.exp(-0.5 * (f_at_obs[2] - mu1) ** 2 / (sig1 ** 2)) + a / (f_at_obs[2] + b) + c)
        gauss_3 = pm.Deterministic('gauss_3', w1 * np.exp(-0.5 * (f_at_obs[3] - mu1) ** 2 / (sig1 ** 2)) + a / (f_at_obs[3] + b) + c)
        gauss_4 = pm.Deterministic('gauss_4', w1 * np.exp(-0.5 * (f_at_obs[4] - mu1) ** 2 / (sig1 ** 2)) + a / (f_at_obs[4] + b) + c)
        gauss_5 = pm.Deterministic('gauss_5', w1 * np.exp(-0.5 * (f_at_obs[5] - mu1) ** 2 / (sig1 ** 2)) + a / (f_at_obs[5] + b) + c)
        gauss_6 = pm.Deterministic('gauss_6', w1 * np.exp(-0.5 * (f_at_obs[6] - mu1) ** 2 / (sig1 ** 2)) + a / (f_at_obs[6] + b) + c)
        gauss_7 = pm.Deterministic('gauss_7', w1 * np.exp(-0.5 * (f_at_obs[7] - mu1) ** 2 / (sig1 ** 2)) + a / (f_at_obs[7] + b) + c)
        gauss_8 = pm.Deterministic('gauss_8', w1 * np.exp(-0.5 * (f_at_obs[8] - mu1) ** 2 / (sig1 ** 2)) + a / (f_at_obs[8] + b) + c)
        gauss_9 = pm.Deterministic('gauss_9', w1 * np.exp(-0.5 * (f_at_obs[9] - mu1) ** 2 / (sig1 ** 2)) + a / (f_at_obs[9] + b) + c)
        gauss_10 = pm.Deterministic('gauss_10', w1 * np.exp(-0.5 * (f_at_obs[10] - mu1) ** 2 / (sig1 ** 2)) + a / (f_at_obs[10] + b) + c)
        gauss_11 = pm.Deterministic('gauss_11', w1 * np.exp(-0.5 * (f_at_obs[11] - mu1) ** 2 / (sig1 ** 2)) + a / (f_at_obs[11] + b) + c)
        gauss_12 = pm.Deterministic('gauss_12', w1 * np.exp(-0.5 * (f_at_obs[12] - mu1) ** 2 / (sig1 ** 2)) + a / (f_at_obs[12] + b) + c)
        gauss_13 = pm.Deterministic('gauss_13', w1 * np.exp(-0.5 * (f_at_obs[13] - mu1) ** 2 / (sig1 ** 2)) + a / (f_at_obs[13] + b) + c)
        gauss_14 = pm.Deterministic('gauss_14', w1 * np.exp(-0.5 * (f_at_obs[14] - mu1) ** 2 / (sig1 ** 2)) + a / (f_at_obs[14] + b) + c)
        gauss_15 = pm.Deterministic('gauss_15', w1 * np.exp(-0.5 * (f_at_obs[15] - mu1) ** 2 / (sig1 ** 2)) + a / (f_at_obs[15] + b) + c)
        gauss_16 = pm.Deterministic('gauss_16', w1 * np.exp(-0.5 * (f_at_obs[16] - mu1) ** 2 / (sig1 ** 2)) + a / (f_at_obs[16] + b) + c)
        gauss_17 = pm.Deterministic('gauss_17', w1 * np.exp(-0.5 * (f_at_obs[17] - mu1) ** 2 / (sig1 ** 2)) + a / (f_at_obs[17] + b) + c)


        mu_  = tt.stack([gauss_0, gauss_1, gauss_2, gauss_3, gauss_4, gauss_5, gauss_6, gauss_7, gauss_8, gauss_9,
                         gauss_10, gauss_11, gauss_12, gauss_13, gauss_14, gauss_15, gauss_16, gauss_17]).reshape((18, ))
    
        return mu_


def calc_cov_mat(sigma, ro, l_freq, mode_cov):
    import theano.tensor as tt
    import pymc3.math as math

    if l_freq==9:
        v_1 = tt.stack([math.dot(sigma**2, ro**0), math.dot(sigma**2, ro**1), math.dot(sigma**2, ro**2), math.dot(sigma**2, ro**3), math.dot(sigma**2, ro**4),
                       math.dot(sigma**2, ro**5), math.dot(sigma**2, ro**6), math.dot(sigma**2, ro**7), math.dot(sigma**2, ro**8)]).reshape((9, ))
        v_2 = tt.stack([math.dot(sigma**2, ro ** 1), math.dot(sigma**2, ro ** 0), math.dot(sigma**2, ro ** 1), math.dot(sigma**2, ro ** 2), math.dot(sigma**2, ro ** 3),
                        math.dot(sigma**2, ro ** 4), math.dot(sigma**2, ro ** 5), math.dot(sigma**2, ro ** 6), math.dot(sigma**2, ro ** 7)]).reshape((9,))
        v_3 = tt.stack([math.dot(sigma**2, ro ** 2), math.dot(sigma**2, ro ** 1), math.dot(sigma**2, ro ** 0), math.dot(sigma**2, ro ** 1), math.dot(sigma**2, ro ** 2),
                        math.dot(sigma**2, ro ** 3), math.dot(sigma**2, ro ** 4), math.dot(sigma**2, ro ** 5), math.dot(sigma**2, ro ** 6)]).reshape((9,))
        v_4 = tt.stack([math.dot(sigma**2, ro ** 3), math.dot(sigma**2, ro ** 2), math.dot(sigma**2, ro ** 1), math.dot(sigma**2, ro ** 0), math.dot(sigma**2, ro ** 1),
                        math.dot(sigma**2, ro ** 2), math.dot(sigma**2, ro ** 3), math.dot(sigma**2, ro ** 4), math.dot(sigma**2, ro ** 5)]).reshape((9,))
        v_5 = tt.stack([math.dot(sigma**2, ro ** 4), math.dot(sigma**2, ro ** 3), math.dot(sigma**2, ro ** 2), math.dot(sigma**2, ro ** 1), math.dot(sigma**2,ro ** 0),
                        math.dot(sigma**2, ro ** 1), math.dot(sigma**2, ro ** 2), math.dot(sigma**2, ro ** 3), math.dot(sigma**2, ro ** 4)]).reshape((9,))
        v_6 = tt.stack([math.dot(sigma**2, ro ** 5), math.dot(sigma**2, ro ** 4), math.dot(sigma**2, ro ** 3), math.dot(sigma**2, ro ** 2), math.dot(sigma**2,ro ** 1),
                        math.dot(sigma**2,ro ** 0), math.dot(sigma**2, ro ** 1), math.dot(sigma**2, ro ** 2), math.dot(sigma**2, ro ** 3)]).reshape((9,))
        v_7 = tt.stack([math.dot(sigma**2, ro ** 6), math.dot(sigma**2, ro ** 5), math.dot(sigma**2, ro ** 4), math.dot(sigma**2, ro ** 3), math.dot(sigma**2,ro ** 2),
                        math.dot(sigma**2, ro ** 1), math.dot(sigma**2, ro ** 0), math.dot(sigma**2, ro ** 1), math.dot(sigma**2, ro ** 2)]).reshape((9,))
        v_8 = tt.stack([math.dot(sigma**2, ro ** 7), math.dot(sigma**2, ro ** 6), math.dot(sigma**2, ro ** 5), math.dot(sigma**2, ro ** 4), math.dot(sigma**2,ro ** 3),
                        math.dot(sigma**2, ro ** 2),math.dot(sigma**2, ro ** 1), math.dot(sigma**2, ro ** 0), math.dot(sigma**2, ro ** 1)]).reshape((9,))
        v_9 = tt.stack([math.dot(sigma**2, ro ** 8), math.dot(sigma**2, ro ** 7), math.dot(sigma**2, ro ** 6), math.dot(sigma**2, ro ** 5), math.dot(sigma**2, ro ** 4),
                        math.dot(sigma**2, ro ** 3), math.dot(sigma**2, ro ** 2), math.dot(sigma**2, ro ** 1), math.dot(sigma**2, ro ** 0)]).reshape((9,))

        cov_ = tt.stack([v_1, v_2, v_3, v_4, v_5, v_6, v_7, v_8, v_9]).reshape((9, 9))
    
    elif l_freq==18:
        
        
        if mode_cov ==-1: 
            v_1 = tt.stack([math.dot(sigma**2, ro**0), math.dot(sigma**2, ro**1), math.dot(sigma**2, ro**2), math.dot(sigma**2, ro**3), math.dot(sigma**2, ro**4),
                            math.dot(sigma**2, ro**5), math.dot(sigma**2, ro**6), math.dot(sigma**2, ro**7), math.dot(sigma**2, ro**8),  math.dot(sigma**2, ro**9),
                            math.dot(sigma**2, ro**10), math.dot(sigma**2, ro**11), math.dot(sigma**2, ro**12), math.dot(sigma**2, ro**13), math.dot(sigma**2, ro**14),
                            math.dot(sigma**2, ro**15), math.dot(sigma**2, ro**16), math.dot(sigma**2, ro**17)]).reshape((l_freq, ))
            
            v_2 = tt.stack([math.dot(sigma**2, ro**1), math.dot(sigma**2, ro**0), math.dot(sigma**2, ro**1), math.dot(sigma**2, ro**2), math.dot(sigma**2, ro**3),
                            math.dot(sigma**2, ro**4), math.dot(sigma**2, ro**5), math.dot(sigma**2, ro**6), math.dot(sigma**2, ro**7), math.dot(sigma**2, ro**8), 
                            math.dot(sigma**2, ro**9), math.dot(sigma**2, ro**10), math.dot(sigma**2, ro**11), math.dot(sigma**2, ro**12), math.dot(sigma**2, ro**13), math.dot(sigma**2, ro**14),
                            math.dot(sigma**2, ro**15), math.dot(sigma**2, ro**16)]).reshape((l_freq, ))
            
            v_3 = tt.stack([math.dot(sigma**2, ro**2), math.dot(sigma**2, ro**1), math.dot(sigma**2, ro**0), math.dot(sigma**2, ro**1), math.dot(sigma**2, ro**2), 
                            math.dot(sigma**2, ro**3), math.dot(sigma**2, ro**4),
                            math.dot(sigma**2, ro**5), math.dot(sigma**2, ro**6), math.dot(sigma**2, ro**7), math.dot(sigma**2, ro**8),  math.dot(sigma**2, ro**9),
                            math.dot(sigma**2, ro**10), math.dot(sigma**2, ro**11), math.dot(sigma**2, ro**12), math.dot(sigma**2, ro**13), math.dot(sigma**2, ro**14),
                            math.dot(sigma**2, ro**15)]).reshape((l_freq, ))
            
            v_4 = tt.stack([math.dot(sigma**2, ro**3), math.dot(sigma**2, ro**2), math.dot(sigma**2, ro**1), math.dot(sigma**2, ro**0), math.dot(sigma**2, ro**1), 
                            math.dot(sigma**2, ro**2), math.dot(sigma**2, ro**3), math.dot(sigma**2, ro**4),
                            math.dot(sigma**2, ro**5), math.dot(sigma**2, ro**6), math.dot(sigma**2, ro**7), math.dot(sigma**2, ro**8),  math.dot(sigma**2, ro**9),
                            math.dot(sigma**2, ro**10), math.dot(sigma**2, ro**11), math.dot(sigma**2, ro**12), math.dot(sigma**2, ro**13), math.dot(sigma**2, ro**14)]).reshape((l_freq, ))
            
            v_5 = tt.stack([math.dot(sigma**2, ro**4), math.dot(sigma**2, ro**3), math.dot(sigma**2, ro**2), math.dot(sigma**2, ro**1), math.dot(sigma**2, ro**0), 
                            math.dot(sigma**2, ro**1), math.dot(sigma**2, ro**2), math.dot(sigma**2, ro**3), math.dot(sigma**2, ro**4),
                            math.dot(sigma**2, ro**5), math.dot(sigma**2, ro**6), math.dot(sigma**2, ro**7), math.dot(sigma**2, ro**8),  math.dot(sigma**2, ro**9),
                            math.dot(sigma**2, ro**10), math.dot(sigma**2, ro**11), math.dot(sigma**2, ro**12), math.dot(sigma**2, ro**13)]).reshape((l_freq, ))
            
            v_6 = tt.stack([math.dot(sigma**2, ro**5), math.dot(sigma**2, ro**4), math.dot(sigma**2, ro**3), math.dot(sigma**2, ro**2), math.dot(sigma**2, ro**1), 
                            math.dot(sigma**2, ro**0),math.dot(sigma**2, ro**1),  math.dot(sigma**2, ro**2), math.dot(sigma**2, ro**3), math.dot(sigma**2, ro**4),
                            math.dot(sigma**2, ro**5), math.dot(sigma**2, ro**6), math.dot(sigma**2, ro**7), math.dot(sigma**2, ro**8),  math.dot(sigma**2, ro**9),
                            math.dot(sigma**2, ro**10), math.dot(sigma**2, ro**11), math.dot(sigma**2, ro**12)]).reshape((l_freq, ))
            
            v_7 = tt.stack([math.dot(sigma**2, ro**6), math.dot(sigma**2, ro**5), math.dot(sigma**2, ro**4), math.dot(sigma**2, ro**3), math.dot(sigma**2, ro**2), 
                            math.dot(sigma**2, ro**1), math.dot(sigma**2, ro**0), 
                            math.dot(sigma**2, ro**1), math.dot(sigma**2, ro**2), math.dot(sigma**2, ro**3), math.dot(sigma**2, ro**4),
                            math.dot(sigma**2, ro**5), math.dot(sigma**2, ro**6), math.dot(sigma**2, ro**7), math.dot(sigma**2, ro**8),  math.dot(sigma**2, ro**9),
                            math.dot(sigma**2, ro**10), math.dot(sigma**2, ro**11)]).reshape((l_freq, ))
            
            v_8 = tt.stack([ math.dot(sigma**2, ro**7),  math.dot(sigma**2, ro**6), math.dot(sigma**2, ro**5), math.dot(sigma**2, ro**4), math.dot(sigma**2, ro**3), 
                            math.dot(sigma**2, ro**2), math.dot(sigma**2, ro**1), math.dot(sigma**2, ro**0), 
                            math.dot(sigma**2, ro**1), math.dot(sigma**2, ro**2), math.dot(sigma**2, ro**3), math.dot(sigma**2, ro**4),
                            math.dot(sigma**2, ro**5), math.dot(sigma**2, ro**6), math.dot(sigma**2, ro**7), math.dot(sigma**2, ro**8),  math.dot(sigma**2, ro**9),
                            math.dot(sigma**2, ro**10)]).reshape((l_freq, ))
            
            v_9 = tt.stack([math.dot(sigma**2, ro**8), math.dot(sigma**2, ro**7),  math.dot(sigma**2, ro**6), math.dot(sigma**2, ro**5), math.dot(sigma**2, ro**4), 
                            math.dot(sigma**2, ro**3), math.dot(sigma**2, ro**2), math.dot(sigma**2, ro**1), 
                            math.dot(sigma**2, ro**0), math.dot(sigma**2, ro**1), math.dot(sigma**2, ro**2), math.dot(sigma**2, ro**3), math.dot(sigma**2, ro**4),
                            math.dot(sigma**2, ro**5), math.dot(sigma**2, ro**6), math.dot(sigma**2, ro**7), math.dot(sigma**2, ro**8),  math.dot(sigma**2, ro**9)]).reshape((l_freq, ))
            
            v_10 = tt.stack([math.dot(sigma**2, ro**9), math.dot(sigma**2, ro**8), math.dot(sigma**2, ro**7),  math.dot(sigma**2, ro**6), math.dot(sigma**2, ro**5), 
                              math.dot(sigma**2, ro**4), math.dot(sigma**2, ro**3), math.dot(sigma**2, ro**2), math.dot(sigma**2, ro**1), 
                            math.dot(sigma**2, ro**0), math.dot(sigma**2, ro**1), math.dot(sigma**2, ro**2), math.dot(sigma**2, ro**3), math.dot(sigma**2, ro**4),
                            math.dot(sigma**2, ro**5), math.dot(sigma**2, ro**6), math.dot(sigma**2, ro**7), math.dot(sigma**2, ro**8)]).reshape((l_freq, ))
            
            v_11 = tt.stack([math.dot(sigma**2, ro**10), math.dot(sigma**2, ro**9), math.dot(sigma**2, ro**8), math.dot(sigma**2, ro**7),  math.dot(sigma**2, ro**6), 
                              math.dot(sigma**2, ro**5), math.dot(sigma**2, ro**4), math.dot(sigma**2, ro**3), math.dot(sigma**2, ro**2), math.dot(sigma**2, ro**1), 
                            math.dot(sigma**2, ro**0), math.dot(sigma**2, ro**1), math.dot(sigma**2, ro**2), math.dot(sigma**2, ro**3), math.dot(sigma**2, ro**4),
                            math.dot(sigma**2, ro**5), math.dot(sigma**2, ro**6), math.dot(sigma**2, ro**7)]).reshape((l_freq, ))
            
            v_12 = tt.stack([math.dot(sigma**2, ro**11), math.dot(sigma**2, ro**10), math.dot(sigma**2, ro**9), math.dot(sigma**2, ro**8), math.dot(sigma**2, ro**7), 
                              math.dot(sigma**2, ro**6), math.dot(sigma**2, ro**5), math.dot(sigma**2, ro**4), math.dot(sigma**2, ro**3), math.dot(sigma**2, ro**2), math.dot(sigma**2, ro**1), 
                            math.dot(sigma**2, ro**0), math.dot(sigma**2, ro**1), math.dot(sigma**2, ro**2), math.dot(sigma**2, ro**3), math.dot(sigma**2, ro**4),
                            math.dot(sigma**2, ro**5), math.dot(sigma**2, ro**6)]).reshape((l_freq, ))
            
            v_13 = tt.stack([math.dot(sigma**2, ro**12), math.dot(sigma**2, ro**11), math.dot(sigma**2, ro**10), math.dot(sigma**2, ro**9), math.dot(sigma**2, ro**8), 
                              math.dot(sigma**2, ro**7),  math.dot(sigma**2, ro**6), math.dot(sigma**2, ro**5), math.dot(sigma**2, ro**4), math.dot(sigma**2, ro**3), math.dot(sigma**2, ro**2), math.dot(sigma**2, ro**1), 
                            math.dot(sigma**2, ro**0), math.dot(sigma**2, ro**1), math.dot(sigma**2, ro**2), math.dot(sigma**2, ro**3), math.dot(sigma**2, ro**4),
                            math.dot(sigma**2, ro**5)]).reshape((l_freq, ))
            
            v_14 = tt.stack([math.dot(sigma**2, ro**13), math.dot(sigma**2, ro**12), math.dot(sigma**2, ro**11), math.dot(sigma**2, ro**10), math.dot(sigma**2, ro**9), 
                              math.dot(sigma**2, ro**8), math.dot(sigma**2, ro**7),  math.dot(sigma**2, ro**6), math.dot(sigma**2, ro**5), math.dot(sigma**2, ro**4), 
                              math.dot(sigma**2, ro**3), math.dot(sigma**2, ro**2), math.dot(sigma**2, ro**1), 
                            math.dot(sigma**2, ro**0), math.dot(sigma**2, ro**1), math.dot(sigma**2, ro**2), math.dot(sigma**2, ro**3), math.dot(sigma**2, ro**4)]).reshape((l_freq, ))
            
            v_15 = tt.stack([math.dot(sigma**2, ro**14), math.dot(sigma**2, ro**13), math.dot(sigma**2, ro**12), math.dot(sigma**2, ro**11), math.dot(sigma**2, ro**10), 
                              math.dot(sigma**2, ro**9), math.dot(sigma**2, ro**8), math.dot(sigma**2, ro**7),  math.dot(sigma**2, ro**6), math.dot(sigma**2, ro**5), 
                              math.dot(sigma**2, ro**4), math.dot(sigma**2, ro**3), math.dot(sigma**2, ro**2), math.dot(sigma**2, ro**1), 
                            math.dot(sigma**2, ro**0), math.dot(sigma**2, ro**1), math.dot(sigma**2, ro**2), math.dot(sigma**2, ro**3)]).reshape((l_freq, ))
            
            v_16 = tt.stack([math.dot(sigma**2, ro**15), math.dot(sigma**2, ro**14), math.dot(sigma**2, ro**13), math.dot(sigma**2, ro**12), math.dot(sigma**2, ro**11), 
                              math.dot(sigma**2, ro**10), math.dot(sigma**2, ro**9), math.dot(sigma**2, ro**8), math.dot(sigma**2, ro**7),  math.dot(sigma**2, ro**6), 
                              math.dot(sigma**2, ro**5), math.dot(sigma**2, ro**4), math.dot(sigma**2, ro**3), math.dot(sigma**2, ro**2), math.dot(sigma**2, ro**1), 
                            math.dot(sigma**2, ro**0), math.dot(sigma**2, ro**1), math.dot(sigma**2, ro**2)]).reshape((l_freq, ))
            
            v_17 = tt.stack([math.dot(sigma**2, ro**16), math.dot(sigma**2, ro**15), math.dot(sigma**2, ro**14), math.dot(sigma**2, ro**13), math.dot(sigma**2, ro**12), 
                              math.dot(sigma**2, ro**11), math.dot(sigma**2, ro**10), math.dot(sigma**2, ro**9), math.dot(sigma**2, ro**8), math.dot(sigma**2, ro**7),  
                              math.dot(sigma**2, ro**6), math.dot(sigma**2, ro**5), math.dot(sigma**2, ro**4), math.dot(sigma**2, ro**3), math.dot(sigma**2, ro**2), math.dot(sigma**2, ro**1), 
                            math.dot(sigma**2, ro**0), math.dot(sigma**2, ro**1)]).reshape((l_freq, ))
            
            v_18 = tt.stack([math.dot(sigma**2, ro**17), math.dot(sigma**2, ro**16), math.dot(sigma**2, ro**15), math.dot(sigma**2, ro**14), math.dot(sigma**2, ro**13), 
                              math.dot(sigma**2, ro**12), math.dot(sigma**2, ro**11), math.dot(sigma**2, ro**10), math.dot(sigma**2, ro**9), math.dot(sigma**2, ro**8), math.dot(sigma**2, ro**7), 
                              math.dot(sigma**2, ro**6), math.dot(sigma**2, ro**5), math.dot(sigma**2, ro**4), math.dot(sigma**2, ro**3), math.dot(sigma**2, ro**2), math.dot(sigma**2, ro**1), 
                            math.dot(sigma**2, ro**0)]).reshape((l_freq, ))
        
        elif mode_cov == 0:
        
            v_1 = tt.stack([math.dot(sigma**2, ro**0), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).reshape((l_freq, ))
            
            v_2 = tt.stack([0, math.dot(sigma**2, ro**0), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).reshape((l_freq, ))
            
            v_3 = tt.stack([0, 0, math.dot(sigma**2, ro**0), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).reshape((l_freq, ))
            
            v_4 = tt.stack([0, 0, 0, math.dot(sigma**2, ro**0), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).reshape((l_freq, ))
            
            v_5 = tt.stack([0, 0, 0, 0, math.dot(sigma**2, ro**0),  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).reshape((l_freq, ))
            
            v_6 = tt.stack([0, 0, 0, 0, 0, 
                            math.dot(sigma**2, ro**0), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).reshape((l_freq, ))
            
            v_7 = tt.stack([ 0, 0, 0, 0, 0, 0, math.dot(sigma**2, ro**0), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,]).reshape((l_freq, ))
            
            v_8 = tt.stack([ 0, 0, 0, 0, 0, 0, 0, math.dot(sigma**2, ro**0),  0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).reshape((l_freq, ))
            
            v_9 = tt.stack([0, 0, 0, 0, 0, 0, 0, 0, math.dot(sigma**2, ro**0), 0, 0, 0, 0, 0, 0, 0, 0, 0]).reshape((l_freq, ))
            
            v_10 = tt.stack([0, 0, 0, 0, 0, 0, 0, 0, 0, math.dot(sigma**2, ro**0), 0, 0, 0, 0, 0, 0, 0, 0]).reshape((l_freq, ))
            
            v_11 = tt.stack([ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, math.dot(sigma**2, ro**0), 0, 0, 0, 0, 0, 0, 0]).reshape((l_freq, ))
            
            v_12 = tt.stack([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                            math.dot(sigma**2, ro**0), 0, 0, 0, 0, 0, 0]).reshape((l_freq, ))
            
            v_13 = tt.stack([ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                            math.dot(sigma**2, ro**0), 0, 0, 0, 0, 0]).reshape((l_freq, ))
            
            v_14 = tt.stack([ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                            math.dot(sigma**2, ro**0), 0, 0, 0, 0]).reshape((l_freq, ))
            
            v_15 = tt.stack([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                            math.dot(sigma**2, ro**0), 0, 0, 0]).reshape((l_freq, ))
            
            v_16 = tt.stack([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                            math.dot(sigma**2, ro**0), 0, 0]).reshape((l_freq, ))
            
            v_17 = tt.stack([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                            math.dot(sigma**2, ro**0), 0]).reshape((l_freq, ))
            
            v_18 = tt.stack([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                            math.dot(sigma**2, ro**0)]).reshape((l_freq, ))
        
        elif mode_cov == 1:
            
            v_1 = tt.stack([math.dot(sigma**2, ro**0), math.dot(sigma**2, ro ** 1), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).reshape((l_freq, ))
            
            v_2 = tt.stack([math.dot(sigma**2, ro ** 1), math.dot(sigma**2, ro**0), math.dot(sigma**2, ro ** 1), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).reshape((l_freq, ))
            
            v_3 = tt.stack([0, math.dot(sigma**2, ro ** 1), math.dot(sigma**2, ro**0), math.dot(sigma**2, ro ** 1), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).reshape((l_freq, ))
            
            v_4 = tt.stack([0, 0, math.dot(sigma**2, ro ** 1), math.dot(sigma**2, ro**0), math.dot(sigma**2, ro ** 1), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).reshape((l_freq, ))
            
            v_5 = tt.stack([0, 0, 0, math.dot(sigma**2, ro ** 1), math.dot(sigma**2, ro**0),  math.dot(sigma**2, ro ** 1), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).reshape((l_freq, ))
            
            v_6 = tt.stack([0, 0, 0, 0, math.dot(sigma**2, ro ** 1), 
                            math.dot(sigma**2, ro**0), math.dot(sigma**2, ro ** 1), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).reshape((l_freq, ))
            
            v_7 = tt.stack([ 0, 0, 0, 0, 0, math.dot(sigma**2, ro ** 1), math.dot(sigma**2, ro**0), math.dot(sigma**2, ro ** 1), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,]).reshape((l_freq, ))
            
            v_8 = tt.stack([ 0, 0, 0, 0, 0, 0, math.dot(sigma**2, ro ** 1), math.dot(sigma**2, ro**0),  math.dot(sigma**2, ro ** 1), 0, 0, 0, 0, 0, 0, 0, 0, 0]).reshape((l_freq, ))
            
            v_9 = tt.stack([0, 0, 0, 0, 0, 0, 0, math.dot(sigma**2, ro ** 1), math.dot(sigma**2, ro**0), math.dot(sigma**2, ro ** 1), 0, 0, 0, 0, 0, 0, 0, 0]).reshape((l_freq, ))
            
            v_10 = tt.stack([0, 0, 0, 0, 0, 0, 0, 0, math.dot(sigma**2, ro ** 1), math.dot(sigma**2, ro**0), math.dot(sigma**2, ro ** 1), 0, 0, 0, 0, 0, 0, 0]).reshape((l_freq, ))
            
            v_11 = tt.stack([ 0, 0, 0, 0, 0, 0, 0, 0, 0, math.dot(sigma**2, ro ** 1), math.dot(sigma**2, ro**0), math.dot(sigma**2, ro ** 1), 0, 0, 0, 0, 0, 0]).reshape((l_freq, ))
            
            v_12 = tt.stack([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, math.dot(sigma**2, ro ** 1), 
                            math.dot(sigma**2, ro**0), math.dot(sigma**2, ro ** 1), 0, 0, 0, 0, 0]).reshape((l_freq, ))
            
            v_13 = tt.stack([ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, math.dot(sigma**2, ro ** 1),
                            math.dot(sigma**2, ro**0), math.dot(sigma**2, ro ** 1), 0, 0, 0, 0]).reshape((l_freq, ))
            
            v_14 = tt.stack([ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, math.dot(sigma**2, ro ** 1),
                            math.dot(sigma**2, ro**0), math.dot(sigma**2, ro ** 1), 0, 0, 0]).reshape((l_freq, ))
            
            v_15 = tt.stack([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, math.dot(sigma**2, ro ** 1), 
                            math.dot(sigma**2, ro**0), math.dot(sigma**2, ro ** 1), 0, 0]).reshape((l_freq, ))
            
            v_16 = tt.stack([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, math.dot(sigma**2, ro ** 1), 
                            math.dot(sigma**2, ro**0), math.dot(sigma**2, ro ** 1), 0]).reshape((l_freq, ))
            
            v_17 = tt.stack([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, math.dot(sigma**2, ro ** 1), 
                            math.dot(sigma**2, ro**0), math.dot(sigma**2, ro ** 1)]).reshape((l_freq, ))
            
            v_18 = tt.stack([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, math.dot(sigma**2, ro ** 1), 
                            math.dot(sigma**2, ro**0)]).reshape((l_freq, ))
            
        elif mode_cov == 2:
        
            v_1 = tt.stack([math.dot(sigma**2, ro**0), math.dot(sigma**2, ro ** 1), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).reshape((l_freq, ))
            
            v_2 = tt.stack([math.dot(sigma**2, ro ** 1), math.dot(sigma**2, ro**0), math.dot(sigma**2, ro ** 1), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).reshape((l_freq, ))
            
            v_3 = tt.stack([0, math.dot(sigma**2, ro ** 1), math.dot(sigma**2, ro**0), math.dot(sigma**2, ro ** 1), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).reshape((l_freq, ))
            
            v_4 = tt.stack([0, 0, math.dot(sigma**2, ro ** 1), math.dot(sigma**2, ro**0), math.dot(sigma**2, ro ** 1), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).reshape((l_freq, ))
            
            v_5 = tt.stack([0, 0, 0, math.dot(sigma**2, ro ** 1), math.dot(sigma**2, ro**0),  math.dot(sigma**2, ro ** 1), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).reshape((l_freq, ))
            
            v_6 = tt.stack([0, 0, 0, 0, math.dot(sigma**2, ro ** 1), 
                            math.dot(sigma**2, ro**0), math.dot(sigma**2, ro ** 1), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).reshape((l_freq, ))
            
            v_7 = tt.stack([ 0, 0, 0, 0, 0, math.dot(sigma**2, ro ** 1), math.dot(sigma**2, ro**0), math.dot(sigma**2, ro ** 1), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,]).reshape((l_freq, ))
            
            v_8 = tt.stack([ 0, 0, 0, 0, 0, 0, math.dot(sigma**2, ro ** 1), math.dot(sigma**2, ro**0),  math.dot(sigma**2, ro ** 1), 0, 0, 0, 0, 0, 0, 0, 0, 0]).reshape((l_freq, ))
            
            v_9 = tt.stack([0, 0, 0, 0, 0, 0, 0, math.dot(sigma**2, ro ** 1), math.dot(sigma**2, ro**0), math.dot(sigma**2, ro ** 1), 0, 0, 0, 0, 0, 0, 0, 0]).reshape((l_freq, ))
            
            v_10 = tt.stack([0, 0, 0, 0, 0, 0, 0, 0, math.dot(sigma**2, ro ** 1), math.dot(sigma**2, ro**0), math.dot(sigma**2, ro ** 1), 0, 0, 0, 0, 0, 0, 0]).reshape((l_freq, ))
            
            v_11 = tt.stack([ 0, 0, 0, 0, 0, 0, 0, 0, 0, math.dot(sigma**2, ro ** 1), math.dot(sigma**2, ro**0), math.dot(sigma**2, ro ** 1), 0, 0, 0, 0, 0, 0]).reshape((l_freq, ))
            
            v_12 = tt.stack([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, math.dot(sigma**2, ro ** 1), 
                            math.dot(sigma**2, ro**0), math.dot(sigma**2, ro ** 1), 0, 0, 0, 0, 0]).reshape((l_freq, ))
            
            v_13 = tt.stack([ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, math.dot(sigma**2, ro ** 1),
                            math.dot(sigma**2, ro**0), math.dot(sigma**2, ro ** 1), 0, 0, 0, 0]).reshape((l_freq, ))
            
            v_14 = tt.stack([ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, math.dot(sigma**2, ro ** 1),
                            math.dot(sigma**2, ro**0), math.dot(sigma**2, ro ** 1), 0, 0, 0]).reshape((l_freq, ))
            
            v_15 = tt.stack([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, math.dot(sigma**2, ro ** 1), 
                            math.dot(sigma**2, ro**0), math.dot(sigma**2, ro ** 1), 0, 0]).reshape((l_freq, ))
            
            v_16 = tt.stack([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, math.dot(sigma**2, ro ** 1), 
                            math.dot(sigma**2, ro**0), math.dot(sigma**2, ro ** 1), 0]).reshape((l_freq, ))
            
            v_17 = tt.stack([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, math.dot(sigma**2, ro ** 1), 
                            math.dot(sigma**2, ro**0), math.dot(sigma**2, ro ** 1)]).reshape((l_freq, ))
            
            v_18 = tt.stack([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, math.dot(sigma**2, ro ** 1), 
                            math.dot(sigma**2, ro**0)]).reshape((l_freq, ))
        
        cov_ = tt.stack([v_1, v_2, v_3, v_4, v_5, v_6, v_7, v_8, v_9, v_10, 
                         v_11, v_12, v_13, v_14, v_15, v_16, v_17, v_18]).reshape((l_freq, l_freq))
        

        return cov_


    import theano.tensor as tt
    import pymc3.math as math

    if l_freq==9:
        v_1 = tt.stack([math.dot(s_d, ro**0), math.dot(s_d, ro**1), math.dot(s_d, ro**2), math.dot(s_d, ro**3), math.dot(s_d, ro**4),
                       math.dot(s_d, ro**5), math.dot(s_d, ro**6), math.dot(s_d, ro**7), math.dot(s_d, ro**8)]).reshape((9, ))
        v_2 = tt.stack([math.dot(s_d, ro ** 1), math.dot(s_d, ro ** 0), math.dot(s_d, ro ** 1), math.dot(s_d, ro ** 2), math.dot(s_d, ro ** 3),
                        math.dot(s_d, ro ** 4), math.dot(s_d, ro ** 5), math.dot(s_d, ro ** 6), math.dot(s_d, ro ** 7)]).reshape((9,))
        v_3 = tt.stack([math.dot(s_d, ro ** 2), math.dot(s_d, ro ** 1), math.dot(s_d, ro ** 0), math.dot(s_d, ro ** 1), math.dot(s_d, ro ** 2),
                        math.dot(s_d, ro ** 3), math.dot(s_d, ro ** 4), math.dot(s_d, ro ** 5), math.dot(s_d, ro ** 6)]).reshape((9,))
        v_4 = tt.stack([math.dot(s_d, ro ** 3), math.dot(s_d, ro ** 2), math.dot(s_d, ro ** 1), math.dot(s_d, ro ** 0), math.dot(s_d, ro ** 1),
                        math.dot(s_d, ro ** 2), math.dot(s_d, ro ** 3), math.dot(s_d, ro ** 4), math.dot(s_d, ro ** 5)]).reshape((9,))
        v_5 = tt.stack([math.dot(s_d, ro ** 4), math.dot(s_d, ro ** 3), math.dot(s_d, ro ** 2), math.dot(s_d, ro ** 1), math.dot(s_d,ro ** 0),
                        math.dot(s_d, ro ** 1), math.dot(s_d, ro ** 2), math.dot(s_d, ro ** 3), math.dot(s_d, ro ** 4)]).reshape((9,))
        v_6 = tt.stack([math.dot(s_d, ro ** 5), math.dot(s_d, ro ** 4), math.dot(s_d, ro ** 3), math.dot(s_d, ro ** 2), math.dot(s_d,ro ** 1),
                        math.dot(s_d,ro ** 0), math.dot(s_d, ro ** 1), math.dot(s_d, ro ** 2), math.dot(s_d, ro ** 3)]).reshape((9,))
        v_7 = tt.stack([math.dot(s_d, ro ** 6), math.dot(s_d, ro ** 5), math.dot(s_d, ro ** 4), math.dot(s_d, ro ** 3), math.dot(s_d,ro ** 2),
                        math.dot(s_d, ro ** 1), math.dot(s_d, ro ** 0), math.dot(s_d, ro ** 1), math.dot(s_d, ro ** 2)]).reshape((9,))
        v_8 = tt.stack([math.dot(s_d, ro ** 7), math.dot(s_d, ro ** 6), math.dot(s_d, ro ** 5), math.dot(s_d, ro ** 4), math.dot(s_d,ro ** 3),
                        math.dot(s_d, ro ** 2),math.dot(s_d, ro ** 1), math.dot(s_d, ro ** 0), math.dot(s_d, ro ** 1)]).reshape((9,))
        v_9 = tt.stack([math.dot(s_d, ro ** 8), math.dot(s_d, ro ** 7), math.dot(s_d, ro ** 6), math.dot(s_d, ro ** 5), math.dot(s_d, ro ** 4),
                        math.dot(s_d, ro ** 3), math.dot(s_d, ro ** 2), math.dot(s_d, ro ** 1), math.dot(s_d, ro ** 0)]).reshape((9,))

        cov_ = tt.stack([v_1, v_2, v_3, v_4, v_5, v_6, v_7, v_8, v_9]).reshape((9, 9))
    
    elif l_freq==18:
        
        
        if mode_cov ==-1: 
            v_1 = tt.stack([math.dot(s_d, ro**0), math.dot(s_d, ro**1), math.dot(s_d, ro**2), math.dot(s_d, ro**3), math.dot(s_d, ro**4),
                            math.dot(s_d, ro**5), math.dot(s_d, ro**6), math.dot(s_d, ro**7), math.dot(s_d, ro**8),  math.dot(s_d, ro**9),
                            math.dot(s_d, ro**10), math.dot(s_d, ro**11), math.dot(s_d, ro**12), math.dot(s_d, ro**13), math.dot(s_d, ro**14),
                            math.dot(s_d, ro**15), math.dot(s_d, ro**16), math.dot(s_d, ro**17)]).reshape((l_freq, ))
            
            v_2 = tt.stack([math.dot(s_d, ro**1), math.dot(s_d, ro**0), math.dot(s_d, ro**1), math.dot(s_d, ro**2), math.dot(s_d, ro**3),
                            math.dot(s_d, ro**4), math.dot(s_d, ro**5), math.dot(s_d, ro**6), math.dot(s_d, ro**7), math.dot(s_d, ro**8), 
                            math.dot(s_d, ro**9), math.dot(s_d, ro**10), math.dot(s_d, ro**11), math.dot(s_d, ro**12), math.dot(s_d, ro**13), math.dot(s_d, ro**14),
                            math.dot(s_d, ro**15), math.dot(s_d, ro**16)]).reshape((l_freq, ))
            
            v_3 = tt.stack([math.dot(s_d, ro**2), math.dot(s_d, ro**1), math.dot(s_d, ro**0), math.dot(s_d, ro**1), math.dot(s_d, ro**2), 
                            math.dot(s_d, ro**3), math.dot(s_d, ro**4),
                            math.dot(s_d, ro**5), math.dot(s_d, ro**6), math.dot(s_d, ro**7), math.dot(s_d, ro**8),  math.dot(s_d, ro**9),
                            math.dot(s_d, ro**10), math.dot(s_d, ro**11), math.dot(s_d, ro**12), math.dot(s_d, ro**13), math.dot(s_d, ro**14),
                            math.dot(s_d, ro**15)]).reshape((l_freq, ))
            
            v_4 = tt.stack([math.dot(s_d, ro**3), math.dot(s_d, ro**2), math.dot(s_d, ro**1), math.dot(s_d, ro**0), math.dot(s_d, ro**1), 
                            math.dot(s_d, ro**2), math.dot(s_d, ro**3), math.dot(s_d, ro**4),
                            math.dot(s_d, ro**5), math.dot(s_d, ro**6), math.dot(s_d, ro**7), math.dot(s_d, ro**8),  math.dot(s_d, ro**9),
                            math.dot(s_d, ro**10), math.dot(s_d, ro**11), math.dot(s_d, ro**12), math.dot(s_d, ro**13), math.dot(s_d, ro**14)]).reshape((l_freq, ))
            
            v_5 = tt.stack([math.dot(s_d, ro**4), math.dot(s_d, ro**3), math.dot(s_d, ro**2), math.dot(s_d, ro**1), math.dot(s_d, ro**0), 
                            math.dot(s_d, ro**1), math.dot(s_d, ro**2), math.dot(s_d, ro**3), math.dot(s_d, ro**4),
                            math.dot(s_d, ro**5), math.dot(s_d, ro**6), math.dot(s_d, ro**7), math.dot(s_d, ro**8),  math.dot(s_d, ro**9),
                            math.dot(s_d, ro**10), math.dot(s_d, ro**11), math.dot(s_d, ro**12), math.dot(s_d, ro**13)]).reshape((l_freq, ))
            
            v_6 = tt.stack([math.dot(s_d, ro**5), math.dot(s_d, ro**4), math.dot(s_d, ro**3), math.dot(s_d, ro**2), math.dot(s_d, ro**1), 
                            math.dot(s_d, ro**0),math.dot(s_d, ro**1),  math.dot(s_d, ro**2), math.dot(s_d, ro**3), math.dot(s_d, ro**4),
                            math.dot(s_d, ro**5), math.dot(s_d, ro**6), math.dot(s_d, ro**7), math.dot(s_d, ro**8),  math.dot(s_d, ro**9),
                            math.dot(s_d, ro**10), math.dot(s_d, ro**11), math.dot(s_d, ro**12)]).reshape((l_freq, ))
            
            v_7 = tt.stack([math.dot(s_d, ro**6), math.dot(s_d, ro**5), math.dot(s_d, ro**4), math.dot(s_d, ro**3), math.dot(s_d, ro**2), 
                            math.dot(s_d, ro**1), math.dot(s_d, ro**0), 
                            math.dot(s_d, ro**1), math.dot(s_d, ro**2), math.dot(s_d, ro**3), math.dot(s_d, ro**4),
                            math.dot(s_d, ro**5), math.dot(s_d, ro**6), math.dot(s_d, ro**7), math.dot(s_d, ro**8),  math.dot(s_d, ro**9),
                            math.dot(s_d, ro**10), math.dot(s_d, ro**11)]).reshape((l_freq, ))
            
            v_8 = tt.stack([ math.dot(s_d, ro**7),  math.dot(s_d, ro**6), math.dot(s_d, ro**5), math.dot(s_d, ro**4), math.dot(s_d, ro**3), 
                            math.dot(s_d, ro**2), math.dot(s_d, ro**1), math.dot(s_d, ro**0), 
                            math.dot(s_d, ro**1), math.dot(s_d, ro**2), math.dot(s_d, ro**3), math.dot(s_d, ro**4),
                            math.dot(s_d, ro**5), math.dot(s_d, ro**6), math.dot(s_d, ro**7), math.dot(s_d, ro**8),  math.dot(s_d, ro**9),
                            math.dot(s_d, ro**10)]).reshape((l_freq, ))
            
            v_9 = tt.stack([math.dot(s_d, ro**8), math.dot(s_d, ro**7),  math.dot(s_d, ro**6), math.dot(s_d, ro**5), math.dot(s_d, ro**4), 
                            math.dot(s_d, ro**3), math.dot(s_d, ro**2), math.dot(s_d, ro**1), 
                            math.dot(s_d, ro**0), math.dot(s_d, ro**1), math.dot(s_d, ro**2), math.dot(s_d, ro**3), math.dot(s_d, ro**4),
                            math.dot(s_d, ro**5), math.dot(s_d, ro**6), math.dot(s_d, ro**7), math.dot(s_d, ro**8),  math.dot(s_d, ro**9)]).reshape((l_freq, ))
            
            v_10 = tt.stack([math.dot(s_d, ro**9), math.dot(s_d, ro**8), math.dot(s_d, ro**7),  math.dot(s_d, ro**6), math.dot(s_d, ro**5), 
                              math.dot(s_d, ro**4), math.dot(s_d, ro**3), math.dot(s_d, ro**2), math.dot(s_d, ro**1), 
                            math.dot(s_d, ro**0), math.dot(s_d, ro**1), math.dot(s_d, ro**2), math.dot(s_d, ro**3), math.dot(s_d, ro**4),
                            math.dot(s_d, ro**5), math.dot(s_d, ro**6), math.dot(s_d, ro**7), math.dot(s_d, ro**8)]).reshape((l_freq, ))
            
            v_11 = tt.stack([math.dot(s_d, ro**10), math.dot(s_d, ro**9), math.dot(s_d, ro**8), math.dot(s_d, ro**7),  math.dot(s_d, ro**6), 
                              math.dot(s_d, ro**5), math.dot(s_d, ro**4), math.dot(s_d, ro**3), math.dot(s_d, ro**2), math.dot(s_d, ro**1), 
                            math.dot(s_d, ro**0), math.dot(s_d, ro**1), math.dot(s_d, ro**2), math.dot(s_d, ro**3), math.dot(s_d, ro**4),
                            math.dot(s_d, ro**5), math.dot(s_d, ro**6), math.dot(s_d, ro**7)]).reshape((l_freq, ))
            
            v_12 = tt.stack([math.dot(s_d, ro**11), math.dot(s_d, ro**10), math.dot(s_d, ro**9), math.dot(s_d, ro**8), math.dot(s_d, ro**7), 
                              math.dot(s_d, ro**6), math.dot(s_d, ro**5), math.dot(s_d, ro**4), math.dot(s_d, ro**3), math.dot(s_d, ro**2), math.dot(s_d, ro**1), 
                            math.dot(s_d, ro**0), math.dot(s_d, ro**1), math.dot(s_d, ro**2), math.dot(s_d, ro**3), math.dot(s_d, ro**4),
                            math.dot(s_d, ro**5), math.dot(s_d, ro**6)]).reshape((l_freq, ))
            
            v_13 = tt.stack([math.dot(s_d, ro**12), math.dot(s_d, ro**11), math.dot(s_d, ro**10), math.dot(s_d, ro**9), math.dot(s_d, ro**8), 
                              math.dot(s_d, ro**7),  math.dot(s_d, ro**6), math.dot(s_d, ro**5), math.dot(s_d, ro**4), math.dot(s_d, ro**3), math.dot(s_d, ro**2), math.dot(s_d, ro**1), 
                            math.dot(s_d, ro**0), math.dot(s_d, ro**1), math.dot(s_d, ro**2), math.dot(s_d, ro**3), math.dot(s_d, ro**4),
                            math.dot(s_d, ro**5)]).reshape((l_freq, ))
            
            v_14 = tt.stack([math.dot(s_d, ro**13), math.dot(s_d, ro**12), math.dot(s_d, ro**11), math.dot(s_d, ro**10), math.dot(s_d, ro**9), 
                              math.dot(s_d, ro**8), math.dot(s_d, ro**7),  math.dot(s_d, ro**6), math.dot(s_d, ro**5), math.dot(s_d, ro**4), 
                              math.dot(s_d, ro**3), math.dot(s_d, ro**2), math.dot(s_d, ro**1), 
                            math.dot(s_d, ro**0), math.dot(s_d, ro**1), math.dot(s_d, ro**2), math.dot(s_d, ro**3), math.dot(s_d, ro**4)]).reshape((l_freq, ))
            
            v_15 = tt.stack([math.dot(s_d, ro**14), math.dot(s_d, ro**13), math.dot(s_d, ro**12), math.dot(s_d, ro**11), math.dot(s_d, ro**10), 
                              math.dot(s_d, ro**9), math.dot(s_d, ro**8), math.dot(s_d, ro**7),  math.dot(s_d, ro**6), math.dot(s_d, ro**5), 
                              math.dot(s_d, ro**4), math.dot(s_d, ro**3), math.dot(s_d, ro**2), math.dot(s_d, ro**1), 
                            math.dot(s_d, ro**0), math.dot(s_d, ro**1), math.dot(s_d, ro**2), math.dot(s_d, ro**3)]).reshape((l_freq, ))
            
            v_16 = tt.stack([math.dot(s_d, ro**15), math.dot(s_d, ro**14), math.dot(s_d, ro**13), math.dot(s_d, ro**12), math.dot(s_d, ro**11), 
                              math.dot(s_d, ro**10), math.dot(s_d, ro**9), math.dot(s_d, ro**8), math.dot(s_d, ro**7),  math.dot(s_d, ro**6), 
                              math.dot(s_d, ro**5), math.dot(s_d, ro**4), math.dot(s_d, ro**3), math.dot(s_d, ro**2), math.dot(s_d, ro**1), 
                            math.dot(s_d, ro**0), math.dot(s_d, ro**1), math.dot(s_d, ro**2)]).reshape((l_freq, ))
            
            v_17 = tt.stack([math.dot(s_d, ro**16), math.dot(s_d, ro**15), math.dot(s_d, ro**14), math.dot(s_d, ro**13), math.dot(s_d, ro**12), 
                              math.dot(s_d, ro**11), math.dot(s_d, ro**10), math.dot(s_d, ro**9), math.dot(s_d, ro**8), math.dot(s_d, ro**7),  
                              math.dot(s_d, ro**6), math.dot(s_d, ro**5), math.dot(s_d, ro**4), math.dot(s_d, ro**3), math.dot(s_d, ro**2), math.dot(s_d, ro**1), 
                            math.dot(s_d, ro**0), math.dot(s_d, ro**1)]).reshape((l_freq, ))
            
            v_18 = tt.stack([math.dot(s_d, ro**17), math.dot(s_d, ro**16), math.dot(s_d, ro**15), math.dot(s_d, ro**14), math.dot(s_d, ro**13), 
                              math.dot(s_d, ro**12), math.dot(s_d, ro**11), math.dot(s_d, ro**10), math.dot(s_d, ro**9), math.dot(s_d, ro**8), math.dot(s_d, ro**7), 
                              math.dot(s_d, ro**6), math.dot(s_d, ro**5), math.dot(s_d, ro**4), math.dot(s_d, ro**3), math.dot(s_d, ro**2), math.dot(s_d, ro**1), 
                            math.dot(s_d, ro**0)]).reshape((l_freq, ))
        
        elif mode_cov == 0:
        
            v_1 = tt.stack([math.dot(s_d, ro**0), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).reshape((l_freq, ))
            
            v_2 = tt.stack([0, math.dot(s_d, ro**0), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).reshape((l_freq, ))
            
            v_3 = tt.stack([0, 0, math.dot(s_d, ro**0), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).reshape((l_freq, ))
            
            v_4 = tt.stack([0, 0, 0, math.dot(s_d, ro**0), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).reshape((l_freq, ))
            
            v_5 = tt.stack([0, 0, 0, 0, math.dot(s_d, ro**0),  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).reshape((l_freq, ))
            
            v_6 = tt.stack([0, 0, 0, 0, 0, 
                            math.dot(s_d, ro**0), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).reshape((l_freq, ))
            
            v_7 = tt.stack([ 0, 0, 0, 0, 0, 0, math.dot(s_d, ro**0), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,]).reshape((l_freq, ))
            
            v_8 = tt.stack([ 0, 0, 0, 0, 0, 0, 0, math.dot(s_d, ro**0),  0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).reshape((l_freq, ))
            
            v_9 = tt.stack([0, 0, 0, 0, 0, 0, 0, 0, math.dot(s_d, ro**0), 0, 0, 0, 0, 0, 0, 0, 0, 0]).reshape((l_freq, ))
            
            v_10 = tt.stack([0, 0, 0, 0, 0, 0, 0, 0, 0, math.dot(s_d, ro**0), 0, 0, 0, 0, 0, 0, 0, 0]).reshape((l_freq, ))
            
            v_11 = tt.stack([ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, math.dot(s_d, ro**0), 0, 0, 0, 0, 0, 0, 0]).reshape((l_freq, ))
            
            v_12 = tt.stack([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                            math.dot(s_d, ro**0), 0, 0, 0, 0, 0, 0]).reshape((l_freq, ))
            
            v_13 = tt.stack([ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                            math.dot(s_d, ro**0), 0, 0, 0, 0, 0]).reshape((l_freq, ))
            
            v_14 = tt.stack([ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                            math.dot(s_d, ro**0), 0, 0, 0, 0]).reshape((l_freq, ))
            
            v_15 = tt.stack([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                            math.dot(s_d, ro**0), 0, 0, 0]).reshape((l_freq, ))
            
            v_16 = tt.stack([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                            math.dot(s_d, ro**0), 0, 0]).reshape((l_freq, ))
            
            v_17 = tt.stack([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                            math.dot(s_d, ro**0), 0]).reshape((l_freq, ))
            
            v_18 = tt.stack([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                            math.dot(s_d, ro**0)]).reshape((l_freq, ))
        
        elif mode_cov == 1:
            
            v_1 = tt.stack([math.dot(s_d, ro**0), math.dot(s_d, ro ** 1), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).reshape((l_freq, ))
            
            v_2 = tt.stack([math.dot(s_d, ro ** 1), math.dot(s_d, ro**0), math.dot(s_d, ro ** 1), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).reshape((l_freq, ))
            
            v_3 = tt.stack([0, math.dot(s_d, ro ** 1), math.dot(s_d, ro**0), math.dot(s_d, ro ** 1), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).reshape((l_freq, ))
            
            v_4 = tt.stack([0, 0, math.dot(s_d, ro ** 1), math.dot(s_d, ro**0), math.dot(s_d, ro ** 1), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).reshape((l_freq, ))
            
            v_5 = tt.stack([0, 0, 0, math.dot(s_d, ro ** 1), math.dot(s_d, ro**0),  math.dot(s_d, ro ** 1), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).reshape((l_freq, ))
            
            v_6 = tt.stack([0, 0, 0, 0, math.dot(s_d, ro ** 1), 
                            math.dot(s_d, ro**0), math.dot(s_d, ro ** 1), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).reshape((l_freq, ))
            
            v_7 = tt.stack([ 0, 0, 0, 0, 0, math.dot(s_d, ro ** 1), math.dot(s_d, ro**0), math.dot(s_d, ro ** 1), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,]).reshape((l_freq, ))
            
            v_8 = tt.stack([ 0, 0, 0, 0, 0, 0, math.dot(s_d, ro ** 1), math.dot(s_d, ro**0),  math.dot(s_d, ro ** 1), 0, 0, 0, 0, 0, 0, 0, 0, 0]).reshape((l_freq, ))
            
            v_9 = tt.stack([0, 0, 0, 0, 0, 0, 0, math.dot(s_d, ro ** 1), math.dot(s_d, ro**0), math.dot(s_d, ro ** 1), 0, 0, 0, 0, 0, 0, 0, 0]).reshape((l_freq, ))
            
            v_10 = tt.stack([0, 0, 0, 0, 0, 0, 0, 0, math.dot(s_d, ro ** 1), math.dot(s_d, ro**0), math.dot(s_d, ro ** 1), 0, 0, 0, 0, 0, 0, 0]).reshape((l_freq, ))
            
            v_11 = tt.stack([ 0, 0, 0, 0, 0, 0, 0, 0, 0, math.dot(s_d, ro ** 1), math.dot(s_d, ro**0), math.dot(s_d, ro ** 1), 0, 0, 0, 0, 0, 0]).reshape((l_freq, ))
            
            v_12 = tt.stack([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, math.dot(s_d, ro ** 1), 
                            math.dot(s_d, ro**0), math.dot(s_d, ro ** 1), 0, 0, 0, 0, 0]).reshape((l_freq, ))
            
            v_13 = tt.stack([ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, math.dot(s_d, ro ** 1),
                            math.dot(s_d, ro**0), math.dot(s_d, ro ** 1), 0, 0, 0, 0]).reshape((l_freq, ))
            
            v_14 = tt.stack([ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, math.dot(s_d, ro ** 1),
                            math.dot(s_d, ro**0), math.dot(s_d, ro ** 1), 0, 0, 0]).reshape((l_freq, ))
            
            v_15 = tt.stack([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, math.dot(s_d, ro ** 1), 
                            math.dot(s_d, ro**0), math.dot(s_d, ro ** 1), 0, 0]).reshape((l_freq, ))
            
            v_16 = tt.stack([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, math.dot(s_d, ro ** 1), 
                            math.dot(s_d, ro**0), math.dot(s_d, ro ** 1), 0]).reshape((l_freq, ))
            
            v_17 = tt.stack([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, math.dot(s_d, ro ** 1), 
                            math.dot(s_d, ro**0), math.dot(s_d, ro ** 1)]).reshape((l_freq, ))
            
            v_18 = tt.stack([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, math.dot(s_d, ro ** 1), 
                            math.dot(s_d, ro**0)]).reshape((l_freq, ))
            
        elif mode_cov == 2:
        
            v_1 = tt.stack([math.dot(s_d, ro**0), math.dot(s_d, ro ** 1), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).reshape((l_freq, ))
            
            v_2 = tt.stack([math.dot(s_d, ro ** 1), math.dot(s_d, ro**0), math.dot(s_d, ro ** 1), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).reshape((l_freq, ))
            
            v_3 = tt.stack([0, math.dot(s_d, ro ** 1), math.dot(s_d, ro**0), math.dot(s_d, ro ** 1), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).reshape((l_freq, ))
            
            v_4 = tt.stack([0, 0, math.dot(s_d, ro ** 1), math.dot(s_d, ro**0), math.dot(s_d, ro ** 1), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).reshape((l_freq, ))
            
            v_5 = tt.stack([0, 0, 0, math.dot(s_d, ro ** 1), math.dot(s_d, ro**0),  math.dot(s_d, ro ** 1), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).reshape((l_freq, ))
            
            v_6 = tt.stack([0, 0, 0, 0, math.dot(s_d, ro ** 1), 
                            math.dot(s_d, ro**0), math.dot(s_d, ro ** 1), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).reshape((l_freq, ))
            
            v_7 = tt.stack([ 0, 0, 0, 0, 0, math.dot(s_d, ro ** 1), math.dot(s_d, ro**0), math.dot(s_d, ro ** 1), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,]).reshape((l_freq, ))
            
            v_8 = tt.stack([ 0, 0, 0, 0, 0, 0, math.dot(s_d, ro ** 1), math.dot(s_d, ro**0),  math.dot(s_d, ro ** 1), 0, 0, 0, 0, 0, 0, 0, 0, 0]).reshape((l_freq, ))
            
            v_9 = tt.stack([0, 0, 0, 0, 0, 0, 0, math.dot(s_d, ro ** 1), math.dot(s_d, ro**0), math.dot(s_d, ro ** 1), 0, 0, 0, 0, 0, 0, 0, 0]).reshape((l_freq, ))
            
            v_10 = tt.stack([0, 0, 0, 0, 0, 0, 0, 0, math.dot(s_d, ro ** 1), math.dot(s_d, ro**0), math.dot(s_d, ro ** 1), 0, 0, 0, 0, 0, 0, 0]).reshape((l_freq, ))
            
            v_11 = tt.stack([ 0, 0, 0, 0, 0, 0, 0, 0, 0, math.dot(s_d, ro ** 1), math.dot(s_d, ro**0), math.dot(s_d, ro ** 1), 0, 0, 0, 0, 0, 0]).reshape((l_freq, ))
            
            v_12 = tt.stack([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, math.dot(s_d, ro ** 1), 
                            math.dot(s_d, ro**0), math.dot(s_d, ro ** 1), 0, 0, 0, 0, 0]).reshape((l_freq, ))
            
            v_13 = tt.stack([ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, math.dot(s_d, ro ** 1),
                            math.dot(s_d, ro**0), math.dot(s_d, ro ** 1), 0, 0, 0, 0]).reshape((l_freq, ))
            
            v_14 = tt.stack([ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, math.dot(s_d, ro ** 1),
                            math.dot(s_d, ro**0), math.dot(s_d, ro ** 1), 0, 0, 0]).reshape((l_freq, ))
            
            v_15 = tt.stack([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, math.dot(s_d, ro ** 1), 
                            math.dot(s_d, ro**0), math.dot(s_d, ro ** 1), 0, 0]).reshape((l_freq, ))
            
            v_16 = tt.stack([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, math.dot(s_d, ro ** 1), 
                            math.dot(s_d, ro**0), math.dot(s_d, ro ** 1), 0]).reshape((l_freq, ))
            
            v_17 = tt.stack([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, math.dot(s_d, ro ** 1), 
                            math.dot(s_d, ro**0), math.dot(s_d, ro ** 1)]).reshape((l_freq, ))
            
            v_18 = tt.stack([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, math.dot(s_d, ro ** 1), 
                            math.dot(s_d, ro**0)]).reshape((l_freq, ))
        
        cov_ = tt.stack([v_1, v_2, v_3, v_4, v_5, v_6, v_7, v_8, v_9, v_10, 
                         v_11, v_12, v_13, v_14, v_15, v_16, v_17, v_18]).reshape((l_freq, l_freq))
        

        return cov_


def calc_cov_pre_analysis(signal, dt):
    
    # data = data_cosine(N=2048, A=0.1, sampling=1024, freq=200)

    NW=9
    N_ = signal.shape[0]

    xf = np.linspace(0.0, 1.0/(2.0*dt), N_//2)
    [tapers, eigen] = dpss(signal.shape[0], NW)

    Sk_complex, weights, eigenvalues= pmtm(signal, e=eigen, v=tapers, show=False)
    Sk = abs(Sk_complex)**2

    mean_val_Sk = np.mean(Sk * np.transpose(weights), axis=0) * dt

    f_max = 50
    ind_ = np.asarray(np.where(np.min(np.abs(xf-f_max)) == np.abs(xf-f_max))).reshape((1,))
    ind_f_max = ind_[0]

    plt_ = 0
    if plt_:
        plt.figure()
        plt.plot(xf[0:ind_f_max], np.log(mean_val_Sk[0:ind_f_max]))
        plt.title('all tapers')
        plt.show()

    plt_ = 0
    if plt_:
        for ii in range(5):
            plt.figure()
            plt.plot(xf[0:ind_f_max], np.log(Sk[ii, 0:ind_f_max]))
            plt.title('taper= ' + str(ii))
            plt.show()


    # calculate covariance matrix of adjancent frequencies
    Sk_ = np.transpose(np.log(Sk[:, 0:ind_f_max]))
    f_ind = xf[0:ind_f_max]
    cov_mat = np.cov(Sk_)

    plt_ = 0
    if plt_:
        plt.figure()
        plt.imshow(cov_mat, extent=[f_ind[0], f_ind[-1], f_ind[-1], f_ind[0]])
        plt.colorbar()
        plt.xlabel('freq')
        plt.ylabel('freq')
        plt.title('covariance matrix')
        plt.show()

    k=1

### 3 - Example of simulated EEG using AR($p$), generated by specifying $R$ AR(1) and $C$ AR(2) components
def calc_ken_data_generation(dt): 
    import buildsignal as bs
    import nitime.algorithms as tsa
    
    f1 = 21.  # hz
    coh1 = 0.982  # how coherent or sinusoidal is f1 osc?  0~1    1 near sinusoidal, > 0.95 neural-like

    obsnpz = 0.8  # make this bigger to add more observation noise
    comps = [[[f1, coh1]], [0.96]]  # describe AR(p) in terms of AR(2) and AR(1) components.
    N = 4096
    #
    signal = bs.build_signal(N, dt, [comps], [1])
    signal += obsnpz * np.random.randn(N)  # add observation noise (white noise)


    # multi-taper: for now, let's work on the average of tapers
    # focus on 1 to 30 Hz
    bw = 9
    max_f = 50
    fs = 1 / dt
    f, psd_mt, nu = tsa.multi_taper_psd(signal, Fs=fs, NW=bw, low_bias=False, adaptive=False, jackknife=False)



    num_tapers = nu[0]/2

    plt_ = 0
    if plt_:
        fig = plt.figure(figsize=(14, 5))
        plt.subplot2grid((1, 4), (0, 0), colspan=3)
        plt.plot(signal)
        plt.subplot2grid((1, 4), (0, 3), colspan=1)
        plt.plot(f, np.log(psd_mt), '*')
        plt.plot(f, np.log(psd_mt))
        plt.xlim(0, max_f)
        plt.show()

    theseInds = np.where(f < max_f)[0]
    f_psd = f[theseInds]
    k=1
    
    return f, psd_mt, nu