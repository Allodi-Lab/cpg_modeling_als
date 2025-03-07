#!/usr/bin/env python

import nest
import numpy as np
import pathlib, sys
import pylab
import math
import matplotlib.pyplot as plt
import random
import pickle, yaml
import time, datetime
import re

#Import parameters for network
file = open(r'configuration_run_nest.yaml')
args = yaml.load(file, Loader=yaml.FullLoader)
print(f'\nLoading parameters from configuration file:\n')
if len(sys.argv) > 2:  
    input_string = sys.argv[2]
    match = re.match(r"P(\d+)_D(\d+)", input_string)
    if match:
        days = int(match.group(1))  # Number between P and _
        freq_test = int(match.group(2))  # Number after D
    else:
        print("Invalid test name.")
else:
    days = int(args['days_after_onset'])
    freq_test = int(args['freq_test'])

class neural_network():
    def __init__(self):
        self.args = args
        self.save_all_pops = args['save_all_pops']
        stabilized = int(args['synaptically_stabilized'])
        compensation = int(args['synaptic_compensation'])
        self.sparsity_rg_layer = 0.1 #Use this to update the RG-V1/V2b-RG connectivity 
        self.sparsity_v1_rg = 0.11 if args['v1v2b_mn_connected']==1 else 0.1 #Use this to update the V1-RG connectivity/change ext BD
        self.sparsity_v1s_outside_rg_layer = 0.3 
        
        if days==0:
            print('Running simulation P0 (healthy)')
            #Feedback (1a afferents)
            self.fb_multiplier = 0.5
            #V1 synaptic connectivity         
            self.sparsity_custom_v1_rg = self.sparsity_v1_rg   
            self.sparsity_custom_v1_v2a = self.sparsity_v1s_outside_rg_layer 
            self.sparsity_custom_v1_mn = self.sparsity_v1s_outside_rg_layer 
            self.sparsity_custom_rc_mn = self.sparsity_v1s_outside_rg_layer 
            self.sparsity_custom_v1a_mn = self.sparsity_v1s_outside_rg_layer
            self.sparsity_custom_rc_v1a = self.sparsity_v1s_outside_rg_layer 
            self.sparsity_custom_rc_rc = self.sparsity_v1s_outside_rg_layer
            self.sparsity_custom_v1a_v1a = self.sparsity_v1s_outside_rg_layer  
            #V1 synaptic strength
            self.w_1a_multiplier = 1 #P45 D1/D2=2.5; P63 D1/D2=3.5; P112 D1/D2=4.75 
            self.w_rc_multiplier = 1 #P45 D1/D2=2.5; P63 D1/D2=3.5; P112 D1/D2=4.75 
            self.w_v1_multiplier = 1 #P45 D1=1.6, D2=1.8; P63 D1=3.1, D2=2.9;P112 D1=4.75, D2=5.5
            #Population size
            self.inh_inter_pop_size= 102 #Healthy(V1/V2b)=102, use v1_pct_surviving to adjust cell death of V1 interneurons
            self.v1_pct_surviving= 1. #Healthy=1., P63=.8084, P112=.5465 
            self.v2a_tonic_pop_size= 54 #Healthy=54, P63=54, P112=23
            self.v2a_bursting_pop_size= 0 
            self.v1a_pop_size= 21 #Healthy=21, P63=12, P112=11
            self.v0c_pop_size= 6 #Healthy=6
            self.rc_pop_size= 12 #Healthy=12, P63=6, P112=2
            self.num_motor_neurons= 45 #Healthy=45, P63=45, P112=32
        elif days==45:
            print('Running simulation P45')
            #Feedback (1a afferents)
            self.fb_multiplier = 0.5
            #Synaptic connectivity        
            self.sparsity_custom_v1_rg = self.sparsity_v1_rg/2
            self.sparsity_custom_v1_v2a = self.sparsity_v1s_outside_rg_layer/2
            self.sparsity_custom_v1_mn = self.sparsity_v1s_outside_rg_layer/2 
            self.sparsity_custom_rc_mn = self.sparsity_v1s_outside_rg_layer/2 
            self.sparsity_custom_v1a_mn = self.sparsity_v1s_outside_rg_layer/2
            self.sparsity_custom_rc_v1a = self.sparsity_v1s_outside_rg_layer/2
            self.sparsity_custom_rc_rc = self.sparsity_v1s_outside_rg_layer/2
            self.sparsity_custom_v1a_v1a = self.sparsity_v1s_outside_rg_layer/2
            #V1 synaptic strength
            self.w_1a_multiplier = 2.75 if compensation==1 else 1.  #TESTING 2.5
            self.w_rc_multiplier = 2.75 if compensation==1 else 1.  #TESTING 2.5
            self.w_v1_multiplier = 1.8 if compensation==1 and freq_test==1 else 1.6 if compensation==1 and freq_test==2 else 1. #TESTING 1.6, 1.8
            #Population size
            self.inh_inter_pop_size= 102 
            self.v1_pct_surviving= 1.  
            self.v2a_tonic_pop_size= 54 
            self.v2a_bursting_pop_size= 0 
            self.v1a_pop_size= 21 
            self.v0c_pop_size= 6 
            self.rc_pop_size= 12 
            self.num_motor_neurons= 45      
        elif days==63:
            print('Running simulation P63')
            #Feedback (1a afferents)
            self.fb_multiplier = 0.5           
            #Synaptic connectivity
            self.sparsity_custom_v1_rg = self.sparsity_v1_rg/2
            self.sparsity_custom_v1_v2a = self.sparsity_v1s_outside_rg_layer/2
            self.sparsity_custom_v1_mn = self.sparsity_v1s_outside_rg_layer/2
            self.sparsity_custom_rc_mn = self.sparsity_v1s_outside_rg_layer/2 
            self.sparsity_custom_v1a_mn = self.sparsity_v1s_outside_rg_layer/2
            self.sparsity_custom_rc_v1a = self.sparsity_v1s_outside_rg_layer/2
            self.sparsity_custom_rc_rc = self.sparsity_v1s_outside_rg_layer/2
            self.sparsity_custom_v1a_v1a = self.sparsity_v1s_outside_rg_layer/2
            #V1 synaptic strength
            self.w_1a_multiplier = 3.25 if compensation==1 and freq_test==1 else 4. if compensation==1 and freq_test==2 else 1.  #P112 D1/D2=4.75 TESTING 3.5
            self.w_rc_multiplier = 3.25 if compensation==1 and freq_test==1 else 4. if compensation==1 and freq_test==2 else 1. #P112 D1/D2=4.75 TESTING 3.5
            self.w_v1_multiplier = 2.5 if compensation==1 and freq_test==1 else 3.5 if compensation==1 and freq_test==2 else 1. #P112 D1=4.75; D2=5.5 TESTING 3.1, 2.5           
            #Population size
            self.inh_inter_pop_size= 102 
            self.v1_pct_surviving= .8084  
            self.v2a_tonic_pop_size= 54 
            self.v2a_bursting_pop_size= 0 
            self.v1a_pop_size= 12            
            self.v0c_pop_size= 6 
            self.rc_pop_size= 6 
            self.num_motor_neurons= 45   
        elif days==112 and stabilized==0:
            print('Running simulation P112')
            #Feedback (1a afferents)
            self.fb_multiplier = 0.5      
            #Synaptic connectivity 
            self.sparsity_custom_v1_rg = self.sparsity_v1_rg/2 
            self.sparsity_custom_v1_v2a = self.sparsity_v1s_outside_rg_layer/2
            self.sparsity_custom_v1_mn = self.sparsity_v1s_outside_rg_layer/2
            self.sparsity_custom_rc_mn = self.sparsity_v1s_outside_rg_layer/2           
            self.sparsity_custom_v1a_mn = self.sparsity_v1s_outside_rg_layer/2          
            self.sparsity_custom_rc_v1a = self.sparsity_v1s_outside_rg_layer/2                   
            self.sparsity_custom_rc_rc = self.sparsity_v1s_outside_rg_layer/2                   
            self.sparsity_custom_v1a_v1a = self.sparsity_v1s_outside_rg_layer/2
            #V1 synaptic strength
            self.w_1a_multiplier = 4.75 if compensation==1 and args['v2a_mn_intact']==0 else 3.5 if compensation==1 and args['v2a_mn_intact']==1 else 1.  #TESTING 4.75
            self.w_rc_multiplier = 4.75 if compensation==1 and args['v2a_mn_intact']==0 else 3.5 if compensation==1 and args['v2a_mn_intact']==1 else 1. #TESTING 4.75
            self.w_v1_multiplier = 10. if compensation==1 and freq_test==1 and args['v2a_mn_intact']==0 else 10. if compensation==1 and freq_test==2 and args['v2a_mn_intact']==0 else 4. if compensation==1 and freq_test==1 and args['v2a_mn_intact']==1 else 5. if compensation==1 and freq_test==2 and args['v2a_mn_intact']==1 else 1. #TESTING 4.75, 5.5, 3.5, 3.25 
            #Population size
            self.inh_inter_pop_size= 102 
            self.v1_pct_surviving= .5465 
            self.v2a_tonic_pop_size= 23 if args['v2a_mn_intact']==0 else 54
            self.v2a_bursting_pop_size= 0 
            self.v1a_pop_size= 11            
            self.v0c_pop_size= 6        
            self.rc_pop_size= 2 
            self.num_motor_neurons= 32 if args['v2a_mn_intact']==0 else 45     
        elif days==112 and stabilized==1:
            print('Running simulation P112 with synaptic stabilization')
            #Feedback (1a afferents)
            self.fb_multiplier = 0.5      
            #Synaptic connectivity                     
            self.sparsity_custom_v1_rg = self.sparsity_v1_rg
            self.sparsity_custom_v1_v2a = self.sparsity_v1s_outside_rg_layer
            self.sparsity_custom_v1_mn = self.sparsity_v1s_outside_rg_layer
            self.sparsity_custom_rc_mn = self.sparsity_v1s_outside_rg_layer 
            self.sparsity_custom_v1a_mn = self.sparsity_v1s_outside_rg_layer
            self.sparsity_custom_rc_v1a = self.sparsity_v1s_outside_rg_layer 
            self.sparsity_custom_rc_rc = self.sparsity_v1s_outside_rg_layer 
            self.sparsity_custom_v1a_v1a = self.sparsity_v1s_outside_rg_layer
            #V1 synaptic strength
            self.w_1a_multiplier = 4.75 if compensation==1 else 1.  
            self.w_rc_multiplier = 4.75 if compensation==1 else 1. 
            self.w_v1_multiplier = 4.75 if compensation==1 and freq_test==1 else 5.5 if compensation==1 and freq_test==2 else 1.
            self.w_v2a_multiplier = 2. if freq_test==2 else 1.
            #Population size             
            self.inh_inter_pop_size= 102 
            self.v1_pct_surviving= 1. 
            self.v2a_tonic_pop_size= 23     
            self.v2a_bursting_pop_size= 0 
            self.v1a_pop_size= 21 
            self.v0c_pop_size= 6 
            self.rc_pop_size= 12 
            self.num_motor_neurons= 32
        #Shared synaptic strengths across timepoints and frequencies
        self.w_exc_multiplier = 0.5 #Use this to change the RG balance through excitatory weights
        self.ratio_exc_inh = 4
        self.w_exc_mean = 0.6/self.ratio_exc_inh+self.w_exc_multiplier*0.6 
        self.w_exc_std = 0.12 
        self.w_inh_mean = -0.6 if args['remove_inhibition'] == 0 else 0.6
        self.w_inh_std = 0.12 
        self.w_custom_rg_rg_mean = 0.01        
        self.w_custom_rg_rg_std = 0.09  
        self.w_custom_rg_v1a_mean = 0.45    
        self.w_custom_rg_v1a_std = 0.12 
        self.w_custom_rc_v1a_mean = -0.024*self.w_rc_multiplier if compensation==1 else -0.024 
        self.w_custom_rc_v1a_std = 0.09 
        self.w_custom_rc_rc_mean = -4.32*self.w_rc_multiplier if compensation==1 else -4.32    
        self.w_custom_rc_rc_std = 0.36  
        self.w_custom_rc_mn_mean= -0.2*self.w_rc_multiplier if compensation==1 else -0.2     
        self.w_custom_rc_mn_std= 0.12   
        self.w_custom_mn_rc_mean= 1.8    
        self.w_custom_mn_rc_std= 0.12
        self.w_custom_rg_v1_mean= 0.01  
        self.w_custom_rg_v1_std= 0.09          
        self.w_custom_rg_v2b_mean= 0.01 
        self.w_custom_rg_v2b_std= 0.09         
        self.w_custom_v1a_mn_mean= -4.14*self.w_1a_multiplier if compensation==1 else -4.14        
        self.w_custom_v1a_mn_std= .12
        self.w_custom_v1a_v1a_mean= -0.69*self.w_1a_multiplier if compensation==1 else -0.69     
        self.w_custom_v1a_v1a_std= 0.12
        self.w_custom_v2b_v2a_mean= -1.3     
        self.w_custom_v2b_v2a_std= .045     
        self.w_custom_v1_v2a_mean= -1.495*self.w_v1_multiplier if compensation==1 else -1.495   
        self.w_custom_v1_v2a_std= .07
        self.w_custom_v2b_mn_mean= -0.6   
        self.w_custom_v2b_mn_std= .07     
        self.w_custom_v1_mn_mean= -0.69*self.w_v1_multiplier if compensation==1 else -0.69    
        self.w_custom_v1_mn_std= .07      
        self.w_custom_v2b_rg_mean= -0.01      
        self.w_custom_v2b_rg_std= .08           
        self.w_custom_rg_v2a_mean = 5.28         
        self.w_custom_rg_v2a_std = 0.12         
        self.w_custom_rg_v0c_mean = 3.6         
        self.w_custom_rg_v0c_std = 0.12
        self.w_custom_v0c_mn_mean= 10.56 if compensation==1 and days==112 else 5.28
        self.w_custom_v0c_mn_std= 0.24 if compensation==1 and days==112 else 0.12
        self.w_custom_v2a_mn_mean = 3.6*self.w_v2a_multiplier if stabilized==1 else 3.6          
        self.w_custom_v2a_mn_std = 0.12         
        self.w_custom_v2a_selfexc_mean = 0.5    
        self.w_custom_v2a_selfexc_std = 0.12
        #Shared synaptic connectivity % across timepoints and frequencies
        self.sparsity_rg = 0.03 #0.03                                               
        self.sparsity_rg1_rg2 = 0.01 #0.01                                           
        #self.sparsity_cpg = 0.09 #0.09      
        self.sparsity_custom_rg_v1 = self.sparsity_rg_layer #0.01                                   
        self.sparsity_custom_rg_v2b = self.sparsity_rg_layer #0.01        
        self.sparsity_custom_rg_v2a = 0.5 #0.5
        self.sparsity_custom_rg_v1a = 0.7 #0.7   
        self.sparsity_custom_v2b_rg = self.sparsity_rg_layer #0.01                                 
        self.sparsity_custom_v2b_v2a = 0.5 #0.5
        self.sparsity_custom_v2b_mn = 0.5 #0.5
        self.sparsity_custom_rg_v0c = 0.5 #0.5         
        self.sparsity_custom_v0c_mn = 0.5 #0.5         
        self.sparsity_custom_v2a_mn = 0.5 #0.5       
        self.sparsity_custom_mn_rc = 0.5 #0.5
        self.selfexc_flx = 0.12  #0.12
        self.selfexc_ext = 0.12  #0.12   
        #Rise and decay times for synapses
        inh_syn_multiplier = 1 if days==0 or args['slow_syn_dyn']==0 else 1.6 
        exc_syn_multiplier = 1 if days==0 or args['slow_syn_dyn']==0 else 1.6 
        self.tau_syn_e_rise = 0.2              
        self.tau_syn_e_decay = 1.0 
        self.tau_syn_e_rise_rc = 0.2 * exc_syn_multiplier #set Exc tau for effect from MNs to RCs           
        self.tau_syn_e_decay_rc = 1.0 * exc_syn_multiplier #set Exc tau for effect from MNs to RCs
        self.tau_syn_i_rise = 0.5 
        self.tau_syn_i_decay = 20.0
        self.tau_syn_i_rise_mn = 0.5 * inh_syn_multiplier #set Inh tau for effect from RCs, 1as to MNs 
        self.tau_syn_i_decay_mn = 20.0 * inh_syn_multiplier #set Inh tau for effect from RCs, 1as to MNs
        print('Excitatory synaptic rise/decay (RC)',self.tau_syn_e_rise_rc,self.tau_syn_e_decay_rc)
        print('Inhibitory synaptic rise/decay (MN)',self.tau_syn_i_rise_mn,self.tau_syn_i_decay_mn)
        #Shared population characteristics across timepoints and frequencies
        self.rg_pop_neurons= 1950   #1950        
        self.rg_ext_exc_pct_tonic= 0.9 #0.9     
        self.rg_ext_inh_pct_tonic= 0.9 #0.9
        self.inh_inter_pct_tonic= 1. 
        
        if freq_test==1:            
            self.I_e_bursting_mean = 300 #pA    1pct 200          
            self.I_e_tonic_mean = 480 #pA       1pct 320                     
            self.rg_flx_exc_pct_tonic= 0.7            
            self.rg_flx_inh_pct_tonic= 0.7
            self.w_custom_v1_rg_mean= -0.23*self.w_v1_multiplier if compensation == 1 else -0.4 if stabilized == 1 else -0.23 #1pct -2.0; 10pct -0.2 TESTING -0.1
            self.w_custom_v1_rg_std= .08
        elif freq_test==2:
            self.I_e_bursting_mean = 450 #pA    1pct 400  
            self.I_e_tonic_mean = 720 #pA       1pct 640    
            self.rg_flx_exc_pct_tonic= 0.7            
            self.rg_flx_inh_pct_tonic= 0.7      
            self.w_custom_v1_rg_mean= -0.125*self.w_v1_multiplier if compensation==1 else -0.225 if stabilized == 1 else -0.125 #1pct -0.9; 10pct -0.125 TESTING -0.1
            self.w_custom_v1_rg_std= .08                   
        
        self.I_fb_bursting_mean = self.I_e_bursting_mean*self.fb_multiplier      
        self.I_fb_tonic_mean = self.I_e_tonic_mean*self.fb_multiplier 
        
        #Set parameters for network
        if len(sys.argv) > 1:
            self.rng_seed = int(sys.argv[1]) 
        else:
            self.rng_seed = args['seed']
        #self.rng_seed = np.random.randint(10**7) if args['seed'] == 0 else args['seed'] #set seed for NEST 	
        self.time_resolution = args['delta_clock'] 		#equivalent to "delta_clock"
        self.num_inh_inter_tonic_v2b = round(self.inh_inter_pop_size*self.inh_inter_pct_tonic)
        self.num_inh_inter_bursting_v2b = self.inh_inter_pop_size-self.num_inh_inter_tonic_v2b
        self.num_inh_inter_tonic_v1 = int(self.num_inh_inter_tonic_v2b*self.v1_pct_surviving)
        self.num_inh_inter_bursting_v1 = int(self.num_inh_inter_bursting_v2b*self.v1_pct_surviving)
        self.exc_neurons_count = int(np.round(self.rg_pop_neurons * (self.ratio_exc_inh / (self.ratio_exc_inh + 1)))) # N_E = N*(r / (r+1))
        self.inh_neurons_count = int(np.round(self.rg_pop_neurons * ( 1 / (self.ratio_exc_inh + 1)))) # N_I = N*(1 / (r+1))
        self.flx_exc_tonic_count = round(self.exc_neurons_count*self.rg_flx_exc_pct_tonic)
        self.flx_exc_bursting_count = self.exc_neurons_count-self.flx_exc_tonic_count
        self.flx_inh_tonic_count = round(self.inh_neurons_count*self.rg_flx_inh_pct_tonic)
        self.flx_inh_bursting_count = self.inh_neurons_count-self.flx_inh_tonic_count
        self.ext_exc_tonic_count = round(self.exc_neurons_count*self.rg_ext_exc_pct_tonic)
        self.ext_exc_bursting_count = self.exc_neurons_count-self.ext_exc_tonic_count
        self.ext_inh_tonic_count = round(self.inh_neurons_count*self.rg_ext_inh_pct_tonic)
        self.ext_inh_bursting_count = self.inh_neurons_count-self.ext_inh_tonic_count
        self.sim_time = args['t_steps']         #time in ms
        
        #Initialize neuronal parameters
        self.V_th_mean_tonic = -50.0 #mV  
        self.V_th_std_tonic = 1.0 #mV
        self.V_th_v1v2b_mean_tonic = -50.0 #mV  
        self.V_th_v1v2b_std_tonic = 1.0 #mV
        self.V_th_mean_bursting = -52.0 #mV  
        self.V_th_std_bursting = 1.0 #mV
        self.V_m_mean = -60.0 #mV 
        self.V_m_std = 10.0 #mV
        self.C_m_bursting_mean = 400.0 #pF  
        self.C_m_bursting_std = 80.0 #pF   
        self.C_m_bursting_ext_mean = 500.0 #pF  
        self.C_m_bursting_ext_std = 100.0 #pF   
        self.C_m_tonic_mean = 200.0 #pF         
        self.C_m_tonic_std = 40.0 #pF      
        self.C_m_v1v2b_tonic_mean = 200.0 #pF         
        self.C_m_v1v2b_tonic_std = 40.0 #pF        
        self.t_ref_mean = 1.0 #ms
        self.t_ref_std = 0.2 #ms
        
        self.synaptic_delay = 2. #args['synaptic_delay']
        #self.noise_std_dev_tonic = args['noise_amplitude_tonic'] #pA
        #self.noise_std_dev_bursting = args['noise_amplitude_bursting'] #pA
        self.rgs_connected = args['rgs_connected']
        self.v1v2b_mn_connected = args['v1v2b_mn_connected']
       
        print('Running freq test ',freq_test,', Mean desc current (T,B), Mean fb current (T,B): ',self.I_e_tonic_mean,self.I_e_bursting_mean,self.I_fb_tonic_mean,self.I_fb_bursting_mean)
        self.I_e_bursting_std = 0.25*self.I_e_bursting_mean #pA 
        self.I_e_tonic_std = 0.25*self.I_e_tonic_mean #pA
        self.I_fb_bursting_std = 0.25*self.I_fb_bursting_mean #pA 
        self.I_fb_tonic_std = 0.25*self.I_fb_tonic_mean #pA
        self.noise_std_dev_tonic = self.I_e_tonic_mean #pA
        self.noise_std_dev_bursting = self.I_e_bursting_mean #pA
        print('Noise standard deviation (T,B) ',self.noise_std_dev_tonic,self.noise_std_dev_bursting)

        #Set data evaluation parameters
        self.convstd_rate = args['convstd_rate']
        self.convstd_pca = args['convstd_pca']
        self.chop_edges_amount = args['chop_edges_amount']
        self.remove_mean = args['remove_mean']
        self.high_pass_filtered = args['high_pass_filtered']
        self.downsampling_convolved = args['downsampling_convolved']
        self.remove_silent = args['remove_silent']
        self.PCA_components = args['PCA_components']
        self.calculate_balance = args['calculate_balance']               
        self.raster_plot = args['raster_plot']
        self.rate_coded_plot = args['rate_coded_plot']
        self.spike_distribution_plot = args['spike_distribution_plot']
        self.pca_plot = args['pca_plot']
        self.phase_ordered_plot = args['phase_ordered_plot']
        self.membrane_potential_plot = args['membrane_potential_plot']
        self.time_window = args['smoothing_window']

        #Set spike detector parameters 
        self.sd_params = {"withtime" : True, "withgid" : True, 'to_file' : False, 'flush_after_simulate' : False, 'flush_records' : True}
        
        #Set connection parameters
        self.conn_dict_custom_selfexc_flx = {'rule': 'pairwise_bernoulli', 'p': self.selfexc_flx}
        self.conn_dict_custom_selfexc_ext = {'rule': 'pairwise_bernoulli', 'p': self.selfexc_ext}
        self.conn_dict_custom_rg = {'rule': 'pairwise_bernoulli', 'p': self.sparsity_rg}		
        self.conn_dict_custom_rg1_rg2 = {'rule': 'pairwise_bernoulli', 'p': self.sparsity_rg1_rg2}
        #self.conn_dict_custom_cpg = {'rule': 'pairwise_bernoulli', 'p': self.sparsity_cpg}
        self.conn_dict_custom_v1a_mn = {'rule': 'pairwise_bernoulli', 'p': self.sparsity_custom_v1a_mn}
        self.conn_dict_custom_v1a_v1a = {'rule': 'pairwise_bernoulli', 'p': self.sparsity_custom_v1a_v1a}
        self.conn_dict_custom_rg_v1 = {'rule': 'pairwise_bernoulli', 'p': self.sparsity_custom_rg_v1}
        self.conn_dict_custom_rg_v2b = {'rule': 'pairwise_bernoulli', 'p': self.sparsity_custom_rg_v2b}
        self.conn_dict_custom_rg_v2a = {'rule': 'pairwise_bernoulli', 'p': self.sparsity_custom_rg_v2a}
        self.conn_dict_custom_rg_v1a = {'rule': 'pairwise_bernoulli', 'p': self.sparsity_custom_rg_v1a}
        self.conn_dict_custom_v1_rg = {'rule': 'pairwise_bernoulli', 'p': self.sparsity_custom_v1_rg}
        self.conn_dict_custom_v2b_rg = {'rule': 'pairwise_bernoulli', 'p': self.sparsity_custom_v2b_rg}
        self.conn_dict_custom_v1_v2a = {'rule': 'pairwise_bernoulli', 'p': self.sparsity_custom_v1_v2a}
        self.conn_dict_custom_v2b_v2a = {'rule': 'pairwise_bernoulli', 'p': self.sparsity_custom_v2b_v2a}
        self.conn_dict_custom_v1_mn = {'rule': 'pairwise_bernoulli', 'p': self.sparsity_custom_v1_mn}
        self.conn_dict_custom_v2b_mn = {'rule': 'pairwise_bernoulli', 'p': self.sparsity_custom_v2b_mn}
        self.conn_dict_custom_rg_v0c = {'rule': 'pairwise_bernoulli', 'p': self.sparsity_custom_rg_v0c}
        self.conn_dict_custom_v0c_mn = {'rule': 'pairwise_bernoulli', 'p': self.sparsity_custom_v0c_mn}
        self.conn_dict_custom_v2a_mn = {'rule': 'pairwise_bernoulli', 'p': self.sparsity_custom_v2a_mn}
        self.conn_dict_custom_rc_v1a = {'rule': 'pairwise_bernoulli', 'p': self.sparsity_custom_rc_v1a}
        self.conn_dict_custom_rc_rc = {'rule': 'pairwise_bernoulli', 'p': self.sparsity_custom_rc_rc}
        self.conn_dict_custom_rc_mn = {'rule': 'pairwise_bernoulli', 'p': self.sparsity_custom_rc_mn}
        self.conn_dict_custom_mn_rc = {'rule': 'pairwise_bernoulli', 'p': self.sparsity_custom_mn_rc}

        #Set multimeter parameters
        self.mm_params = {'interval': 1., 'record_from': ['V_m']}

        #Set noise parameters
        self.noise_params_tonic = {"dt": self.time_resolution, "std":self.noise_std_dev_tonic}
        self.noise_params_bursting = {"dt": self.time_resolution, "std":self.noise_std_dev_bursting}
   
    ################
    # Save results #
    ################
    if args['save_results'] and not args['optimizing']:
        #id_ = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        if len(sys.argv) > 1:
            id_ = str(int(sys.argv[1]))
        else:
            id_ = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        path = 'saved_simulations' + '/' + 'P'+str(days)+'_D'+str(freq_test) + '/' + id_ 
        pathFigures = 'saved_simulations' + '/' + 'P'+str(days)+'_D'+str(freq_test) + '/' + id_ + '/Figures'
        pathlib.Path(path).mkdir(parents=True, exist_ok=False)
        pathlib.Path(pathFigures).mkdir(parents=True, exist_ok=False)
        with open(path + '/args_' + id_ + '.yaml', 'w') as yamlfile:
            #args['seed'] = simulation_config['seed']
            yaml.dump(args, yamlfile)
      

