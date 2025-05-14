#!/usr/bin/env python

#include <static_connection.h>
import nest
import nest.raster_plot
import numpy as np
import sys
import pylab
import math
import matplotlib.pyplot as plt
import pickle, yaml
import random
import scipy
import scipy.fftpack
from scipy.signal import find_peaks, peak_widths, peak_prominences
import time
import numpy as np
import copy
from set_network_params import neural_network
netparams = neural_network()

class interneuron_population():
    def __init__(self):        
        pass
        
    def create_interneuron_population(self,pop_type,self_connection,firing_behavior,pop_size,input_type):
        #Create population
        self.spike_detector = nest.Create("spike_recorder",pop_size) 
        self.multimeter = nest.Create("multimeter",netparams.mm_params)
        if firing_behavior == 'tonic':
            self.I_e = nest.random.normal(mean=netparams.I_e_tonic_mean, std=netparams.I_e_tonic_std) if input_type=='descending' else nest.random.normal(mean=netparams.I_fb_tonic_mean, std=netparams.I_fb_tonic_std) if input_type=='sensory_feedback' else 0 
            self.tonic_neuronparams = {'C_m':nest.random.normal(mean=netparams.C_m_tonic_mean, std=netparams.C_m_tonic_std), 'g_L':10.,'E_L':-70.,'V_th':nest.random.normal(mean=netparams.V_th_mean_tonic, std=netparams.V_th_std_tonic),'Delta_T':2.,'tau_w':30., 'a':3., 'b':0., 'V_reset':-58., 'I_e':self.I_e,'t_ref':nest.random.normal(mean=netparams.t_ref_mean, std=netparams.t_ref_std),'V_m':nest.random.normal(mean=netparams.V_m_mean, std=netparams.V_m_std),"tau_syn_rise_I": netparams.tau_syn_i_rise, "tau_syn_decay_I": netparams.tau_syn_i_decay, "tau_syn_rise_E": netparams.tau_syn_e_rise, "tau_syn_decay_E": netparams.tau_syn_e_decay}
            self.interneuron_pop = nest.Create('aeif_cond_beta_aeif_cond_beta_nestml',pop_size,self.tonic_neuronparams)            
            self.white_noise = nest.Create("noise_generator",netparams.noise_params_tonic)                       
        elif firing_behavior == 'bursting' and pop_type == 'rc_slow_syn_enabled':
            self.I_e = nest.random.normal(mean=netparams.I_e_bursting_mean, std=netparams.I_e_bursting_std) if input_type=='descending' else nest.random.normal(mean=netparams.I_fb_bursting_mean, std=netparams.I_fb_bursting_std) if input_type=='sensory_feedback' else 0
            self.bursting_neuronparams = {'C_m':nest.random.normal(mean=netparams.C_m_bursting_mean, std=netparams.C_m_bursting_std), 'g_L':26.,'E_L':-60.,'V_th':nest.random.normal(mean=netparams.V_th_mean_bursting, std=netparams.V_th_std_bursting),'Delta_T':2.,'tau_w':130., 'a':-11., 'b':30., 'V_reset':-48., 'I_e':nest.random.normal(mean=netparams.I_e_bursting_mean, std=netparams.I_e_bursting_std),'t_ref':nest.random.normal(mean=netparams.t_ref_mean, std=netparams.t_ref_std),'V_m':nest.random.normal(mean=netparams.V_m_mean, std=netparams.V_m_std),"tau_syn_rise_I": netparams.tau_syn_i_rise, "tau_syn_decay_I": netparams.tau_syn_i_decay, "tau_syn_rise_E": netparams.tau_syn_e_rise_rc, "tau_syn_decay_E": netparams.tau_syn_e_decay_rc} #bursting, Naud et al. 2008, C = pF; g_L = nS    
            self.interneuron_pop = nest.Create('aeif_cond_beta_aeif_cond_beta_nestml',pop_size,self.bursting_neuronparams)
            self.white_noise = nest.Create("noise_generator",netparams.noise_params_bursting)
        elif firing_behavior == 'bursting' and pop_type == 'rc_slow_syn_disabled':
            self.I_e = nest.random.normal(mean=netparams.I_e_bursting_mean, std=netparams.I_e_bursting_std) if input_type=='descending' else nest.random.normal(mean=netparams.I_fb_bursting_mean, std=netparams.I_fb_bursting_std) if input_type=='sensory_feedback' else 0
            self.bursting_neuronparams = {'C_m':nest.random.normal(mean=netparams.C_m_bursting_mean, std=netparams.C_m_bursting_std), 'g_L':26.,'E_L':-60.,'V_th':nest.random.normal(mean=netparams.V_th_mean_bursting, std=netparams.V_th_std_bursting),'Delta_T':2.,'tau_w':130., 'a':-11., 'b':30., 'V_reset':-48., 'I_e':nest.random.normal(mean=netparams.I_e_bursting_mean, std=netparams.I_e_bursting_std),'t_ref':nest.random.normal(mean=netparams.t_ref_mean, std=netparams.t_ref_std),'V_m':nest.random.normal(mean=netparams.V_m_mean, std=netparams.V_m_std),"tau_syn_rise_I": netparams.tau_syn_i_rise, "tau_syn_decay_I": netparams.tau_syn_i_decay, "tau_syn_rise_E": netparams.tau_syn_e_rise, "tau_syn_decay_E": netparams.tau_syn_e_decay} #bursting, Naud et al. 2008, C = pF; g_L = nS    
            self.interneuron_pop = nest.Create('aeif_cond_beta_aeif_cond_beta_nestml',pop_size,self.bursting_neuronparams)
            self.white_noise = nest.Create("noise_generator",netparams.noise_params_bursting)    
        
        if pop_type=='V1a_1' and netparams.fb_1a_flx == 1:
            #Create poisson generator for feedback
            self.v1a_1_pg = nest.Create("poisson_generator",netparams.num_pgs, params={"rate": 0.0})
            nest.Connect(self.v1a_1_pg,self.interneuron_pop,{'rule': 'pairwise_bernoulli', 'p': 1.})
        if pop_type=='V1a_2' and netparams.fb_1a_ext == 1:
            #Create poisson generator for feedback
            self.v1a_2_pg = nest.Create("poisson_generator",netparams.num_pgs, params={"rate": 0.0})
            nest.Connect(self.v1a_2_pg,self.interneuron_pop,{'rule': 'pairwise_bernoulli', 'p': 1.})
	
        #Connect white noise to neurons
        nest.Connect(self.white_noise,self.interneuron_pop,"all_to_all")

        #Connect spike detectors to neuron populations
        nest.Connect(self.interneuron_pop,self.spike_detector,"one_to_one")

        #Connect multimeters to neuron populations
        nest.Connect(self.multimeter,self.interneuron_pop)
        
        if self_connection == 'inh':
            #Connect neurons within interneuron pop
            self.inh_syn_params = {"synapse_model":"static_synapse",
                "weight" : nest.random.normal(mean=netparams.w_inh_mean,std=netparams.w_inh_std), #nS            
                "delay" : netparams.synaptic_delay}	#ms
            nest.Connect(self.interneuron_pop,self.interneuron_pop,'all_to_all',self.inh_syn_params)
        elif self_connection == 'exc':
            self.exc_syn_params = {"synapse_model":"static_synapse",
                "weight" : nest.random.normal(mean=netparams.w_exc_mean,std=netparams.w_exc_std), #nS
                "delay" : netparams.synaptic_delay}	#ms
            nest.Connect(self.interneuron_pop,self.interneuron_pop,'all_to_all',self.exc_syn_params)    
        else:
            #print('Interneuron population is not self-connected.')
            pass        
                
