#!/usr/bin/env python

#include <static_connection.h>
import nest
import nest.raster_plot
import numpy as np
import sys
import pylab
import math
import matplotlib.pyplot as pyplot
import pickle, yaml
import random
import scipy
import scipy.fftpack
from scipy.ndimage import gaussian_filter
from scipy.signal import find_peaks, peak_widths, peak_prominences
import time
import copy
from set_network_params import neural_network
netparams = neural_network()

class create_rg_population():
    def __init__(self):
        self.senders = []
        self.spiketimes = []
        self.saved_spiketimes = []
        self.saved_senders = []
        self.time_window = 50		#50*0.1=5ms time window, based on time resolution of 0.1
        self.count = 0
        self.current_multiplier_bursting = netparams.current_multiplier_bursting_ext
        self.current_multiplier_tonic = netparams.current_multiplier_tonic_ext
        
        #Create populations for rg
        '''
        self.bursting_neuronparams = {'C_m':nest.random.normal(mean=netparams.C_m_bursting_mean, std=netparams.C_m_bursting_std), 'g_L':26.,'E_L':-60.,'V_th':nest.random.normal(mean=netparams.V_th_mean_bursting, std=netparams.V_th_std_bursting),'Delta_T':2.,'tau_w':130., 'a':-11., 'b':30., 'V_reset':-48., 'I_e':nest.random.normal(mean=self.current_multiplier_bursting*netparams.I_e_bursting_mean, std=self.current_multiplier_bursting*netparams.I_e_bursting_std),'t_ref':nest.random.normal(mean=netparams.t_ref_bursting_mean, std=netparams.t_ref_bursting_std),'V_m':nest.random.normal(mean=netparams.V_m_mean, std=netparams.V_m_std),"tau_syn_rise_I": netparams.tau_syn_i_rise, "tau_syn_decay_I": netparams.tau_syn_i_decay, "tau_syn_rise_E": netparams.tau_syn_e_rise, "tau_syn_decay_E": netparams.tau_syn_e_decay} #bursting, Naud et al. 2008, C = pF; g_L = nS
        '''
        self.bursting_neuronparams = {'C_m':nest.random.normal(mean=netparams.C_m_bursting_mean, std=netparams.C_m_bursting_std), 'g_L':26.,'E_L':-60.,'V_th':nest.random.normal(mean=netparams.V_th_mean_bursting, std=netparams.V_th_std_bursting),'Delta_T':3.,'tau_w':260., 'a':-11., 'b':60., 'V_reset':-48., 'I_e':nest.random.normal(mean=self.current_multiplier_bursting*netparams.I_e_bursting_mean, std=self.current_multiplier_bursting*netparams.I_e_bursting_std),'t_ref':nest.random.normal(mean=netparams.t_ref_bursting_mean, std=netparams.t_ref_bursting_std),'V_m':nest.random.normal(mean=netparams.V_m_mean, std=netparams.V_m_std),"tau_syn_rise_I": netparams.tau_syn_i_rise, "tau_syn_decay_I": netparams.tau_syn_i_decay, "tau_syn_rise_E": netparams.tau_syn_e_rise, "tau_syn_decay_E": netparams.tau_syn_e_decay} 
        self.tonic_neuronparams = {'C_m':nest.random.normal(mean=netparams.C_m_tonic_mean, std=netparams.C_m_tonic_std), 'g_L':10.,'E_L':-70.,'V_th':nest.random.normal(mean=netparams.V_th_mean_tonic, std=netparams.V_th_std_tonic),'Delta_T':2.,'tau_w':30., 'a':3., 'b':0., 'V_reset':-58., 'I_e':nest.random.normal(mean=self.current_multiplier_tonic*netparams.I_e_tonic_mean, std=self.current_multiplier_tonic*netparams.I_e_tonic_std),'t_ref':nest.random.normal(mean=netparams.t_ref_mean, std=netparams.t_ref_std),'V_m':nest.random.normal(mean=netparams.V_m_mean, std=netparams.V_m_std),"tau_syn_rise_I": netparams.tau_syn_i_rise, "tau_syn_decay_I": netparams.tau_syn_i_decay, "tau_syn_rise_E": netparams.tau_syn_e_rise, "tau_syn_decay_E": netparams.tau_syn_e_decay}
        
        self.rg_exc_bursting = nest.Create('aeif_cond_beta_aeif_cond_beta_nestml',netparams.ext_exc_bursting_count,self.bursting_neuronparams) 
        neuron_ids = self.rg_exc_bursting.tolist() 
        print('Neuron ID (RG Ext)',neuron_ids[0],neuron_ids[-1])
        self.rg_inh_bursting = nest.Create('aeif_cond_beta_aeif_cond_beta_nestml',netparams.ext_inh_bursting_count,self.bursting_neuronparams) 
        if netparams.ext_exc_tonic_count != 0: self.rg_exc_tonic = nest.Create('aeif_cond_beta_aeif_cond_beta_nestml',netparams.ext_exc_tonic_count,self.tonic_neuronparams) 	
        if netparams.ext_inh_tonic_count != 0: self.rg_inh_tonic = nest.Create('aeif_cond_beta_aeif_cond_beta_nestml',netparams.ext_inh_tonic_count,self.tonic_neuronparams) 
        
        #Create noise
        self.white_noise_tonic = nest.Create("noise_generator",netparams.noise_params_tonic) 
        self.white_noise_bursting = nest.Create("noise_generator",netparams.noise_params_bursting)     
        if netparams.fb_rg_ext == 1:
            #Create poisson generator for feedback
            self.rg_ext_pg = nest.Create("poisson_generator",netparams.num_pgs, params={"rate": 0.0})
            nest.Connect(self.rg_ext_pg,self.rg_exc_bursting,{'rule': 'pairwise_bernoulli', 'p': 1.})
        
        #Create spike detectors (for recording spikes)
        self.spike_detector_rg_exc_bursting = nest.Create("spike_recorder",netparams.ext_exc_bursting_count)
        self.spike_detector_rg_inh_bursting = nest.Create("spike_recorder",netparams.ext_inh_bursting_count)
        if netparams.ext_exc_tonic_count != 0: 
            self.spike_detector_rg_exc_tonic = nest.Create("spike_recorder",netparams.ext_exc_tonic_count)
        if netparams.ext_inh_tonic_count != 0: 
            self.spike_detector_rg_inh_tonic = nest.Create("spike_recorder",netparams.ext_inh_tonic_count)
                
        #Create multimeters (for recording membrane potential)
        self.mm_rg_exc_bursting = nest.Create("multimeter",netparams.mm_params)
        self.mm_rg_inh_bursting = nest.Create("multimeter",netparams.mm_params)
        self.mm_rg_exc_tonic = nest.Create("multimeter",netparams.mm_params)
        self.mm_rg_inh_tonic = nest.Create("multimeter",netparams.mm_params)
	
        #Connect white noise to neurons
        nest.Connect(self.white_noise_bursting,self.rg_exc_bursting,"all_to_all")
        nest.Connect(self.white_noise_bursting,self.rg_inh_bursting,"all_to_all")
        if netparams.ext_exc_tonic_count != 0: nest.Connect(self.white_noise_tonic,self.rg_exc_tonic,"all_to_all") 
        if netparams.ext_inh_tonic_count != 0: nest.Connect(self.white_noise_tonic,self.rg_inh_tonic,"all_to_all") 
	
        #Connect neurons within rg
        self.inh_syn_params = {"synapse_model":"static_synapse",
            "weight" : nest.random.normal(mean=netparams.w_inh_mean,std=netparams.w_inh_std), #nS            
            "delay" : netparams.synaptic_delay}	#ms
        self.exc_syn_params = {"synapse_model":"static_synapse",
            "weight" : nest.random.normal(mean=netparams.w_exc_mean,std=netparams.w_exc_std), #nS
            "delay" : netparams.synaptic_delay}	#ms
            
        self.coupling_exc_inh = nest.Connect(self.rg_exc_bursting,self.rg_inh_bursting,netparams.conn_dict_custom_rg,self.exc_syn_params)
        self.coupling_exc_exc = nest.Connect(self.rg_exc_bursting,self.rg_exc_bursting,netparams.conn_dict_custom_selfexc_ext,self.exc_syn_params)  	  
        self.coupling_inh_exc = nest.Connect(self.rg_inh_bursting,self.rg_exc_bursting,netparams.conn_dict_custom_rg,self.inh_syn_params)  
        self.coupling_inh_inh = nest.Connect(self.rg_inh_bursting,self.rg_inh_bursting,netparams.conn_dict_custom_selfexc_ext,self.inh_syn_params)
        if netparams.ext_exc_tonic_count != 0: 
            self.coupling_exc_tonic_inh = nest.Connect(self.rg_exc_tonic,self.rg_inh_bursting,netparams.conn_dict_custom_rg,self.exc_syn_params)
            self.coupling_exc_tonic_exc = nest.Connect(self.rg_exc_tonic,self.rg_exc_bursting,netparams.conn_dict_custom_rg,self.exc_syn_params)
            self.coupling_exc_inh_tonic = nest.Connect(self.rg_exc_bursting,self.rg_inh_tonic,netparams.conn_dict_custom_rg,self.exc_syn_params)
            self.coupling_exc_exc_tonic = nest.Connect(self.rg_exc_bursting,self.rg_exc_tonic,netparams.conn_dict_custom_rg,self.exc_syn_params)
            self.coupling_exc_tonic_exc_tonic = nest.Connect(self.rg_exc_tonic,self.rg_exc_tonic,netparams.conn_dict_custom_rg,self.exc_syn_params)
            self.coupling_exc_tonic_inh_tonic = nest.Connect(self.rg_exc_tonic,self.rg_inh_tonic,netparams.conn_dict_custom_rg,self.exc_syn_params)            
        if netparams.ext_inh_tonic_count != 0: 
            self.coupling_inh_tonic_inh = nest.Connect(self.rg_inh_tonic,self.rg_exc_bursting,netparams.conn_dict_custom_rg,self.inh_syn_params)
            self.coupling_inh_tonic_exc = nest.Connect(self.rg_inh_tonic,self.rg_inh_bursting,netparams.conn_dict_custom_rg,self.inh_syn_params)
            self.coupling_inh_exc_tonic = nest.Connect(self.rg_inh_bursting,self.rg_exc_tonic,netparams.conn_dict_custom_rg,self.inh_syn_params)
            self.coupling_inh_inh_tonic = nest.Connect(self.rg_inh_bursting,self.rg_inh_tonic,netparams.conn_dict_custom_rg,self.inh_syn_params)
            self.coupling_inh_tonic_inh_tonic = nest.Connect(self.rg_inh_tonic,self.rg_inh_tonic,netparams.conn_dict_custom_rg,self.inh_syn_params)           
            self.coupling_exc_tonic_inh_tonic = nest.Connect(self.rg_inh_tonic,self.rg_exc_tonic,netparams.conn_dict_custom_rg,self.inh_syn_params) 

        #Connect spike detectors to neuron populations
        nest.Connect(self.rg_exc_bursting,self.spike_detector_rg_exc_bursting,"one_to_one")
        nest.Connect(self.rg_inh_bursting,self.spike_detector_rg_inh_bursting,"one_to_one")
        self.spike_detector_rg_exc_bursting.n_events = 0		#ensure no spikes left from previous simulations
        self.spike_detector_rg_inh_bursting.n_events = 0		#ensure no spikes left from previous simulations
        if netparams.ext_exc_tonic_count != 0: 
            nest.Connect(self.rg_exc_tonic,self.spike_detector_rg_exc_tonic,"one_to_one")
            self.spike_detector_rg_exc_tonic.n_events = 0	#ensure no spikes left from previous simulations
        if netparams.ext_inh_tonic_count != 0: 
            nest.Connect(self.rg_inh_tonic,self.spike_detector_rg_inh_tonic,"one_to_one")
            self.spike_detector_rg_inh_tonic.n_events = 0	#ensure no spikes left from previous simulations
                    
        #Connect multimeters to neuron populations
        nest.Connect(self.mm_rg_exc_bursting,self.rg_exc_bursting)
        nest.Connect(self.mm_rg_inh_bursting,self.rg_inh_bursting)
        if netparams.ext_exc_tonic_count != 0: 
            nest.Connect(self.mm_rg_exc_tonic,self.rg_exc_tonic)
        if netparams.ext_inh_tonic_count != 0: 
            nest.Connect(self.mm_rg_inh_tonic,self.rg_inh_tonic)     	        
