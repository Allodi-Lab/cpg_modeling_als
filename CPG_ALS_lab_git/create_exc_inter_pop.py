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

class create_exc_inter_population():
    def __init__(self):
        self.senders = []
        self.spiketimes = []
        self.saved_spiketimes = []
        self.saved_senders = []
        self.time_window = 50		#50*0.1=5ms time window, based on time resolution of 0.1
        self.count = 0
        self.current_multiplier = 1.
        
        #Create population
        if netparams.remove_descending_drive==0:
            self.tonic_neuronparams = {'C_m':nest.random.normal(mean=netparams.C_m_tonic_mean, std=netparams.C_m_tonic_std), 'g_L':10.,'E_L':-70.,'V_th':nest.random.normal(mean=netparams.V_th_mean_tonic, std=netparams.V_th_std_tonic),'Delta_T':2.,'tau_w':30., 'a':3., 'b':0., 'V_reset':-58., 'I_e':nest.random.normal(mean=self.current_multiplier*netparams.I_e_tonic_mean, std=self.current_multiplier*netparams.I_e_tonic_std),'t_ref':nest.random.normal(mean=netparams.t_ref_mean, std=netparams.t_ref_std),'V_m':nest.random.normal(mean=netparams.V_m_mean, std=netparams.V_m_std),"tau_syn_rise_I": netparams.tau_syn_i_rise, "tau_syn_decay_I": netparams.tau_syn_i_decay, "tau_syn_rise_E": netparams.tau_syn_e_rise, "tau_syn_decay_E": netparams.tau_syn_e_decay}
        if netparams.remove_descending_drive==1:
            self.tonic_neuronparams = {'C_m':nest.random.normal(mean=netparams.C_m_tonic_mean, std=netparams.C_m_tonic_std), 'g_L':10.,'E_L':-70.,'V_th':nest.random.normal(mean=netparams.V_th_mean_tonic, std=netparams.V_th_std_tonic),'Delta_T':2.,'tau_w':30., 'a':3., 'b':0., 'V_reset':-58., 'I_e':0,'t_ref':nest.random.normal(mean=netparams.t_ref_mean, std=netparams.t_ref_std),'V_m':nest.random.normal(mean=netparams.V_m_mean, std=netparams.V_m_std),"tau_syn_rise_I": netparams.tau_syn_i_rise, "tau_syn_decay_I": netparams.tau_syn_i_decay, "tau_syn_rise_E": netparams.tau_syn_e_rise, "tau_syn_decay_E": netparams.tau_syn_e_decay}

        self.exc_syn_params = {"synapse_model":"static_synapse",
            "weight" : nest.random.normal(mean=netparams.w_custom_v2a_selfexc_mean,std=netparams.w_custom_v2a_selfexc_std), #nS
            "delay" : netparams.synaptic_delay}	#ms
        
        self.exc_inter_tonic = nest.Create('aeif_cond_beta_aeif_cond_beta_nestml',netparams.v2a_tonic_pop_size,self.tonic_neuronparams)
        #Create noise
        self.white_noise_tonic = nest.Create("noise_generator",netparams.noise_params_tonic)       
        
        #Create spike detectors (for recording spikes)
        self.spike_detector_exc_inter_tonic = nest.Create("spike_recorder",netparams.v2a_tonic_pop_size)       
                
        #Create multimeters (for recording membrane potential)
        self.mm_exc_inter_tonic = nest.Create("multimeter",netparams.mm_params)        
	
        #Connect white noise to neurons
        nest.Connect(self.white_noise_tonic,self.exc_inter_tonic,"all_to_all")        

        #Connect spike detectors to neuron populations
        nest.Connect(self.exc_inter_tonic,self.spike_detector_exc_inter_tonic,"one_to_one")        

        #Connect multimeters to neuron populations
        nest.Connect(self.mm_exc_inter_tonic,self.exc_inter_tonic)      
        
        #Connect neurons within exc interneuron pop        
        nest.Connect(self.exc_inter_tonic,self.exc_inter_tonic,'all_to_all',self.exc_syn_params)
        print('Self-excitatory V2a connection created, connectivity %, weight (mean/std) = 1. ',netparams.w_custom_v2a_selfexc_mean,netparams.w_custom_v2a_selfexc_std)
        

        
                
