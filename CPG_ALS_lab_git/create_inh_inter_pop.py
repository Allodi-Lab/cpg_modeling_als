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

class create_inh_inter_population():
    def __init__(self,neuron_type):
        self.senders = []
        self.spiketimes = []
        self.saved_spiketimes = []
        self.saved_senders = []
        self.time_window = 50		#50*0.1=5ms time window, based on time resolution of 0.1
        self.count = 0
        self.v2b_current_multiplier = netparams.v2b_current_multiplier 
        self.v1_current_multiplier = netparams.v1_current_multiplier 
        
        #Create population
        if neuron_type=='V2b' and self.v2b_current_multiplier>0:
            print('Creating a V2b population')
            self.tonic_neuronparams = {'C_m':nest.random.normal(mean=netparams.C_m_v1v2b_tonic_mean, std=netparams.C_m_v1v2b_tonic_std), 'g_L':10.,'E_L':-70.,'V_th':nest.random.normal(mean=netparams.V_th_v1v2b_mean_tonic, std=netparams.V_th_v1v2b_std_tonic),'Delta_T':2.,'tau_w':30., 'a':3., 'b':0., 'V_reset':-58., 'I_e':nest.random.normal(mean=self.v2b_current_multiplier*netparams.I_e_tonic_mean, std=self.v2b_current_multiplier*netparams.I_e_tonic_std),'t_ref':nest.random.normal(mean=netparams.t_ref_mean, std=netparams.t_ref_std),'V_m':nest.random.normal(mean=netparams.V_m_mean, std=netparams.V_m_std),"tau_syn_rise_I": netparams.tau_syn_i_rise, "tau_syn_decay_I": netparams.tau_syn_i_decay, "tau_syn_rise_E": netparams.tau_syn_e_rise, "tau_syn_decay_E": netparams.tau_syn_e_decay}
        elif neuron_type=='V1' and self.v1_current_multiplier>0:
            print('Creating a V1 population')
            self.tonic_neuronparams = {'C_m':nest.random.normal(mean=netparams.C_m_v1v2b_tonic_mean, std=netparams.C_m_v1v2b_tonic_std), 'g_L':10.,'E_L':-70.,'V_th':nest.random.normal(mean=netparams.V_th_v1v2b_mean_tonic, std=netparams.V_th_v1v2b_std_tonic),'Delta_T':2.,'tau_w':30., 'a':3., 'b':0., 'V_reset':-58., 'I_e':nest.random.normal(mean=self.v1_current_multiplier*netparams.I_e_tonic_mean, std=self.v1_current_multiplier*netparams.I_e_tonic_std),'t_ref':nest.random.normal(mean=netparams.t_ref_mean, std=netparams.t_ref_std),'V_m':nest.random.normal(mean=netparams.V_m_mean, std=netparams.V_m_std),"tau_syn_rise_I": netparams.tau_syn_i_rise, "tau_syn_decay_I": netparams.tau_syn_i_decay, "tau_syn_rise_E": netparams.tau_syn_e_rise, "tau_syn_decay_E": netparams.tau_syn_e_decay}
        elif neuron_type=='V2b' or neuron_type=='V1' and self.v1_current_multiplier==0:
            print('Creating a V1 population')
            self.tonic_neuronparams = {'C_m':nest.random.normal(mean=netparams.C_m_v1v2b_tonic_mean, std=netparams.C_m_v1v2b_tonic_std), 'g_L':10.,'E_L':-70.,'V_th':nest.random.normal(mean=netparams.V_th_v1v2b_mean_tonic, std=netparams.V_th_v1v2b_std_tonic),'Delta_T':2.,'tau_w':30., 'a':3., 'b':0., 'V_reset':-58., 'I_e':0.,'t_ref':nest.random.normal(mean=netparams.t_ref_mean, std=netparams.t_ref_std),'V_m':nest.random.normal(mean=netparams.V_m_mean, std=netparams.V_m_std),"tau_syn_rise_I": netparams.tau_syn_i_rise, "tau_syn_decay_I": netparams.tau_syn_i_decay, "tau_syn_rise_E": netparams.tau_syn_e_rise, "tau_syn_decay_E": netparams.tau_syn_e_decay}    
        
        num_tonic_neurons = netparams.num_inh_inter_tonic_v2b if neuron_type=='V2b' else netparams.num_inh_inter_tonic_v1 
        print('Neuron count (type) ',neuron_type,num_tonic_neurons)
        
        self.inh_inter_tonic = nest.Create('aeif_cond_beta_aeif_cond_beta_nestml',num_tonic_neurons,self.tonic_neuronparams)
        neuron_ids = self.inh_inter_tonic.tolist() 
        print('Neuron ID (InhI)',neuron_ids[0],neuron_ids[-1])
        self.white_noise_tonic = nest.Create("noise_generator",netparams.noise_params_tonic)
        self.spike_detector_inh_inter_tonic = nest.Create("spike_recorder",num_tonic_neurons)
        self.mm_inh_inter_tonic = nest.Create("multimeter",netparams.mm_params)
        nest.Connect(self.white_noise_tonic,self.inh_inter_tonic,"all_to_all")
        nest.Connect(self.inh_inter_tonic,self.spike_detector_inh_inter_tonic,"one_to_one")
        nest.Connect(self.mm_inh_inter_tonic,self.inh_inter_tonic)
        
        if neuron_type=='V2b' and netparams.fb_v2b == 1:
            #Create poisson generator for feedback
            self.v2b_pg = nest.Create("poisson_generator",netparams.num_pgs, params={"rate": 0.0})
            nest.Connect(self.v2b_pg,self.inh_inter_tonic,{'rule': 'pairwise_bernoulli', 'p': 1.})
            
        if neuron_type=='V1' and netparams.fb_v1 == 1:
            #Create poisson generator for feedback
            self.v1_pg = nest.Create("poisson_generator",netparams.num_pgs, params={"rate": 0.0})
            nest.Connect(self.v1_pg,self.inh_inter_tonic,{'rule': 'pairwise_bernoulli', 'p': 1.})
                
