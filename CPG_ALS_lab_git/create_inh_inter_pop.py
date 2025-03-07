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
        self.v2b_current_multiplier = 1. 
        self.v1_current_multiplier = 1. 
        
        #Create population
        if neuron_type=='V2b':
            print('Creating a V2b population')
            #self.bursting_neuronparams = {'C_m':nest.random.normal(mean=netparams.C_m_v1v2b_bursting_mean, std=netparams.C_m_v1v2b_bursting_std), 'g_L':26.,'E_L':-60.,'V_th':nest.random.normal(mean=netparams.V_th_v1v2b_mean_bursting, std=netparams.V_th_v1v2b_std_bursting),'Delta_T':2.,'tau_w':130., 'a':-11., 'b':30., 'V_reset':-48., 'I_e':nest.random.normal(mean=self.v2b_current_multiplier*netparams.I_e_bursting_mean, std=self.v2b_current_multiplier*netparams.I_e_bursting_std),'t_ref':nest.random.normal(mean=netparams.t_ref_mean, std=netparams.t_ref_std),'V_m':nest.random.normal(mean=netparams.V_m_mean, std=netparams.V_m_std),"tau_syn_rise_I": netparams.tau_syn_i_rise, "tau_syn_decay_I": netparams.tau_syn_i_decay, "tau_syn_rise_E": netparams.tau_syn_e_rise, "tau_syn_decay_E": netparams.tau_syn_e_decay} #bursting, Naud et al. 2008, C = pF; g_L = nS
            self.tonic_neuronparams = {'C_m':nest.random.normal(mean=netparams.C_m_v1v2b_tonic_mean, std=netparams.C_m_v1v2b_tonic_std), 'g_L':10.,'E_L':-70.,'V_th':nest.random.normal(mean=netparams.V_th_v1v2b_mean_tonic, std=netparams.V_th_v1v2b_std_tonic),'Delta_T':2.,'tau_w':30., 'a':3., 'b':0., 'V_reset':-58., 'I_e':nest.random.normal(mean=self.v2b_current_multiplier*netparams.I_e_tonic_mean, std=self.v2b_current_multiplier*netparams.I_e_tonic_std),'t_ref':nest.random.normal(mean=netparams.t_ref_mean, std=netparams.t_ref_std),'V_m':nest.random.normal(mean=netparams.V_m_mean, std=netparams.V_m_std),"tau_syn_rise_I": netparams.tau_syn_i_rise, "tau_syn_decay_I": netparams.tau_syn_i_decay, "tau_syn_rise_E": netparams.tau_syn_e_rise, "tau_syn_decay_E": netparams.tau_syn_e_decay}
        elif neuron_type=='V1' and self.v1_current_multiplier>0:
            print('Creating a V1 population')
            #self.bursting_neuronparams = {'C_m':nest.random.normal(mean=netparams.C_m_v1v2b_bursting_mean, std=netparams.C_m_v1v2b_bursting_std), 'g_L':26.,'E_L':-60.,'V_th':nest.random.normal(mean=netparams.V_th_v1v2b_mean_bursting, std=netparams.V_th_v1v2b_std_bursting),'Delta_T':2.,'tau_w':130., 'a':-11., 'b':30., 'V_reset':-48., 'I_e':nest.random.normal(mean=self.v1_current_multiplier*netparams.I_e_bursting_mean, std=self.v1_current_multiplier*netparams.I_e_bursting_std),'t_ref':nest.random.normal(mean=netparams.t_ref_mean, std=netparams.t_ref_std),'V_m':nest.random.normal(mean=netparams.V_m_mean, std=netparams.V_m_std),"tau_syn_rise_I": netparams.tau_syn_i_rise, "tau_syn_decay_I": netparams.tau_syn_i_decay, "tau_syn_rise_E": netparams.tau_syn_e_rise, "tau_syn_decay_E": netparams.tau_syn_e_decay} #bursting, Naud et al. 2008, C = pF; g_L = nS
            self.tonic_neuronparams = {'C_m':nest.random.normal(mean=netparams.C_m_v1v2b_tonic_mean, std=netparams.C_m_v1v2b_tonic_std), 'g_L':10.,'E_L':-70.,'V_th':nest.random.normal(mean=netparams.V_th_v1v2b_mean_tonic, std=netparams.V_th_v1v2b_std_tonic),'Delta_T':2.,'tau_w':30., 'a':3., 'b':0., 'V_reset':-58., 'I_e':nest.random.normal(mean=self.v1_current_multiplier*netparams.I_e_tonic_mean, std=self.v1_current_multiplier*netparams.I_e_tonic_std),'t_ref':nest.random.normal(mean=netparams.t_ref_mean, std=netparams.t_ref_std),'V_m':nest.random.normal(mean=netparams.V_m_mean, std=netparams.V_m_std),"tau_syn_rise_I": netparams.tau_syn_i_rise, "tau_syn_decay_I": netparams.tau_syn_i_decay, "tau_syn_rise_E": netparams.tau_syn_e_rise, "tau_syn_decay_E": netparams.tau_syn_e_decay}
        elif neuron_type=='V1' and self.v1_current_multiplier==0:
            print('Creating a V1 population')
            #self.bursting_neuronparams = {'C_m':nest.random.normal(mean=netparams.C_m_v1v2b_bursting_mean, std=netparams.C_m_v1v2b_bursting_std), 'g_L':26.,'E_L':-60.,'V_th':nest.random.normal(mean=netparams.V_th_v1v2b_mean_bursting, std=netparams.V_th_v1v2b_std_bursting),'Delta_T':2.,'tau_w':130., 'a':-11., 'b':30., 'V_reset':-48., 'I_e':0.,'t_ref':nest.random.normal(mean=netparams.t_ref_mean, std=netparams.t_ref_std),'V_m':nest.random.normal(mean=netparams.V_m_mean, std=netparams.V_m_std),"tau_syn_rise_I": netparams.tau_syn_i_rise, "tau_syn_decay_I": netparams.tau_syn_i_decay, "tau_syn_rise_E": netparams.tau_syn_e_rise, "tau_syn_decay_E": netparams.tau_syn_e_decay} #bursting, Naud et al. 2008, C = pF; g_L = nS
            self.tonic_neuronparams = {'C_m':nest.random.normal(mean=netparams.C_m_v1v2b_tonic_mean, std=netparams.C_m_v1v2b_tonic_std), 'g_L':10.,'E_L':-70.,'V_th':nest.random.normal(mean=netparams.V_th_v1v2b_mean_tonic, std=netparams.V_th_v1v2b_std_tonic),'Delta_T':2.,'tau_w':30., 'a':3., 'b':0., 'V_reset':-58., 'I_e':0.,'t_ref':nest.random.normal(mean=netparams.t_ref_mean, std=netparams.t_ref_std),'V_m':nest.random.normal(mean=netparams.V_m_mean, std=netparams.V_m_std),"tau_syn_rise_I": netparams.tau_syn_i_rise, "tau_syn_decay_I": netparams.tau_syn_i_decay, "tau_syn_rise_E": netparams.tau_syn_e_rise, "tau_syn_decay_E": netparams.tau_syn_e_decay}    
        
        num_tonic_neurons = netparams.num_inh_inter_tonic_v2b if neuron_type=='V2b' else netparams.num_inh_inter_tonic_v1 
        num_bursting_neurons = netparams.num_inh_inter_bursting_v2b if neuron_type=='V2b' else netparams.num_inh_inter_bursting_v1
        print('Neuron count (type,tonic,bursting) ',neuron_type,num_tonic_neurons,num_bursting_neurons)
        
        self.inh_inter_tonic = nest.Create('aeif_cond_beta_aeif_cond_beta_nestml',num_tonic_neurons,self.tonic_neuronparams)
        self.white_noise_tonic = nest.Create("noise_generator",netparams.noise_params_tonic)
        self.spike_detector_inh_inter_tonic = nest.Create("spike_recorder",num_tonic_neurons)
        self.mm_inh_inter_tonic = nest.Create("multimeter",netparams.mm_params)
        nest.Connect(self.white_noise_tonic,self.inh_inter_tonic,"all_to_all")
        nest.Connect(self.inh_inter_tonic,self.spike_detector_inh_inter_tonic,"one_to_one")
        nest.Connect(self.mm_inh_inter_tonic,self.inh_inter_tonic)
        
        if num_bursting_neurons>0: 
            self.inh_inter_bursting = nest.Create('aeif_cond_beta_aeif_cond_beta_nestml',num_bursting_neurons,self.bursting_neuronparams)		
            self.white_noise_bursting = nest.Create("noise_generator",netparams.noise_params_bursting)
            self.spike_detector_inh_inter_bursting = nest.Create("spike_recorder",num_bursting_neurons)
            self.mm_inh_inter_bursting = nest.Create("multimeter",netparams.mm_params)
            nest.Connect(self.white_noise_bursting,self.inh_inter_bursting,"all_to_all")
            nest.Connect(self.inh_inter_bursting,self.spike_detector_inh_inter_bursting,"one_to_one")
            nest.Connect(self.mm_inh_inter_bursting,self.inh_inter_bursting) 	
                
