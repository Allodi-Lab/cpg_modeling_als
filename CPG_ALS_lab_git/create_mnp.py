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

class mnp():
    def __init__(self):
        self.senders = []
        self.spiketimes = []
        self.saved_spiketimes = []
        self.saved_senders = []
        self.time_window = 50		#50*0.1=5ms time window, based on time resolution of 0.1
        self.count = 0
        
    def create_mnp(self,pop_type):    
        #Create population
        if pop_type == 'mnp_slow_syn_enabled':
            self.tonic_neuronparams = {'C_m':nest.random.normal(mean=netparams.C_m_tonic_mean, std=netparams.C_m_tonic_std), 'g_L':10.,'E_L':-70.,'V_th':nest.random.normal(mean=netparams.V_th_mean_tonic, std=netparams.V_th_std_tonic),'Delta_T':2.,'tau_w':30., 'a':3., 'b':0., 'V_reset':-58., 'I_e':nest.random.normal(mean=netparams.I_e_tonic_mean, std=netparams.I_e_tonic_std),'t_ref':nest.random.normal(mean=netparams.t_ref_mean, std=netparams.t_ref_std),'V_m':nest.random.normal(mean=netparams.V_m_mean, std=netparams.V_m_std),"tau_syn_rise_I": netparams.tau_syn_i_rise_mn, "tau_syn_decay_I": netparams.tau_syn_i_decay_mn, "tau_syn_rise_E": netparams.tau_syn_e_rise, "tau_syn_decay_E": netparams.tau_syn_e_decay}
        elif pop_type == 'mnp_slow_syn_disabled':
            self.tonic_neuronparams = {'C_m':nest.random.normal(mean=netparams.C_m_tonic_mean, std=netparams.C_m_tonic_std), 'g_L':10.,'E_L':-70.,'V_th':nest.random.normal(mean=netparams.V_th_mean_tonic, std=netparams.V_th_std_tonic),'Delta_T':2.,'tau_w':30., 'a':3., 'b':0., 'V_reset':-58., 'I_e':nest.random.normal(mean=netparams.I_e_tonic_mean, std=netparams.I_e_tonic_std),'t_ref':nest.random.normal(mean=netparams.t_ref_mean, std=netparams.t_ref_std),'V_m':nest.random.normal(mean=netparams.V_m_mean, std=netparams.V_m_std),"tau_syn_rise_I": netparams.tau_syn_i_rise, "tau_syn_decay_I": netparams.tau_syn_i_decay, "tau_syn_rise_E": netparams.tau_syn_e_rise, "tau_syn_decay_E": netparams.tau_syn_e_decay}

        self.motor_neuron_pop = nest.Create('aeif_cond_beta_aeif_cond_beta_nestml',netparams.num_motor_neurons,self.tonic_neuronparams)	
        
        motor_neuron_ids = self.motor_neuron_pop.tolist()
        random_id = random.choice(motor_neuron_ids)
        self.neuron_to_sample = random_id  
        print('Neuron ID (MNP)',motor_neuron_ids[0],motor_neuron_ids[-1])
        
        #Create noise
        self.white_noise = nest.Create("noise_generator",netparams.noise_params_tonic)
        
        #Create spike detectors (for recording spikes)
        self.spike_detector_motor = nest.Create("spike_recorder",netparams.num_motor_neurons)
                
        #Create multimeters (for recording membrane potential)
        self.mm_motor = nest.Create("multimeter",netparams.mm_params)
	
        #Connect white noise to neurons
        nest.Connect(self.white_noise,self.motor_neuron_pop,"all_to_all")

        #Connect spike detectors to neuron populations
        nest.Connect(self.motor_neuron_pop,self.spike_detector_motor,"one_to_one")
        self.spike_detector_motor.n_events = 0		#ensure no spikes left from previous simulations

        #Connect multimeters to neuron populations
        nest.Connect(self.mm_motor,self.motor_neuron_pop)
       	        
