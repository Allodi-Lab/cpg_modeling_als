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

def create_synapse_params(mean, std, delay):
    """
    Helper function to create synapse parameters with random weight and fixed delay.
    
    Args:
        mean (float): Mean of the weight distribution.
        std (float): Standard deviation of the weight distribution.
        delay (float): Synaptic delay.
    
    Returns:
        dict: Synapse parameters including weight and delay.
    """
    return {
        "synapse_model": "static_synapse",
        "weight": nest.random.normal(mean=mean, std=std),  # nS
        "delay": delay  # ms
    }

class ConnectNetwork():
    def __init__(self):
        self.total_weight_exc = 0
        self.total_weight_inh = 0
        self.balance_pct = 0
        self.num_of_synapses = []
        self.name_of_pops = []

        self.synapse_params = {
            'custom_rg_v1': {
                'conn_dict': 'conn_dict_custom_rg_v1',
                'syn_params': create_synapse_params(netparams.w_custom_rg_v1_mean, netparams.w_custom_rg_v1_std, netparams.synaptic_delay),
                'sparsity': 'sparsity_custom_rg_v1',
                'conn_type': 'rg_layer'
            },
            'custom_rg_v2b': {
                'conn_dict': 'conn_dict_custom_rg_v2b',
                'syn_params': create_synapse_params(netparams.w_custom_rg_v2b_mean, netparams.w_custom_rg_v2b_std, netparams.synaptic_delay),
                'sparsity': 'sparsity_custom_rg_v2b'
            },
            'custom_rg_rg': {
                'conn_dict': 'conn_dict_custom_rg1_rg2',
                'syn_params': create_synapse_params(netparams.w_custom_rg_rg_mean, netparams.w_custom_rg_rg_std, netparams.synaptic_delay),
                'sparsity': 'sparsity_rg1_rg2'
            },
            'custom_rg_v1a': {
                'conn_dict': 'conn_dict_custom_rg_v1a',
                'syn_params': create_synapse_params(netparams.w_custom_rg_v1a_mean, netparams.w_custom_rg_v1a_std, netparams.synaptic_delay),
                'sparsity': 'sparsity_custom_rg_v1a'
            },
            'custom_mn_rc': {
                'conn_dict': 'conn_dict_custom_mn_rc',
                'syn_params': create_synapse_params(netparams.w_custom_mn_rc_mean, netparams.w_custom_mn_rc_std, netparams.synaptic_delay),
                'sparsity': 'sparsity_custom_mn_rc'
            },
            'custom_rc_rc': {
                'conn_dict': 'conn_dict_custom_rc_rc',
                'syn_params': create_synapse_params(netparams.w_custom_rc_rc_mean, netparams.w_custom_rc_rc_std, netparams.synaptic_delay),
                'sparsity': 'sparsity_custom_rc_rc'
            },
            'custom_rc_v1a': {
                'conn_dict': 'conn_dict_custom_rc_v1a',
                'syn_params': create_synapse_params(netparams.w_custom_rc_v1a_mean, netparams.w_custom_rc_v1a_std, netparams.synaptic_delay),
                'sparsity': 'sparsity_custom_rc_v1a'
            },
            'custom_rc_mn': {
                'conn_dict': 'conn_dict_custom_rc_mn',
                'syn_params': create_synapse_params(netparams.w_custom_rc_mn_mean, netparams.w_custom_rc_mn_std, netparams.synaptic_delay),
                'sparsity': 'sparsity_custom_rc_mn'
            },
            'custom_v1a_mn': {
                'conn_dict': 'conn_dict_custom_v1a_mn',
                'syn_params': create_synapse_params(netparams.w_custom_v1a_mn_mean, netparams.w_custom_v1a_mn_std, netparams.synaptic_delay),
                'sparsity': 'sparsity_custom_v1a_mn'
            },
            'custom_v1a_v1a': {
                'conn_dict': 'conn_dict_custom_v1a_v1a',
                'syn_params': create_synapse_params(netparams.w_custom_v1a_v1a_mean, netparams.w_custom_v1a_v1a_std, netparams.synaptic_delay),
                'sparsity': 'sparsity_custom_v1a_v1a'
            },
            'custom_v1_rg': {
                'conn_dict': 'conn_dict_custom_v1_rg',
                'syn_params': create_synapse_params(netparams.w_custom_v1_rg_mean, netparams.w_custom_v1_rg_std, netparams.synaptic_delay),
                'sparsity': 'sparsity_custom_v1_rg'
            },
            'custom_v2b_rg': {
                'conn_dict': 'conn_dict_custom_v2b_rg',
                'syn_params': create_synapse_params(netparams.w_custom_v2b_rg_mean, netparams.w_custom_v2b_rg_std, netparams.synaptic_delay),
                'sparsity': 'sparsity_custom_v2b_rg'
            },
            'custom_v1_v2a': {
                'conn_dict': 'conn_dict_custom_v1_v2a',
                'syn_params': create_synapse_params(netparams.w_custom_v1_v2a_mean, netparams.w_custom_v1_v2a_std, netparams.synaptic_delay),
                'sparsity': 'sparsity_custom_v1_v2a'
            },
            'custom_v2b_v2a': {
                'conn_dict': 'conn_dict_custom_v2b_v2a',
                'syn_params': create_synapse_params(netparams.w_custom_v2b_v2a_mean, netparams.w_custom_v2b_v2a_std, netparams.synaptic_delay),
                'sparsity': 'sparsity_custom_v2b_v2a'
            },
            'custom_v0c_mn': {
                'conn_dict': 'conn_dict_custom_v0c_mn',
                'syn_params': create_synapse_params(netparams.w_custom_v0c_mn_mean, netparams.w_custom_v0c_mn_std, netparams.synaptic_delay),
                'sparsity': 'sparsity_custom_v0c_mn'
            },
            'custom_rg_v2a': {
                'conn_dict': 'conn_dict_custom_rg_v2a',
                'syn_params': create_synapse_params(netparams.w_custom_rg_v2a_mean, netparams.w_custom_rg_v2a_std, netparams.synaptic_delay),
                'sparsity': 'sparsity_custom_rg_v2a'
            },
            'custom_rg_v0c': {
                'conn_dict': 'conn_dict_custom_rg_v0c',
                'syn_params': create_synapse_params(netparams.w_custom_rg_v0c_mean, netparams.w_custom_rg_v0c_std, netparams.synaptic_delay),
                'sparsity': 'sparsity_custom_rg_v0c'
            },
            'custom_v2a_mn': {
                'conn_dict': 'conn_dict_custom_v2a_mn',
                'syn_params': create_synapse_params(netparams.w_custom_v2a_mn_mean, netparams.w_custom_v2a_mn_std, netparams.synaptic_delay),
                'sparsity': 'sparsity_custom_v2a_mn'
            }
        }
    
    def create_connections(self, pop1, pop2, syn_type):
        """
        Creates connections between populations with specified synapse type.

        Args:
            pop1: Source population.
            pop2: Target population.
            syn_type: Type of synapse (must be a valid key in `self.synapse_params`).
        """
        if syn_type in self.synapse_params:
            conn_dict = getattr(netparams, self.synapse_params[syn_type]['conn_dict'])
            syn_params = self.synapse_params[syn_type]['syn_params']
            sparsity = getattr(netparams, self.synapse_params[syn_type]['sparsity'])

            # Create the connection
            nest.Connect(pop1, pop2, conn_dict, syn_params)
            self.local_connections = len(nest.GetConnections(source=pop1, target=pop2))
            self.name_of_pops.append(syn_type)
            self.num_of_synapses.append(self.local_connections)
            weight_mean_name = str('w_'+syn_type+'_mean')
            weight_std_name = str('w_'+syn_type+'_std')
            print(f"{syn_type} connection created, connectivity %, weight (mean,std) = {sparsity},",
      getattr(netparams, weight_mean_name), getattr(netparams, weight_std_name))
        else:
            print(f"Invalid synapse type: {syn_type}")        
        return
    
    def calculate_synapse_percentage(self):
        self.all_connections = len(nest.GetConnections())
        self.percentage_of_connections = [x//self.all_connections for x in self.num_of_synapses]       
        print('Total connections: ',self.all_connections)
        print('Name of connections: ',self.name_of_pops)
        print('Local connections: ',self.num_of_synapses)
    
    def sum_weights_per_source(self,population):
        synapse_data = nest.GetConnections(population).get(['source', 'weight'])
        weights_per_source = {}
        for connection in synapse_data:
            source_neuron = synapse_data['source']
            weights = synapse_data['weight']
            for s in set(source_neuron):
                if s not in weights_per_source:
                    weights_per_source[s] = sum([w for i, w in enumerate(weights) if source_neuron[i] == s])
                else:
                    weights_per_source[s] += sum([w for i, w in enumerate(weights) if source_neuron[i] == s])
        return weights_per_source
    
    def count_spikes_per_source(self,spike_detector):
        sender_counts = {}
        spike_data = spike_detector.get('events', 'senders')
        #print('Sender data: ',spike_data)
        for sender_list in spike_data:
            for sender in sender_list:
                if sender not in sender_counts:
                    sender_counts[sender] = 1
                else:
                    sender_counts[sender] += 1
        return sender_counts
    
    def calculate_weighted_balance(self, pop1,spike_detector):
        self.total_weight = 0 
        self.weights_by_source = self.sum_weights_per_source(pop1)
        self.sender_counts = self.count_spikes_per_source(spike_detector)
        #print('Count per neuron ID: ',self.sender_counts)        
        #print('Weights by source: ',self.weights_by_source)
        for source in self.weights_by_source:
            #print('Neuron ID: ',source)
            if source in self.sender_counts:
                weighted_weight = self.weights_by_source[source] * self.sender_counts[source]
            else:
                weighted_weight = 0
            self.total_weight += weighted_weight
        #self.total_weight = self.total_weight*2 if self.total_weight < 0 else self.total_weight*.2
        self.total_weight = self.total_weight*2.9 if self.total_weight < 0 else self.total_weight
        return round(self.total_weight,2) 
