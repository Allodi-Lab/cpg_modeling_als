#!/usr/bin/env python

import nest
import numpy as np
import sys
import pylab
import math
import matplotlib.pyplot as plt
import random
import time
from set_network_params import neural_network
netparams = neural_network()

class feedback():
    def __init__(self):
        self.activation_tracking_flx = []
        self.activation_tracking_ext = []
        self.simulated_feedback_tracking_flx = []
        self.simulated_feedback_tracking_ext = []
        self.a_flx = 0
        self.a_ext = 0
        self.simulation_chunk = 1
        
        # Creat a sine wave input for testing
        self.f = netparams.sim_fb_freq  # Frequency in Hz
        self.time_points = np.arange(0, netparams.sim_time, self.simulation_chunk)  # Time vector
        self.flx_rate_values = np.sin(2 * np.pi * self.f * self.time_points / 1000.0)  # Sine wave in Hz
        self.ext_rate_values = np.cos(2 * np.pi * self.f * self.time_points / 1000.0)  # Cosine wave in Hz
        
    def send_muscle_activation(self, flx_spikes, ext_spikes):
        """
        This function approximates the firing rate and sends an analog value between [0, 1] to the muscle
        using the Forward Euler method.

        Parameters:
        flx_spikes -- Number of spikes from the flexor motor neuron pool
        ext_spikes -- Number of spikes from the extensor motor neuron pool
        dt         -- Simulation chunk size (Minimum is 1ms)
        """
        tau_flx = 20   # time constant for flexor
        tau_ext = 20   # time constant for extensor
        w_flx = 0.1       # weight for flexor spikes 
        w_ext = 0.1       # weight for extensor spikes 
        spike_count_threshold = 0
        dt = self.simulation_chunk     #ms
        
        self.a_flx += w_flx * flx_spikes
        self.a_ext += w_ext * ext_spikes
        # Compute change in activation using Forward Euler method
        dA_flx = - self.a_flx / tau_flx 
        dA_ext = - self.a_ext / tau_ext 

        # Update activation with Euler integration
        self.a_flx += dA_flx * dt
        self.a_ext += dA_ext * dt

        # Ensure activation stays in [0,1]
        self.a_flx = max(0, min(1, self.a_flx))
        self.a_ext = max(0, min(1, self.a_ext))

        #print('Muscle activation (flx, ext)', flx_spikes, self.a_flx, ext_spikes, self.a_ext, end="\n")

        # Track activation over time
        self.activation_tracking_flx.append(self.a_flx)
        self.activation_tracking_ext.append(self.a_ext)
    
    def receive_muscle_afferents(self):
        """
        This function reads feedback values from the muscle model.
        """
        current_time = nest.biological_time
        idx = np.searchsorted(self.time_points, current_time)  # Find closest index
        idx_flx = min(max(idx, 0), len(self.flx_rate_values) - 1)  # Ensure index is within range
        idx_ext = min(max(idx, 0), len(self.ext_rate_values) - 1)  
        simulated_feedback_flx = self.flx_rate_values[idx_flx] if self.flx_rate_values[idx_flx]>0 else 0
        simulated_feedback_ext = self.ext_rate_values[idx_ext] if self.ext_rate_values[idx_ext]>0 else 0
        feedback_weight_flx = 100
        feedback_weight_ext = 100
        
        flx_1a_feedback = feedback_weight_flx*simulated_feedback_flx #Velocity
        ext_1a_feedback = feedback_weight_ext*simulated_feedback_ext
        flx_1b_feedback = feedback_weight_flx*simulated_feedback_flx #Velocity
        ext_1b_feedback = feedback_weight_ext*simulated_feedback_ext
        flx_11_feedback = feedback_weight_flx*simulated_feedback_flx #Stretch (length)
        ext_11_feedback = feedback_weight_ext*simulated_feedback_ext
        
        self.simulated_feedback_tracking_flx.append(simulated_feedback_flx)
        self.simulated_feedback_tracking_ext.append(simulated_feedback_ext)

        return flx_1a_feedback, ext_1a_feedback, flx_1b_feedback, ext_1b_feedback, flx_11_feedback, ext_11_feedback