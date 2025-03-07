#!/usr/bin/env python

import nest
import numpy as np
import sys
import pylab
import math
import matplotlib.pyplot as plt
import random
import time
import start_simulation as ss
import pickle, yaml
import pandas as pd
#import elephant
from scipy.signal import find_peaks,correlate
from scipy.fft import fft, fftfreq
import set_network_params as netparams
from phase_ordering import order_by_phase
from pca import run_PCA
from connect_populations import ConnectNetwork
import population_functions as popfunc
ss.nest_start()
nn=netparams.neural_network()
conn=ConnectNetwork() 

import create_flx_rg as flx_rg
import create_ext_rg as ext_rg
import create_exc_inter_pop as exc
import create_inh_inter_pop as inh
import create_interneuron_pop as inter 
import create_mnp as mnp
import calculate_stability_metrics as calc

#Create neuron populations - NEST
rg1 = flx_rg.create_rg_population()
rg2 = ext_rg.create_rg_population()

exc1 = exc.create_exc_inter_population()
exc2 = exc.create_exc_inter_population()

V0c_1 = inter.interneuron_population()
V0c_1.create_interneuron_population(pop_type='V0c_1',self_connection='none',firing_behavior='tonic',pop_size=nn.v0c_pop_size,input_type='descending')
V0c_2 = inter.interneuron_population() 
V0c_2.create_interneuron_population(pop_type='V0c_2',self_connection='none',firing_behavior='tonic',pop_size=nn.v0c_pop_size,input_type='descending')

V1a_1 = inter.interneuron_population()
V1a_1.create_interneuron_population(pop_type='V1a_1',self_connection='none',firing_behavior='tonic',pop_size=nn.v1a_pop_size,input_type='sensory_feedback')
V1a_2 = inter.interneuron_population() 
V1a_2.create_interneuron_population(pop_type='V1a_2',self_connection='none',firing_behavior='tonic',pop_size=nn.v1a_pop_size,input_type='sensory_feedback')

rc_1 = inter.interneuron_population()
rc_1.create_interneuron_population(pop_type='rc_1',self_connection='none',firing_behavior='bursting',pop_size=nn.rc_pop_size,input_type='none')
rc_2 = inter.interneuron_population() 
rc_2.create_interneuron_population(pop_type='rc_2',self_connection='none',firing_behavior='bursting',pop_size=nn.rc_pop_size,input_type='none')

mnp1 = mnp.mnp()
mnp1.create_mnp(pop_type='flx')
mnp2 = mnp.mnp()
mnp2.create_mnp(pop_type='ext')

#Connect rg neurons to V2a excitatory interneuron populations
conn.create_connections(rg1.rg_exc_bursting,exc1.exc_inter_tonic,'custom_rg_v2a')
conn.create_connections(rg1.rg_exc_tonic,exc1.exc_inter_tonic,'custom_rg_v2a')

conn.create_connections(rg2.rg_exc_bursting,exc2.exc_inter_tonic,'custom_rg_v2a')
conn.create_connections(rg2.rg_exc_tonic,exc2.exc_inter_tonic,'custom_rg_v2a')

#Connect V2a excitatory interneuron populations to motor neurons
conn.create_connections(exc1.exc_inter_tonic,mnp1.motor_neuron_pop,'custom_v2a_mn') 
conn.create_connections(exc2.exc_inter_tonic,mnp2.motor_neuron_pop,'custom_v2a_mn') 

#Connect rg neurons to V1a excitatory interneuron populations
conn.create_connections(rg1.rg_exc_bursting,V1a_2.interneuron_pop,'custom_rg_v1a') 
conn.create_connections(rg1.rg_exc_tonic,V1a_2.interneuron_pop,'custom_rg_v1a')

conn.create_connections(rg2.rg_exc_bursting,V1a_1.interneuron_pop,'custom_rg_v1a')
conn.create_connections(rg2.rg_exc_tonic,V1a_1.interneuron_pop,'custom_rg_v1a')

#Connect rg neurons to V0c interneurons
conn.create_connections(rg1.rg_exc_bursting,V0c_1.interneuron_pop,'custom_rg_v0c') 
conn.create_connections(rg1.rg_exc_tonic,V0c_1.interneuron_pop,'custom_rg_v0c')

conn.create_connections(rg2.rg_exc_bursting,V0c_2.interneuron_pop,'custom_rg_v0c')
conn.create_connections(rg2.rg_exc_tonic,V0c_2.interneuron_pop,'custom_rg_v0c')

#Connect V0c to motor neurons
conn.create_connections(V0c_1.interneuron_pop,mnp1.motor_neuron_pop,'custom_v0c_mn') 
conn.create_connections(V0c_2.interneuron_pop,mnp2.motor_neuron_pop,'custom_v0c_mn') 

#Connect V1a interneurons to contralateral V1a interneurons
conn.create_connections(V1a_2.interneuron_pop,V1a_1.interneuron_pop,'custom_v1a_v1a')
conn.create_connections(V1a_1.interneuron_pop,V1a_2.interneuron_pop,'custom_v1a_v1a')

#Connect V1a to motor neurons
conn.create_connections(V1a_1.interneuron_pop,mnp1.motor_neuron_pop,'custom_v1a_mn')
conn.create_connections(V1a_2.interneuron_pop,mnp2.motor_neuron_pop,'custom_v1a_mn')

#Connect RC interneurons to V1a interneurons
conn.create_connections(rc_1.interneuron_pop,V1a_1.interneuron_pop,'custom_rc_v1a') 
conn.create_connections(rc_2.interneuron_pop,V1a_2.interneuron_pop,'custom_rc_v1a')

#Connect RC interneurons to contralateral RC interneurons
conn.create_connections(rc_1.interneuron_pop,rc_2.interneuron_pop,'custom_rc_rc')
conn.create_connections(rc_2.interneuron_pop,rc_1.interneuron_pop,'custom_rc_rc') 

#Connect RC interneurons to motor neurons
conn.create_connections(rc_1.interneuron_pop,mnp1.motor_neuron_pop,'custom_rc_mn') 
conn.create_connections(rc_2.interneuron_pop,mnp2.motor_neuron_pop,'custom_rc_mn')
conn.create_connections(mnp1.motor_neuron_pop,rc_1.interneuron_pop,'custom_mn_rc')
conn.create_connections(mnp2.motor_neuron_pop,rc_2.interneuron_pop,'custom_mn_rc')

if nn.rgs_connected == 1:
    inh1 = inh.create_inh_inter_population('V2b')  # V2b
    inh2 = inh.create_inh_inter_population('V1')  # V1

    # Connect excitatory rg neurons to V1/V2b inhibitory populations
    conn.create_connections(rg1.rg_exc_bursting, inh1.inh_inter_tonic, 'custom_rg_v2b')
    conn.create_connections(rg1.rg_exc_tonic, inh1.inh_inter_tonic, 'custom_rg_v2b')
    if nn.num_inh_inter_bursting_v2b > 0:
        conn.create_connections(rg1.rg_exc_bursting, inh1.inh_inter_bursting, 'custom_rg_v2b')
        conn.create_connections(rg1.rg_exc_tonic, inh1.inh_inter_bursting, 'custom_rg_v2b')
    conn.create_connections(rg2.rg_exc_bursting, inh2.inh_inter_tonic, 'custom_rg_v1')
    conn.create_connections(rg2.rg_exc_tonic, inh2.inh_inter_tonic, 'custom_rg_v1')
    if nn.num_inh_inter_bursting_v1 > 0:
        conn.create_connections(rg2.rg_exc_bursting, inh2.inh_inter_bursting, 'custom_rg_v1')
        conn.create_connections(rg2.rg_exc_tonic, inh2.inh_inter_bursting, 'custom_rg_v1')

    #Connect V1/V2b inhibitory populations to all rg neurons
    conn.create_connections(inh1.inh_inter_tonic,rg2.rg_exc_bursting,'custom_v2b_rg') 
    conn.create_connections(inh1.inh_inter_tonic,rg2.rg_exc_tonic,'custom_v2b_rg')
    conn.create_connections(inh1.inh_inter_tonic,rg2.rg_inh_bursting,'custom_v2b_rg')
    conn.create_connections(inh1.inh_inter_tonic,rg2.rg_inh_tonic,'custom_v2b_rg')
    if nn.num_inh_inter_bursting_v2b>0:
        conn.create_connections(inh1.inh_inter_bursting,rg2.rg_exc_bursting,'custom_v2b_rg') 
        conn.create_connections(inh1.inh_inter_bursting,rg2.rg_exc_tonic,'custom_v2b_rg')
        conn.create_connections(inh1.inh_inter_bursting,rg2.rg_inh_bursting,'custom_v2b_rg')
        conn.create_connections(inh1.inh_inter_bursting,rg2.rg_inh_tonic,'custom_v2b_rg')
    conn.create_connections(inh2.inh_inter_tonic,rg1.rg_exc_bursting,'custom_v1_rg') 
    conn.create_connections(inh2.inh_inter_tonic,rg1.rg_exc_tonic,'custom_v1_rg')
    conn.create_connections(inh2.inh_inter_tonic,rg1.rg_inh_bursting,'custom_v1_rg')
    conn.create_connections(inh2.inh_inter_tonic,rg1.rg_inh_tonic,'custom_v1_rg')
    if nn.num_inh_inter_bursting_v1>0:
        conn.create_connections(inh2.inh_inter_bursting,rg1.rg_exc_bursting,'custom_v1_rg') 
        conn.create_connections(inh2.inh_inter_bursting,rg1.rg_exc_tonic,'custom_v1_rg')
        conn.create_connections(inh2.inh_inter_bursting,rg1.rg_inh_bursting,'custom_v1_rg')
        conn.create_connections(inh2.inh_inter_bursting,rg1.rg_inh_tonic,'custom_v1_rg')
	
    #Connect V1/V2b inhibitory populations to V2a
    conn.create_connections(inh1.inh_inter_tonic,exc2.exc_inter_tonic,'custom_v2b_v2a') 
    if nn.num_inh_inter_bursting_v2b>0:
        conn.create_connections(inh1.inh_inter_bursting,exc2.exc_inter_tonic,'custom_v2b_v2a')
    conn.create_connections(inh2.inh_inter_tonic,exc1.exc_inter_tonic,'custom_v1_v2a') 
    if nn.num_inh_inter_bursting_v1>0:
        conn.create_connections(inh2.inh_inter_bursting,exc1.exc_inter_tonic,'custom_v1_v2a')
    
    if nn.v1v2b_mn_connected==1:
        #Connect V1/V2b inhibitory populations to motor neurons
        conn.create_connections(inh1.inh_inter_tonic,mnp2.motor_neuron_pop,'custom_v2b_mn') 
        if nn.num_inh_inter_bursting_v2b>0:
            conn.create_connections(inh1.inh_inter_bursting,mnp2.motor_neuron_pop,'custom_v2b_mn')
        conn.create_connections(inh2.inh_inter_tonic,mnp1.motor_neuron_pop,'custom_v1_mn') 
        if nn.num_inh_inter_bursting_v1>0:
            conn.create_connections(inh2.inh_inter_bursting,mnp1.motor_neuron_pop,'custom_v1_mn')
	
    #Connect excitatory rg neurons
    conn.create_connections(rg1.rg_exc_bursting,rg2.rg_exc_bursting,'custom_rg_rg')
    conn.create_connections(rg1.rg_exc_bursting,rg2.rg_exc_tonic,'custom_rg_rg')
    conn.create_connections(rg1.rg_exc_tonic,rg2.rg_exc_bursting,'custom_rg_rg')
    conn.create_connections(rg1.rg_exc_tonic,rg2.rg_exc_tonic,'custom_rg_rg')

    conn.create_connections(rg2.rg_exc_bursting,rg1.rg_exc_bursting,'custom_rg_rg')
    conn.create_connections(rg2.rg_exc_bursting,rg1.rg_exc_tonic,'custom_rg_rg')
    conn.create_connections(rg2.rg_exc_tonic,rg1.rg_exc_bursting,'custom_rg_rg')
    conn.create_connections(rg2.rg_exc_tonic,rg1.rg_exc_tonic,'custom_rg_rg')	

conn.calculate_synapse_percentage()    
    
print("Seed#: ",nn.rng_seed)
print("RG Flx: # exc (bursting, tonic): ",nn.flx_exc_bursting_count,nn.flx_exc_tonic_count,"; # inh(bursting, tonic): ",nn.flx_inh_bursting_count,nn.flx_inh_tonic_count)
print("RG Ext: # exc (bursting, tonic): ",nn.ext_exc_bursting_count,nn.ext_exc_tonic_count,"; # inh(bursting, tonic): ",nn.ext_inh_bursting_count,nn.ext_inh_tonic_count)
print("V2b/V1: # inh (bursting): ",nn.num_inh_inter_bursting_v2b,nn.num_inh_inter_bursting_v1,"; (tonic): ",nn.num_inh_inter_tonic_v2b,nn.num_inh_inter_tonic_v1)
print("V2a: # exc (bursting, tonic): ",nn.v2a_bursting_pop_size,nn.v2a_tonic_pop_size,"; # MNs: ",nn.num_motor_neurons)

init_time=50
nest.Simulate(init_time)
num_steps = int(nn.sim_time/nn.time_resolution)
t_start = time.perf_counter()
for i in range(int(num_steps/10)-init_time):	
    nest.Simulate(nn.time_resolution*10)
    print("t = " + str(nest.biological_time),end="\r")        
                
t_stop = time.perf_counter()    
print('Simulation completed. It took ',round(t_stop-t_start,2),' seconds.')

spike_count_array = []
#Read spike data - rg populations
senders_exc1,spiketimes_exc1 = popfunc.read_spike_data(rg1.spike_detector_rg_exc_bursting)
senders_inh1,spiketimes_inh1 = popfunc.read_spike_data(rg1.spike_detector_rg_inh_bursting)
senders_exc_tonic1,spiketimes_exc_tonic1 = popfunc.read_spike_data(rg1.spike_detector_rg_exc_tonic)
senders_inh_tonic1,spiketimes_inh_tonic1 = popfunc.read_spike_data(rg1.spike_detector_rg_inh_tonic)

senders_exc2,spiketimes_exc2 = popfunc.read_spike_data(rg2.spike_detector_rg_exc_bursting)
senders_inh2,spiketimes_inh2 = popfunc.read_spike_data(rg2.spike_detector_rg_inh_bursting)
senders_exc_tonic2,spiketimes_exc_tonic2 = popfunc.read_spike_data(rg2.spike_detector_rg_exc_tonic)
senders_inh_tonic2,spiketimes_inh_tonic2 = popfunc.read_spike_data(rg2.spike_detector_rg_inh_tonic)

#Read spike data - V2a excitatory interneurons
senders_exc_inter_tonic1,spiketimes_exc_inter_tonic1 = popfunc.read_spike_data(exc1.spike_detector_exc_inter_tonic)
senders_exc_inter_tonic2,spiketimes_exc_inter_tonic2 = popfunc.read_spike_data(exc2.spike_detector_exc_inter_tonic)

#Read spike data - interneurons
senders_V0c_1,spiketimes_V0c_1 = popfunc.read_spike_data(V0c_1.spike_detector)
senders_V0c_2,spiketimes_V0c_2 = popfunc.read_spike_data(V0c_2.spike_detector)
senders_V1a_1,spiketimes_V1a_1 = popfunc.read_spike_data(V1a_1.spike_detector)
senders_V1a_2,spiketimes_V1a_2 = popfunc.read_spike_data(V1a_2.spike_detector)
senders_rc_1,spiketimes_rc_1 = popfunc.read_spike_data(rc_1.spike_detector)
senders_rc_2,spiketimes_rc_2 = popfunc.read_spike_data(rc_2.spike_detector)

#Read spike data - MNPs
senders_mnp1,spiketimes_mnp1 = popfunc.read_spike_data(mnp1.spike_detector_motor)
senders_mnp2,spiketimes_mnp2 = popfunc.read_spike_data(mnp2.spike_detector_motor)

#Read spike data - V1/V2b inhibitory populations
if nn.rgs_connected==1:
    senders_inh_inter_tonic1,spiketimes_inh_inter_tonic1 = popfunc.read_spike_data(inh1.spike_detector_inh_inter_tonic)
    senders_inh_inter_tonic2,spiketimes_inh_inter_tonic2 = popfunc.read_spike_data(inh2.spike_detector_inh_inter_tonic)
    if nn.num_inh_inter_bursting_v2b>0:
        senders_inh_inter_bursting1,spiketimes_inh_inter_bursting1 = popfunc.read_spike_data(inh1.spike_detector_inh_inter_bursting)
    if nn.num_inh_inter_bursting_v1>0:
        senders_inh_inter_bursting2,spiketimes_inh_inter_bursting2 = popfunc.read_spike_data(inh2.spike_detector_inh_inter_bursting)

#Calculate synaptic balance of rg populations and total CPG network - missing interneurons
if nn.calculate_balance==1:
		
	rg1_exc_burst_weight = conn.calculate_weighted_balance(rg1.rg_exc_bursting,rg1.spike_detector_rg_exc_bursting)
	rg1_inh_burst_weight = conn.calculate_weighted_balance(rg1.rg_inh_bursting,rg1.spike_detector_rg_inh_bursting)
	rg1_exc_tonic_weight = conn.calculate_weighted_balance(rg1.rg_exc_tonic,rg1.spike_detector_rg_exc_tonic)
	rg1_inh_tonic_weight = conn.calculate_weighted_balance(rg1.rg_inh_tonic,rg1.spike_detector_rg_inh_tonic)
	weights_per_pop1 = [rg1_exc_burst_weight,rg1_inh_burst_weight,rg1_exc_tonic_weight,rg1_inh_tonic_weight]
	absolute_weights_per_pop1 = [rg1_exc_burst_weight,abs(rg1_inh_burst_weight),rg1_exc_tonic_weight,abs(rg1_inh_tonic_weight)]
	rg1_balance_pct = (sum(weights_per_pop1)/sum(absolute_weights_per_pop1))*100
	#print('RG1 balance %: ',round(rg1_balance_pct,2),' >0 skew excitatory; <0 skew inhibitory')
	
	rg2_exc_burst_weight = conn.calculate_weighted_balance(rg2.rg_exc_bursting,rg2.spike_detector_rg_exc_bursting)
	rg2_inh_burst_weight = conn.calculate_weighted_balance(rg2.rg_inh_bursting,rg2.spike_detector_rg_inh_bursting)
	rg2_exc_tonic_weight = conn.calculate_weighted_balance(rg2.rg_exc_tonic,rg2.spike_detector_rg_exc_tonic)
	rg2_inh_tonic_weight = conn.calculate_weighted_balance(rg2.rg_inh_tonic,rg2.spike_detector_rg_inh_tonic)
	weights_per_pop2 = [rg2_exc_burst_weight,rg2_inh_burst_weight,rg2_exc_tonic_weight,rg2_inh_tonic_weight]
	absolute_weights_per_pop2 = [rg2_exc_burst_weight,abs(rg2_inh_burst_weight),rg2_exc_tonic_weight,abs(rg2_inh_tonic_weight)]
	rg2_balance_pct = (sum(weights_per_pop2)/sum(absolute_weights_per_pop2))*100
	#print('RG2 balance %: ',round(rg2_balance_pct,2),' >0 skew excitatory; <0 skew inhibitory')
	
	exc_tonic1_weight = conn.calculate_weighted_balance(exc1.exc_inter_tonic,exc1.spike_detector_exc_inter_tonic)
	exc_tonic2_weight = conn.calculate_weighted_balance(exc2.exc_inter_tonic,exc2.spike_detector_exc_inter_tonic)
	mnp1_weight = conn.calculate_weighted_balance(mnp1.motor_neuron_pop,mnp1.spike_detector_motor)
	mnp2_weight = conn.calculate_weighted_balance(mnp2.motor_neuron_pop,mnp2.spike_detector_motor)
	weights_per_pop_side1 = [rg1_exc_burst_weight,rg1_inh_burst_weight,rg1_exc_tonic_weight,rg1_inh_tonic_weight,exc_tonic1_weight,exc_bursting1_weight,mnp1_weight]
	absolute_weights_per_pop_side1 = [rg1_exc_burst_weight,abs(rg1_inh_burst_weight),rg1_exc_tonic_weight,abs(rg1_inh_tonic_weight),exc_tonic1_weight,exc_bursting1_weight,mnp1_weight]
	weights_per_pop_side2 = [rg2_exc_burst_weight,rg2_inh_burst_weight,rg2_exc_tonic_weight,rg2_inh_tonic_weight,exc_tonic2_weight,exc_bursting2_weight,mnp2_weight]
	absolute_weights_per_pop_side2 = [rg2_exc_burst_weight,abs(rg2_inh_burst_weight),rg2_exc_tonic_weight,abs(rg2_inh_tonic_weight),exc_tonic2_weight,exc_bursting2_weight,mnp2_weight]
	side1_balance_pct = (sum(weights_per_pop_side1)/sum(absolute_weights_per_pop_side1))*100
	side2_balance_pct = (sum(weights_per_pop_side2)/sum(absolute_weights_per_pop_side2))*100
	print('Balance % (RG1, RG2, Side1, Side2): ',round(rg1_balance_pct,2),round(rg2_balance_pct,2),round(side1_balance_pct,2),round(side2_balance_pct,2))
	
	if nn.rgs_connected==1:
		inh1_weight = conn.calculate_weighted_balance(inh1.inh_pop,inh1.spike_detector_inh)
		inh2_weight = conn.calculate_weighted_balance(inh2.inh_pop,inh2.spike_detector_inh)
		weights_per_pop = [rg1_exc_burst_weight,rg1_inh_burst_weight,rg1_exc_tonic_weight,rg1_inh_tonic_weight,rg2_exc_burst_weight,rg2_inh_burst_weight,rg2_exc_tonic_weight,rg2_inh_tonic_weight,inh1_weight,inh1_weight,exc_tonic1_weight,exc_bursting1_weight,mnp1_weight,exc_tonic2_weight,exc_bursting2_weight,mnp2_weight]
		absolute_weights_per_pop = [rg1_exc_burst_weight,abs(rg1_inh_burst_weight),rg1_exc_tonic_weight,abs(rg1_inh_tonic_weight),rg2_exc_burst_weight,abs(rg2_inh_burst_weight),rg2_exc_tonic_weight,abs(rg2_inh_tonic_weight),abs(inh1_weight),abs(inh1_weight),exc_tonic1_weight,exc_bursting1_weight,mnp1_weight,exc_tonic2_weight,exc_bursting2_weight,mnp2_weight]
		total_balance_pct = (sum(weights_per_pop)/sum(absolute_weights_per_pop))*100
		print('Balance % (complete network): ',round(total_balance_pct,2),' >0 skew excitatory; <0 skew inhibitory')

if nn.phase_ordered_plot==1:
    t_start = time.perf_counter()
    #Convolve spike data - rg populations
    rg_exc_convolved1 = popfunc.convolve_spiking_activity(nn.flx_exc_bursting_count,spiketimes_exc1)
    rg_exc_tonic_convolved1 = popfunc.convolve_spiking_activity(nn.flx_exc_tonic_count,spiketimes_exc_tonic1)
    rg_inh_convolved1 = popfunc.convolve_spiking_activity(nn.flx_inh_bursting_count,spiketimes_inh1)
    rg_inh_tonic_convolved1 = popfunc.convolve_spiking_activity(nn.flx_inh_tonic_count,spiketimes_inh_tonic1)
    spikes_convolved_all1 = np.vstack([rg_exc_convolved1,rg_inh_convolved1])
    spikes_convolved_all1 = np.vstack([spikes_convolved_all1,rg_exc_tonic_convolved1])
    spikes_convolved_all1 = np.vstack([spikes_convolved_all1,rg_inh_tonic_convolved1])

    rg_exc_convolved2 = popfunc.convolve_spiking_activity(nn.ext_exc_bursting_count,spiketimes_exc2)
    rg_exc_tonic_convolved2 = popfunc.convolve_spiking_activity(nn.ext_exc_tonic_count,spiketimes_exc_tonic2)
    rg_inh_convolved2 = popfunc.convolve_spiking_activity(nn.ext_inh_bursting_count,spiketimes_inh2)
    rg_inh_tonic_convolved2 = popfunc.convolve_spiking_activity(nn.ext_inh_tonic_count,spiketimes_inh_tonic2)
    spikes_convolved_all2 = np.vstack([rg_exc_convolved2,rg_inh_convolved2])
    spikes_convolved_all2 = np.vstack([spikes_convolved_all2,rg_exc_tonic_convolved2])
    spikes_convolved_all2 = np.vstack([spikes_convolved_all2,rg_inh_tonic_convolved2])
    spikes_convolved_rgs = np.vstack([spikes_convolved_all1,spikes_convolved_all2])	

    #Convolve spike data - V2a excitatory interneuron populations
    exc_inter_tonic_convolved1 = popfunc.convolve_spiking_activity(nn.v2a_tonic_pop_size,spiketimes_exc_inter_tonic1)
    exc_inter_tonic_convolved2 = popfunc.convolve_spiking_activity(nn.v2a_tonic_pop_size,spiketimes_exc_inter_tonic2)

    #Convolve spike data - interneuron populations
    V0c_convolved1 = popfunc.convolve_spiking_activity(nn.v0c_pop_size,spiketimes_V0c_1)
    V0c_convolved2 = popfunc.convolve_spiking_activity(nn.v0c_pop_size,spiketimes_V0c_2)
    V1a_convolved1 = popfunc.convolve_spiking_activity(nn.v1a_pop_size,spiketimes_V1a_1)
    V1a_convolved2 = popfunc.convolve_spiking_activity(nn.v1a_pop_size,spiketimes_V1a_2)
    rc_convolved1 = popfunc.convolve_spiking_activity(nn.rc_pop_size,spiketimes_rc_1)
    rc_convolved2 = popfunc.convolve_spiking_activity(nn.rc_pop_size,spiketimes_rc_2)

    #Convolve spike data - MNPs
    motor_convolved1 = popfunc.convolve_spiking_activity(nn.num_motor_neurons,spiketimes_mnp1)
    motor_convolved2 = popfunc.convolve_spiking_activity(nn.num_motor_neurons,spiketimes_mnp2)

    spikes_convolved_side1 = np.vstack([spikes_convolved_all1,exc_inter_tonic_convolved1])
    spikes_convolved_side1 = np.vstack([spikes_convolved_all1,V0c_convolved1])
    spikes_convolved_side1 = np.vstack([spikes_convolved_all1,V1a_convolved1])
    spikes_convolved_side1 = np.vstack([spikes_convolved_all1,rc_convolved1])
    spikes_convolved_side1 = np.vstack([spikes_convolved_side1,motor_convolved1])
    spikes_convolved_side2 = np.vstack([spikes_convolved_all2,exc_inter_tonic_convolved2])
    spikes_convolved_side2 = np.vstack([spikes_convolved_all2,V0c_convolved2])
    spikes_convolved_side2 = np.vstack([spikes_convolved_all2,V1a_convolved2])
    spikes_convolved_side2 = np.vstack([spikes_convolved_all2,rc_convolved2])
    spikes_convolved_side2 = np.vstack([spikes_convolved_side2,motor_convolved2])
    spikes_convolved_both_sides = np.vstack([spikes_convolved_side1,spikes_convolved_side2])
    spikes_convolved_complete_network = spikes_convolved_both_sides
    
    # Convolve spike data - inh populations
    if nn.rgs_connected == 1:
        inh_inter_tonic_convolved1 = popfunc.convolve_spiking_activity(nn.num_inh_inter_tonic_v2b, spiketimes_inh_inter_tonic1)
        inh_inter_tonic_convolved2 = popfunc.convolve_spiking_activity(nn.num_inh_inter_tonic_v1, spiketimes_inh_inter_tonic2)
        spikes_convolved_inh = np.vstack([inh_inter_tonic_convolved1, inh_inter_tonic_convolved2])
        if nn.num_inh_inter_bursting_v2b > 0:
            inh_inter_bursting_convolved1 = popfunc.convolve_spiking_activity(nn.num_inh_inter_bursting_v2b, spiketimes_inh_inter_bursting1)
            spikes_convolved_inh = np.vstack([spikes_convolved_inh, inh_inter_bursting_convolved1])        
        if nn.num_inh_inter_bursting_v1 > 0:
            inh_inter_bursting_convolved2 = popfunc.convolve_spiking_activity(nn.num_inh_inter_bursting_v1, spiketimes_inh_inter_bursting2)
            spikes_convolved_inh = np.vstack([spikes_convolved_inh, inh_inter_bursting_convolved2])        
        spikes_convolved_complete_network = np.vstack([spikes_convolved_both_sides, spikes_convolved_inh])

    if nn.remove_silent:
        print('Removing silent neurons')
        spikes_convolved_all1 = spikes_convolved_all1[~np.all(spikes_convolved_all1 == 0, axis=1)]
        spikes_convolved_all2 = spikes_convolved_all2[~np.all(spikes_convolved_all2 == 0, axis=1)]
        spikes_convolved_rgs = spikes_convolved_rgs[~np.all(spikes_convolved_rgs == 0, axis=1)]
        spikes_convolved_side1 = spikes_convolved_side1[~np.all(spikes_convolved_side1 == 0, axis=1)]
        spikes_convolved_side2 = spikes_convolved_side2[~np.all(spikes_convolved_side2 == 0, axis=1)]
        spikes_convolved_both_sides = spikes_convolved_both_sides[~np.all(spikes_convolved_both_sides == 0, axis=1)]
        spikes_convolved_complete_network = spikes_convolved_complete_network[~np.all(spikes_convolved_complete_network == 0, axis=1)]
    t_stop = time.perf_counter()
    spikes_convolved_all1 = popfunc.normalize_rows(spikes_convolved_all1)
    spikes_convolved_all2 = popfunc.normalize_rows(spikes_convolved_all2)
    spikes_convolved_rgs = popfunc.normalize_rows(spikes_convolved_rgs)
    spikes_convolved_side1 = popfunc.normalize_rows(spikes_convolved_side1)
    spikes_convolved_side2 = popfunc.normalize_rows(spikes_convolved_side2)
    spikes_convolved_both_sides = popfunc.normalize_rows(spikes_convolved_both_sides)
    spikes_convolved_complete_network = popfunc.normalize_rows(spikes_convolved_complete_network)    
    print('Convolved spiking activity complete, taking ',int(t_stop-t_start),' seconds.') 

#Run PCA - rg populations
if nn.pca_plot==1 and nn.phase_ordered_plot==1:
	run_PCA(spikes_convolved_complete_network,'all_pops')
	print('PCA complete')
if nn.pca_plot==1 and nn.phase_ordered_plot==0:
	print('The convolved spiking activity is required to run a PCA, ensure "phase_ordered_plot" is selected.')

#Create Rate Coded Output
if nn.rate_coded_plot==1:
    t_start = time.perf_counter()
    spike_bins_rg_exc1 = popfunc.rate_code_spikes(nn.flx_exc_bursting_count,spiketimes_exc1)
    spike_bins_rg_inh1 = popfunc.rate_code_spikes(nn.flx_inh_bursting_count,spiketimes_inh1)
    spike_bins_rg_exc_tonic1 = popfunc.rate_code_spikes(nn.flx_exc_tonic_count,spiketimes_exc_tonic1)
    spike_bins_rg_inh_tonic1 = popfunc.rate_code_spikes(nn.flx_inh_tonic_count,spiketimes_inh_tonic1)
    spike_bins_rg1 = spike_bins_rg_exc1+spike_bins_rg_exc_tonic1+spike_bins_rg_inh1+spike_bins_rg_inh_tonic1
    spike_bins_rg1_true = spike_bins_rg1
    print('Max spike count RG_F: ',max(spike_bins_rg1))
    spike_bins_rg1 = (spike_bins_rg1-np.min(spike_bins_rg1))/(np.max(spike_bins_rg1)-np.min(spike_bins_rg1))

    spike_bins_rg_exc2 = popfunc.rate_code_spikes(nn.ext_exc_bursting_count,spiketimes_exc2)
    spike_bins_rg_inh2 = popfunc.rate_code_spikes(nn.ext_inh_bursting_count,spiketimes_inh2)
    spike_bins_rg_exc_tonic2 = popfunc.rate_code_spikes(nn.ext_exc_tonic_count,spiketimes_exc_tonic2)
    spike_bins_rg_inh_tonic2 = popfunc.rate_code_spikes(nn.ext_inh_tonic_count,spiketimes_inh_tonic2)
    spike_bins_rg2 = spike_bins_rg_exc2+spike_bins_rg_exc_tonic2+spike_bins_rg_inh2+spike_bins_rg_inh_tonic2
    spike_bins_rg2_true = spike_bins_rg2
    print('Max spike count RG_E: ',max(spike_bins_rg2))
    spike_bins_rg2 = (spike_bins_rg2-np.min(spike_bins_rg2))/(np.max(spike_bins_rg2)-np.min(spike_bins_rg2))
    spike_bins_rgs = spike_bins_rg1+spike_bins_rg2

    spike_bins_exc_inter_tonic1 = popfunc.rate_code_spikes(nn.v2a_tonic_pop_size,spiketimes_exc_inter_tonic1)
    spike_bins_exc_inter1 = spike_bins_exc_inter_tonic1
    spike_bins_exc_inter1_true = spike_bins_exc_inter1
    spike_bins_exc_inter1 = (spike_bins_exc_inter1-np.min(spike_bins_exc_inter1))/(np.max(spike_bins_exc_inter1)-np.min(spike_bins_exc_inter1))
    spike_bins_exc_inter_tonic2 = popfunc.rate_code_spikes(nn.v2a_tonic_pop_size,spiketimes_exc_inter_tonic2)
    spike_bins_exc_inter2 = spike_bins_exc_inter_tonic2
    spike_bins_exc_inter2_true = spike_bins_exc_inter2
    spike_bins_exc_inter2 = (spike_bins_exc_inter2-np.min(spike_bins_exc_inter2))/(np.max(spike_bins_exc_inter2)-np.min(spike_bins_exc_inter2))

    spike_bins_V0c_1 = popfunc.rate_code_spikes(nn.v0c_pop_size,spiketimes_V0c_1)
    spike_bins_V0c_1_true = spike_bins_V0c_1
    spike_bins_V0c_1 = (spike_bins_V0c_1-np.min(spike_bins_V0c_1))/(np.max(spike_bins_V0c_1)-np.min(spike_bins_V0c_1))
    spike_bins_V0c_2 = popfunc.rate_code_spikes(nn.v0c_pop_size,spiketimes_V0c_2)
    spike_bins_V0c_2_true = spike_bins_V0c_2  
    spike_bins_V0c_2 = (spike_bins_V0c_2-np.min(spike_bins_V0c_2))/(np.max(spike_bins_V0c_2)-np.min(spike_bins_V0c_2))
    spike_bins_V1a_1 = popfunc.rate_code_spikes(nn.v1a_pop_size,spiketimes_V1a_1)
    spike_bins_V1a_1_true = spike_bins_V1a_1
    spike_bins_V1a_1 = (spike_bins_V1a_1-np.min(spike_bins_V1a_1))/(np.max(spike_bins_V1a_1)-np.min(spike_bins_V1a_1))
    spike_bins_V1a_2 = popfunc.rate_code_spikes(nn.v1a_pop_size,spiketimes_V1a_2)
    spike_bins_V1a_2_true = spike_bins_V1a_2
    spike_bins_V1a_2 = (spike_bins_V1a_2-np.min(spike_bins_V1a_2))/(np.max(spike_bins_V1a_2)-np.min(spike_bins_V1a_2))
    spike_bins_rc_1 = popfunc.rate_code_spikes(nn.rc_pop_size,spiketimes_rc_1)
    spike_bins_rc_1_true = spike_bins_rc_1
    spike_bins_rc_1 = (spike_bins_rc_1-np.min(spike_bins_rc_1))/(np.max(spike_bins_rc_1)-np.min(spike_bins_rc_1))
    spike_bins_rc_2 = popfunc.rate_code_spikes(nn.rc_pop_size,spiketimes_rc_2)
    spike_bins_rc_2_true = spike_bins_rc_2
    spike_bins_rc_2 = (spike_bins_rc_2-np.min(spike_bins_rc_2))/(np.max(spike_bins_rc_2)-np.min(spike_bins_rc_2))

    spike_bins_mnp1 = popfunc.rate_code_spikes(nn.num_motor_neurons,spiketimes_mnp1)
    spike_bins_mnp1_true = spike_bins_mnp1
    print('Max spike count FLX: ',max(spike_bins_mnp1))
    spike_bins_mnp1 = (spike_bins_mnp1-np.min(spike_bins_mnp1))/(np.max(spike_bins_mnp1)-np.min(spike_bins_mnp1))
    spike_bins_mnp2 = popfunc.rate_code_spikes(nn.num_motor_neurons,spiketimes_mnp2)
    spike_bins_mnp2_true = spike_bins_mnp2
    print('Max spike count EXT: ',max(spike_bins_mnp2))
    spike_bins_mnp2 = (spike_bins_mnp2-np.min(spike_bins_mnp2))/(np.max(spike_bins_mnp2)-np.min(spike_bins_mnp2))
    spike_bins_mnps = spike_bins_mnp1+spike_bins_mnp2

    if nn.rgs_connected==1:
        spike_bins_inh_inter_tonic1 = popfunc.rate_code_spikes(nn.num_inh_inter_tonic_v2b,spiketimes_inh_inter_tonic1)
        spike_bins_inh_inter1 = spike_bins_inh_inter_tonic1
        if nn.num_inh_inter_bursting_v2b>0: 
            spike_bins_inh_inter_bursting1 = popfunc.rate_code_spikes(nn.num_inh_inter_bursting_v2b,spiketimes_inh_inter_bursting1)
            spike_bins_inh_inter1 = spike_bins_inh_inter_tonic1+spike_bins_inh_inter_bursting1
        spike_bins_inh_inter1_true = spike_bins_inh_inter1
        spike_bins_inh_inter1 = (spike_bins_inh_inter1-np.min(spike_bins_inh_inter1))/(np.max(spike_bins_inh_inter1)-np.min(spike_bins_inh_inter1))
        spike_bins_inh_inter_tonic2 = popfunc.rate_code_spikes(nn.num_inh_inter_tonic_v1,spiketimes_inh_inter_tonic2)
        spike_bins_inh_inter2 = spike_bins_inh_inter_tonic2
        if nn.num_inh_inter_bursting_v1>0:
            spike_bins_inh_inter_bursting2 = popfunc.rate_code_spikes(nn.num_inh_inter_bursting_v1,spiketimes_inh_inter_bursting2)
            spike_bins_inh_inter2 = spike_bins_inh_inter_tonic2+spike_bins_inh_inter_bursting2
        spike_bins_inh_inter2_true = spike_bins_inh_inter2
        spike_bins_inh_inter2 = (spike_bins_inh_inter2-np.min(spike_bins_inh_inter2))/(np.max(spike_bins_inh_inter2)-np.min(spike_bins_inh_inter2))

    t_stop = time.perf_counter()
    print('Rate coded activity complete, taking ',int(t_stop-t_start),' seconds.')

#Plot phase sorted activity
if nn.phase_ordered_plot==1 and nn.rate_coded_plot==1:
    order_by_phase(spikes_convolved_all1, spike_bins_rg1, 'rg1',avg_rg1_peaks) #ADD AVG_RG1_PEAKS CALCULATION (LOOK AT DOPA NETWORK)
    order_by_phase(spikes_convolved_all2, spike_bins_rg2, 'rg2',avg_rg2_peaks)
    order_by_phase(spikes_convolved_side1, spike_bins_mnp1, 'side1',avg_mnp1_peaks)
    order_by_phase(spikes_convolved_side2, spike_bins_mnp2, 'side2',avg_mnp2_peaks)
    order_by_phase(spikes_convolved_complete_network, spike_bins_mnp1, 'all_pops',4300) #Compare to one network output

    neuron_num_to_plot = int(spikes_convolved_all1.shape[0]/5)
    #pylab.figure()
    #pylab.subplot(211)
    fig, ax = plt.subplots(5, sharex=True, figsize=(15, 8))	
    ax[0].plot(spikes_convolved_all1[neuron_num_to_plot])
    ax[1].plot(spikes_convolved_all1[neuron_num_to_plot*2])
    ax[2].plot(spikes_convolved_all1[neuron_num_to_plot*3])
    ax[3].plot(spikes_convolved_all1[neuron_num_to_plot*4])
    ax[4].plot(spike_bins_rg1,label='RG1')
    ax[0].set_title('Firing rate individual neurons vs Population activity (RG1)')
    ax[0].set_ylabel('Exc Bursting')
    ax[1].set_ylabel('Inh Bursting')
    ax[2].set_ylabel('Exc Tonic')
    ax[3].set_ylabel('Inh Tonic')
    ax[4].set_ylabel('RG1')
    ax[4].set_xlabel('Time steps')
    if nn.args['save_results']: plt.savefig(nn.pathFigures + '/' + 'single_neuron_firing_rate.pdf',bbox_inches="tight")
if nn.phase_ordered_plot==1 and nn.rate_coded_plot==0:
    print('The rate-coded output must be calculated in order to produce a phase-ordered plot, ensure "rate_coded_plot" is selected.')

#Plot rate-coded output
if nn.rate_coded_plot==1:
    t = np.arange(0,len(spike_bins_rg1),1)
  
    fig, ax = plt.subplots(4,sharex='all')
    ax[0].plot(t, spike_bins_V0c_1_true)
    ax[0].plot(t, spike_bins_V0c_2_true)
    ax[1].plot(t, spike_bins_V1a_1_true)
    ax[1].plot(t, spike_bins_V1a_2_true)		
    ax[2].plot(t, spike_bins_rc_1_true)
    ax[2].plot(t, spike_bins_rc_2_true) 
    ax[3].plot(t, spike_bins_mnp1_true)
    ax[3].plot(t, spike_bins_mnp2_true)
    for i in range(2):
        ax[i].set_xticks([])
        ax[i].set_xlim(0,len(spike_bins_rg1_true))
    ax[3].set_xlabel('Time (ms)')
    ax[3].set_xticks([0,10000,20000,30000,40000,50000,60000,70000,80000,90000])
    ax[3].set_xticklabels([0,1000,2000,3000,4000,5000,6000,7000,8000,9000])
    ax[3].set_xlim(0,len(spike_bins_rg1_true))
    ax[0].legend(['V0c_F', 'V0c_E'],loc='upper right',fontsize='x-small') 
    ax[1].legend(['1a_F', '1a_E'],loc='upper right',fontsize='x-small') 
    ax[2].legend(['RC_F', 'RC_E'],loc='upper right',fontsize='x-small') 
    ax[3].legend(['FLX', 'EXT'],loc='upper right',fontsize='x-small')
    ax[0].set_title("Population output (V0c)")
    ax[1].set_title("Population output (1a)")
    ax[2].set_title("Population output (RC)")
    ax[3].set_title("Population output (MNP)")
    figure = plt.gcf() # get current figure
    figure.set_size_inches(8, 6)
    plt.tight_layout()
    if nn.args['save_results']: plt.savefig(nn.pathFigures + '/' + 'rate_coded_output_interneurons.pdf',bbox_inches="tight")
	
    fig, ax = plt.subplots(4,sharex='all')
    ax[0].plot(t, spike_bins_rg1_true)
    ax[0].plot(t, spike_bins_rg2_true)
    ax[1].plot(t, spike_bins_inh_inter1_true)
    ax[1].plot(t, spike_bins_inh_inter2_true)		
    ax[2].plot(t, spike_bins_exc_inter1_true)
    ax[2].plot(t, spike_bins_exc_inter2_true) 
    ax[3].plot(t, spike_bins_mnp1_true)
    ax[3].plot(t, spike_bins_mnp2_true)
    for i in range(2):
        ax[i].set_xticks([])
        ax[i].set_xlim(0,len(spike_bins_rg1_true))
    ax[3].set_xlabel('Time (ms)')
    ax[3].set_xticks([0,10000,20000,30000,40000,50000,60000,70000,80000,90000])
    ax[3].set_xticklabels([0,1000,2000,3000,4000,5000,6000,7000,8000,9000])
    ax[3].set_xlim(0,len(spike_bins_rg1_true))
    ax[0].legend(['RG_F', 'RG_E'],loc='upper right',fontsize='x-small') 
    ax[1].legend(['V2b', 'V1'],loc='upper right',fontsize='x-small') 
    ax[2].legend(['V2a_F', 'V2a_E'],loc='upper right',fontsize='x-small') 
    ax[3].legend(['FLX', 'EXT'],loc='upper right',fontsize='x-small')
    ax[0].set_title("Population output (RG)")
    ax[1].set_title("Population output (V1/V2b)")
    ax[2].set_title("Population output (V2a)")
    ax[3].set_title("Population output (MNP)")
    figure = plt.gcf() # get current figure
    figure.set_size_inches(8, 6)
    plt.tight_layout()
    if nn.args['save_results']: plt.savefig(nn.pathFigures + '/' + 'rate_coded_output.pdf',bbox_inches="tight")
        
    fig, ax = plt.subplots(2,sharex='all')
    ax[0].plot(t, spike_bins_rg1_true)
    ax[0].plot(t, spike_bins_rg2_true)
    ax[1].plot(t, spike_bins_mnp1_true)
    ax[1].plot(t, spike_bins_mnp2_true)
    ax[0].set_xticks([])
    ax[0].set_xlim(0,len(spike_bins_rg1_true))
    ax[1].set_xlabel('Time (ms)')
    ax[1].set_xticks([0,10000,20000,30000,40000,50000,60000,70000,80000,90000])
    ax[1].set_xticklabels([0,1000,2000,3000,4000,5000,6000,7000,8000,9000])
    ax[1].set_xlim(0,len(spike_bins_rg1_true))
    ax[0].legend(['RG_F', 'RG_E'],loc='upper right',fontsize='x-small')  
    ax[1].legend(['FLX', 'EXT'],loc='upper right',fontsize='x-small')
    ax[0].set_title("Population output (RG)")
    ax[1].set_title("Population output (MNP)")
    figure = plt.gcf() # get current figure
    figure.set_size_inches(8, 6)
    plt.tight_layout()
    if nn.args['save_results']: plt.savefig(nn.pathFigures + '/' + 'rate_coded_output_rg_mnp.pdf',bbox_inches="tight")

    if max(spike_bins_mnp1)>0 and max(spike_bins_mnp2)>0: 
        avg_freq, avg_phase, bd_comparison = calc.analyze_output(spike_bins_mnp1,spike_bins_mnp2,'MNP',y_line_bd=0.4,y_line_phase=0.7)
        
if nn.spike_distribution_plot==1:
    #Count spikes per neuron
    # Define parameters and senders for each neuron group
    neuron_params = [
        (nn.flx_exc_bursting_count, senders_exc1), 
        (nn.flx_inh_bursting_count, senders_inh1), 
        (nn.flx_exc_tonic_count, senders_exc_tonic1), 
        (nn.flx_inh_tonic_count, senders_inh_tonic1),
        (nn.ext_exc_bursting_count, senders_exc2), 
        (nn.ext_inh_bursting_count, senders_inh2), 
        (nn.ext_exc_tonic_count, senders_exc_tonic2), 
        (nn.ext_inh_tonic_count, senders_inh_tonic2),
        (nn.v2a_tonic_pop_size, senders_exc_inter_tonic1), 
        (nn.v2a_tonic_pop_size, senders_exc_inter_tonic2),
        (nn.v0c_pop_size,senders_V0c_1),
        (nn.v0c_pop_size,senders_V0c_2),
        (nn.v1a_pop_size,senders_V1a_1),
        (nn.v1a_pop_size,senders_V1a_2),
        (nn.rc_pop_size,senders_rc_1),
        (nn.rc_pop_size,senders_rc_2),
        (nn.num_motor_neurons, senders_mnp1), 
        (nn.num_motor_neurons, senders_mnp2)
    ]

    # If RGs are connected, add inhibitory inter-neurons
    if nn.rgs_connected == 1:
        neuron_params.extend([
            (nn.num_inh_inter_tonic_v2b, senders_inh_inter_tonic1), 
            (nn.num_inh_inter_tonic_v1, senders_inh_inter_tonic2)
        ])

        # Conditional addition based on neuron count
        if nn.num_inh_inter_bursting_v2b > 0:
            neuron_params.append((nn.num_inh_inter_bursting_v2b, senders_inh_inter_bursting1))
        if nn.num_inh_inter_bursting_v1 > 0:
            neuron_params.append((nn.num_inh_inter_bursting_v1, senders_inh_inter_bursting2))

    # Initialize counters for spikes, sparse firing, and silent neurons
    all_indiv_spike_counts = []
    sparse_firing_count = 0
    silent_neuron_count = 0

    # Iterate through all neuron groups and compute spike data
    for param, senders in neuron_params:
        indiv_spikes, _, sparse_count, silent_count = popfunc.count_indiv_spikes(param, senders, avg_freq)
        all_indiv_spike_counts.extend(indiv_spikes)
        sparse_firing_count += sparse_count
        silent_neuron_count += silent_count

    # Calculate and print sparse firing statistics
    active_neuron_count = len(all_indiv_spike_counts) - silent_neuron_count
    if len(all_indiv_spike_counts) > 0:
        sparse_firing_percentage = round(sparse_firing_count * 100 / (len(all_indiv_spike_counts) - silent_neuron_count), 2)
        print('Active neuron count, sparsely firing count, % sparse firing:', active_neuron_count, sparse_firing_count, sparse_firing_percentage, '%')
    else:
        print("No active neurons found; all neurons are silent.")       
      
    spike_distribution = [all_indiv_spike_counts.count(i) for i in range(max(all_indiv_spike_counts))]
    '''
    pylab.figure()
    pylab.plot(spike_distribution[2:])
    pylab.xscale('log')
    pylab.xlabel('Total Spike Count')
    pylab.ylabel('Number of Neurons')
    pylab.title('Spike Distribution')
    if nn.args['save_results']: plt.savefig(nn.pathFigures + '/' + 'spike_distribution.pdf',bbox_inches="tight")
    '''
if nn.args['save_results'] and nn.save_all_pops==0:
	# Save population output
	np.savetxt(nn.pathFigures + '/output_mnp1.csv',spike_bins_mnp1_true,delimiter=',')
	np.savetxt(nn.pathFigures + '/output_mnp2.csv',spike_bins_mnp2_true,delimiter=',')
    
if nn.args['save_results'] and nn.save_all_pops==1:
    np.savetxt(nn.pathFigures + '/output_rg1.csv',spike_bins_rg1_true,delimiter=',')
    np.savetxt(nn.pathFigures + '/output_rg2.csv',spike_bins_rg2_true,delimiter=',')
    np.savetxt(nn.pathFigures + '/output_v2b.csv',spike_bins_inh_inter1_true,delimiter=',')
    np.savetxt(nn.pathFigures + '/output_v1.csv',spike_bins_inh_inter2_true,delimiter=',')
    np.savetxt(nn.pathFigures + '/output_v2a1.csv',spike_bins_exc_inter1_true,delimiter=',')
    np.savetxt(nn.pathFigures + '/output_v2a2.csv',spike_bins_exc_inter2_true,delimiter=',')
    np.savetxt(nn.pathFigures + '/output_v0c1.csv',spike_bins_V0c_1_true,delimiter=',')
    np.savetxt(nn.pathFigures + '/output_v0c2.csv',spike_bins_V0c_2_true,delimiter=',')
    np.savetxt(nn.pathFigures + '/output_1a1.csv',spike_bins_V1a_1_true,delimiter=',')
    np.savetxt(nn.pathFigures + '/output_1a2.csv',spike_bins_V1a_2_true,delimiter=',')
    np.savetxt(nn.pathFigures + '/output_rc1.csv',spike_bins_rc_1_true,delimiter=',')
    np.savetxt(nn.pathFigures + '/output_rc2.csv',spike_bins_rc_2_true,delimiter=',')
    np.savetxt(nn.pathFigures + '/output_mnp1.csv',spike_bins_mnp1_true,delimiter=',')
    np.savetxt(nn.pathFigures + '/output_mnp2.csv',spike_bins_mnp2_true,delimiter=',')
