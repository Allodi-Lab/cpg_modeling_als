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
import warnings
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
import send_receive_feedback as interface_fb

warnings.filterwarnings('ignore')

#Create neuron populations - NEST
rg1 = flx_rg.create_rg_population()
rg2 = ext_rg.create_rg_population()

exc1 = exc.create_exc_inter_population()
exc2 = exc.create_exc_inter_population()

V0c_1 = inter.interneuron_population()
V0c_2 = inter.interneuron_population() 
V1a_1 = inter.interneuron_population()
V1a_2 = inter.interneuron_population() 
rc_1 = inter.interneuron_population()
rc_2 = inter.interneuron_population()
mnp1 = mnp.mnp()
mnp2 = mnp.mnp()

fb = interface_fb.feedback()

if nn.remove_descending_drive==0:
    V0c_1.create_interneuron_population(pop_type='V0c_1',self_connection='none',firing_behavior='tonic',pop_size=nn.v0c_pop_size,input_type='descending')
    V0c_2.create_interneuron_population(pop_type='V0c_2',self_connection='none',firing_behavior='tonic',pop_size=nn.v0c_pop_size,input_type='descending')
    V1a_1.create_interneuron_population(pop_type='V1a_1',self_connection='none',firing_behavior='tonic',pop_size=nn.v1a_pop_size,input_type='sensory_feedback') 
    V1a_2.create_interneuron_population(pop_type='V1a_2',self_connection='none',firing_behavior='tonic',pop_size=nn.v1a_pop_size,input_type='sensory_feedback') 
elif nn.remove_descending_drive==1:
    V0c_1.create_interneuron_population(pop_type='V0c_1',self_connection='none',firing_behavior='tonic',pop_size=nn.v0c_pop_size,input_type='none')
    V0c_2.create_interneuron_population(pop_type='V0c_2',self_connection='none',firing_behavior='tonic',pop_size=nn.v0c_pop_size,input_type='none')
    V1a_1.create_interneuron_population(pop_type='V1a_1',self_connection='none',firing_behavior='tonic',pop_size=nn.v1a_pop_size,input_type='none')
    V1a_2.create_interneuron_population(pop_type='V1a_2',self_connection='none',firing_behavior='tonic',pop_size=nn.v1a_pop_size,input_type='none')

if nn.slow_syn_bias == 'flx':
    print('Slow synaptic dynamics applied to Flexor side only.')
    rc_1.create_interneuron_population(pop_type='rc_slow_syn_enabled',self_connection='none',firing_behavior='bursting',pop_size=nn.rc_pop_size,input_type='none')
    rc_2.create_interneuron_population(pop_type='rc_slow_syn_disabled',self_connection='none',firing_behavior='bursting',pop_size=nn.rc_pop_size,input_type='none') 
    mnp1.create_mnp(pop_type='mnp_slow_syn_enabled') 
    mnp2.create_mnp(pop_type='mnp_slow_syn_disabled')
elif nn.slow_syn_bias == 'ext':
    print('Slow synaptic dynamics applied to Extensor side only.')
    rc_1.create_interneuron_population(pop_type='rc_slow_syn_disabled',self_connection='none',firing_behavior='bursting',pop_size=nn.rc_pop_size,input_type='none')
    rc_2.create_interneuron_population(pop_type='rc_slow_syn_enabled',self_connection='none',firing_behavior='bursting',pop_size=nn.rc_pop_size,input_type='none') 
    mnp1.create_mnp(pop_type='mnp_slow_syn_disabled') 
    mnp2.create_mnp(pop_type='mnp_slow_syn_enabled')
else:
    print('Slow synaptic dynamics applied to Flexor and Extensor.')
    rc_1.create_interneuron_population(pop_type='rc_slow_syn_enabled',self_connection='none',firing_behavior='bursting',pop_size=nn.rc_pop_size,input_type='none')
    rc_2.create_interneuron_population(pop_type='rc_slow_syn_enabled',self_connection='none',firing_behavior='bursting',pop_size=nn.rc_pop_size,input_type='none') 
    mnp1.create_mnp(pop_type='mnp_slow_syn_enabled') 
    mnp2.create_mnp(pop_type='mnp_slow_syn_enabled')     

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
conn.create_connections(rc_1.interneuron_pop,V1a_2.interneuron_pop,'custom_rc_v1a') 
conn.create_connections(rc_2.interneuron_pop,V1a_1.interneuron_pop,'custom_rc_v1a') 

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
    #conn.create_connections(rg1.rg_exc_tonic, inh1.inh_inter_tonic, 'custom_rg_v2b')
    conn.create_connections(rg2.rg_exc_bursting, inh2.inh_inter_tonic, 'custom_rg_v1')
    #conn.create_connections(rg2.rg_exc_tonic, inh2.inh_inter_tonic, 'custom_rg_v1')
        
    #Connect V1/V2b inhibitory populations to all rg neurons
    conn.create_connections(inh1.inh_inter_tonic,rg2.rg_exc_bursting,'custom_v2b_rg') 
    conn.create_connections(inh1.inh_inter_tonic,rg2.rg_exc_tonic,'custom_v2b_rg')
    conn.create_connections(inh1.inh_inter_tonic,rg2.rg_inh_bursting,'custom_v2b_rg')  
    conn.create_connections(inh1.inh_inter_tonic,rg2.rg_inh_tonic,'custom_v2b_rg')
    
    conn.create_connections(inh2.inh_inter_tonic,rg1.rg_exc_bursting,'custom_v1_rg') 
    conn.create_connections(inh2.inh_inter_tonic,rg1.rg_exc_tonic,'custom_v1_rg')
    conn.create_connections(inh2.inh_inter_tonic,rg1.rg_inh_bursting,'custom_v1_rg')
    conn.create_connections(inh2.inh_inter_tonic,rg1.rg_inh_tonic,'custom_v1_rg')
	
    #Connect V1/V2b inhibitory populations to V2a
    conn.create_connections(inh1.inh_inter_tonic,exc2.exc_inter_tonic,'custom_v2b_v2a') 
    conn.create_connections(inh2.inh_inter_tonic,exc1.exc_inter_tonic,'custom_v1_v2a') 
    
    if nn.v1v2b_mn_connected==1:
        #Connect V1/V2b inhibitory populations to motor neurons
        conn.create_connections(inh1.inh_inter_tonic,mnp2.motor_neuron_pop,'custom_v2b_mn') 
        conn.create_connections(inh2.inh_inter_tonic,mnp1.motor_neuron_pop,'custom_v1_mn') 
    
    #Connect excitatory rg neurons
    conn.create_connections(rg1.rg_exc_bursting,rg2.rg_exc_bursting,'custom_rg_rg')
    conn.create_connections(rg1.rg_exc_bursting,rg2.rg_exc_tonic,'custom_rg_rg')
    conn.create_connections(rg1.rg_exc_tonic,rg2.rg_exc_bursting,'custom_rg_rg')
    conn.create_connections(rg1.rg_exc_tonic,rg2.rg_exc_tonic,'custom_rg_rg')

    conn.create_connections(rg2.rg_exc_bursting,rg1.rg_exc_bursting,'custom_rg_rg')
    conn.create_connections(rg2.rg_exc_bursting,rg1.rg_exc_tonic,'custom_rg_rg')
    conn.create_connections(rg2.rg_exc_tonic,rg1.rg_exc_bursting,'custom_rg_rg')
    conn.create_connections(rg2.rg_exc_tonic,rg1.rg_exc_tonic,'custom_rg_rg')	
    
#conn.calculate_synapse_percentage()    
    
print("Seed#: ",nn.rng_seed)
print("RG Flx: # exc (bursting, tonic): ",nn.flx_exc_bursting_count,nn.flx_exc_tonic_count,"; # inh(bursting, tonic): ",nn.flx_inh_bursting_count,nn.flx_inh_tonic_count)
print("RG Ext: # exc (bursting, tonic): ",nn.ext_exc_bursting_count,nn.ext_exc_tonic_count,"; # inh(bursting, tonic): ",nn.ext_inh_bursting_count,nn.ext_inh_tonic_count)
print("V2b/V1: # inh (tonic): ",nn.num_inh_inter_tonic_v2b,nn.num_inh_inter_tonic_v1)
print("V2a: # exc (tonic): ",nn.v2a_tonic_pop_size,"; # MNs: ",nn.num_motor_neurons)

init_time=50
nest.Simulate(init_time)
num_steps = int(nn.sim_time/nn.time_resolution)
t_start = time.perf_counter()
for i in range(int(num_steps/10)-init_time):	
    nest.Simulate(nn.time_resolution*10)
    num_spikes_flx = popfunc.read_recent_spike_data(mnp1.spike_detector_motor)
    num_spikes_ext = popfunc.read_recent_spike_data(mnp2.spike_detector_motor)
    fb.send_muscle_activation(num_spikes_flx,num_spikes_ext)
    flx_1a_feedback, ext_1a_feedback, flx_1b_feedback, ext_1b_feedback, flx_11_feedback, ext_11_feedback = fb.receive_muscle_afferents()
    if nn.fb_rg_flx == 1:
        nest.SetStatus(rg1.rg_flx_pg, {"rate": flx_1a_feedback})
    if nn.fb_rg_ext == 1:
        nest.SetStatus(rg2.rg_ext_pg, {"rate": ext_1b_feedback})
    if nn.fb_v2b == 1:
        nest.SetStatus(inh1.v2b_pg, {"rate": flx_1a_feedback})
    if nn.fb_v1 == 1:
        nest.SetStatus(inh2.v1_pg, {"rate": ext_1b_feedback})    
    if nn.fb_1a_flx == 1:
        nest.SetStatus(V1a_1.v1a_1_pg, {"rate": ext_1b_feedback}) #1a are inhibitory, receive "opposite" excitation from feedback
    if nn.fb_1a_ext == 1:
        nest.SetStatus(V1a_2.v1a_2_pg, {"rate": flx_1a_feedback})
    #print("t = " + str(nest.biological_time),end="\r\u001b[1A")
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

if nn.fb_rg_flx == 1:
    #Read spike data - poisson generators
    senders_rg_flx_pg,spiketimes_rg_flx_pg = popfunc.read_spike_data(rg1.spike_detector_rg_flx_pg)

#Read spike data - V1/V2b inhibitory populations
if nn.rgs_connected==1:
    senders_inh_inter_tonic1,spiketimes_inh_inter_tonic1 = popfunc.read_spike_data(inh1.spike_detector_inh_inter_tonic)
    senders_inh_inter_tonic2,spiketimes_inh_inter_tonic2 = popfunc.read_spike_data(inh2.spike_detector_inh_inter_tonic)

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
        spike_bins_inh_inter1_true = spike_bins_inh_inter1
        spike_bins_inh_inter1 = (spike_bins_inh_inter1-np.min(spike_bins_inh_inter1))/(np.max(spike_bins_inh_inter1)-np.min(spike_bins_inh_inter1))
        spike_bins_inh_inter_tonic2 = popfunc.rate_code_spikes(nn.num_inh_inter_tonic_v1,spiketimes_inh_inter_tonic2)
        spike_bins_inh_inter2 = spike_bins_inh_inter_tonic2
        spike_bins_inh_inter2_true = spike_bins_inh_inter2
        spike_bins_inh_inter2 = (spike_bins_inh_inter2-np.min(spike_bins_inh_inter2))/(np.max(spike_bins_inh_inter2)-np.min(spike_bins_inh_inter2))
        
    t_stop = time.perf_counter()
    print('Rate coded activity complete, taking ',int(t_stop-t_start),' seconds.')

    #Plot rate-coded output
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
    if nn.args['save_results']: plt.savefig(nn.pathFigures + '/' + 'rate_coded_output_interneurons.png',bbox_inches="tight")
	
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
    if nn.args['save_results']: plt.savefig(nn.pathFigures + '/' + 'rate_coded_output.png',bbox_inches="tight")
        
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
    if nn.args['save_results']: plt.savefig(nn.pathFigures + '/' + 'rate_coded_output_rg_mnp.png',bbox_inches="tight")
        
    if max(spike_bins_mnp1)>0 and max(spike_bins_mnp2)>0: 
        avg_freq, avg_phase, bd_comparison = calc.analyze_output(spike_bins_mnp1,spike_bins_mnp2,spike_bins_mnp1_true,spike_bins_mnp2_true,'MNP',y_line_bd=0.4,y_line_phase=0.7)    

if nn.isf_output==1:
    t_start = time.perf_counter()
    #Calculate instantaneous spiking frequency
    rgexc1_bursting_freq, rgexc1_bursting_times = popfunc.calculate_interspike_frequency(nn.flx_exc_bursting_count,spiketimes_exc1)
    rgexc2_bursting_freq, rgexc2_bursting_times = popfunc.calculate_interspike_frequency(nn.ext_exc_bursting_count,spiketimes_exc2)
    rgexc1_tonic_freq, rgexc1_tonic_times = popfunc.calculate_interspike_frequency(nn.flx_exc_tonic_count,spiketimes_exc_tonic1)
    rgexc2_tonic_freq, rgexc2_tonic_times = popfunc.calculate_interspike_frequency(nn.ext_exc_tonic_count,spiketimes_exc_tonic2)
    rginh1_bursting_freq, rginh1_bursting_times = popfunc.calculate_interspike_frequency(nn.flx_inh_bursting_count,spiketimes_inh1)
    rginh2_bursting_freq, rginh2_bursting_times = popfunc.calculate_interspike_frequency(nn.ext_inh_bursting_count,spiketimes_inh2)
    rginh1_tonic_freq, rginh1_tonic_times = popfunc.calculate_interspike_frequency(nn.flx_inh_tonic_count,spiketimes_inh_tonic1)
    rginh2_tonic_freq, rginh2_tonic_times = popfunc.calculate_interspike_frequency(nn.ext_inh_tonic_count,spiketimes_inh_tonic2)

    v2a1_freq, v2a1_times = popfunc.calculate_interspike_frequency(nn.v2a_tonic_pop_size,spiketimes_exc_inter_tonic1)
    v2a2_freq, v2a2_times =popfunc.calculate_interspike_frequency(nn.v2a_tonic_pop_size,spiketimes_exc_inter_tonic2)
    
    v0c1_freq, v0c1_times = popfunc.calculate_interspike_frequency(nn.v0c_pop_size,spiketimes_V0c_1)
    v0c2_freq, v0c2_times = popfunc.calculate_interspike_frequency(nn.v0c_pop_size,spiketimes_V0c_2)
    
    v1a1_freq, v1a1_times = popfunc.calculate_interspike_frequency(nn.v1a_pop_size,spiketimes_V1a_1)
    v1a2_freq, v1a2_times = popfunc.calculate_interspike_frequency(nn.v1a_pop_size,spiketimes_V1a_2)
    
    rc1_freq, rc1_times = popfunc.calculate_interspike_frequency(nn.rc_pop_size,spiketimes_rc_1)
    rc2_freq, rc2_times = popfunc.calculate_interspike_frequency(nn.rc_pop_size,spiketimes_rc_2)
    
    if nn.rgs_connected:
        v2b_freq, v2b_times =popfunc.calculate_interspike_frequency(nn.num_inh_inter_tonic_v2b,spiketimes_inh_inter_tonic1)
        v1_freq, v1_times =popfunc.calculate_interspike_frequency(nn.num_inh_inter_tonic_v1,spiketimes_inh_inter_tonic2)
    
    mnp1_freq, mnp1_times = popfunc.calculate_interspike_frequency(nn.num_motor_neurons,spiketimes_mnp1)
    mnp2_freq, mnp2_times = popfunc.calculate_interspike_frequency(nn.num_motor_neurons,spiketimes_mnp2)
    
    t_stop = time.perf_counter()    
    print('Calculating ISF complete, taking ',int(t_stop-t_start),' seconds.')
    
    t_start = time.perf_counter()
    #Convolve spike data - RG populations
    rg_exc_convolved1, _ = popfunc.convolve_spiking_activity(nn.flx_exc_bursting_count,spiketimes_exc1)
    rg_exc_tonic_convolved1, _ = popfunc.convolve_spiking_activity(nn.flx_exc_tonic_count,spiketimes_exc_tonic1)
    rg_inh_convolved1, _ = popfunc.convolve_spiking_activity(nn.flx_inh_bursting_count,spiketimes_inh1)
    rg_inh_tonic_convolved1, _ = popfunc.convolve_spiking_activity(nn.flx_inh_tonic_count,spiketimes_inh_tonic1)
    rg1_convolved = np.vstack([rg_exc_convolved1,rg_inh_convolved1])
    rg1_convolved = np.vstack([rg1_convolved,rg_exc_tonic_convolved1])
    rg1_convolved = np.vstack([rg1_convolved,rg_inh_tonic_convolved1])
    rg1_convolved = rg1_convolved.mean(axis=0)

    rg_exc_convolved2, _  = popfunc.convolve_spiking_activity(nn.ext_exc_bursting_count,spiketimes_exc2)
    rg_exc_tonic_convolved2, _  = popfunc.convolve_spiking_activity(nn.ext_exc_tonic_count,spiketimes_exc_tonic2)
    rg_inh_convolved2, _  = popfunc.convolve_spiking_activity(nn.ext_inh_bursting_count,spiketimes_inh2)
    rg_inh_tonic_convolved2, _  = popfunc.convolve_spiking_activity(nn.ext_inh_tonic_count,spiketimes_inh_tonic2)
    rg2_convolved = np.vstack([rg_exc_convolved2,rg_inh_convolved2])
    rg2_convolved = np.vstack([rg2_convolved,rg_exc_tonic_convolved2])
    rg2_convolved = np.vstack([rg2_convolved,rg_inh_tonic_convolved2])
    rg2_convolved = rg2_convolved.mean(axis=0)

    #Convolve spike data - V2a excitatory interneuron populations
    v2a1_convolved, _  = popfunc.convolve_spiking_activity(nn.v2a_tonic_pop_size,spiketimes_exc_inter_tonic1)
    v2a2_convolved, _  = popfunc.convolve_spiking_activity(nn.v2a_tonic_pop_size,spiketimes_exc_inter_tonic2)
    
    #Convolve spike data - V2a excitatory interneuron populations
    v0c1_convolved, _  = popfunc.convolve_spiking_activity(nn.v0c_pop_size,spiketimes_V0c_1)
    v0c2_convolved, _  = popfunc.convolve_spiking_activity(nn.v0c_pop_size,spiketimes_V0c_2)
    
    v1a1_convolved, _ = popfunc.convolve_spiking_activity(nn.v1a_pop_size,spiketimes_V1a_1)
    v1a2_convolved, _ = popfunc.convolve_spiking_activity(nn.v1a_pop_size,spiketimes_V1a_2)
    
    rc1_convolved, _ = popfunc.convolve_spiking_activity(nn.rc_pop_size,spiketimes_rc_1)
    rc2_convolved, _ = popfunc.convolve_spiking_activity(nn.rc_pop_size,spiketimes_rc_2)

    #Convolve spike data - MNPs
    mnp1_convolved, convolved_time = popfunc.convolve_spiking_activity(nn.num_motor_neurons,spiketimes_mnp1)
    mnp2_convolved, _ = popfunc.convolve_spiking_activity(nn.num_motor_neurons,spiketimes_mnp2)
    
    # Convolve spike data - inh populations
    if nn.rgs_connected == 1:
        v2b_convolved, _  = popfunc.convolve_spiking_activity(nn.num_inh_inter_tonic_v2b, spiketimes_inh_inter_tonic1)
        v1_convolved, _  = popfunc.convolve_spiking_activity(nn.num_inh_inter_tonic_v1, spiketimes_inh_inter_tonic2)

    t_stop = time.perf_counter()    
    print('Convolved spiking activity complete, taking ',int(t_stop-t_start),' seconds.')

    rg1_isf_max = np.nanmax(np.array([np.nanmean(neuron_freq) for neuron_freq in rgexc1_bursting_freq]))
    rg2_isf_max = np.nanmax(np.array([np.nanmean(neuron_freq) for neuron_freq in rgexc2_bursting_freq]))
    rg1_conv_max = np.nanmax(rg1_convolved)
    rg2_conv_max = np.nanmax(rg2_convolved)
    rg1_scale = rg1_isf_max / rg1_conv_max
    rg2_scale = rg2_isf_max / rg2_conv_max
    
    print('Max firing rate of a Flx RG (ISF):',round(rg1_isf_max,2),'Ext RG:',round(rg2_isf_max,2))
    print('Max firing rate of a Flx RG (Convolved):',round(rg1_conv_max,2),'Ext RG:',round(rg2_conv_max,2))
    print('Convolved max is',round(rg1_scale,3),round(rg2_scale,3), 'times the size of ISF max (Flx, Ext).')
    
    v2b_isf_max = np.nanmax(np.array([np.nanmean(neuron_freq) for neuron_freq in v2b_freq]))
    v1_isf_max = np.nanmax(np.array([np.nanmean(neuron_freq) for neuron_freq in v1_freq]))
    v2b_conv_max = np.nanmax(v2b_convolved)
    v1_conv_max = np.nanmax(v1_convolved)
    v2b_scale = v2b_isf_max / v2b_conv_max
    v1_scale = v1_isf_max / v1_conv_max
    
    v2a1_isf_max = np.nanmax(np.array([np.nanmean(neuron_freq) for neuron_freq in v2a1_freq]))
    v2a2_isf_max = np.nanmax(np.array([np.nanmean(neuron_freq) for neuron_freq in v2a2_freq]))
    v2a1_conv_max = np.nanmax(v2a1_convolved)
    v2a2_conv_max = np.nanmax(v2a2_convolved)
    v2a1_scale = v2a1_isf_max / v2a1_conv_max
    v2a2_scale = v2a2_isf_max / v2a2_conv_max
    
    v0c1_isf_max = np.nanmax(np.array([np.nanmean(neuron_freq) for neuron_freq in v0c1_freq]))
    v0c2_isf_max = np.nanmax(np.array([np.nanmean(neuron_freq) for neuron_freq in v0c2_freq]))
    v0c1_conv_max = np.nanmax(v0c1_convolved)
    v0c2_conv_max = np.nanmax(v0c2_convolved)
    v0c1_scale = v0c1_isf_max / v0c1_conv_max
    v0c2_scale = v0c2_isf_max / v0c2_conv_max
    
    v1a1_isf_max = np.nanmax(np.array([np.nanmean(neuron_freq) for neuron_freq in v1a1_freq]))
    v1a2_isf_max = np.nanmax(np.array([np.nanmean(neuron_freq) for neuron_freq in v1a2_freq]))
    v1a1_conv_max = np.nanmax(v1a1_convolved)
    v1a2_conv_max = np.nanmax(v1a2_convolved)
    v1a1_scale = v1a1_isf_max / v1a1_conv_max
    v1a2_scale = v1a2_isf_max / v1a2_conv_max
    
    rc1_isf_max = np.nanmax(np.array([np.nanmean(neuron_freq) for neuron_freq in rc1_freq]))
    rc2_isf_max = np.nanmax(np.array([np.nanmean(neuron_freq) for neuron_freq in rc2_freq]))
    rc1_conv_max = np.nanmax(rc1_convolved)
    rc2_conv_max = np.nanmax(rc2_convolved)
    rc1_scale = rc1_isf_max / rc1_conv_max
    rc2_scale = rc2_isf_max / rc2_conv_max
    
    mnp1_isf_max = np.nanmax(np.array([np.nanmean(neuron_freq) for neuron_freq in mnp1_freq]))
    mnp2_isf_max = np.nanmax(np.array([np.nanmean(neuron_freq) for neuron_freq in mnp2_freq]))
    mnp1_conv_max = np.nanmax(mnp1_convolved)
    mnp2_conv_max = np.nanmax(mnp2_convolved)
    mnp1_scale = mnp1_isf_max / mnp1_conv_max
    mnp2_scale = mnp2_isf_max / mnp2_conv_max
    
    print('Max firing rate of a Flx MN (ISF):',round(mnp1_isf_max,2),'Ext MN:',round(mnp2_isf_max,2))
    print('Max firing rate of a Flx MN (Convolved):',round(mnp1_conv_max,2),'Ext MN:',round(mnp2_conv_max,2))
    print('Convolved max is',round(mnp1_scale,3),round(mnp2_scale,3), 'times the size of ISF max (Flx, Ext).')
    
    mnp1_convolved_scaled = mnp1_convolved * mnp1_scale
    mnp2_convolved_scaled = mnp2_convolved * mnp2_scale
    mnp1_convolved_scaled_mean = np.nanmean(mnp1_convolved_scaled)
    mnp2_convolved_scaled_mean = np.nanmean(mnp2_convolved_scaled)
    mnp1_convolved_max_scaled = np.nanmax(mnp1_convolved * mnp1_scale)
    mnp2_convolved_max_scaled = np.nanmax(mnp2_convolved * mnp2_scale)
    
    print('After scaling max firing rate of a Flx MN (Convolved):',round(mnp1_convolved_max_scaled,2),'Ext MN:',round(mnp2_convolved_max_scaled,2))
    print('After scaling mean firing rate of a Flx MN (Convolved):',round(mnp1_convolved_scaled_mean,2),'Ext MN:',round(mnp2_convolved_scaled_mean,2))
    
    mnp1_avg_norm = (mnp1_convolved-np.min(mnp1_convolved))/(np.max(mnp1_convolved)-np.min(mnp1_convolved))
    mnp2_avg_norm = (mnp2_convolved-np.min(mnp2_convolved))/(np.max(mnp2_convolved)-np.min(mnp2_convolved))
    if max(mnp1_avg_norm)>0 and max(mnp2_avg_norm)>0: 
        avg_freq, avg_phase, bd_comparison = calc.analyze_output(mnp1_avg_norm,mnp2_avg_norm,mnp1_convolved_scaled,mnp2_convolved_scaled,'MNP',y_line_bd=0.4,y_line_phase=0.7)
    
    t = convolved_time
    xticks = np.arange(start=np.ceil(t[0] / 1000) * 1000, stop=t[-1], step=1000)
    fig, ax = plt.subplots(4,sharex='all',figsize=(18, 12))    
    ax[0].plot(t, rg1_convolved* rg1_scale)
    ax[0].plot(t, rg2_convolved* rg2_scale)
    ax[1].plot(t, v2b_convolved* v2b_scale)
    ax[1].plot(t, v1_convolved* v1_scale)
    ax[2].plot(t, v2a1_convolved* v2a1_scale)
    ax[2].plot(t, v2a2_convolved* v2a2_scale)
    ax[3].plot(t, mnp1_convolved* mnp1_scale)
    ax[3].plot(t, mnp2_convolved* mnp2_scale)
    ax[0].set_xticks([])
    ax[0].legend(['RG_F', 'RG_E'],loc='upper right',fontsize='x-small') 
    ax[1].legend(['V2b', 'V1'],loc='upper right',fontsize='x-small') 
    ax[2].legend(['V2a_F', 'V2a_E'],loc='upper right',fontsize='x-small') 
    ax[3].legend(['FLX', 'EXT'],loc='upper right',fontsize='x-small')
    ax[3].set_xlabel('Time (ms)')
    ax[3].set_xticks(xticks)
    ax[3].set_xticklabels([f'{int(x)}' for x in xticks])
    ax[0].set_ylabel('Freq (Hz)')
    ax[1].set_ylabel('Freq (Hz)')
    ax[2].set_ylabel('Freq (Hz)')
    ax[3].set_ylabel('Freq (Hz)')
    ax[0].set_title('Average Spike Rate')
    if nn.args['save_results']: plt.savefig(nn.pathFigures + '/' + 'spike_rate_rg_mnp.png',bbox_inches="tight")
    
    fig, ax = plt.subplots(4,sharex='all',figsize=(18, 12))    
    ax[0].plot(t, v0c1_convolved* v0c1_scale)
    ax[0].plot(t, v0c2_convolved* v0c2_scale)
    ax[1].plot(t, v1a1_convolved* v1a1_scale)
    ax[1].plot(t, v1a2_convolved* v1a2_scale)
    ax[2].plot(t, rc1_convolved* rc1_scale)
    ax[2].plot(t, rc2_convolved* rc2_scale)
    ax[3].plot(t, mnp1_convolved* mnp1_scale)
    ax[3].plot(t, mnp2_convolved* mnp2_scale)
    ax[0].set_xticks([])
    ax[0].legend(['V0c_F', 'V0c_E'],loc='upper right',fontsize='x-small') 
    ax[1].legend(['1a_F', '1a_E'],loc='upper right',fontsize='x-small') 
    ax[2].legend(['RC_F', 'RC_E'],loc='upper right',fontsize='x-small') 
    ax[3].legend(['FLX', 'EXT'],loc='upper right',fontsize='x-small')
    ax[3].set_xlabel('Time (ms)')
    ax[3].set_xticks(xticks)
    ax[3].set_xticklabels([f'{int(x)}' for x in xticks])
    ax[0].set_ylabel('Freq (Hz)')
    ax[1].set_ylabel('Freq (Hz)')
    ax[2].set_ylabel('Freq (Hz)')
    ax[3].set_ylabel('Freq (Hz)')
    ax[0].set_title('Average Spike Rate')
    if nn.args['save_results']: plt.savefig(nn.pathFigures + '/' + 'spike_rate_interneurons.png',bbox_inches="tight")
        
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
    if nn.args['save_results']: plt.savefig(nn.pathFigures + '/' + 'spike_distribution.png',bbox_inches="tight")
    '''
if nn.membrane_potential_plot==1:
    v_m1,t_m1 = popfunc.read_membrane_potential(mnp1.mm_motor,nn.num_motor_neurons,mnp1.neuron_to_sample)
    v_m2,t_m2 = popfunc.read_membrane_potential(mnp2.mm_motor,nn.num_motor_neurons,mnp2.neuron_to_sample)

    pylab.figure(figsize=(18, 12))
    pylab.subplot(211)
    pylab.plot(t_m1,v_m1)
    pylab.xlim(1000,1500)
    pylab.title('Individual Neuron Membrane Potential (Flx)')
    pylab.ylabel('Membrane potential (mV)')
    pylab.subplot(212)
    pylab.plot(t_m2,v_m2)
    pylab.xlim(1000,1500)
    pylab.title('Individual Neuron Membrane Potential (Ext)')
    pylab.xlabel('Time (ms)')
    pylab.ylabel('Membrane potential (mV)')
    if nn.args['save_results']: plt.savefig(nn.pathFigures + '/' + 'membrane_potential_mns.png',bbox_inches="tight")

#Plot individual spikes
if nn.raster_plot==1:
    pylab.figure(figsize=(18, 12))
    pylab.subplot(211)
    for i in range(nn.num_motor_neurons-1): 
        pylab.plot(spiketimes_mnp1[0][i],senders_mnp1[0][i],'.',label='Flx')
    pylab.xlim(1000,1500)
    pylab.title('Spike Output (Flx)')
    pylab.ylabel('Neuron ID')
    pylab.subplot(212)
    for i in range(nn.num_motor_neurons-1):
        pylab.plot(spiketimes_mnp2[0][i],senders_mnp2[0][i],'.',label='Ext')  
    pylab.xlim(1000,1500)
    pylab.title('Spike Output (Ext)')
    pylab.xlabel('Time (ms)')
    pylab.ylabel('Neuron ID')
    if nn.args['save_results']: plt.savefig(nn.pathFigures + '/' + 'raster_plot_mns.png',bbox_inches="tight")

if nn.fb_rg_flx == 1:
    #Plot poisson generator spikes
    fig,ax = plt.subplots(figsize=(18, 12))
    for i in range(nn.num_pgs-1):
        if nn.num_pgs != 0: ax.plot(spiketimes_rg_flx_pg[0][i],senders_rg_flx_pg[0][i],'.')
    #ax.set_ylim(2,12)
    #ax.set_xlim(500,4000)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Neuron #')
    plt.title('Poisson Spiking Input')    
    if nn.args['save_results']: plt.savefig(nn.pathFigures + '/' + 'poisson_spike_input.png',bbox_inches="tight")        

if nn.args['save_results']:        
    np.savetxt(nn.pathFigures + '/output_mnp1.csv',mnp1_convolved_scaled,delimiter=',')
    np.savetxt(nn.pathFigures + '/output_mnp2.csv',mnp2_convolved_scaled,delimiter=',')    

if nn.args['save_results'] and nn.save_all_pops==1:
    np.savetxt(nn.pathFigures + '/output_rg1.csv',rg1_convolved_scaled,delimiter=',')
    np.savetxt(nn.pathFigures + '/output_rg2.csv',rg2_convolved_scaled,delimiter=',')
    np.savetxt(nn.pathFigures + '/output_v2b.csv',v2b_convolved_scaled,delimiter=',')
    np.savetxt(nn.pathFigures + '/output_v1.csv',v1_convolved_scaled,delimiter=',')
    np.savetxt(nn.pathFigures + '/output_v2a1.csv',v2a1_convolved_scaled,delimiter=',')
    np.savetxt(nn.pathFigures + '/output_v2a2.csv',v2a2_convolved_scaled,delimiter=',')
    np.savetxt(nn.pathFigures + '/output_v0c1.csv',v0c1_convolved_scaled,delimiter=',')
    np.savetxt(nn.pathFigures + '/output_v0c2.csv',v0c2_convolved_scaled,delimiter=',')
    np.savetxt(nn.pathFigures + '/output_1a1.csv',v1a1_convolved_scaled,delimiter=',')
    np.savetxt(nn.pathFigures + '/output_1a2.csv',v1a2_convolved_scaled,delimiter=',')
    np.savetxt(nn.pathFigures + '/output_rc1.csv',rc1_convolved_scaled,delimiter=',')
    np.savetxt(nn.pathFigures + '/output_rc2.csv',rc2_convolved_scaleddelimiter=',')  
    
if nn.args['save_results'] and nn.rate_coded_plot == 1 and nn.isf_output == 0:
    # Save population rate output
    np.savetxt(nn.pathFigures + '/output_mnp1.csv',spike_bins_mnp1_true,delimiter=',')
    np.savetxt(nn.pathFigures + '/output_mnp2.csv',spike_bins_mnp2_true,delimiter=',')
    
if nn.args['save_results'] and nn.save_all_pops==1 and nn.rate_coded_plot == 1 and nn.isf_output == 0:
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
