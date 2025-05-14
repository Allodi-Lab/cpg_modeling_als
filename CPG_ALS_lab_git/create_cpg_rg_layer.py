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

warnings.filterwarnings('ignore')

#Create neuron populations - NEST
rg1 = flx_rg.create_rg_population()
rg2 = ext_rg.create_rg_population()

if nn.rgs_connected == 1:
    inh1 = inh.create_inh_inter_population('V2b')  # V2b
    inh2 = inh.create_inh_inter_population('V1')  # V1

    # Connect excitatory rg neurons to V1/V2b inhibitory populations
    conn.create_connections(rg1.rg_exc_bursting, inh1.inh_inter_tonic, 'custom_rg_v2b')
    #conn.create_connections(rg1.rg_exc_tonic, inh1.inh_inter_tonic, 'custom_rg_v2b')
    conn.create_connections(rg2.rg_exc_bursting, inh2.inh_inter_tonic, 'custom_rg_v1')
    #conn.create_connections(rg2.rg_exc_tonic, inh2.inh_inter_tonic, 'custom_rg_v1')

    #Connect V1/V2b inhibitory populations to all rg neurons
    conn.create_connections(inh1.inh_inter_tonic,rg2.rg_exc_bursting,'custom_v2b_rg') #Healthy = inh_strong, degenerated = custom_v2b_rg
    conn.create_connections(inh1.inh_inter_tonic,rg2.rg_exc_tonic,'custom_v2b_rg')
    conn.create_connections(inh1.inh_inter_tonic,rg2.rg_inh_bursting,'custom_v2b_rg')
    conn.create_connections(inh1.inh_inter_tonic,rg2.rg_inh_tonic,'custom_v2b_rg')
    conn.create_connections(inh2.inh_inter_tonic,rg1.rg_exc_bursting,'custom_v1_rg') #Healthy = inh_strong, degenerated = custom_v1_rg
    conn.create_connections(inh2.inh_inter_tonic,rg1.rg_exc_tonic,'custom_v1_rg')
    conn.create_connections(inh2.inh_inter_tonic,rg1.rg_inh_bursting,'custom_v1_rg')
    conn.create_connections(inh2.inh_inter_tonic,rg1.rg_inh_tonic,'custom_v1_rg')
    
    #Connect excitatory rg neurons
    conn.create_connections(rg1.rg_exc_bursting,rg2.rg_exc_bursting,'custom_rg_rg')
    conn.create_connections(rg1.rg_exc_bursting,rg2.rg_exc_tonic,'custom_rg_rg')
    conn.create_connections(rg1.rg_exc_tonic,rg2.rg_exc_bursting,'custom_rg_rg')
    conn.create_connections(rg1.rg_exc_tonic,rg2.rg_exc_tonic,'custom_rg_rg')

    conn.create_connections(rg2.rg_exc_bursting,rg1.rg_exc_bursting,'custom_rg_rg')
    conn.create_connections(rg2.rg_exc_bursting,rg1.rg_exc_tonic,'custom_rg_rg')
    conn.create_connections(rg2.rg_exc_tonic,rg1.rg_exc_bursting,'custom_rg_rg')
    conn.create_connections(rg2.rg_exc_tonic,rg1.rg_exc_tonic,'custom_rg_rg')	
    
print("Seed#: ",nn.rng_seed)
print("RG Flx: # exc (bursting, tonic): ",nn.flx_exc_bursting_count,nn.flx_exc_tonic_count,"; # inh(bursting, tonic): ",nn.flx_inh_bursting_count,nn.flx_inh_tonic_count)
print("RG Ext: # exc (bursting, tonic): ",nn.ext_exc_bursting_count,nn.ext_exc_tonic_count,"; # inh(bursting, tonic): ",nn.ext_inh_bursting_count,nn.ext_inh_tonic_count)
print("V2b/V1: # inh (bursting): ",nn.num_inh_inter_bursting_v2b,nn.num_inh_inter_bursting_v1,"; (tonic): ",nn.num_inh_inter_tonic_v2b,nn.num_inh_inter_tonic_v1)

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

#Read spike data - V1/V2b inhibitory populations
if nn.rgs_connected==1:
    senders_inh_inter_tonic1,spiketimes_inh_inter_tonic1 = popfunc.read_spike_data(inh1.spike_detector_inh_inter_tonic)
    senders_inh_inter_tonic2,spiketimes_inh_inter_tonic2 = popfunc.read_spike_data(inh2.spike_detector_inh_inter_tonic)
    if nn.num_inh_inter_bursting_v2b>0:
        senders_inh_inter_bursting1,spiketimes_inh_inter_bursting1 = popfunc.read_spike_data(inh1.spike_detector_inh_inter_bursting)
    if nn.num_inh_inter_bursting_v1>0:
        senders_inh_inter_bursting2,spiketimes_inh_inter_bursting2 = popfunc.read_spike_data(inh2.spike_detector_inh_inter_bursting)

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
        spikes_convolved_complete_network = np.vstack([spikes_convolved_rgs, spikes_convolved_inh])

    if nn.remove_silent:
        print('Removing silent neurons')
        spikes_convolved_all1 = spikes_convolved_all1[~np.all(spikes_convolved_all1 == 0, axis=1)]
        spikes_convolved_all2 = spikes_convolved_all2[~np.all(spikes_convolved_all2 == 0, axis=1)]
        spikes_convolved_rgs = spikes_convolved_rgs[~np.all(spikes_convolved_rgs == 0, axis=1)]
        
    t_stop = time.perf_counter()
    spikes_convolved_all1 = popfunc.normalize_rows(spikes_convolved_all1)
    spikes_convolved_all2 = popfunc.normalize_rows(spikes_convolved_all2)
    spikes_convolved_rgs = popfunc.normalize_rows(spikes_convolved_rgs)   
    print('Convolved spiking activity complete, taking ',int(t_stop-t_start),' seconds.') 

#Calculate balance

if nn.calculate_balance == 1:
    rg1_exc_burst_weight = conn.calculate_weighted_balance(rg1.rg_exc_bursting,rg1.spike_detector_rg_exc_bursting)
    rg1_inh_burst_weight = conn.calculate_weighted_balance(rg1.rg_inh_bursting,rg1.spike_detector_rg_inh_bursting)
    rg1_exc_tonic_weight = conn.calculate_weighted_balance(rg1.rg_exc_tonic,rg1.spike_detector_rg_exc_tonic)
    rg1_inh_tonic_weight = conn.calculate_weighted_balance(rg1.rg_inh_tonic,rg1.spike_detector_rg_inh_tonic)
    weights_per_pop2 = [rg1_exc_burst_weight,rg1_inh_burst_weight,rg1_exc_tonic_weight,rg1_inh_tonic_weight]
    absolute_weights_per_pop2 = [rg1_exc_burst_weight,abs(rg1_inh_burst_weight),rg1_exc_tonic_weight,abs(rg1_inh_tonic_weight)]
    rg1_balance_pct = (sum(weights_per_pop2)/sum(absolute_weights_per_pop2))*100
    print('Flx balance %: ',round(rg1_balance_pct,2),' >0 skew excitatory; <0 skew inhibitory')
    rg2_exc_burst_weight = conn.calculate_weighted_balance(rg2.rg_exc_bursting,rg2.spike_detector_rg_exc_bursting)
    rg2_inh_burst_weight = conn.calculate_weighted_balance(rg2.rg_inh_bursting,rg2.spike_detector_rg_inh_bursting)
    rg2_exc_tonic_weight = conn.calculate_weighted_balance(rg2.rg_exc_tonic,rg2.spike_detector_rg_exc_tonic)
    rg2_inh_tonic_weight = conn.calculate_weighted_balance(rg2.rg_inh_tonic,rg2.spike_detector_rg_inh_tonic)
    weights_per_pop2 = [rg2_exc_burst_weight,rg2_inh_burst_weight,rg2_exc_tonic_weight,rg2_inh_tonic_weight]
    absolute_weights_per_pop2 = [rg2_exc_burst_weight,abs(rg2_inh_burst_weight),rg2_exc_tonic_weight,abs(rg2_inh_tonic_weight)]
    rg2_balance_pct = (sum(weights_per_pop2)/sum(absolute_weights_per_pop2))*100
    print('Ext balance %: ',round(rg2_balance_pct,2),' >0 skew excitatory; <0 skew inhibitory')

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
    t = np.arange(0,len(spike_bins_rg1_true),1)
    if nn.rgs_connected == 1:
        fig, ax = plt.subplots(2,sharex='all')
        ax[0].plot(t, spike_bins_rg1_true)
        ax[0].plot(t, spike_bins_rg2_true)
        ax[1].plot(t, spike_bins_inh_inter1_true)
        ax[1].plot(t, spike_bins_inh_inter2_true)		
        for i in range(1):
            ax[i].set_xticks([])
            ax[i].set_xlim(0,len(spike_bins_rg1_true))
        ax[1].set_xlabel('Time (ms)')
        ax[1].set_xticks([0,10000,20000,30000,40000,50000,60000,70000,80000,90000])
        ax[1].set_xticklabels([0,1000,2000,3000,4000,5000,6000,7000,8000,9000])
        ax[1].set_xlim(0,len(spike_bins_rg1_true))
        ax[0].legend(['RG_Flx', 'RG_Ext'],loc='upper right',fontsize='x-small') 
        ax[1].legend(['V2b', 'V1'],loc='upper right',fontsize='x-small') 
        ax[0].set_title("Population output (RG)")
        ax[1].set_title("Population output (V1/V2b)")
        figure = plt.gcf() # get current figure
        figure.set_size_inches(8, 6)
        plt.tight_layout()
        if nn.args['save_results']: plt.savefig(nn.pathFigures + '/' + 'rate_coded_output.pdf',bbox_inches="tight")
    else:
        fig, ax = plt.subplots(2,sharex='all')
        ax[0].plot(t, spike_bins_rg1_true)
        ax[1].plot(t, spike_bins_rg2_true)	
        for i in range(1):
            ax[i].set_xticks([])
            ax[i].set_xlim(0,len(spike_bins_rg1_true))
        ax[1].set_xlabel('Time (ms)')
        ax[1].set_xticks([0,10000,20000,30000,40000,50000,60000,70000,80000,90000])
        ax[1].set_xticklabels([0,1000,2000,3000,4000,5000,6000,7000,8000,9000])
        ax[1].set_xlim(0,len(spike_bins_rg1_true))
        ax[0].legend(['RG_F'],loc='upper right',fontsize='x-small') 
        ax[1].legend(['RG_E'],loc='upper right',fontsize='x-small') 
        ax[0].set_title("Population output (RG Flx)")
        ax[1].set_title("Population output (RG Ext)")
        figure = plt.gcf() # get current figure
        figure.set_size_inches(8, 6)
        plt.tight_layout()
        if nn.args['save_results']: plt.savefig(nn.pathFigures + '/' + 'rate_coded_output.pdf',bbox_inches="tight")
        
        pylab.figure()
        pylab.plot(t, spike_bins_rg1_true)
        pylab.plot(t, spike_bins_rg2_true)
        pylab.xlabel('Time (ms)')
        pylab.xticks([0,10000,20000,30000,40000,50000,60000,70000,80000,90000],[0,1000,2000,3000,4000,5000,6000,7000,8000,9000])
        pylab.ylabel('# of spikes')
        pylab.title('Population output Isolated RG Flx vs RG Ext')
        pylab.legend(['RG_Flx','RG_Ext'],loc='upper right',fontsize='x-small')
        if nn.args['save_results']: plt.savefig(nn.pathFigures + '/' + 'rate_coded_output_compared.pdf',bbox_inches="tight")

    if max(spike_bins_rg1)>0 and max(spike_bins_rg2)>0: 
        avg_freq, avg_phase, bd_diff = calc.analyze_output(spike_bins_rg1,spike_bins_rg2,'RG',y_line_bd=0.4,y_line_phase=0.7)
        
if nn.spike_distribution_plot==1:
    #Count spikes per neuron - MUST ADD INTERNEURONS TO THIS CALCULATION
    indiv_spikes_exc1,neuron_to_sample_rg1_burst,sparse_count1,silent_count1 = popfunc.count_indiv_spikes(nn.flx_exc_bursting_count,senders_exc1,avg_freq)
    indiv_spikes_inh1,neuron_to_sample_rg1_burst_inh,sparse_count2,silent_count2 = popfunc.count_indiv_spikes(nn.flx_inh_bursting_count,senders_inh1,avg_freq)
    indiv_spikes_exc_tonic1,neuron_to_sample_rg1_ton,sparse_count3,silent_count3 = popfunc.count_indiv_spikes(nn.flx_exc_tonic_count,senders_exc_tonic1,avg_freq)
    indiv_spikes_inh_tonic1,neuron_to_sample_rg1_ton_inh,sparse_count4,silent_count4 = popfunc.count_indiv_spikes(nn.flx_inh_tonic_count,senders_inh_tonic1,avg_freq)
    rg1_indiv_spike_count = indiv_spikes_exc1+indiv_spikes_inh1+indiv_spikes_exc_tonic1+indiv_spikes_inh_tonic1
    rg1_sparse_firing_count = sparse_count1+sparse_count2+sparse_count3+sparse_count4
    rg1_silent_neuron_count = silent_count1+silent_count2+silent_count3+silent_count4
    print('Flx sparse firing %',round(rg1_sparse_firing_count*100/(len(rg1_indiv_spike_count)-rg1_silent_neuron_count),2))
    
    indiv_spikes_exc2,neuron_to_sample_rg2_burst,sparse_count5,silent_count5 = popfunc.count_indiv_spikes(nn.ext_exc_bursting_count,senders_exc2,avg_freq)
    indiv_spikes_inh2,neuron_to_sample_rg2_burst_inh,sparse_count6,silent_count6 = popfunc.count_indiv_spikes(nn.ext_inh_bursting_count,senders_inh2,avg_freq)
    indiv_spikes_exc_tonic2,neuron_to_sample_rg2_ton,sparse_count7,silent_count7 = popfunc.count_indiv_spikes(nn.ext_exc_tonic_count,senders_exc_tonic2,avg_freq)
    indiv_spikes_inh_tonic2,neuron_to_sample_rg2_ton_inh,sparse_count8,silent_count8 = popfunc.count_indiv_spikes(nn.ext_inh_tonic_count,senders_inh_tonic2,avg_freq)
    rg2_indiv_spike_count = indiv_spikes_exc2+indiv_spikes_inh2+indiv_spikes_exc_tonic2+indiv_spikes_inh_tonic2
    rg2_sparse_firing_count = sparse_count5+sparse_count6+sparse_count7+sparse_count8
    rg2_silent_neuron_count = silent_count5+silent_count6+silent_count7+silent_count8
    print('Ext sparse firing %',round(rg2_sparse_firing_count*100/(len(rg2_indiv_spike_count)-rg2_silent_neuron_count),2))
    all_indiv_spike_counts=indiv_spikes_exc1+indiv_spikes_inh1+indiv_spikes_exc_tonic1+indiv_spikes_inh_tonic1+indiv_spikes_exc2+indiv_spikes_inh2+indiv_spikes_exc_tonic2+indiv_spikes_inh_tonic2
    sparse_firing_count = sparse_count1+sparse_count2+sparse_count3+sparse_count4+sparse_count5+sparse_count6+sparse_count7+sparse_count8
    silent_neuron_count = silent_count1+silent_count2+silent_count3+silent_count4+silent_count5+silent_count6+silent_count7+silent_count8
    #print('RG sparse firing, % sparse firing in RGs',sparse_firing_count,round(sparse_firing_count*100/(len(all_indiv_spike_counts)-silent_neuron_count),2),'%')

    if nn.rgs_connected==1:
        indiv_spikes_inh_inter_tonic1,neuron_to_sample_inh_inter_tonic1,sparse_count15,silent_count15 = popfunc.count_indiv_spikes(nn.num_inh_inter_tonic_v2b,senders_inh_inter_tonic1,avg_freq)
        indiv_spikes_inh_inter_tonic2,neuron_to_sample_inh_inter_tonic2,sparse_count17,silent_count17 = popfunc.count_indiv_spikes(nn.num_inh_inter_tonic_v1,senders_inh_inter_tonic2,avg_freq)
        all_indiv_spike_counts=all_indiv_spike_counts+indiv_spikes_inh_inter_tonic1+indiv_spikes_inh_inter_tonic2
        sparse_firing_count=sparse_firing_count+sparse_count15+sparse_count17
        silent_neuron_count=silent_neuron_count+silent_count15+silent_count17
        if nn.num_inh_inter_bursting_v2b>0:
            indiv_spikes_inh_inter_bursting1,neuron_to_sample_inh_inter_bursting1,sparse_count16,silent_count16 = popfunc.count_indiv_spikes(nn.num_inh_inter_bursting_v2b,senders_inh_inter_bursting1,avg_freq)
            all_indiv_spike_counts=all_indiv_spike_counts+indiv_spikes_inh_inter_bursting1
            sparse_firing_count=sparse_firing_count+sparse_count16
            silent_neuron_count=silent_neuron_count+silent_count16
        if nn.num_inh_inter_bursting_v1>0:
            indiv_spikes_inh_inter_bursting2,neuron_to_sample_inh_inter_bursting2,sparse_count18,silent_count18 = popfunc.count_indiv_spikes(nn.num_inh_inter_bursting_v1,senders_inh_inter_bursting2,avg_freq)
            all_indiv_spike_counts=all_indiv_spike_counts+indiv_spikes_inh_inter_bursting2
            sparse_firing_count=sparse_firing_count+sparse_count18
            silent_neuron_count=silent_neuron_count+silent_count18
      
    print('Length of spike count array (all) ',len(all_indiv_spike_counts))
    print('Total sparse firing, % sparse firing',sparse_firing_count,round(sparse_firing_count*100/(len(all_indiv_spike_counts)-silent_neuron_count),2),'%')
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
    
    if nn.rgs_connected:
        v2b_freq, v2b_times =popfunc.calculate_interspike_frequency(nn.num_inh_inter_tonic_v2b,spiketimes_inh_inter_tonic1)
        v1_freq, v1_times =popfunc.calculate_interspike_frequency(nn.num_inh_inter_tonic_v1,spiketimes_inh_inter_tonic2)
    
    t_stop = time.perf_counter()    
    print('Calculating ISF complete, taking ',int(t_stop-t_start),' seconds.')
    
    t_start = time.perf_counter()
    #Convolve spike data - RG populations
    rg_exc_convolved1, convolved_time = popfunc.convolve_spiking_activity(nn.flx_exc_bursting_count,spiketimes_exc1)
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
    
    rg1_avg_norm = (rg1_convolved-np.min(rg1_convolved))/(np.max(rg1_convolved)-np.min(rg1_convolved))
    rg2_avg_norm = (rg2_convolved-np.min(rg2_convolved))/(np.max(rg2_convolved)-np.min(rg2_convolved))
    rg1_convolved_scaled = rg1_convolved*rg1_scale
    rg2_convolved_scaled = rg2_convolved*rg2_scale
    if max(rg1_avg_norm)>0 and max(rg2_avg_norm)>0: 
        avg_freq, avg_phase, bd_comparison = calc.analyze_output(rg1_avg_norm,rg2_avg_norm,rg1_convolved_scaled,rg2_convolved_scaled,'RG',y_line_bd=0.4,y_line_phase=0.7)
    
    print('Max firing rate of a Flx RG (ISF):',round(rg1_isf_max,2),'Ext RG:',round(rg2_isf_max,2))
    print('Max firing rate of a Flx RG (Convolved):',round(rg1_conv_max,2),'Ext RG:',round(rg2_conv_max,2))
    print('Convolved max is',round(rg1_scale,3),round(rg2_scale,3), 'times the size of ISF max (Flx, Ext).')
    
    if nn.rgs_connected==1:
        v2b_isf_max = np.nanmax(np.array([np.nanmean(neuron_freq) for neuron_freq in v2b_freq]))
        v1_isf_max = np.nanmax(np.array([np.nanmean(neuron_freq) for neuron_freq in v1_freq]))
        v2b_conv_max = np.nanmax(v2b_convolved)
        v1_conv_max = np.nanmax(v1_convolved)
        v2b_scale = v2b_isf_max / v2b_conv_max
        v1_scale = v1_isf_max / v1_conv_max
    
    
    if nn.rgs_connected==0:
        t = convolved_time
        xticks = np.arange(start=np.ceil(t[0] / 1000) * 1000, stop=t[-1], step=1000)
        fig, ax = plt.subplots(1,sharex='all',figsize=(18, 12))    
        ax.plot(t, rg1_convolved*rg1_scale)
        ax.plot(t, rg2_convolved*rg2_scale)
        ax.legend(['RG_F', 'RG_E'],loc='upper right',fontsize='x-small') 
        ax.set_xlabel('Time (ms)')
        ax.set_xticks(xticks)
        ax.set_xticklabels([f'{int(x)}' for x in xticks])
        ax.set_ylabel('Freq (Hz)')
        ax.set_title('Average Spike Rate')
        if nn.args['save_results']: plt.savefig(nn.pathFigures + '/' + 'spike_rate_isolated_rgs.pdf',bbox_inches="tight")
    if nn.rgs_connected==1:
        t = convolved_time
        xticks = np.arange(start=np.ceil(t[0] / 1000) * 1000, stop=t[-1], step=1000)
        fig, ax = plt.subplots(2,sharex='all',figsize=(18, 12))    
        ax[0].plot(t, rg1_convolved*rg1_scale)
        ax[0].plot(t, rg2_convolved*rg2_scale)
        ax[1].plot(t, v2b_convolved*v2b_scale)
        ax[1].plot(t, v1_convolved*v1_scale)
        ax[0].set_xticks([])
        ax[0].legend(['RG_F', 'RG_E'],loc='upper right',fontsize='x-small') 
        ax[1].legend(['V2b', 'V1'],loc='upper right',fontsize='x-small') 
        ax[1].set_xlabel('Time (ms)')
        ax[1].set_xticks(xticks)
        ax[1].set_xticklabels([f'{int(x)}' for x in xticks])
        ax[0].set_ylabel('Freq (Hz)')
        ax[1].set_ylabel('Freq (Hz)')
        ax[0].set_title('Average Spike Rate')
        if nn.args['save_results']: plt.savefig(nn.pathFigures + '/' + 'spike_rate_rg_v1v2b.pdf',bbox_inches="tight")
    
if nn.args['save_results']:
	# Save rate-coded output
	np.savetxt(nn.pathFigures + '/output_rg1.csv',rg1_convolved_scaled,delimiter=',')
	np.savetxt(nn.pathFigures + '/output_rg2.csv',rg2_convolved_scaled,delimiter=',')
plt.show()