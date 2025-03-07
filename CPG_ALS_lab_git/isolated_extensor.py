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
from scipy.signal import find_peaks,correlate
from scipy.fft import fft, fftfreq
import set_network_params as netparams
from phase_ordering import order_by_phase
from pca import run_PCA
import population_functions as popfunc
from connect_populations import connect
ss.nest_start()
nn=netparams.neural_network()
conn=connect()

import create_ext_rg as rg
import calculate_output_metrics as calc

#Create neuron populations - NEST
rg2 = rg.create_rg_population()

print("Seed#: ",nn.rng_seed)
print("RG Ext: # exc (bursting, tonic): ",nn.ext_exc_bursting_count,nn.ext_exc_tonic_count,"; # inh(bursting, tonic): ",nn.ext_inh_bursting_count,nn.ext_inh_tonic_count)

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

#Read spike data
senders_exc2,spiketimes_exc2 = popfunc.read_spike_data(rg2.spike_detector_rg_exc_bursting)
senders_inh2,spiketimes_inh2 = popfunc.read_spike_data(rg2.spike_detector_rg_inh_bursting)
senders_exc_tonic2,spiketimes_exc_tonic2 = popfunc.read_spike_data(rg2.spike_detector_rg_exc_tonic)
senders_inh_tonic2,spiketimes_inh_tonic2 = popfunc.read_spike_data(rg2.spike_detector_rg_inh_tonic)

#Calculate balance
rg2_exc_burst_weight = conn.calculate_weighted_balance(rg2.rg_exc_bursting,rg2.spike_detector_rg_exc_bursting)
rg2_inh_burst_weight = conn.calculate_weighted_balance(rg2.rg_inh_bursting,rg2.spike_detector_rg_inh_bursting)
rg2_exc_tonic_weight = conn.calculate_weighted_balance(rg2.rg_exc_tonic,rg2.spike_detector_rg_exc_tonic)
rg2_inh_tonic_weight = conn.calculate_weighted_balance(rg2.rg_inh_tonic,rg2.spike_detector_rg_inh_tonic)
weights_per_pop2 = [rg2_exc_burst_weight,rg2_inh_burst_weight,rg2_exc_tonic_weight,rg2_inh_tonic_weight]
absolute_weights_per_pop2 = [rg2_exc_burst_weight,abs(rg2_inh_burst_weight),rg2_exc_tonic_weight,abs(rg2_inh_tonic_weight)]
rg2_balance_pct = (sum(weights_per_pop2)/sum(absolute_weights_per_pop2))*100
print('RG2 balance %: ',round(rg2_balance_pct,2),' >0 skew excitatory; <0 skew inhibitory')

#Create and plot population rate output
spike_bins_rg_exc2 = popfunc.rate_code_spikes(nn.ext_exc_bursting_count,spiketimes_exc2)
spike_bins_rg_inh2 = popfunc.rate_code_spikes(nn.ext_inh_bursting_count,spiketimes_inh2)
spike_bins_rg_exc_tonic2 = popfunc.rate_code_spikes(nn.ext_exc_tonic_count,spiketimes_exc_tonic2)
spike_bins_rg_inh_tonic2 = popfunc.rate_code_spikes(nn.ext_inh_tonic_count,spiketimes_inh_tonic2)
spike_bins_rg2 = spike_bins_rg_exc2+spike_bins_rg_exc_tonic2+spike_bins_rg_inh2+spike_bins_rg_inh_tonic2
print('Max spike count RG2: ',max(spike_bins_rg2))
#spike_bins_rg2 = (spike_bins_rg2-np.min(spike_bins_rg2))/(np.max(spike_bins_rg2)-np.min(spike_bins_rg2))

t = np.arange(0,len(spike_bins_rg2),1)
fig, ax = plt.subplots(1,sharex='all')
ax.plot(t, spike_bins_rg2)
ax.set_xlabel('Time (ms)')
ax.set_xticks([0,10000,20000,30000,40000,50000,60000,70000,80000,90000])
ax.set_xticklabels([0,1000,2000,3000,4000,5000,6000,7000,8000,9000])
ax.set_xlim(0,len(spike_bins_rg2))
ax.legend(['RG2'],loc='upper right',fontsize='x-small') 
ax.set_title("Population output (RG)")
figure = plt.gcf() # get current figure
figure.set_size_inches(8, 6)
plt.tight_layout()
if nn.args['save_results']: plt.savefig(nn.pathFigures + '/' + 'population_rate_output.pdf',bbox_inches="tight")

#Plot firing sparsity
indiv_spikes_exc2,neuron_to_sample_rg2_burst,sparse_count5,silent_count5 = popfunc.count_indiv_spikes(nn.ext_exc_bursting_count,senders_exc2)
indiv_spikes_inh2,neuron_to_sample_rg2_burst_inh,sparse_count6,silent_count6 = popfunc.count_indiv_spikes(nn.ext_inh_bursting_count,senders_inh2)
indiv_spikes_exc_tonic2,neuron_to_sample_rg2_ton,sparse_count7,silent_count7 = popfunc.count_indiv_spikes(nn.ext_exc_tonic_count,senders_exc_tonic2)
indiv_spikes_inh_tonic2,neuron_to_sample_rg2_ton_inh,sparse_count8,silent_count8 = popfunc.count_indiv_spikes(nn.ext_inh_tonic_count,senders_inh_tonic2)

all_indiv_spike_counts=indiv_spikes_exc2+indiv_spikes_inh2+indiv_spikes_exc_tonic2+indiv_spikes_inh_tonic2
sparse_firing_count = sparse_count5+sparse_count6+sparse_count7+sparse_count8
silent_neuron_count = silent_count5+silent_count6+silent_count7+silent_count8
spike_distribution = [all_indiv_spike_counts.count(i) for i in range(max(all_indiv_spike_counts))]
print('RG sparse firing, % sparse firing in RGs',sparse_firing_count,round(sparse_firing_count*100/(len(all_indiv_spike_counts)-silent_neuron_count),2),'%')

pylab.figure()
pylab.plot(spike_distribution[2:])
pylab.xscale('log')
pylab.xlabel('Total Spike Count')
pylab.ylabel('Number of Neurons')
pylab.title('Spike Distribution')
if nn.args['save_results']: plt.savefig(nn.pathFigures + '/' + 'spike_distribution.pdf',bbox_inches="tight")
    
plt.show()