#!/usr/bin/env python

import nest
import numpy as np
import pathlib, sys
import pylab
import math
import matplotlib.pyplot as plt
import random
import time, datetime
import start_simulation as ss
import pickle, yaml
import pandas as pd
from scipy.signal import find_peaks,correlate
from scipy.fft import fft, fftfreq
import set_network_params as netparams
from phase_ordering import order_by_phase
from pca import run_PCA
from connect_populations import connect
import population_functions as popfunc
import calculate_stability_metrics as calc

desired_freq = 4.           #Hz
desired_phase_diff = 180.   #degrees
desired_bd_diff = 1.5       #ms
desired_firing_sparsity = 0.2      #0.2 = 20%

def run_rg_simulation(testing_parameters):
    
    rg_v1v2b_conn = testing_parameters[0]
    v1v2b_rg_conn = testing_parameters[1]
    rg_rg_conn = testing_parameters[2]
    rg_v1v2b_w = testing_parameters[3] 
    v1v2b_rg_w = testing_parameters[4] 
    rg_rg_w = testing_parameters[5] 
    print('Received parameters: ',testing_parameters)
    
    ss.nest_start()
    nn=netparams.neural_network()
    conn=connect() 
    
    import create_flx_rg as flx_rg
    import create_ext_rg as ext_rg
    import create_exc_inter_pop as exc
    import create_inh_inter_pop as inh
    import create_interneuron_pop as inter 
    import create_mnp as mnp
    #Create neuron populations - NEST
    rg1 = flx_rg.create_rg_population()
    rg2 = ext_rg.create_rg_population()

    if nn.rgs_connected == 1:
        inh1 = inh.create_inh_inter_population('V2b')  # V2b
        inh2 = inh.create_inh_inter_population('V1')  # V1

        # Connect excitatory rg neurons to V1/V2b inhibitory populations        
        nest.Connect(rg1.rg_exc_bursting,inh1.inh_inter_tonic,{'rule': 'pairwise_bernoulli', 'p':rg_v1v2b_conn},
            {"synapse_model":"static_synapse",
            "weight" : rg_v1v2b_w, #nS            
            "delay" : nn.synaptic_delay})
        nest.Connect(rg1.rg_exc_tonic,inh1.inh_inter_tonic,{'rule': 'pairwise_bernoulli', 'p':rg_v1v2b_conn},
            {"synapse_model":"static_synapse",
            "weight" : rg_v1v2b_w, #nS            
            "delay" : nn.synaptic_delay})
        nest.Connect(rg2.rg_exc_bursting,inh2.inh_inter_tonic,{'rule': 'pairwise_bernoulli', 'p':rg_v1v2b_conn},
            {"synapse_model":"static_synapse",
            "weight" : rg_v1v2b_w, #nS            
            "delay" : nn.synaptic_delay})
        nest.Connect(rg2.rg_exc_tonic,inh2.inh_inter_tonic,{'rule': 'pairwise_bernoulli', 'p':rg_v1v2b_conn},
            {"synapse_model":"static_synapse",
            "weight" : rg_v1v2b_w, #nS            
            "delay" : nn.synaptic_delay})

        #Connect V1/V2b inhibitory populations to all rg neurons
        nest.Connect(inh1.inh_inter_tonic,rg2.rg_exc_bursting,{'rule': 'pairwise_bernoulli', 'p':v1v2b_rg_conn},
            {"synapse_model":"static_synapse",
            "weight" : v1v2b_rg_w, #nS            
            "delay" : nn.synaptic_delay})
        nest.Connect(inh1.inh_inter_tonic,rg2.rg_exc_tonic,{'rule': 'pairwise_bernoulli', 'p':v1v2b_rg_conn},
            {"synapse_model":"static_synapse",
            "weight" : v1v2b_rg_w, #nS            
            "delay" : nn.synaptic_delay})
        nest.Connect(inh1.inh_inter_tonic,rg2.rg_inh_bursting,{'rule': 'pairwise_bernoulli', 'p':v1v2b_rg_conn},
            {"synapse_model":"static_synapse",
            "weight" : v1v2b_rg_w, #nS            
            "delay" : nn.synaptic_delay})
        nest.Connect(inh1.inh_inter_tonic,rg2.rg_inh_tonic,{'rule': 'pairwise_bernoulli', 'p':v1v2b_rg_conn},
            {"synapse_model":"static_synapse",
            "weight" : v1v2b_rg_w, #nS            
            "delay" : nn.synaptic_delay})
        
        nest.Connect(inh2.inh_inter_tonic,rg1.rg_exc_bursting,{'rule': 'pairwise_bernoulli', 'p':v1v2b_rg_conn},
            {"synapse_model":"static_synapse",
            "weight" : v1v2b_rg_w, #nS            
            "delay" : nn.synaptic_delay})
        nest.Connect(inh2.inh_inter_tonic,rg1.rg_exc_tonic,{'rule': 'pairwise_bernoulli', 'p':v1v2b_rg_conn},
            {"synapse_model":"static_synapse",
            "weight" : v1v2b_rg_w, #nS            
            "delay" : nn.synaptic_delay})
        nest.Connect(inh2.inh_inter_tonic,rg1.rg_inh_bursting,{'rule': 'pairwise_bernoulli', 'p':v1v2b_rg_conn},
            {"synapse_model":"static_synapse",
            "weight" : v1v2b_rg_w, #nS            
            "delay" : nn.synaptic_delay})
        nest.Connect(inh2.inh_inter_tonic,rg1.rg_inh_tonic,{'rule': 'pairwise_bernoulli', 'p':v1v2b_rg_conn},
            {"synapse_model":"static_synapse",
            "weight" : v1v2b_rg_w, #nS            
            "delay" : nn.synaptic_delay})

        #Connect excitatory rg neurons
        nest.Connect(rg1.rg_exc_bursting,rg2.rg_exc_bursting,{'rule': 'pairwise_bernoulli', 'p':rg_rg_conn},
            {"synapse_model":"static_synapse",
            "weight" : rg_rg_w, #nS            
            "delay" : nn.synaptic_delay})
        nest.Connect(rg1.rg_exc_bursting,rg2.rg_exc_tonic,{'rule': 'pairwise_bernoulli', 'p':rg_rg_conn},
            {"synapse_model":"static_synapse",
            "weight" : rg_rg_w, #nS            
            "delay" : nn.synaptic_delay})
        nest.Connect(rg1.rg_exc_tonic,rg2.rg_exc_bursting,{'rule': 'pairwise_bernoulli', 'p':rg_rg_conn},
            {"synapse_model":"static_synapse",
            "weight" : rg_rg_w, #nS            
            "delay" : nn.synaptic_delay})
        nest.Connect(rg1.rg_exc_tonic,rg2.rg_exc_tonic,{'rule': 'pairwise_bernoulli', 'p':rg_rg_conn},
            {"synapse_model":"static_synapse",
            "weight" : rg_rg_w, #nS            
            "delay" : nn.synaptic_delay})
        
        nest.Connect(rg2.rg_exc_bursting,rg1.rg_exc_bursting,{'rule': 'pairwise_bernoulli', 'p':rg_rg_conn},
            {"synapse_model":"static_synapse",
            "weight" : rg_rg_w, #nS            
            "delay" : nn.synaptic_delay})
        nest.Connect(rg2.rg_exc_bursting,rg1.rg_exc_tonic,{'rule': 'pairwise_bernoulli', 'p':rg_rg_conn},
            {"synapse_model":"static_synapse",
            "weight" : rg_rg_w, #nS            
            "delay" : nn.synaptic_delay})
        nest.Connect(rg2.rg_exc_tonic,rg1.rg_exc_bursting,{'rule': 'pairwise_bernoulli', 'p':rg_rg_conn},
            {"synapse_model":"static_synapse",
            "weight" : rg_rg_w, #nS            
            "delay" : nn.synaptic_delay})
        nest.Connect(rg2.rg_exc_tonic,rg1.rg_exc_tonic,{'rule': 'pairwise_bernoulli', 'p':rg_rg_conn},
            {"synapse_model":"static_synapse",
            "weight" : rg_rg_w, #nS            
            "delay" : nn.synaptic_delay})	

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

    ################
    # Save results #
    ################
    if nn.args['save_results']:
        id_ = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        #simulation_config['date'] = datetime.date.today()
        #simulation_config['id'] = id_
        path = 'saved_simulations' + '/' + id_ 
        pathFigures = 'saved_simulations' + '/' + id_ + '/Figures'
        pathlib.Path(path).mkdir(parents=True, exist_ok=False)
        pathlib.Path(pathFigures).mkdir(parents=True, exist_ok=False)
        with open(path + '/args_' + id_ + '.yaml', 'w') as yamlfile:
            #args['seed'] = simulation_config['seed']
            yaml.dump(nn.args, yamlfile)
    
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

        t = np.arange(0,len(spike_bins_rg1_true),1)
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
        ax[0].legend(['RG_F', 'RG_E'],loc='upper right',fontsize='x-small') 
        ax[1].legend(['V2b', 'V1'],loc='upper right',fontsize='x-small') 
        ax[0].set_title("Population output (RG)")
        ax[1].set_title("Population output (V1/V2b)")
        figure = plt.gcf() # get current figure
        figure.set_size_inches(8, 6)
        plt.tight_layout()
        if nn.args['save_results']: plt.savefig(pathFigures + '/' + 'rate_coded_output.pdf',bbox_inches="tight")
        #plt.show()
        
           

    if nn.spike_distribution_plot==1:
        #Count spikes per neuron
        indiv_spikes_exc1,neuron_to_sample_rg1_burst,sparse_count1,silent_count1 = popfunc.count_indiv_spikes(nn.flx_exc_bursting_count,senders_exc1)
        indiv_spikes_inh1,neuron_to_sample_rg1_burst_inh,sparse_count2,silent_count2 = popfunc.count_indiv_spikes(nn.flx_inh_bursting_count,senders_inh1)
        indiv_spikes_exc_tonic1,neuron_to_sample_rg1_ton,sparse_count3,silent_count3 = popfunc.count_indiv_spikes(nn.flx_exc_tonic_count,senders_exc_tonic1)
        indiv_spikes_inh_tonic1,neuron_to_sample_rg1_ton_inh,sparse_count4,silent_count4 = popfunc.count_indiv_spikes(nn.flx_inh_tonic_count,senders_inh_tonic1)

        indiv_spikes_exc2,neuron_to_sample_rg2_burst,sparse_count5,silent_count5 = popfunc.count_indiv_spikes(nn.ext_exc_bursting_count,senders_exc2)
        indiv_spikes_inh2,neuron_to_sample_rg2_burst_inh,sparse_count6,silent_count6 = popfunc.count_indiv_spikes(nn.ext_inh_bursting_count,senders_inh2)
        indiv_spikes_exc_tonic2,neuron_to_sample_rg2_ton,sparse_count7,silent_count7 = popfunc.count_indiv_spikes(nn.ext_exc_tonic_count,senders_exc_tonic2)
        indiv_spikes_inh_tonic2,neuron_to_sample_rg2_ton_inh,sparse_count8,silent_count8 = popfunc.count_indiv_spikes(nn.ext_inh_tonic_count,senders_inh_tonic2)
        all_indiv_spike_counts=indiv_spikes_exc1+indiv_spikes_inh1+indiv_spikes_exc_tonic1+indiv_spikes_inh_tonic1+indiv_spikes_exc2+indiv_spikes_inh2+indiv_spikes_exc_tonic2+indiv_spikes_inh_tonic2
        sparse_firing_count = sparse_count1+sparse_count2+sparse_count3+sparse_count4+sparse_count5+sparse_count6+sparse_count7+sparse_count8
        silent_neuron_count = silent_count1+silent_count2+silent_count3+silent_count4+silent_count5+silent_count6+silent_count7+silent_count8
        print('RG sparse firing, % sparse firing in RGs',sparse_firing_count,round(sparse_firing_count*100/(len(all_indiv_spike_counts)-silent_neuron_count),2),'%')

        if nn.rgs_connected==1:
            indiv_spikes_inh_inter_tonic1,neuron_to_sample_inh_inter_tonic1,sparse_count15,silent_count15 = popfunc.count_indiv_spikes(nn.num_inh_inter_tonic_v2b,senders_inh_inter_tonic1)
            indiv_spikes_inh_inter_tonic2,neuron_to_sample_inh_inter_tonic2,sparse_count17,silent_count17 = popfunc.count_indiv_spikes(nn.num_inh_inter_tonic_v1,senders_inh_inter_tonic2)
            all_indiv_spike_counts=all_indiv_spike_counts+indiv_spikes_inh_inter_tonic1+indiv_spikes_inh_inter_tonic2
            sparse_firing_count=sparse_firing_count+sparse_count15+sparse_count17
            silent_neuron_count=silent_neuron_count+silent_count15+silent_count17
            if nn.num_inh_inter_bursting_v2b>0:
                indiv_spikes_inh_inter_bursting1,neuron_to_sample_inh_inter_bursting1,sparse_count16,silent_count16 = popfunc.count_indiv_spikes(nn.num_inh_inter_bursting_v2b,senders_inh_inter_bursting1)
                all_indiv_spike_counts=all_indiv_spike_counts+indiv_spikes_inh_inter_bursting1
                sparse_firing_count=sparse_firing_count+sparse_count16
                silent_neuron_count=silent_neuron_count+silent_count16
            if nn.num_inh_inter_bursting_v1>0:
                indiv_spikes_inh_inter_bursting2,neuron_to_sample_inh_inter_bursting2,sparse_count18,silent_count18 = popfunc.count_indiv_spikes(nn.num_inh_inter_bursting_v1,senders_inh_inter_bursting2)
                all_indiv_spike_counts=all_indiv_spike_counts+indiv_spikes_inh_inter_bursting2
                sparse_firing_count=sparse_firing_count+sparse_count18
                silent_neuron_count=silent_neuron_count+silent_count18
        
        firing_sparsity = round(sparse_firing_count/(len(all_indiv_spike_counts)-silent_neuron_count),4)
        #print('Length of spike count array (all) ',len(all_indiv_spike_counts))
        print('Total # sparse firing, % sparse firing',sparse_firing_count,firing_sparsity*100,'%')
        spike_distribution = [all_indiv_spike_counts.count(i) for i in range(max(all_indiv_spike_counts))]

    if nn.args['save_results']:
        # Save rate-coded output
        np.savetxt(pathFigures + '/output_rg1.csv',spike_bins_rg1,delimiter=',')
        np.savetxt(pathFigures + '/output_rg2.csv',spike_bins_rg2,delimiter=',')
    
    avg_rg_freq, avg_phase_diff, avg_bd_diff = calc.analyze_output(spike_bins_rg1,spike_bins_rg2,'RG',y_line_bd=0.4,y_line_phase=0.5)
    freq_diff = 10 if math.isnan(avg_rg_freq) else abs(desired_freq-avg_rg_freq)/(desired_freq+avg_rg_freq) #Normalize so that freq and phase have equal influence on optimization
    phase_diff = 10 if math.isnan(avg_phase_diff) else abs(desired_phase_diff-avg_phase_diff)/(desired_phase_diff+avg_phase_diff)
    bd_diff = 10 if math.isnan(avg_bd_diff) or avg_bd_diff<0  else abs(desired_bd_diff-avg_bd_diff)/(desired_bd_diff+avg_bd_diff)
    amp_diff = 10 if math.isnan(max(spike_bins_rg1_true)) or math.isnan(max(spike_bins_rg2_true)) else abs(max(spike_bins_rg1_true)-max(spike_bins_rg2_true))/(max(spike_bins_rg1_true)+max(spike_bins_rg2_true))
    sparsity_diff = 10 if math.isnan(firing_sparsity)  else abs(desired_firing_sparsity-firing_sparsity)/(desired_firing_sparsity+firing_sparsity)  
    total_diff = freq_diff + phase_diff + bd_diff + amp_diff + sparsity_diff

    total_diff = 50 if math.isnan(total_diff) else total_diff
    print('Indiv parameter diff (freq,phase,bd,amp,spar): ',round(freq_diff,2),round(phase_diff,2),round(bd_diff,2),round(amp_diff,2),round(sparsity_diff,2))
    print('Calculated difference from desired characteristics: ',round(total_diff,2))     
        
    other_metrics = [spike_bins_rg1_true, spike_bins_rg2_true, spike_bins_inh_inter1_true, spike_bins_inh_inter2_true]    
        
    return total_diff, other_metrics

#run_rg_simulation([0.3,0.1,0.03,0.34,-1.2,0.34])
#plt.show()
