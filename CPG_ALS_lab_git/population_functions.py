#!/usr/bin/env python

#include <static_connection.h>
import nest
import nest.raster_plot
import numpy as np
import sys
import pylab
import math
import matplotlib.pyplot as pyplot
import pickle, yaml
import random
import scipy
import scipy.fftpack
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from scipy.signal import find_peaks, peak_widths, peak_prominences
from scipy.signal import convolve2d, windows, butter, filtfilt, decimate
import time
import copy
import set_network_params as netparams

nn=netparams.neural_network()

def update_neuronal_characteristic(update_charac,neuron_population,leakage_value):
	neuron_charac = update_charac
	for neuron in neuron_population:
	    nest.SetStatus(neuron, {neuron_charac: leakage_value})
	new_val = nest.GetStatus(neuron_population, keys=neuron_charac)[0]
	return new_val

def read_recent_spike_data(spike_detector):
    total_spikes = 0
    spiketimes = [spike_detector.get('events', 'times')]
    spike_detection_window = 1 #ms
    current_time = nest.biological_time
    for i in range(len(spiketimes)):
        total_spikes += int(([np.sum((arr[i] >= current_time-spike_detection_window) & (arr[i] <= current_time)) for arr in spiketimes])[0])
    return total_spikes

def read_spike_data(spike_detector):
	senders = []
	spiketimes = []
	spike_detector = spike_detector
	senders += [spike_detector.get('events', 'senders')]
	spiketimes += [spike_detector.get('events', 'times')]
	return senders,spiketimes

def read_membrane_potential(multimeter,pop_size,neuron_num):
    mm = nest.GetStatus(multimeter,keys="events")[0]
    vm =  mm['V_m']
    t_vm = mm['times']
    vm = vm[neuron_num::pop_size]
    t_vm = t_vm[neuron_num::pop_size]
    
    # Ensure chronological order
    sorted_indices = np.argsort(t_vm)
    t_vm = t_vm[sorted_indices]
    vm = vm[sorted_indices]
    
    return vm,t_vm

def count_indiv_spikes(total_neurons,neuron_id_data,calc_freq):
        total_spikes_per_second = 6 if math.isnan(calc_freq) else int(calc_freq*2) #Spiking 2 times per period
        spike_count_array = [len(neuron_id_data[0][i]) for i in range(total_neurons)]
        sparse_count_max = total_spikes_per_second*(nn.sim_time/1000)	
        sparse_firing_count = [i for i, count in enumerate(spike_count_array) if count>=1 and count<=sparse_count_max]
        silent_neuron_count = [i for i, count in enumerate(spike_count_array) if count==0]
        neuron_to_sample = sparse_firing_count[1] if len(sparse_firing_count) > 1 else 0
        #print('Max for sparse firing for this trial: ',sparse_count_max)
        return spike_count_array,neuron_to_sample,len(sparse_firing_count),len(silent_neuron_count) 

def save_spike_data(num_neurons,population,neuron_num_offset):
	spike_time = []
	all_spikes = []
	for i in range(num_neurons):
	    spike_data = population[0][i]
	    neuron_num = [i+neuron_num_offset]*spike_data.shape[0]
	    for j in range(spike_data.shape[0]):
	        spike_time.append(spike_data[j])    
	    indiv_spikes = list(zip(neuron_num,spike_time))
	    all_spikes.extend(indiv_spikes)  
	    spike_time = []     
	return all_spikes

def single_neuron_spikes(neuron_number,population):
	spike_time = [0]*int(nn.sim_time/nn.time_resolution)
	spike_data = population[0][neuron_number]
	for j in range(spike_data.shape[0]):
	    spike_time_index = int(spike_data[j]*(1/nn.time_resolution))-1
	    spike_time[spike_time_index]=spike_data[j]        
	return spike_time

def single_neuron_spikes_binary(neuron_number,population):
	spike_time = [0]*int(nn.sim_time/nn.time_resolution)
	spike_data = population[0][neuron_number]
	for j in range(spike_data.shape[0]):
	    spike_time_index = int(spike_data[j]*(1/nn.time_resolution))-1
	    spike_time[spike_time_index]=1        
	return spike_time
'''
def calculate_interspike_frequency(neuron_count, output_spiketimes):
    frequencies = []
    times = []
    for i in range(neuron_count):
        t_spikes = output_spiketimes[0][i]
        if len(t_spikes) > 0:
            # Sort spikes by time
            sorted_indices = np.argsort(t_spikes)
            spike_times = t_spikes[sorted_indices]
            
            isi = np.diff(t_spikes)
            frequencies.append(1000.0 / isi)
            times.append(spike_times[1:])
    return frequencies, times
'''

def calculate_interspike_frequency(neuron_count, output_spiketimes):
    frequencies = []
    times = []
    for i in range(neuron_count):
        t_spikes = output_spiketimes[0][i]
        if len(t_spikes) > 1:  # Need at least two spikes for ISI
            # Sort spikes by time
            sorted_indices = np.argsort(t_spikes)
            spike_times = t_spikes[sorted_indices]
            
            isi = np.diff(spike_times)
            # Filter out NaNs
            valid_mask = ~np.isnan(isi)
            valid_isi = isi[valid_mask]
            valid_times = spike_times[1:][valid_mask]
            
            if len(valid_isi) > 0:
                frequencies.append(1000.0 / valid_isi)
                times.append(valid_times)
            else:
                frequencies.append(np.array([np.nan]))
                times.append(np.array([np.nan]))
        else:
            frequencies.append(np.array([np.nan]))
            times.append(np.array([np.nan]))
    return frequencies, times


def calculate_avg_interspike_frequencies(output_spiketimes):
    total_time = nn.sim_time
    bin_width = nn.time_window*nn.time_resolution
    bin_edges = np.arange(0, total_time + bin_width, bin_width)
    num_bins = len(bin_edges) - 1

    all_times = []
    all_freqs = []

    # Step 1–2: Collect all frequencies and their associated times
    for neuron_spikes in output_spiketimes[0]:
        if len(neuron_spikes) > 1:
            sorted_spikes = np.sort(neuron_spikes)
            isi = np.diff(sorted_spikes)
            freqs = 1000.0 / isi  # Hz
            times = sorted_spikes[1:]  # Time of the second spike in the ISI
            all_freqs.extend(freqs)
            all_times.extend(times)

    all_freqs = np.array(all_freqs)
    all_times = np.array(all_times)

    # Step 3: Bin the data
    bin_sums = np.zeros(num_bins)
    bin_counts = np.zeros(num_bins)

    bin_indices = np.digitize(all_times, bin_edges) - 1
    for i, bin_idx in enumerate(bin_indices):
        if 0 <= bin_idx < num_bins:
            bin_sums[bin_idx] += all_freqs[i]
            bin_counts[bin_idx] += 1

    # Step 4: Compute averages
    with np.errstate(invalid='ignore'):
        avg_freqs = np.divide(bin_sums, bin_counts, where=bin_counts != 0)

    # Step 5: Fill empty bins by averaging neighbors
    for i in range(num_bins):
        if bin_counts[i] == 0:
            prev_val = next_val = None
            # Search left
            for j in range(i - 1, -1, -1):
                if bin_counts[j] != 0:
                    prev_val = avg_freqs[j]
                    break
            # Search right
            for j in range(i + 1, num_bins):
                if bin_counts[j] != 0:
                    next_val = avg_freqs[j]
                    break
            if prev_val is not None and next_val is not None:
                avg_freqs[i] = (prev_val + next_val) / 2
            elif prev_val is not None:
                avg_freqs[i] = prev_val
            elif next_val is not None:
                avg_freqs[i] = next_val
            else:
                avg_freqs[i] = 0  # No data at all

    smoothed_freqs = gaussian_filter(avg_freqs, 2)

    return smoothed_freqs, bin_edges[:-1]


def padded_sliding_time_window(signal, window_size):
    padded_signal = np.pad(signal, (window_size//2, window_size//2), mode='edge')
    windows = np.lib.stride_tricks.sliding_window_view(padded_signal, window_size)
    return np.mean(windows, axis=1)[:len(signal)]  # Trim to original size


def rate_code_spikes(neuron_count, output_spiketimes):
	# Initialize the spike bins array as a 2D array
	bins=np.arange(0, nn.sim_time+nn.time_resolution,nn.time_resolution)
	# Loop over each neuron
	for i in range(neuron_count):
	    t_spikes = output_spiketimes[0][i]
	    # Use numpy's histogram function to assign each spike to its corresponding time bin index
	    spikes_per_bin,bin_edges=np.histogram(t_spikes, bins)
	    # Add the spike counts to the `spike_bins_current` array
	    if i == 0:
	        spike_bins_current = spikes_per_bin
	    else:
	        spike_bins_current += spikes_per_bin
	spike_bins_current = sliding_time_window(spike_bins_current,nn.time_window) #Applies a time window to smooth the output        
	smoothed_spike_bins = gaussian_filter(spike_bins_current, nn.convstd_rate) #Applies a filter to smooth the high frequency noise
	if nn.chop_edges_amount > 0.0:
	    smoothed_spike_bins = smoothed_spike_bins[int(nn.chop_edges_amount):int(-nn.chop_edges_amount)]
	return smoothed_spike_bins

def sliding_time_window(signal, window_size):
	windows = np.lib.stride_tricks.sliding_window_view(signal, window_size)
	return np.sum(windows, axis=1)

def sliding_time_window_matrix(signal, window_size):
	result = []
	for row in signal:
	    windows = np.lib.stride_tricks.sliding_window_view(row, window_size)
	    row_sum = np.sum(windows, axis=1)
	    result.append(row_sum)
	return np.array(result)

def smooth(data, sd):
	data = copy.copy(data)
	from scipy.signal import convolve, windows
	n_bins = data.shape[1]
	w = n_bins - 1 if n_bins % 2 == 0 else n_bins
	window = windows.gaussian(w, std=sd)
	for j in range(data.shape[0]):
	    data[j,:] = convolve(data[j,:], window, mode='same', method='auto') 
	return data

def convolve_spiking_activity(population_size,population):
    time_steps = int(nn.sim_time/nn.time_resolution) 
    binary_spikes = np.vstack([single_neuron_spikes_binary(i, population) for i in range(population_size)])
    binned_spikes = sliding_time_window_matrix(binary_spikes,nn.time_window)
    smoothed_spikes = smooth(binned_spikes, nn.convstd_rate)
    time_vector = np.arange(binned_spikes.shape[1]) * nn.time_resolution
    if nn.chop_edges_amount > 0.0:
        chop = int(nn.chop_edges_amount)
        smoothed_spikes = smoothed_spikes[:, chop:-chop]
        time_vector = time_vector[chop:-chop]
    if nn.remove_mean:
        smoothed_spikes = (smoothed_spikes.T - np.mean(smoothed_spikes, axis=1)).T
    if nn.high_pass_filtered:
        # Same used as in Linden et al, 2022 paper
        b, a = butter(3, .1, 'highpass', fs=1000)		#high pass freq was previously 0.3Hz
        smoothed_spikes = filtfilt(b, a, smoothed_spikes)
    if nn.downsampling_convolved:
        smoothed_spikes = decimate(smoothed_spikes, int(1/nn.time_resolution), n=2, ftype='iir', zero_phase=True)
        time_vector = time_vector[::decimation_factor]
    smoothed_spikes = smoothed_spikes[:, :-nn.time_window+1] #truncate array by the width of the time window 
    time_vector = time_vector[:smoothed_spikes.shape[1]]
    smoothed_spikes = smoothed_spikes.mean(axis=0)
    return smoothed_spikes, time_vector

def inject_current(neuron_population,current):
	for neuron in neuron_population:
	    nest.SetStatus([neuron],{"I_e": current})
	updated_current = nest.GetStatus(neuron_population, keys="I_e")[0]
	return updated_current
	
def normalize_rows(matrix):
    max_values = np.max(matrix, axis=1, keepdims=True)
    normalized_matrix = matrix / max_values
    return normalized_matrix	
