#!/usr/bin/env python

import pandas as pd
import matplotlib.pyplot as plt
import sys
import numpy as np
from scipy.signal import find_peaks
np.set_printoptions(legacy='1.25')

folder_name=sys.argv[1]

plt.rcParams.update({'font.size': 20})
time_resolution=0.1
print_data_array_bd=[]
print_data_array_phase=[]

def analyze_output(input_1,input_2,pop_type,y_line_bd,y_line_phase,min_dist):
    try:
        # Read the CSV file into an array
        pop_data1 = np.loadtxt(input_1, delimiter=',', dtype=float)
        pop_data2 = np.loadtxt(input_2, delimiter=',', dtype=float)
        
        pop_data1_norm = (pop_data1-np.min(pop_data1))/(np.max(pop_data1)-np.min(pop_data1))
        pop_data2_norm = (pop_data2-np.min(pop_data2))/(np.max(pop_data2)-np.min(pop_data2))
        
        up_bd1,down_bd1,burst_duration1,bd_variance1,coeff_bd_variance1=calculate_burst_duration(pop_data1_norm,y_line_bd)
        up_bd2,down_bd2,burst_duration2,bd_variance2,coeff_bd_variance2=calculate_burst_duration(pop_data2_norm,y_line_bd)          
        
        up_bd1_y = [pop_data1_norm[t_thresh] for t_thresh in up_bd1]
        down_bd1_y = [pop_data1_norm[t_thresh] for t_thresh in down_bd1]
        up_bd2_y = [pop_data2_norm[t_thresh] for t_thresh in up_bd2]         
        down_bd2_y = [pop_data2_norm[t_thresh] for t_thresh in down_bd2]
        
        freq_pop1 = calculate_freq(up_bd1)
        freq_pop2 = calculate_freq(up_bd2)
        avg_freq = (freq_pop1 + freq_pop2)/2
        print('Freq (1,2,avg)',freq_pop1,freq_pop2,avg_freq)    
        
        #Calculate phase using peak to peak
        phase_peak,phase_variance_peak,coeff_phase_variance_peak,avg_freq_peak,pop1_peaks,pop2_peaks=calculate_peak_to_peak_phase(pop_data1_norm,pop_data2_norm,y_line_phase,min_dist)
        phase_peak = 360 - phase_peak if phase_peak > 180 else phase_peak  #Ensure phase always between 0-180 degrees
        phase1_y = [pop_data1_norm[t_peak] for t_peak in pop1_peaks]
        phase2_y = [pop_data2_norm[t_peak] for t_peak in pop2_peaks]
        
        total_suppression = find_zero_overlap(pop_data1_norm,pop_data2_norm)
        
        #Calculate CV of amplitude
        coeff_amp1_var,coeff_amp2_var = calculate_amplitude_cv(pop_data1_norm,pop_data2_norm,y_line_phase,min_dist)
        
        print(pop_type+' Freq, Phase, BD Flx, BD Ext: ')
        print(avg_freq,round(phase_peak,2),round(burst_duration1,2),round(burst_duration2,2))  
        print(pop_type,'BD (1,2), BD CV (1,2), Amp CV (1,2), Freq, Phase, Phase CV, % of time suppressed: ')
        print_data_array_bd.extend([round(burst_duration1,2),round(burst_duration2,2),coeff_bd_variance1,coeff_bd_variance2,coeff_amp1_var,coeff_amp2_var])
        print_data_array_phase.extend([avg_freq_peak,round(phase_peak,2),coeff_phase_variance_peak,round(100*(total_suppression/4000),2)])

    except FileNotFoundError:
        print(f"File '{input_1}' not found.")
    except FileNotFoundError:
        print(f"File '{input_2}' not found.")    
    except Exception as e:
        print(f"An error occurred: {str(e)}")  

    return pop_data1_norm,pop_data2_norm,up_bd1,up_bd1_y,up_bd2,up_bd2_y,down_bd1,down_bd1_y,down_bd2,down_bd2_y,pop1_peaks,pop2_peaks,phase1_y,phase2_y
    
def calculate_burst_duration(array, value):
    upward_count = 0
    downward_count = 0
    upward_indices = []  # Store indices of upward crossings
    downward_indices = []  # Store indices of downward crossings

    crossing = False

    for index, item in enumerate(array):
        if item >= value and not crossing:
            upward_count += 1
            upward_indices.append(index)
            crossing = True
        elif item < value and crossing:
            downward_count += 1
            downward_indices.append(index)
            crossing = False
    min_length = min(len(downward_indices), len(upward_indices))
    downward_indices = downward_indices[1:min_length]
    upward_indices = upward_indices[1:min_length]
    burst_duration=(np.subtract(downward_indices,upward_indices))*time_resolution
    bd_variance=np.var(burst_duration)
    coeff_bd_variance=(np.std(burst_duration)/np.mean(burst_duration))
    burst_duration=np.mean(burst_duration)
    return upward_indices,downward_indices,round(burst_duration,2),round(bd_variance,2),round(coeff_bd_variance,2)

def calculate_peak_to_peak_phase(spike_bins1, spike_bins2, min_peak_height, min_dist):
    pop1_peaks = find_peaks(spike_bins1, height=min_peak_height, distance=min_dist, prominence=0.1)[0]
    pop2_peaks = find_peaks(spike_bins2, height=min_peak_height, distance=min_dist, prominence=0.1)[0]

    alternating_peaks1 = []
    alternating_peaks2 = []

    i, j = 0, 0
    last_pop = None

    while i < len(pop1_peaks) and j < len(pop2_peaks):
        if last_pop is None or last_pop == 2:
            if pop1_peaks[i] < pop2_peaks[j]:
                alternating_peaks1.append(pop1_peaks[i])
                last_pop = 1
                i += 1
            else:
                #alternating_peaks2.append(pop2_peaks[j])
                #last_pop = 2
                j += 1
        elif last_pop == 1 and (pop2_peaks[j] < pop1_peaks[i]):
            alternating_peaks2.append(pop2_peaks[j])
            last_pop = 2
            j += 1
        else:
            #alternating_peaks1.append(pop1_peaks[i])
            #last_pop = 1
            i += 1

    # Truncate to the shortest length
    min_length = min(len(alternating_peaks1), len(alternating_peaks2))
    alternating_peaks1 = alternating_peaks1[:min_length]
    alternating_peaks2 = alternating_peaks2[:min_length]

    # Calculate time differences
    time_diff = np.subtract(alternating_peaks2, alternating_peaks1)
    
    period1 = np.mean(np.diff(alternating_peaks1)) * time_resolution
    period2 = np.mean(np.diff(alternating_peaks2)) * time_resolution
    avg_period = (period1 + period2) / 2
    avg_freq = 1000 / avg_period

    phase = (avg_period - (time_diff * time_resolution)) / avg_period
    phase_in_deg = phase * 360
    phase_variance = np.var(phase_in_deg)
    coeff_phase_var = (np.std(phase_in_deg) / np.mean(phase_in_deg)) 
    avg_phase_in_deg = np.mean(phase * 360)
    
    # Normalize the phase to the range [0, 360)
    avg_phase_in_deg = (avg_phase_in_deg + 360) % 360

    return round(avg_phase_in_deg, 2), round(phase_variance, 2), round(coeff_phase_var, 2), round(avg_freq, 2), alternating_peaks1, alternating_peaks2

def calculate_amplitude_cv(spike_bins1, spike_bins2, min_peak_height, min_dist):
    # Extract peaks and their properties (including heights)
    pop1_peaks, pop1_properties = find_peaks(spike_bins1, height=min_peak_height, distance=min_dist, prominence=0.1)
    pop2_peaks, pop2_properties = find_peaks(spike_bins2, height=min_peak_height, distance=min_dist, prominence=0.1)

    # Extract peak heights
    pop1_heights = pop1_properties['peak_heights']
    pop2_heights = pop2_properties['peak_heights']

    # Calculate the variance and coefficient of variation for amplitude
    amp1_variance = np.var(pop1_heights)
    amp2_variance = np.var(pop2_heights)
    coeff_amp1_var = (np.std(pop1_heights) / np.mean(pop1_heights))
    coeff_amp2_var = (np.std(pop2_heights) / np.mean(pop2_heights))

    print('Amplitude CV (1,2)', coeff_amp1_var, coeff_amp2_var)
    
    return round(coeff_amp1_var,4), round(coeff_amp2_var,4)

def calculate_freq(arr):
    period = np.mean(np.diff(arr)) * time_resolution
    freq = 1000 / period
    
    return round(freq,2)

def find_zero_overlap(arr1, arr2):
    if len(arr1) != len(arr2):
        raise ValueError("Arrays must have the same length")

    zero_durations = []  # Store durations of zero intervals
    duration = 0  # Initialize duration

    for i in range(len(arr1)):
        if arr1[i] == 0 and arr2[i] == 0:
            duration += 1  # Increment the duration
        elif duration > 0:
            zero_durations.append(duration)  # Record the duration
            duration = 0
    
    sum_zero_durations = sum(zero_durations)*time_resolution        
    'Time suppression intervals: ',zero_durations
    return sum_zero_durations

#MNP input
mnp_bd_y_line = 0.4
mnp_phase_y_line = 0.4
min_dist_phase_calc = 1000
mnp1_input = folder_name+'/Figures/output_mnp1.csv'
mnp2_input = folder_name+'/Figures/output_mnp2.csv'
mnp1_data,mnp2_data,up_bd1,up_bd1_y,up_bd2,up_bd2_y,down_bd1,down_bd1_y,down_bd2,down_bd2_y,pop1_peaks,pop2_peaks,phase1_y,phase2_y=analyze_output(mnp1_input,mnp2_input,'MNP',mnp_bd_y_line,mnp_phase_y_line,min_dist_phase_calc)

print(print_data_array_bd+print_data_array_phase)

# Plot the MNP data
t = np.arange(0,len(mnp1_data),1)
plt.figure(figsize=(8, 6))
plt.subplot(211)
plt.plot(mnp1_data,label='MNP1')
plt.plot(mnp2_data,label='MNP2')
plt.plot(up_bd1,up_bd1_y,'x',markersize=10,color='r')
plt.plot(up_bd2,up_bd2_y,'x',markersize=10,color='g')
plt.plot(down_bd1,down_bd1_y,'x',markersize=10,color='r')
plt.plot(down_bd2,down_bd2_y,'x',markersize=10,color='g')
plt.axhline(y=mnp_bd_y_line, color='r', linestyle='--', label=f'y_threshold')
plt.ylabel('# of Spikes')
plt.legend()
plt.title(f'Plot of MNP (BD)')
plt.grid(True)  
plt.subplot(212)
plt.plot(mnp1_data,label='MNP1')
plt.plot(mnp2_data,label='MNP2')
plt.plot(pop1_peaks,phase1_y,'x',markersize=10)
plt.plot(pop2_peaks,phase2_y,'x',markersize=10)
plt.axhline(y=mnp_phase_y_line, color='r', linestyle='--', label=f'y_threshold')
plt.xlabel('Steps (1000*Time (ms))')
plt.ylabel('# of Spikes')
plt.legend()
plt.title(f'Plot of MNP (Phase)')
plt.grid(True)  

plt.show()
