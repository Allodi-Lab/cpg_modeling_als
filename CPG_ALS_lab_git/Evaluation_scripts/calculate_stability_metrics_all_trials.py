#!/usr/bin/env python

import pandas as pd
import matplotlib.pyplot as plt
import sys
import numpy as np
import os
import glob
from scipy.signal import find_peaks
np.set_printoptions(legacy='1.25',suppress=True)

folder_name = sys.argv[1]

plt.rcParams.update({'font.size': 20})
time_resolution = 0.1
data_array = []  # To store rows of print_data_array_bd and print_data_array_phase for each file pair

def analyze_output(input_1, input_2, pop_type, y_line_bd, y_line_phase, min_dist, num_motor_neurons):
    print_data_array_bd = []
    print_data_array_phase = []
    try:
        # Read the CSV file into an array
        pop_data1 = np.loadtxt(input_1, delimiter=',', dtype=float)
        pop_data2 = np.loadtxt(input_2, delimiter=',', dtype=float)

        pop_data1_norm = (pop_data1 - np.min(pop_data1)) / (np.max(pop_data1) - np.min(pop_data1))
        pop_data2_norm = (pop_data2 - np.min(pop_data2)) / (np.max(pop_data2) - np.min(pop_data2))
        
        pop_data1 = [x / num_motor_neurons for x in pop_data1]
        pop_data2 = [x / num_motor_neurons for x in pop_data2]
        
        max_spikes_pop1 = np.max(pop_data1)
        max_spikes_pop2 = np.max(pop_data2)

        min_spikes_pop1 = np.min(pop_data1)
        min_spikes_pop2 = np.min(pop_data2)
        
        avg_max_spike_rate_pop1, avg_max_spike_rate_pop2 = calculate_avg_peak(pop_data1, pop_data2, y_line_phase, min_dist)
        
        avg_spike_rate_pop1 = np.nanmean(pop_data1)
        avg_spike_rate_pop2 = np.nanmean(pop_data2)
        
        #up_bd1, down_bd1, burst_duration1, bd_variance1, coeff_bd_variance1 = calculate_burst_duration(pop_data1_norm, y_line_bd)
        #up_bd2, down_bd2, burst_duration2, bd_variance2, coeff_bd_variance2 = calculate_burst_duration(pop_data2_norm, y_line_bd)
        
        avg_duration_flx, avg_duration_ext, avg_on_cycle_flx, avg_off_cycle_flx, avg_on_cycle_ext, avg_off_cycle_ext, avg_frequency_flx, avg_frequency_ext, avg_phase = calculate_burst_duration_crossings(pop_data1,pop_data2)
        
        #freq_pop1 = calculate_freq(up_bd1)
        #freq_pop2 = calculate_freq(up_bd2)
        #avg_freq = (freq_pop1 + freq_pop2) / 2

        # Calculate phase using peak to peak
        phase_peak, phase_variance_peak, coeff_phase_variance_peak, freq1, freq2, pop1_peaks, pop2_peaks, cv_freq1, cv_freq2 = calculate_peak_to_peak_phase(pop_data1_norm, pop_data2_norm, y_line_phase, min_dist)
        phase_peak = 360 - phase_peak if phase_peak > 180 else phase_peak  # Ensure phase always between 0-180 degrees

        #total_suppression = find_zero_overlap(pop_data1_norm, pop_data2_norm)

        # Calculate CV of amplitude
        #coeff_amp1_var, coeff_amp2_var = calculate_amplitude_cv(pop_data1_norm, pop_data2_norm, y_line_phase, min_dist)

        # Store values in the arrays
        #print_data_array_bd.extend([round(burst_duration1, 2), round(burst_duration2, 2), coeff_bd_variance1, coeff_bd_variance2, avg_max_spike_rate_pop1, avg_max_spike_rate_pop2, coeff_amp1_var, coeff_amp2_var])
        #print_data_array_phase.extend([freq1, freq2, cv_freq1, cv_freq2, round(phase_peak, 2), avg_spike_rate_pop1, avg_spike_rate_pop2])

    except FileNotFoundError as e:
        print(f"File not found: {e}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

    #return avg_max_spike_rate_pop1, avg_max_spike_rate_pop2, avg_on_cycle_flx, avg_off_cycle_flx, avg_on_cycle_ext, avg_off_cycle_ext, avg_frequency_flx, avg_frequency_ext, avg_duration_flx, avg_duration_ext, avg_phase
    #return avg_max_spike_rate_pop1, avg_max_spike_rate_pop2, avg_on_cycle_flx, avg_off_cycle_flx, avg_on_cycle_ext, avg_off_cycle_ext, freq1, freq2, avg_duration_flx, avg_duration_ext, round(phase_peak, 2)
    return avg_max_spike_rate_pop1, avg_max_spike_rate_pop2, avg_on_cycle_flx, avg_off_cycle_flx, avg_on_cycle_ext, avg_off_cycle_ext, freq1, freq2, avg_duration_flx, avg_duration_ext, avg_phase

def find_csv_files(folder):
    """
    Search for all output_mnp1.csv and output_mnp2.csv files within the folder and its subfolders.
    
    :param folder: The root folder to search.
    :return: List of tuples containing pairs of file paths (output_mnp1.csv, output_mnp2.csv).
    """
    # Find all output_mnp1.csv and output_mnp2.csv files
    mnp1_files = glob.glob(os.path.join(folder, '**', 'output_mnp1.csv'), recursive=True)
    mnp2_files = glob.glob(os.path.join(folder, '**', 'output_mnp2.csv'), recursive=True)
    
    # Match the mnp1 and mnp2 files by their folder structure
    file_pairs = []
    for mnp1 in mnp1_files:
        folder1 = os.path.dirname(mnp1)
        relative_path = os.path.relpath(folder1, folder)
        subfolder_name = relative_path.split(os.sep)[0]  # Get the first subfolder directly below the provided folder
        #subfolder_name = os.path.basename(folder1)  # Extract subfolder name
        corresponding_mnp2 = os.path.join(folder1, 'output_mnp2.csv')
        if corresponding_mnp2 in mnp2_files:
            file_pairs.append((mnp1, corresponding_mnp2, subfolder_name))
    
    return file_pairs

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
    coeff_bd_variance=(np.std(burst_duration)/np.nanmean(burst_duration))
    burst_duration=np.nanmean(burst_duration)
    return upward_indices,downward_indices,round(burst_duration,2),round(bd_variance,2),round(coeff_bd_variance,2)

def calculate_burst_duration_crossings(array1, array2):
    array1 = np.array(array1)
    array2 = np.array(array2)
    diff = array1 - array2

    crossings = np.where(np.diff(np.sign(diff)))[0]
    burst_durations = np.diff(crossings) * time_resolution

    active_pop = []
    on_cycle_values_flx = []  
    off_cycle_values_flx = []
    on_cycle_values_ext = []  
    off_cycle_values_ext = []
    burst_by_group = {"Ext": [], "Flx": []}
    flx_midpoints = []
    ext_midpoints = []

    for i in range(len(crossings) - 1):
        idx_start = crossings[i]
        idx_end = crossings[i + 1]
        midpoint = (idx_start + idx_end) // 2  # index-based midpoint

        # Determine which population is leading
        if array1[idx_start] > array2[idx_start]:
            active_pop.append("Ext")
            on_cycle_values_ext.append(np.nanmean(array2[idx_start:idx_end]))
            off_cycle_values_flx.append(np.nanmean(array1[idx_start:idx_end]))
            burst_by_group["Ext"].append(burst_durations[i])
            ext_midpoints.append(midpoint)
        else:
            active_pop.append("Flx")
            on_cycle_values_flx.append(np.nanmean(array1[idx_start:idx_end]))
            off_cycle_values_ext.append(np.nanmean(array2[idx_start:idx_end]))
            burst_by_group["Flx"].append(burst_durations[i])
            flx_midpoints.append(midpoint)

    # Convert midpoints to time
    flx_midpoints = np.array(flx_midpoints) * time_resolution
    ext_midpoints = np.array(ext_midpoints) * time_resolution

    # Frequency from flexor-flexor cycle
    if len(flx_midpoints) > 1:
        periods = np.diff(flx_midpoints)
        frequencies = 1000 / periods
        avg_frequency_flx = round(np.nanmean(frequencies), 2)
    else:
        avg_frequency_flx = np.nan
    
    # Frequency from extensor-extensor cycle
    if len(ext_midpoints) > 1:
        periods = np.diff(ext_midpoints)
        frequencies = 1000 / periods
        avg_frequency_ext = round(np.nanmean(frequencies), 2)
    else:
        avg_frequency_ext = np.nan

    # Phase of extensor relative to flexor
    phases = []
    for ext_mid in ext_midpoints:
        prev_flx_idx = np.searchsorted(flx_midpoints, ext_mid) - 1
        if 0 <= prev_flx_idx < len(flx_midpoints) - 1:
            start = flx_midpoints[prev_flx_idx]
            end = flx_midpoints[prev_flx_idx + 1]
            phase = (ext_mid - start) / (end - start)
            phase_deg = phase * 360 #Convert to degrees
            if phase_deg > 180:
                phase_deg = 360 - phase_deg #Ensure max is 180 degrees
            phases.append(phase_deg)

    avg_phase = round(np.nanmean(phases), 2) if phases else np.nan

    # Duration-weighted average firing rates
    on_cycle_values_flx = np.array(on_cycle_values_flx)
    burst_durations_flx = np.array(burst_by_group["Flx"])
    on_cycle_values_ext = np.array(on_cycle_values_ext)
    burst_durations_ext = np.array(burst_by_group["Ext"])

    avg_on_cycle_flx = round(np.nansum(on_cycle_values_flx * burst_durations_flx) / np.nansum(burst_durations_flx), 2) if np.nansum(burst_durations_flx) > 0 else np.nan
    avg_on_cycle_ext = round(np.nansum(on_cycle_values_ext * burst_durations_ext) / np.nansum(burst_durations_ext), 2) if np.nansum(burst_durations_ext) > 0 else np.nan

    avg_off_cycle_flx = round(np.nanmean(off_cycle_values_flx), 2)
    avg_off_cycle_ext = round(np.nanmean(off_cycle_values_ext), 2)
    avg_duration_flx = round(np.nanmean(burst_durations_flx), 2)
    avg_duration_ext = round(np.nanmean(burst_durations_ext), 2)

    #print(f" Avg on-cycle Flx: {avg_on_cycle_flx}, Off-cycle: {avg_off_cycle_flx}")
    #print(f" Avg on-cycle Ext: {avg_on_cycle_ext}, Off-cycle: {avg_off_cycle_ext}")
    #print(f" Avg burst duration Flx, Ext: {avg_duration_flx}, {avg_duration_ext}")
    #print(f" Avg frequency (flx, ext): {avg_frequency_flx} Hz {avg_frequency_ext} Hz, Avg phase: {avg_phase} (Ext rel. to Flx)")

    return avg_duration_flx, avg_duration_ext, avg_on_cycle_flx, avg_off_cycle_flx, avg_on_cycle_ext, avg_off_cycle_ext, avg_frequency_flx, avg_frequency_ext, avg_phase


def calculate_peak_to_peak_phase(spike_bins1, spike_bins2, min_peak_height, min_dist):
    pop1_peaks, pop1_properties = find_peaks(spike_bins1, height=min_peak_height, distance=min_dist, prominence=0.1)
    pop2_peaks, pop2_properties = find_peaks(spike_bins2, height=min_peak_height, distance=min_dist, prominence=0.1)

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
    
    period1 = np.nanmean(np.diff(alternating_peaks1)) * time_resolution
    period2 = np.nanmean(np.diff(alternating_peaks2)) * time_resolution

    # Calculate frequency for period1 and period2
    freq1 = 1000 / period1
    freq2 = 1000 / period2

    # Calculate the standard deviation of periods
    std_period1 = np.std(np.diff(alternating_peaks1)) * time_resolution
    std_period2 = np.std(np.diff(alternating_peaks2)) * time_resolution

    # Coefficient of Variation (CV) for periods
    cv_freq1 = std_period1 / period1
    cv_freq2 = std_period2 / period2
    
    avg_period = (period1 + period2) / 2
    avg_freq = 1000 / avg_period

    phase = (avg_period - (time_diff * time_resolution)) / avg_period
    phase_in_deg = phase * 360
    phase_variance = np.var(phase_in_deg)
    coeff_phase_var = (np.std(phase_in_deg) / np.nanmean(phase_in_deg)) 
    avg_phase_in_deg = np.nanmean(phase * 360)
    
    # Normalize the phase to the range [0, 360)
    avg_phase_in_deg = (avg_phase_in_deg + 360) % 360

    return round(avg_phase_in_deg, 2), round(phase_variance, 2), round(coeff_phase_var, 2), freq1, freq2, alternating_peaks1, alternating_peaks2, cv_freq1, cv_freq2

def calculate_avg_peak(spike_bins1, spike_bins2, min_peak_height, min_dist):
    adjusted_min_peak_height_pop1 = np.max(spike_bins1)*min_peak_height
    adjusted_min_peak_height_pop2 = np.max(spike_bins2)*min_peak_height
    
    pop1_peaks, pop1_properties = find_peaks(spike_bins1, height=adjusted_min_peak_height_pop1, distance=min_dist, prominence=0.1)
    pop2_peaks, pop2_properties = find_peaks(spike_bins2, height=adjusted_min_peak_height_pop2, distance=min_dist, prominence=0.1)
    
    #pop1_peaks, pop1_properties = find_peaks(spike_bins1, height=min_peak_height, distance=min_dist, prominence=0.1)
    #pop2_peaks, pop2_properties = find_peaks(spike_bins2, height=min_peak_height, distance=min_dist, prominence=0.1)

    # Extract peak heights
    pop1_heights = pop1_properties['peak_heights']
    pop2_heights = pop2_properties['peak_heights']
    
    avg_peak_height_pop1 = np.nanmean(pop1_heights)
    avg_peak_height_pop2 = np.nanmean(pop2_heights)
    
    return round(avg_peak_height_pop1,4), round(avg_peak_height_pop2,4)


def calculate_amplitude_cv(spike_bins1, spike_bins2, min_peak_height, min_dist):
    # Extract peaks and their properties
    pop1_peaks, pop1_properties = find_peaks(spike_bins1, height=min_peak_height, distance=min_dist, prominence=0.1)
    pop2_peaks, pop2_properties = find_peaks(spike_bins2, height=min_peak_height, distance=min_dist, prominence=0.1)

    # Extract peak heights
    pop1_heights = pop1_properties['peak_heights']
    pop2_heights = pop2_properties['peak_heights']

    # Calculate the variance and coefficient of variation for amplitude
    amp1_variance = np.var(pop1_heights)
    amp2_variance = np.var(pop2_heights)
    coeff_amp1_var = (np.std(pop1_heights) / np.nanmean(pop1_heights))
    coeff_amp2_var = (np.std(pop2_heights) / np.nanmean(pop2_heights))

    #print('Amplitude CV (1,2)', coeff_amp1_var, coeff_amp2_var)
    
    return round(coeff_amp1_var,4), round(coeff_amp2_var,4)

def calculate_freq(arr):
    period = np.nanmean(np.diff(arr)) * time_resolution
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

def plot_cv_boxplots_with_points(data_array_np):
    """
    This function creates 4 subplots of box and whisker plots for the coefficient of variation (CV)
    of the specified variables and overlays individual data points on top.
    
    Subplot grouping:
    - Subplot 1: Frequency (freq_pop1, freq_pop2)
    - Subplot 2: Burst duration (burst_duration1, burst_duration2)
    - Subplot 3: Amplitude/Pop heights (pop1_heights, pop2_heights)
    - Subplot 4: Phase (phase_peak)
    
    :param data_array_np: Numpy array containing data for plotting.
                          Assumed format: [subfolder, burst_duration1, burst_duration2, cv_bd1, cv_bd2, cv_amp1, cv_amp2, avg_freq_peak, phase_peak, cv_phase]
    """
    # Extract the columns for plotting
    freq_pop1 = data_array_np[:, 8]  # avg_freq_peak (used for both freq_pop1 and freq_pop2)
    freq_pop2 = data_array_np[:, 9]
    burst_duration1 = data_array_np[:, 3]  # burst_duration1
    burst_duration2 = data_array_np[:, 4]  # burst_duration2
    pop1_heights = data_array_np[:, 5]  # CV of amplitude for pop1 (cv_amp1)
    pop2_heights = data_array_np[:, 6]  # CV of amplitude for pop2 (cv_amp2)
    phase_peak = data_array_np[:, 11]  # Phase peak (phase_peak)

    # Create a figure with 2x2 subplots
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))  # Adjust figure size as needed
    scatter_color = 'red'  # Color of the scatter points

    # Subplot 1: Frequency (freq_pop1, freq_pop2)
    data_for_plot_1 = [freq_pop1, freq_pop2]
    x_labels_1 = ['Freq Pop1', 'Freq Pop2']
    axs[0, 0].boxplot(data_for_plot_1, patch_artist=True, notch=False)
    for i, data in enumerate(data_for_plot_1):
        axs[0, 0].scatter(np.ones_like(data) * (i + 1), data, color=scatter_color, alpha=0.6, edgecolors='black', zorder=3)
    axs[0, 0].set_xticks(range(1, len(x_labels_1) + 1))
    axs[0, 0].set_xticklabels(x_labels_1, rotation=45, ha='right')
    axs[0, 0].set_ylabel('Coefficient of Variation (CV)')
    axs[0, 0].set_title('CV of Frequency')

    # Subplot 2: Burst duration (burst_duration1, burst_duration2)
    data_for_plot_2 = [burst_duration1, burst_duration2]
    x_labels_2 = ['Burst Duration1', 'Burst Duration2']
    axs[0, 1].boxplot(data_for_plot_2, patch_artist=True, notch=False)
    for i, data in enumerate(data_for_plot_2):
        axs[0, 1].scatter(np.ones_like(data) * (i + 1), data, color=scatter_color, alpha=0.6, edgecolors='black', zorder=3)
    axs[0, 1].set_xticks(range(1, len(x_labels_2) + 1))
    axs[0, 1].set_xticklabels(x_labels_2, rotation=45, ha='right')
    axs[0, 1].set_ylabel('Coefficient of Variation (CV)')
    axs[0, 1].set_title('CV of Burst Duration')

    # Subplot 3: Amplitude/Pop heights (pop1_heights, pop2_heights)
    data_for_plot_3 = [pop1_heights, pop2_heights]
    x_labels_3 = ['Pop1 Heights', 'Pop2 Heights']
    axs[1, 0].boxplot(data_for_plot_3, patch_artist=True, notch=False)
    for i, data in enumerate(data_for_plot_3):
        axs[1, 0].scatter(np.ones_like(data) * (i + 1), data, color=scatter_color, alpha=0.6, edgecolors='black', zorder=3)
    axs[1, 0].set_xticks(range(1, len(x_labels_3) + 1))
    axs[1, 0].set_xticklabels(x_labels_3, rotation=45, ha='right')
    axs[1, 0].set_ylabel('Coefficient of Variation (CV)')
    axs[1, 0].set_title('CV of Amplitude (Pop Heights)')

    # Subplot 4: Phase (phase_peak)
    data_for_plot_4 = [phase_peak]
    x_labels_4 = ['Phase Peak']
    axs[1, 1].boxplot(data_for_plot_4, patch_artist=True, notch=False)
    axs[1, 1].scatter(np.ones_like(phase_peak), phase_peak, color=scatter_color, alpha=0.6, edgecolors='black', zorder=3)
    axs[1, 1].set_xticks(range(1, len(x_labels_4) + 1))
    axs[1, 1].set_xticklabels(x_labels_4, rotation=45, ha='right')
    axs[1, 1].set_ylabel('Coefficient of Variation (CV)')
    axs[1, 1].set_title('CV of Phase')

    # Adjust layout for better spacing
    plt.tight_layout()    

def main():
    mnp_bd_y_line = 0.4
    mnp_phase_y_line = 0.4
    min_dist_phase_calc = 1000

    file_pairs = find_csv_files(folder_name)

    if not file_pairs:
        print("No matching output_mnp1.csv and output_mnp2.csv files found.")
        return

    global data_array  # To store the entire dataset
    
    # Loop through each file pair (output_mnp1.csv, output_mnp2.csv)
    for mnp1_input, mnp2_input, subfolder_name in file_pairs:
        #print(f"Processing files: {mnp1_input}, {mnp2_input}")
        
        # Analyze the files and append the results as a row in the data array
        data_row = analyze_output(mnp1_input, mnp2_input, 'MNP', mnp_bd_y_line, mnp_phase_y_line, min_dist_phase_calc)
        data_row = np.round(data_row,4)
        data_array.append(np.insert(data_row, 0, subfolder_name))
    
    # Sort the array based on the subfolder names (assuming numerical subfolder names)
    data_array_sorted = sorted(data_array, key=lambda x: x[0])  # Convert subfolder names to float for numerical sorting
    data_array_np = np.array(data_array_sorted, dtype=object)
    plot_cv_boxplots_with_points(data_array_np)
    
    # Convert to numpy array for further analysis or saving
    #print("Final 2D data array:")
    #print(data_array_np)
    
    # Save the data array to a CSV file
    output_csv_path = "output_metrics.csv"
    df = pd.DataFrame(data_array_np)
    df.to_csv(output_csv_path, index=False, header=False)
    print(f"Data array saved to {output_csv_path}")
    
    #plt.show()

if __name__ == '__main__':
    main()
