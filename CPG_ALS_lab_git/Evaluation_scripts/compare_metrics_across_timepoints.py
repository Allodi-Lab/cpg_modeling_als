import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import shapiro
from scipy.stats import ttest_ind, kruskal, ttest_rel, wilcoxon, mannwhitneyu
from scipy import stats
import scikit_posthocs as sp
import seaborn as sns
import sys
import plotly.graph_objects as go
import plotly.subplots
import plotly.io as pio
from calculate_stability_metrics_all_trials import find_csv_files, analyze_output
import warnings
import re

# Suppress all warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
plt.rcParams['svg.fonttype'] = 'none'

folder_containing_data = sys.argv[1]
compare_trials = sys.argv[2]
if len(sys.argv)>3:
    second_folder_containing_data = sys.argv[2]
    compare_trials = sys.argv[3]
no_intervention_data_folder = '1_Random_trials_no_intervention' #Update the name of this folder based on your local path
slow_syn_data_folder = '6_Random_trials_inc_syn_dyn'
#Set test parameters
total_trials = 25                   #Update this based on the number of random seeds 
drive_for_single_timepoint = 'd1'   #Options: 'd1', 'd2', 'both'; Only applies for single timepoint trials
save_motor_neurons = 0              #Use 1 if analyzing a P112 trial that saves the MNs
compare_to_healthy = 1              #Use 1 to compare a single timepoint trial to the healthy control and no intervention state
compare_slow_dyn = 0                #Use 1 to compare P45 slow synaptic dynamics trials to healthy control and no intervention with slow synaptic dynamics
compare_desc_drive = 0              #Use 1 to compare healthy descending drive trials, note, two folders must be specified in the arguments
save_as_svg = 1                     #Use 1 to save as SVG, 0 will save figures as PNG

#Set parameters for figures
title_fontsize = 24
axis_label_fontsize = 20
axis_line_thickness = 2
label_mapping = {
        "1_": "",
        "2_": " T2",
        "3_": " T3",
        "4_": " T4",
        "5_": " T5",
        "6_": " T6",
        "7_": " T7",
        "8_": " T8",
        "9_": " T9",
        "10_": " T10",
        "11_": " T11",
        "12_": " T12",
        "13_": " T13"
    }

def remove_outliers(trial_type,data_array):
    """
    Removes rows from data_array where any column value is more than 3 standard deviations
    away from the mean for that column.

    Parameters:
    - data_array: A 2D list or numpy array to process.

    Returns:
    - A filtered numpy array with outliers removed.
    """
    
    data_array = np.array(data_array)  # Ensure input is a numpy array  
    data_array = data_array[:, :-1].astype(float)
    data_array = data_array[~np.isnan(data_array).any(axis=1)] # Remove rows with NaN values 
    mean = np.nanmean(data_array, axis=0)  # Mean for each column
    std_dev = np.nanstd(data_array, axis=0)  # Standard deviation for each column
    
    # Compute the mask for rows without outliers
    mask = np.all(np.abs(data_array - mean) <= 3 * std_dev, axis=1)
    viable_trials = np.count_nonzero(mask)
    outlier_trial_count = total_trials-viable_trials
    percentage_outliers=round((outlier_trial_count/total_trials)*100,2)
    #print('Viable trials for '+str(trial_type), viable_trials,'Percentage of outliers = ',percentage_outliers,'%')
    
    # Normality check
    is_normal = all(shapiro(data_array[:, i])[1] > 0.05 for i in range(data_array.shape[1]))
    normality_status = 'Normal' if is_normal else 'Not Normal'
    #print(f'Normality check for {trial_type}: {normality_status}')
    
    # Prepare data for CSV
    filtered_data = data_array[mask]
    row_means = np.nanmean(filtered_data, axis=0)
    row_stds = np.nanstd(filtered_data, axis=0)
    # Round row_means and row_stds to two decimal points
    row_means = np.round(row_means, 2)
    row_stds = np.round(row_stds, 2)
    new_row = [trial_type, viable_trials, percentage_outliers, row_means.tolist(), row_stds.tolist(), normality_status]
    csv_path = folder_containing_data + '/metrics_stats_across_timepoints.csv'

    # Read existing data from the CSV file
    updated_rows = []
    found = False
    try:
        with open(csv_path, mode='r', newline='') as file:
            reader = csv.reader(file)
            for row in reader:
                if row and row[0] == trial_type:
                    updated_rows.append(new_row)  # Replace the existing row
                    found = True
                else:
                    updated_rows.append(row)
    except FileNotFoundError:
        # If the file doesn't exist, create a new one
        pass

    # If trial_type was not found, append the new row
    if not found:
        updated_rows.append(new_row)

    # Write updated data back to the CSV file
    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(updated_rows)

    return data_array[mask]

def run_statistics(extracted_values_dict, folders):
    all_results = {}
    healthy_comparison_results = []
    # Flatten the data for each folder into separate numeric arrays
    groups = [np.array([row for row in extracted_values_dict[folder]], dtype=float) for folder in folders]
    # Ensure all groups have data
    groups = [group for group in groups if len(group) > 0]
    column_labels = ['Avg Max Neuron Firing Rate Flx', 'Avg Max Neuron Firing Rate Ext', 'Avg On-Cycle Neuron Firing Rate Flx', 'Avg On-Cycle Neuron Firing Rate Ext', 'Avg Off-Cycle Neuron Firing Rate Flx', 'Avg Off-Cycle Neuron Firing Rate Ext', 'Freq Flx', 'Freq Ext', 'Burst Duration Flx', 'Burst Duration Ext', 'MNP Phase']
    folder_labels = [folder.split('/')[-1].replace('_', ' ') for folder in folders]
    folder_labels = [label.replace("P0", "Healthy") for label in folder_labels]
    # Perform tests column-wise
    if len(groups) == 2 and compare_to_healthy==0:
        print('Only one timepoint provided, statistical analysis between time points is skipped.')
    elif len(groups) == 2 and compare_to_healthy==1:
        print('Comparing two groups using Wilcoxon signed rank test.')
        total_iterations = groups[0].shape[1]
        for i in range(total_iterations):
            healthy_values = groups[0][:, i]
            disease_timepoint_values = groups[1][:, i]
            stat, p_values = wilcoxon(healthy_values, disease_timepoint_values)
            healthy_comparison_results.append(p_values)
            p_value_formatted = f"{p_values:.2e}"
            print(column_labels[i],'p-value compared to healthy: ',p_value_formatted)
    elif len(groups) > 2:
        print('Comparing more than two groups using Kruskal-Wallis test with Dunn posthoc.')
        # Kruskal-Wallis for each column
        p_values = [kruskal(*[group[:, i] for group in groups]).pvalue for i in range(groups[0].shape[1])]
        p_values_formatted = [f"{p:.2e}" for p in p_values]
        
        # Perform Dunn's test for each column to identify which pairs of groups are different        
        for i in range(groups[0].shape[1]):
            # Prepare the data for Dunn's test
            data_for_dunn = [group[:, i] for group in groups]
            dunn_results = sp.posthoc_dunn(data_for_dunn, p_adjust='bonferroni')
            dunn_results = dunn_results.map(lambda x: f"{x:.2e}")
            # Update the Dunn's test result DataFrame to have readable labels
            dunn_results.index = folder_labels
            dunn_results.columns = folder_labels
            # Store the results in the dictionary
            all_results[column_labels[i]] = dunn_results
            
        csv_path = folder_containing_data + '/statistical_comparison_'+compare_trials+'.csv'    
        with open(csv_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            for test_name, df in all_results.items():
                # Write the test name as a header
                writer.writerow([f"Dunn's test results for {test_name}"])
                # Write the DataFrame column labels
                writer.writerow([''] + list(df.columns))
                # Write each row of the DataFrame
                for index, row in df.iterrows():
                    writer.writerow([index] + row.tolist())
                # Add a blank line to separate results
                writer.writerow([])  
    else:
        print("Not enough timepoints for statistical comparison across timepoints.")
    return all_results, healthy_comparison_results

def run_statistics_flx_ext(extracted_values_dict, folders):
    all_results = []
    results_to_save = []
    # Flatten the data for each folder into separate numeric arrays
    groups = [np.array([row for row in extracted_values_dict[folder]], dtype=float) for folder in folders]
    # Ensure all groups have data
    groups = [group for group in groups if len(group) > 0]
    column_labels = ['Avg Max Neuron Firing Rate Flx', 'Avg Max Neuron Firing Rate Ext', 'Avg On-Cycle Neuron Firing Rate Flx', 'Avg On-Cycle Neuron Firing Rate Ext', 'Avg Off-Cycle Neuron Firing Rate Flx', 'Avg Off-Cycle Neuron Firing Rate Ext', 'Freq Flx', 'Freq Ext', 'Burst Duration Flx', 'Burst Duration Ext', 'MNP Phase']
    folder_labels = [folder.split('/')[-1].replace('_', ' ') for folder in folders]
    folder_labels = [label.replace("P0", "Healthy") for label in folder_labels]
    
    # Wilcoxon Signed-Rank Test for each flx/ext data pair
    total_iterations = groups[0].shape[1]-1
    for j in range(len(folders)):
        for i in range(0, total_iterations, 2):
            flx_values = groups[j][:, i]
            ext_values = groups[j][:, i+1]
            stat, p_values = wilcoxon(flx_values, ext_values)               
            all_results.append(p_values)
            p_value_formatted = f"{p_values:.2e}"
            results_to_save.append({
                "Folder": folder_labels[j],
                "Metric": column_labels[i],
                "Wilcoxon_stat": stat,
                "P_value": p_value_formatted
            })
    
    csv_path_flx_ext = folder_containing_data + '/statistical_comparison_flx_ext_'+compare_trials+'.csv'    
    # Save results to CSV
    with open(csv_path_flx_ext, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write header
        writer.writerow(["Folder", "Metric", "Wilcoxon_stat", "P_value"])
        # Write each row of results
        for result in results_to_save:
            writer.writerow([result["Folder"], result["Metric"], result["Wilcoxon_stat"], result["P_value"]])
  
    return all_results

def annotate_significance(ax, cv_data_flx, cv_data_ext, xtick_positions, metric_indices, significance_results, significance_levels, pairs_to_compare, metric_ymax):
    """
    Adds significance annotations to the plot.
    """
    #Data for significance annotations
    if len(folders) == 2:
        metrics = significance_results
    elif len(folders) > 2:    
        metrics = list(significance_results.keys())
    
    y_max_flx = max(max(group, default=0) for group in cv_data_flx)
    y_max_ext = max(max(group, default=0) for group in cv_data_ext)
    buffer = 0.2 * metric_ymax #max(y_max_flx, y_max_ext)
    global_y_max = metric_ymax + buffer #max(y_max_flx + buffer, y_max_ext + buffer)
    ax.set_ylim([0,global_y_max + buffer]) #set_ylim(top=global_y_max + buffer)
    
    metric_flx = metrics[metric_indices[0]]
    metric_ext = metrics[metric_indices[1]]
    for (i, j) in pairs_to_compare:
        if len(folders) == 2:
            p_value_flx = metric_flx
            p_value_ext = metric_ext
        elif len(folders) > 2: 
            p_value_flx = float(significance_results[metric_flx].iloc[i, j])
            p_value_ext = float(significance_results[metric_ext].iloc[i, j])
       
        # Flexor significance
        significance_flx = next((stars for threshold, stars in significance_levels.items() if p_value_flx <= float(threshold)), 'ns')
        if significance_flx:
            x1, x2 = xtick_positions[i * 2], xtick_positions[j * 2]
            ax.plot([x1, x1, x2, x2], [global_y_max, global_y_max + 0.05, global_y_max + 0.05, global_y_max], color='black')
            ax.vlines(x1, global_y_max-(global_y_max*.05), global_y_max, color='blue', linewidth=2)
            ax.vlines(x2, global_y_max-(global_y_max*.05), global_y_max, color='blue', linewidth=2)
            ax.text(x2, global_y_max + 0.1, significance_flx, ha='center', va='bottom', color='blue')

        # Extensor significance
        significance_ext = next((stars for threshold, stars in significance_levels.items() if p_value_ext <= float(threshold)), 'ns')
        if significance_ext:
            x1, x2 = xtick_positions[i * 2 + 1], xtick_positions[j * 2 + 1]
            ax.plot([x1, x1, x2, x2], [global_y_max, global_y_max + 0.05, global_y_max + 0.05, global_y_max], color='black')
            ax.vlines(x1, global_y_max-(global_y_max*.05), global_y_max, color='orange', linewidth=2)
            ax.vlines(x2, global_y_max-(global_y_max*.05), global_y_max, color='orange', linewidth=2)
            ax.text(x2, global_y_max + 0.1, significance_ext, ha='center', va='bottom', color='orange')

def annotate_phase_significance(ax, cv_data, xtick_positions, metric_indices, significance_results, significance_levels, pairs_to_compare, metric_ymax):
    """
    Adds significance annotations to the plot.
    """
    #### Annotate with significance ####
    #Data for significance annotations
    if len(folders) == 2:
        metrics = significance_results
    elif len(folders) > 2:    
        metrics = list(significance_results.keys())
    if len(folders) > 2 or compare_to_healthy==1:    
        # Find max values for placement of significance annotations
        y_max = 190 #max(max(group, default=0) for group in cv_data)
        buffer = 0.2 * y_max  # 20% buffer of the max y-value
        y_max += buffer
        # Set y-axis limit to accommodate significance annotations
        ax.set_ylim([0,y_max + buffer])

        phase_metric = metrics[10]
        for (i, j) in pairs_to_compare:
            # Significance
            if len(folders) == 2:
                p_value = phase_metric
            elif len(folders) > 2: 
                p_value = float(significance_results[phase_metric].iloc[i, j])
            significance = next((stars for threshold, stars in significance_levels.items() if p_value <= float(threshold)), 'ns')
            if significance:
                #x1, x2 = xtick_positions[i * 2], xtick_positions[j * 2]  # Positions
                x1, x2 = xtick_positions[i], xtick_positions[j]
                y, h = y_max, 0.05
                # Plot horizontal significance line, ticks along line and significance annotation
                ax.vlines(x1, y_max-(y_max*.05), y_max, color='black', linewidth=2)
                ax.vlines(x2, y_max-(y_max*.05), y_max, color='black', linewidth=2)
                ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], color='black')
                ax.text(x2, y + h, significance, ha='center', va='bottom', color='black') 

def annotate_significance_flx_ext(ax, cv_data_flx, cv_data_ext, xtick_positions, metric_indices, significance_results, significance_levels, pairs_to_compare, metric_ymax):
    """
    Adds significance annotations to compare each flexor/extensor pair at each time point.
    
    Parameters:
        ax (numpy.ndarray): 2D array of matplotlib.axes corresponding to subplots.
        cv_data_flx (list): List of flexor group data for each metric.
        cv_data_ext (list): List of extensor group data for each metric.
        xtick_positions (list): List of x-axis positions for each time point.
        metric_indices (list): List of metric indices for each subplot position.
        significance_results (list): List of p-values grouped by [metric_0_time_0, metric_1_time_0, ...].
        significance_levels (dict): Dictionary of significance thresholds and star annotations.
        pairs_to_compare (list): List of pairs of indices indicating which groups to compare.
    """
    # Determine overall plot limits and buffer
    y_max_flx = max(max(group, default=0) for group in cv_data_flx)
    y_max_ext = max(max(group, default=0) for group in cv_data_ext)
    buffer = 0.2 * metric_ymax #max(y_max_flx, y_max_ext)
    global_y_max = metric_ymax + buffer #max(y_max_flx + buffer, y_max_ext + buffer)
    ax.set_ylim([0,global_y_max + buffer])
    
    # Iterate through time points for the subplot
    data_labels = ['Avg Max Neuron Firing Rate', 'Avg On-Cycle Neuron Firing Rate', 'Avg Off-Cycle Neuron Firing Rate', 'Freq', 'Burst Duration']
    if metric_indices == [0, 1]:
        offset = 0
    elif metric_indices == [2, 3]:
        offset = 1
    elif metric_indices == [4, 5]:
        offset = 2
    elif metric_indices == [6, 7]:
        offset = 3 
    elif metric_indices == [8, 9]:
        offset = 4     
    else:
        print('Metric indices are not recognized.')
    data_type = data_labels[offset]
    num_time_points = len(cv_data_flx)
    for time_idx in range(num_time_points):
        # Determine the p-value index
        index = time_idx*5+offset
        p_value = significance_results[index]
        print(f"{data_type} p-value flx vs ext, {p_value:.2e}")

        # Determine significance annotation (stars or 'ns')
        significance = next((stars for threshold, stars in significance_levels.items() if p_value <= float(threshold)), 'ns')

        if significance:
            # Define x positions for the flexor and extensor pair
            x1 = xtick_positions[time_idx * 2]       # Flexor x-position
            x2 = xtick_positions[time_idx * 2 + 1]   # Extensor x-position

            # Define the y position for the significance annotation
            y = global_y_max + (0.05 * global_y_max)  # Slightly above the global max

            # Draw annotation lines and text
            ax.plot([x1, x1, x2, x2], [y, y + 0.05, y + 0.05, y], color='black')
            ax.text((x1 + x2) / 2, y + 0.07, significance, ha='center', va='bottom', color='black')

            # Highlight with vertical lines
            ax.vlines(x1, y - (0.05 * global_y_max), y, color='blue', linewidth=2)
            ax.vlines(x2, y - (0.05 * global_y_max), y, color='orange', linewidth=2)
    
    return  
                
def plot_metric(ax, title, ylabel, metric_ymax, metric_indices, folders, extracted_values_dict, scatter_color, group_spacing, pair_offset, significance_results, significance_levels, pairs_to_compare, annotation_type):
    """
    Generic function to plot a single metric pair (Flx and Ext) as boxplots with scatter overlays and significance annotations.
    """
    # Data preparation
    cv_data_flx = []
    cv_data_ext = []
    xticks_combined = []
    subplot_suffix = ""

    for folder in folders:
        folder_flx_data = []
        folder_ext_data = []
        for trial_data in extracted_values_dict[folder]:
            if trial_data[metric_indices[0]] is not None:  # Flx data
                folder_flx_data.append(trial_data[metric_indices[0]])
            if trial_data[metric_indices[1]] is not None:  # Ext data
                folder_ext_data.append(trial_data[metric_indices[1]])
        cv_data_flx.append(folder_flx_data)
        cv_data_ext.append(folder_ext_data)
        test_type = folder.split('/')[0]
        
        folder_name = folder.split('/')[-1].replace("_", " ")
        xtick_label = folder_name.replace("D1", "").replace("D2", "").strip()
        for substring, mapped_label in sorted(label_mapping.items(), key=lambda x: -len(x[0])):
            if substring in test_type:
                xtick_label = xtick_label+mapped_label
                break    
        xticks_combined.extend([f"{xtick_label} (Flx)", f"{xtick_label} (Ext)"])
    xticks_combined = [label.replace("P0", "Healthy") for label in xticks_combined]    

    # Combine data for plotting
    cv_data = []
    xtick_positions = []
    x_pos = 1
    for flx, ext in zip(cv_data_flx, cv_data_ext):
        cv_data.append(flx)
        cv_data.append(ext)
        xtick_positions.extend([x_pos - pair_offset, x_pos + pair_offset])
        x_pos += group_spacing

    # Boxplot and scatter plot
    ax.boxplot(cv_data, positions=np.array(xtick_positions), patch_artist=True, boxprops=dict(facecolor='gray'), medianprops=dict(color='black'))
    for i, (data, pos) in enumerate(zip(cv_data, xtick_positions)):
        color = scatter_color[i % 2]
        ax.scatter(np.ones_like(data) * pos, data, color=color, alpha=0.7, edgecolors='black', zorder=3)

    # Set axis labels and titles
    ax.set_title(title, fontsize=title_fontsize)
    ax.set_ylabel(ylabel, fontsize=axis_label_fontsize)
    ax.set_xticks(xtick_positions)
    ax.set_xticklabels(xticks_combined, rotation=45, ha='right', fontsize=axis_label_fontsize)
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(axis_line_thickness)
        
    # Annotate with significance
    if (annotation_type == 'timepoint' and len(folders) > 2) or (annotation_type == 'timepoint' and compare_to_healthy==1):
        annotate_significance(ax, cv_data_flx, cv_data_ext, xtick_positions, metric_indices, significance_results, significance_levels, pairs_to_compare, metric_ymax)  
    elif annotation_type == 'flx_ext':
        annotate_significance_flx_ext(ax, cv_data_flx, cv_data_ext, xtick_positions, metric_indices, significance_results, significance_levels, pairs_to_compare, metric_ymax)
        

def plot_phase_metric(ax, title, ylabel, metric_ymax, metric_indices, folders, extracted_values_dict, scatter_color, group_spacing, pair_offset, significance_results, significance_levels, pairs_to_compare, annotation_type):
    """
    Generic function to plot a single metric pair (Flx and Ext) as boxplots with scatter overlays and significance annotations.
    """
    cv_data = []
    xtick_positions = [] 
    x_pos = 1  # Initial x-position

    for folder in folders:
        # Collect data for all trials in the current folder
        folder_data = []

        for trial_data in extracted_values_dict[folder]:
            # Extract the specific Flx and Ext metrics for each trial
            #folder_data.append(trial_data[6])  # Phase
            if trial_data[10] is not None:  # Phase
                folder_data.append(trial_data[10])

        # Append all trial data for this folder as a single list (grouped)
        cv_data.append(folder_data)
    
    #Format subplot titles and xtick labels
    xticks_combined = []
    for folder in folders:
        test_type = folder.split('/')[0]
        folder_name = folder.split('/')[-1].replace("_", " ") 
        xtick_label = folder_name.replace("D1", "").replace("D2", "").strip()
        subplot_suffix = ""
        for substring, mapped_label in sorted(label_mapping.items(), key=lambda x: -len(x[0])):
            if substring in test_type:
                xtick_label = xtick_label+mapped_label
                break    
        xticks_combined.append(f"{xtick_label}")  
    xticks_combined = [label.replace("P0", "Healthy") for label in xticks_combined]
    
    for j in range(len(cv_data)):
        xtick_positions.extend([x_pos])
        x_pos += group_spacing  # Increment for next group spacing
    
    # Create boxplots first
    ax.boxplot(cv_data, positions=np.array(xtick_positions), patch_artist=True, boxprops=dict(facecolor='gray'), medianprops=dict(color='black'))
    #title_suffix = f" ({subplot_suffix})" if subplot_suffix else ""
    ax.set_title(title, fontsize=title_fontsize)
    buffer = 0.2 * metric_ymax #max(y_max_flx, y_max_ext)
    global_y_max = metric_ymax + buffer #max(y_max_flx + buffer, y_max_ext + buffer)
    ax.set_ylim([0,global_y_max + buffer])
    ax.set_ylabel(ylabel, fontsize=axis_label_fontsize)
    ax.set_xticks(xtick_positions)
    ax.set_xticklabels(xticks_combined, rotation=45, ha='right', fontsize=axis_label_fontsize)
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(axis_line_thickness)

    # Overlay scatter plot for raw data (dispersion points) on top of the boxplot
    for i, (data, pos) in enumerate(zip(cv_data, xtick_positions)):
        ax.scatter(np.ones_like(data) * pos, data, color=scatter_color[2], alpha=0.7, edgecolors='black', zorder=3)
    
    # Annotate with significance
    if (annotation_type == 'timepoint' and len(folders) > 2) or (annotation_type == 'timepoint' and compare_to_healthy==1):
        annotate_phase_significance(ax, cv_data, xtick_positions, metric_indices, significance_results, significance_levels, pairs_to_compare, metric_ymax) 
    

def plot_comparison_cv_with_dispersion(folders, extracted_values_dict, significance_results):
    """
    Optimized function to plot CV comparisons with dispersion points for multiple metrics.
    """
    #fig, axs = plt.subplots(2, 3, figsize=(24, 18))
    scatter_color = ['blue', 'orange', 'white'] #'purple'
    group_spacing = 1.7
    pair_offset = 0.3
    significance_levels = {'0.001': '***', '0.01': '**', '0.05': '*'}
    pairs_to_compare = [(0, i) for i in range(1, len(folders))] 
    
    #Format subplot titles and xtick labels
    xticks_combined = []
    
    for folder in folders:
        test_type = folder.split('/')[0]
        folder_name = folder.split('/')[-1].replace("_", " ") 
        subplot_suffix = ""
        xtick_label = folder_name.replace("D1", "").replace("D2", "").strip()
        for substring, mapped_label in sorted(label_mapping.items(), key=lambda x: -len(x[0])):
            if substring in test_type:
                xtick_label = xtick_label+mapped_label
                break
        xticks_combined.append(f"{xtick_label}")    
    title_suffix = f" ({subplot_suffix})" if subplot_suffix else ""
    
    # Define metrics and their subplot positions
    metrics = [
        {"title": "Avg Max Neuron Firing Rate"+title_suffix, "ylabel": "Neuron Firing Rate", "metric_ymax": 150, "metric_indices": [0, 1], "subplot_pos": (0, 0)},
        {"title": "Avg On-Cycle Neuron Firing Rate"+title_suffix, "ylabel": "Neuron Firing Rate", "metric_ymax": 150, "metric_indices": [2, 3], "subplot_pos": (0, 1)},
        {"title": "Avg Off-Cycle Neuron Firing Rate"+title_suffix, "ylabel": "Neuron Firing Rate", "metric_ymax": 150, "metric_indices": [4, 5], "subplot_pos": (0, 2)},
        {"title": "Frequency"+title_suffix, "ylabel": "Freq (Hz)", "metric_ymax": 3.5, "metric_indices": [6, 7], "subplot_pos": (1, 0)},
        {"title": "Burst Duration"+title_suffix, "ylabel": "Time (ms)", "metric_ymax": 350, "metric_indices": [8, 9], "subplot_pos": (2, 0)},
    ]
    
    annotation_type = 'timepoint'
    fig, axes = plt.subplots(2, 3, figsize=(24, 18))
    axes = axes.flatten()  # Flatten the 2D array to simplify indexing
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        plot_metric(
            ax=ax,
            title=metric["title"],
            ylabel=metric["ylabel"],
            metric_ymax=metric["metric_ymax"],
            metric_indices=metric["metric_indices"],
            folders=folders,
            extracted_values_dict=extracted_values_dict,
            scatter_color=scatter_color,
            group_spacing=group_spacing,
            pair_offset=pair_offset,
            significance_results=significance_results,
            significance_levels=significance_levels,
            pairs_to_compare=pairs_to_compare,
            annotation_type=annotation_type
        )
    ax = axes[5]
    plot_phase_metric(ax=ax,title="Phase"+title_suffix,ylabel="Phase (deg)",metric_ymax=200,metric_indices=[10], folders=folders,extracted_values_dict=extracted_values_dict,scatter_color=scatter_color,group_spacing=group_spacing, pair_offset=pair_offset,significance_results=significance_results,significance_levels=significance_levels, pairs_to_compare=pairs_to_compare,annotation_type=annotation_type)  
    
    plt.tight_layout()
    if save_as_svg == 0:
        plt.savefig(folder_containing_data + '/' + compare_trials + '_metrics.png',bbox_inches="tight")
    if save_as_svg == 1:    
        plt.savefig(folder_containing_data + '/' + compare_trials + '_metrics.svg', format='svg')
        
    annotation_type = 'flx_ext'
    significance_results_flx_ext = run_statistics_flx_ext(extracted_values_dict, folders)
    fig, axes = plt.subplots(2, 3, figsize=(24, 18))
    axes = axes.flatten()  # Flatten the 2D array to simplify indexing
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        plot_metric(
            ax=ax,
            title=metric["title"],
            ylabel=metric["ylabel"],
            metric_ymax=metric["metric_ymax"],
            metric_indices=metric["metric_indices"],
            folders=folders,
            extracted_values_dict=extracted_values_dict,
            scatter_color=scatter_color,
            group_spacing=group_spacing,
            pair_offset=pair_offset,
            significance_results=significance_results_flx_ext,
            significance_levels=significance_levels,
            pairs_to_compare=pairs_to_compare,
            annotation_type=annotation_type
        )
    ax = axes[5]
    plot_phase_metric(ax=ax,title="Phase"+title_suffix,ylabel="Phase (deg)",metric_ymax=200,metric_indices=[10], folders=folders,extracted_values_dict=extracted_values_dict,scatter_color=scatter_color,group_spacing=group_spacing, pair_offset=pair_offset,significance_results=significance_results_flx_ext,significance_levels=significance_levels, pairs_to_compare=pairs_to_compare,annotation_type=annotation_type)  
    
    plt.tight_layout()
    if save_as_svg == 0:
        plt.savefig(folder_containing_data + '/' + compare_trials + '_flx_ext_metrics.png',bbox_inches="tight")
    if save_as_svg == 1:    
        plt.savefig(folder_containing_data + '/' + compare_trials + '_flx_ext_metrics.svg', format='svg')

if __name__ == "__main__":
    # List of folders to process
    if compare_trials == 'D1':
        folders = [
            no_intervention_data_folder+'/P0_D1',
            folder_containing_data+'/P45_D1',
            folder_containing_data+'/P63_D1',
            folder_containing_data+'/P112_D1'
        ]
    elif compare_trials == 'D2':
        folders = [
            no_intervention_data_folder+'/P0_D2',
            folder_containing_data+'/P45_D2',
            folder_containing_data+'/P63_D2',
            folder_containing_data+'/P112_D2'
        ]
    elif compare_trials == 'P0':
        if compare_desc_drive == 1:
            folders = [
                no_intervention_data_folder+'/P0_D1',
                folder_containing_data+'/P0_D1',
                second_folder_containing_data+'/P0_D1'
            ]
        elif drive_for_single_timepoint == 'both':
            folders = [
                no_intervention_data_folder+'/P0_D1',
                no_intervention_data_folder+'/P0_D2'
            ]
        elif drive_for_single_timepoint == 'd1':
            folders = [
                no_intervention_data_folder+'/P0_D1',
                folder_containing_data+'/P0_D1'
            ]
        elif drive_for_single_timepoint == 'd2':
            folders = [
                no_intervention_data_folder+'/P0_D2'
            ]
    elif compare_trials == 'P45':
        if drive_for_single_timepoint == 'both':
            folders = [
                folder_containing_data+'/P45_D1',
                folder_containing_data+'/P45_D2'
            ]
        elif drive_for_single_timepoint == 'd1' and compare_to_healthy==1 and compare_slow_dyn==0:
            folders = [
                no_intervention_data_folder+'/P0_D1',
                no_intervention_data_folder+'/P45_D1',
                folder_containing_data+'/P45_D1'
            ]
        elif drive_for_single_timepoint == 'd1' and compare_to_healthy==1 and compare_slow_dyn==1:
            folders = [
                no_intervention_data_folder+'/P0_D1',
                slow_syn_data_folder+'/P45_D1',
                folder_containing_data+'/P45_D1'
            ]    
        elif drive_for_single_timepoint == 'd2' and compare_to_healthy==1:
            folders = [
                no_intervention_data_folder+'/P0_D2',
                no_intervention_data_folder+'/P45_D2',
                folder_containing_data+'/P45_D2'
            ] 
    elif compare_trials == 'P63':
        if drive_for_single_timepoint == 'both':
            folders = [
                folder_containing_data+'/P63_D1',
                folder_containing_data+'/P63_D2'
            ]
        elif drive_for_single_timepoint == 'd1' and compare_to_healthy==1:
            folders = [
                no_intervention_data_folder+'/P0_D1',
                no_intervention_data_folder+'/P63_D1',
                folder_containing_data+'/P63_D1'
            ]    
        elif drive_for_single_timepoint == 'd2' and compare_to_healthy==1:
            folders = [
                no_intervention_data_folder+'/P0_D2',
                no_intervention_data_folder+'/P63_D2',
                folder_containing_data+'/P63_D2'
            ] 
    elif compare_trials == 'P112':
        if drive_for_single_timepoint == 'both':
            folders = [
                folder_containing_data+'/P112_D1',
                folder_containing_data+'/P112_D2'
            ]
        elif drive_for_single_timepoint == 'd1' and compare_to_healthy==1:
            folders = [
                no_intervention_data_folder+'/P0_D1',
                no_intervention_data_folder+'/P112_D1',
                folder_containing_data+'/P112_D1'
            ]    
        elif drive_for_single_timepoint == 'd2' and compare_to_healthy==1:
            folders = [
                no_intervention_data_folder+'/P0_D2',
                no_intervention_data_folder+'/P112_D2',
                folder_containing_data+'/P112_D2'
            ] 

    extracted_values_dict = {}

    mnp_bd_y_line = 0.7
    mnp_phase_y_line = 0.4
    min_dist_phase_calc = 1000

    # Extract values for each folder and store in dictionary
    for folder in folders:
        data_array = []  # Reset data_array for each folder
        file_pairs = find_csv_files(folder)
        zero_min_spike_count_flx = 0
        zero_min_spike_count_ext = 0
        current_timepoint = re.search(r"P(\d+)_", folder)
        num_motor_neurons = 1 if current_timepoint.group(1) == '112' and save_motor_neurons == 0 else 1
        print('Analyzing timepoint',current_timepoint.group(1))
        # Loop through each file pair (output_mnp1.csv, output_mnp2.csv)
        for mnp1_input, mnp2_input, subfolder_name in file_pairs:
            # Analyze the files and append the results as a row in the data array
            data_row = analyze_output(mnp1_input, mnp2_input, 'MNP', mnp_bd_y_line, mnp_phase_y_line, min_dist_phase_calc, num_motor_neurons)
            data_row = np.round(data_row, 4)
            # Raw data: [avg_max_spike_rate_pop1, avg_max_spike_rate_pop2, avg_on_cycle_flx, avg_off_cycle_flx, avg_on_cycle_ext, avg_off_cycle_ext, freq1, freq2, avg_duration_flx, avg_duration_ext, phase]
            selected_values = [data_row[i] for i in [0, 1, 2, 4, 3, 5, 6, 7, 8, 9, 10]]
            selected_values.append(str(subfolder_name))  # Append the subfolder name to the data row
            data_array.append(selected_values)    
    
        # Remove outliers from data_array
        trial_type = folder.split('/')[-1].replace("_", " ")
        data_array = remove_outliers(trial_type,data_array)
        extracted_values_dict[folder] = data_array
        print('Completed analyzing: ', folder)
    
    # Run statistics for annotation
    significance_results, significance_healthy_comparison_results = run_statistics(extracted_values_dict, folders)
    
    # Plot comparison across time points
    plot_comparison_cv_with_dispersion(folders, extracted_values_dict, significance_results) 
