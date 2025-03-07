import os
import csv
import numpy as np
import matplotlib.pyplot as plt
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

# Suppress all warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
plt.rcParams['svg.fonttype'] = 'none'

folder_containing_data = sys.argv[1]
no_intervention_data_folder = '1_Random_trials_no_intervention' #Update the name of this folder based on your local path
compare_trials = sys.argv[2]
#Set test parameters
total_trials = 25                   #Update this based on the number of random seeds 
drive_for_single_timepoint = 'd1'   #Options: 'd1', 'd2', 'both'; Only applies for single timepoint trials
compare_to_healthy = 1              #Use 1 to compare a single timepoint trial to the healthy control
save_as_svg = 0                     #Use 1 to save as SVG, 0 will save figures as PDF

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
        "7_": " T7"
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
    mean = np.mean(data_array, axis=0)  # Mean for each column
    std_dev = np.std(data_array, axis=0)  # Standard deviation for each column
    
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
    row_means = np.mean(filtered_data, axis=0)
    row_stds = np.std(filtered_data, axis=0)
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

def run_statistics_flx_ext(extracted_values_dict, folders):
    all_results = []
    results_to_save = []
    # Flatten the data for each folder into separate numeric arrays
    groups = [np.array([row for row in extracted_values_dict[folder]], dtype=float) for folder in folders]
    # Ensure all groups have data
    groups = [group for group in groups if len(group) > 0]
    column_labels = ['Max APs per bin Flx', 'Max APs per bin Ext', 'Freq Flx', 'Freq Ext', 'Burst Duration Flx', 'Burst Duration Ext', 'MNP Phase']
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

def run_statistics(extracted_values_dict, folders):
    all_results = {}
    healthy_comparison_results = []
    # Flatten the data for each folder into separate numeric arrays
    groups = [np.array([row for row in extracted_values_dict[folder]], dtype=float) for folder in folders]
    # Ensure all groups have data
    groups = [group for group in groups if len(group) > 0]
    column_labels = ['Max APs per bin Flx', 'Max APs per bin Ext', 'Freq Flx', 'Freq Ext', 'Burst Duration Flx', 'Burst Duration Ext', 'MNP Phase']
    folder_labels = [folder.split('/')[-1].replace('_', ' ') for folder in folders]
    folder_labels = [label.replace("P0", "Healthy") for label in folder_labels]
    # Perform tests column-wise
    if len(groups) == 2 and compare_to_healthy==0:
        print('Only one timepoint provided, statistical analysis between time points is skipped.')
    elif len(groups) == 2 and compare_to_healthy==1:
        print('Comparing two groups using Mann Whitney U-test.')
        total_iterations = groups[0].shape[1]
        for i in range(total_iterations):
            healthy_values = groups[0][:, i]
            disease_timepoint_values = groups[1][:, i]
            stat, p_values = mannwhitneyu(healthy_values, disease_timepoint_values, alternative='two-sided')  
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
        if compare_trials == "D1" or compare_trials == "D2":
            xtick_label = folder_name.replace("D1", "").replace("D2", "").strip()
        else:
            xtick_label = folder_name
        for substring, mapped_label in label_mapping.items():
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
    ax.boxplot(cv_data, positions=np.array(xtick_positions), patch_artist=True)
    for data, pos in zip(cv_data, xtick_positions):
        ax.scatter(np.ones_like(data) * pos, data, color=scatter_color, alpha=0.7, edgecolors='black', zorder=3)

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
            if trial_data[6] is not None:  # Phase
                folder_data.append(trial_data[6])

        # Append all trial data for this folder as a single list (grouped)
        cv_data.append(folder_data)
    
    #Format subplot titles and xtick labels
    xticks_combined = []
    for folder in folders:
        test_type = folder.split('/')[0]
        folder_name = folder.split('/')[-1].replace("_", " ") 
        if compare_trials == "D1" or compare_trials == "D2":
            if "D1" in folder_name:
                subplot_suffix = "D1"
            elif "D2" in folder_name:
                subplot_suffix = "D2"
            xtick_label = folder_name.replace("D1", "").replace("D2", "").strip()
        else:
            subplot_suffix = ""
            xtick_label = folder_name
        for substring, mapped_label in label_mapping.items():
            if substring in test_type:
                xtick_label = xtick_label+mapped_label
                break    
        xticks_combined.append(f"{xtick_label}")  
    xticks_combined = [label.replace("P0", "Healthy") for label in xticks_combined]
    
    for j in range(len(cv_data)):
        xtick_positions.extend([x_pos])
        x_pos += group_spacing  # Increment for next group spacing
    
    # Create boxplots first
    ax.boxplot(cv_data, positions=np.array(xtick_positions), patch_artist=True)
    #title_suffix = f" ({subplot_suffix})" if subplot_suffix else ""
    ax.set_title(title, fontsize=title_fontsize)
    ax.set_ylabel(ylabel, fontsize=axis_label_fontsize)
    ax.set_xticks(xtick_positions)
    ax.set_xticklabels(xticks_combined, rotation=45, ha='right', fontsize=axis_label_fontsize)
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(axis_line_thickness)

    # Overlay scatter plot for raw data (dispersion points) on top of the boxplot
    for i, (data, pos) in enumerate(zip(cv_data, xtick_positions)):
        ax.scatter(np.ones_like(data) * pos, data, color=scatter_color, alpha=0.7, edgecolors='black', zorder=3)

    # Annotate with significance
    if (annotation_type == 'timepoint' and len(folders) > 2) or (annotation_type == 'timepoint' and compare_to_healthy==1):
        annotate_phase_significance(ax, cv_data, xtick_positions, metric_indices, significance_results, significance_levels, pairs_to_compare, metric_ymax)  
    
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
        #p_value_flx = float(significance_results[flexor_index].iloc[i, j])
        significance_flx = next((stars for threshold, stars in significance_levels.items() if p_value_flx <= float(threshold)), 'ns')
        if significance_flx:
            x1, x2 = xtick_positions[i * 2], xtick_positions[j * 2]
            ax.plot([x1, x1, x2, x2], [global_y_max, global_y_max + 0.05, global_y_max + 0.05, global_y_max], color='black')
            ax.vlines(x1, global_y_max-(global_y_max*.05), global_y_max, color='blue', linewidth=2)
            ax.vlines(x2, global_y_max-(global_y_max*.05), global_y_max, color='blue', linewidth=2)
            ax.text(x2, global_y_max + 0.1, significance_flx, ha='center', va='bottom', color='blue')

        # Extensor significance
        #p_value_ext = float(significance_results[extensor_index].iloc[i, j])
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

        phase_metric = metrics[6]
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
    data_labels = ['Max APs per bin', 'Freq', 'Burst Duration']
    if metric_indices == [0, 1]:
        offset = 0
    elif metric_indices == [2, 3]:
        offset = 1
    elif metric_indices == [4, 5]:
        offset = 2
    else:
        print('Metric indices are not recognized.')
    data_type = data_labels[offset]
    num_time_points = len(cv_data_flx)
    for time_idx in range(num_time_points):
        # Determine the p-value index
        index = time_idx*3+offset
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
            
def plot_variance_svg(folders, extracted_values_dict, significance_results):
    scatter_color = 'purple'
    group_spacing = 1.7
    pair_offset = 0.3
    significance_levels = {'0.001': '***', '0.01': '**', '0.05': '*'}
    pairs_to_compare = [(0, i) for i in range(1, len(folders))] 
    
    #Format subplot titles and xtick labels
    xticks_combined = []
    for folder in folders:
        folder_name = folder.split('/')[-1].replace("_", " ") 
        if compare_trials == "D1" or compare_trials == "D2":
            if "D1" in folder_name:
                subplot_suffix = "D1"
            elif "D2" in folder_name:
                subplot_suffix = "D2"
            xtick_label = folder_name.replace("D1", "").replace("D2", "").strip()
        else:
            subplot_suffix = ""
            xtick_label = folder_name
        xticks_combined.append(f"{xtick_label}")    
    title_suffix = f" ({subplot_suffix})" if subplot_suffix else ""
    
    """
    Creates a grid of subplots, one for each metric, and saves the result to an SVG file.
    """
    # Define metrics and their subplot positions
    metrics = [
        {"title": "Max APs per Bin"+title_suffix, "ylabel": "APs per bin", "metric_ymax": 130, "metric_indices": [0, 1], "subplot_pos": (0, 0)},
        {"title": "Frequency"+title_suffix, "ylabel": "Freq (Hz)", "metric_ymax": 3.5, "metric_indices": [2, 3], "subplot_pos": (0, 1)},
        {"title": "Burst Duration"+title_suffix, "ylabel": "Time (ms)", "metric_ymax": 250, "metric_indices": [4, 5], "subplot_pos": (1, 0)},
    ]
    
    annotation_type = 'timepoint'
    # Create the figure and axes for subplots
    fig, axes = plt.subplots(2, 2, figsize=(18, 11))
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
    #Plot phase subplot
    ax = axes[3]
    plot_phase_metric(
        ax=ax,
        title="Phase"+title_suffix,
        ylabel="Phase (deg)",
        metric_ymax=200,
        metric_indices=[6],
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
    output_file = folder_containing_data + '/' + compare_trials + '_metrics.svg'
    # Adjust layout and save to SVG
    fig.tight_layout()
    plt.savefig(output_file, format='svg')
    #plt.close(fig)
    
def plot_comparison_cv_with_dispersion(folders, extracted_values_dict, significance_results):
    """
    Optimized function to plot CV comparisons with dispersion points for multiple metrics.
    """
    fig, axs = plt.subplots(2, 2, figsize=(18, 11))
    scatter_color = 'purple'
    group_spacing = 1.7
    pair_offset = 0.3
    significance_levels = {'0.001': '***', '0.01': '**', '0.05': '*'}
    pairs_to_compare = [(0, i) for i in range(1, len(folders))] 
    
    #Format subplot titles and xtick labels
    xticks_combined = []
    
    for folder in folders:
        test_type = folder.split('/')[0]
        folder_name = folder.split('/')[-1].replace("_", " ") 
        if compare_trials == "D1" or compare_trials == "D2":
            if "D1" in folder_name:
                subplot_suffix = "D1"
            elif "D2" in folder_name:
                subplot_suffix = "D2"
            xtick_label = folder_name.replace("D1", "").replace("D2", "").strip()
        else:
            subplot_suffix = ""
            xtick_label = folder_name
        for substring, mapped_label in label_mapping.items():
            if substring in test_type:
                xtick_label = xtick_label+mapped_label
                break
        xticks_combined.append(f"{xtick_label}")    
    title_suffix = f" ({subplot_suffix})" if subplot_suffix else ""
    
    func_params = [
        {"indices": [0, 1], "title": f"Max APs per Bin {title_suffix}", "ylabel": "APs per bin", "ymax": 130},
        {"indices": [2, 3], "title": f"Frequency {title_suffix}", "ylabel": "Freq (Hz)", "ymax": 3.5},
        {"indices": [4, 5], "title": f"Burst Duration {title_suffix}", "ylabel": "Time (ms)", "ymax": 250}
    ]
    
    annotation_type = 'timepoint'
    for ax, param in zip(axs.flat, func_params):
        plot_metric(ax, param["title"], param["ylabel"], param["ymax"], param["indices"], folders, extracted_values_dict,
                    scatter_color, group_spacing, pair_offset, significance_results, significance_levels, pairs_to_compare,annotation_type)
    
    # Plot CV boxplots separately for 'MNP Phase' due to different amount of data points
    cv_data = []
    xtick_positions = [] 
    x_pos = 1  # Initial x-position

    for folder in folders:
        # Collect data for all trials in the current folder
        folder_data = []

        for trial_data in extracted_values_dict[folder]:
            # Extract the specific Flx and Ext metrics for each trial
            #folder_data.append(trial_data[6])  # Phase
            if trial_data[6] is not None:  # Phase
                folder_data.append(trial_data[6])

        # Append all trial data for this folder as a single list (grouped)
        cv_data.append(folder_data)
      
    xticks_combined = [label.replace("P0", "Healthy") for label in xticks_combined] 
    
    for j in range(len(cv_data)):
        xtick_positions.extend([x_pos])
        x_pos += group_spacing  # Increment for next group spacing
    
    # Create boxplots first
    axs[1, 1].boxplot(cv_data, positions=np.array(xtick_positions), patch_artist=True)
    #title_suffix = f" ({subplot_suffix})" if subplot_suffix else ""
    axs[1, 1].set_title(f'MNP Phase {title_suffix}', fontsize=title_fontsize)
    axs[1, 1].set_ylabel('Phase (deg)', fontsize=axis_label_fontsize)
    axs[1, 1].set_xticks(xtick_positions)
    axs[1, 1].set_xticklabels(xticks_combined, rotation=45, ha='right', fontsize=axis_label_fontsize)
    for axis in ['top','bottom','left','right']:
        axs[1, 1].spines[axis].set_linewidth(axis_line_thickness)

    # Overlay scatter plot for raw data (dispersion points) on top of the boxplot
    for i, (data, pos) in enumerate(zip(cv_data, xtick_positions)):
        axs[1, 1].scatter(np.ones_like(data) * pos, data, color=scatter_color, alpha=0.7, edgecolors='black', zorder=3)
    
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
        axs[1, 1].set_ylim([0,y_max + buffer])

        phase_metric = metrics[6]
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
                axs[1, 1].vlines(x1, y_max-(y_max*.05), y_max, color='black', linewidth=2)
                axs[1, 1].vlines(x2, y_max-(y_max*.05), y_max, color='black', linewidth=2)
                axs[1, 1].plot([x1, x1, x2, x2], [y, y + h, y + h, y], color='black')
                axs[1, 1].text(x2, y + h, significance, ha='center', va='bottom', color='black')
    plt.tight_layout()
    plt.savefig(folder_containing_data + '/' + compare_trials + '_metrics.pdf',bbox_inches="tight")
    
    fig, axs = plt.subplots(2, 2, figsize=(18, 11))
    annotation_type = 'flx_ext'
    significance_results_flx_ext = run_statistics_flx_ext(extracted_values_dict, folders)
    for ax, param in zip(axs.flat, func_params):
        plot_metric(ax, param["title"], param["ylabel"], param["ymax"], param["indices"], folders, extracted_values_dict,
                    scatter_color, group_spacing, pair_offset, significance_results_flx_ext, significance_levels, pairs_to_compare,annotation_type)
    # Plot CV boxplots separately for 'MNP Phase' due to different amount of data points
    cv_data = []
    xtick_positions = [] 
    x_pos = 1  # Initial x-position

    for folder in folders:
        # Collect data for all trials in the current folder
        folder_data = []

        for trial_data in extracted_values_dict[folder]:
            # Extract the specific Flx and Ext metrics for each trial
            #folder_data.append(trial_data[6])  # Phase
            if trial_data[6] is not None:  # Phase
                folder_data.append(trial_data[6])

        # Append all trial data for this folder as a single list (grouped)
        cv_data.append(folder_data)
      
    xticks_combined = [label.replace("P0", "Healthy") for label in xticks_combined]
    
    for j in range(len(cv_data)):
        xtick_positions.extend([x_pos])
        x_pos += group_spacing  # Increment for next group spacing
    
    # Create boxplots first
    axs[1, 1].boxplot(cv_data, positions=np.array(xtick_positions), patch_artist=True)
    #title_suffix = f" ({subplot_suffix})" if subplot_suffix else ""
    axs[1, 1].set_title(f'MNP Phase {title_suffix}', fontsize=title_fontsize)
    axs[1, 1].set_ylabel('Phase (deg)', fontsize=axis_label_fontsize)
    axs[1, 1].set_ylim([0, 190])
    axs[1, 1].set_xticks(xtick_positions)
    axs[1, 1].set_xticklabels(xticks_combined, rotation=45, ha='right', fontsize=axis_label_fontsize)
    for axis in ['top','bottom','left','right']:
        axs[1, 1].spines[axis].set_linewidth(axis_line_thickness)

    # Overlay scatter plot for raw data (dispersion points) on top of the boxplot
    for i, (data, pos) in enumerate(zip(cv_data, xtick_positions)):
        axs[1, 1].scatter(np.ones_like(data) * pos, data, color=scatter_color, alpha=0.7, edgecolors='black', zorder=3)    
    
    plt.tight_layout()
    plt.savefig(folder_containing_data + '/' + compare_trials + '_flx_ext_metrics.pdf',bbox_inches="tight")
    #plt.show()

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
        if drive_for_single_timepoint == 'both':
            folders = [
                no_intervention_data_folder+'/P0_D1',
                no_intervention_data_folder+'/P0_D2'
            ]
        elif drive_for_single_timepoint == 'd1':
            folders = [
                no_intervention_data_folder+'/P0_D1'
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
        elif drive_for_single_timepoint == 'd1' and compare_to_healthy==1:
            folders = [
                no_intervention_data_folder+'/P0_D1',
                no_intervention_data_folder+'/P45_D1',
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
        # Loop through each file pair (output_mnp1.csv, output_mnp2.csv)
        for mnp1_input, mnp2_input, subfolder_name in file_pairs:
            # Analyze the files and append the results as a row in the data array
            data_row = analyze_output(mnp1_input, mnp2_input, 'MNP', mnp_bd_y_line, mnp_phase_y_line, min_dist_phase_calc)
            data_row = np.round(data_row, 4)
            # Extract specific indices: [Max spike count Flx, Max spike count Ext, Freq Flx, Freq Ext, Burst Duration Flx, Burst Duration Ext, MNP Phase, Min spike count Flx, Min spike count Ext]
            selected_values = [data_row[i] for i in [4, 5, 8, 9, 0, 1, 12]]
            selected_values.append(str(subfolder_name))  # Append the subfolder name to the data row
            data_array.append(selected_values)
            min_spike_values = [data_row[i] for i in [15, 16]]
            if min_spike_values[0] <= 0: 
                zero_min_spike_count_flx += 1
            if min_spike_values[1] <= 0:  
                zero_min_spike_count_ext += 1

        # Remove outliers from data_array
        trial_type = folder.split('/')[-1].replace("_", " ")
        data_array = remove_outliers(trial_type,data_array)
        extracted_values_dict[folder] = data_array
        print('Completed analyzing: ', folder)
        
        # Calculate the percentage of trials with zero min spikes
        total_trials = len(data_array)
        if total_trials > 0:
            zero_min_spike_percentage_flx = round((zero_min_spike_count_flx / total_trials) * 100,2)
            zero_min_spike_percentage_ext = round((zero_min_spike_count_ext / total_trials) * 100,2)
        else:
            zero_min_spike_percentage_flx = 0
            zero_min_spike_percentage_ext = 0
        #print('% of trials that return to zero (flx, ext): ',zero_min_spike_percentage_flx,zero_min_spike_percentage_ext)
    # Run statistics for annotation
    significance_results, significance_healthy_comparison_results = run_statistics(extracted_values_dict, folders)  
    
    # Plot comparison across time points
    if save_as_svg == 0:
        plot_comparison_cv_with_dispersion(folders, extracted_values_dict, significance_results)  
    elif save_as_svg == 1:
        plot_variance_svg(folders, extracted_values_dict, significance_results)
    

