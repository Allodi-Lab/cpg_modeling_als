import os
import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 36})
plt.rcParams['svg.fonttype'] = 'none'

folder_name = sys.argv[1]
drive_to_compare = str(sys.argv[2])
plot_isolated_rgs=0 #0=Plot network output; 1=plot only RG output (relevant for isolated RG output only)
plot_mnp_only=1 #0=Plot all neural populations
timepoints_to_plot="all" #Choose: 'all', 'disease', 'P0', 'P45', 'P63', 'P112'
plot_complete_simulation=0 #0=5 seconds for publication plots; 1=10 seconds for full simulation
save_as_svg = 0

# Define the path to the parent folder that contains subfolders with CSV files
parent_directory = "/Users/wdg562/Documents/NEST_simulations/ALS_network_adex_beta_optimization/saved_simulations/" + folder_name

# Define the specific name patterns you're looking for in pairs
if plot_isolated_rgs==1:
    name_patterns1 = ["output_rg1"]  
    name_patterns2 = ["output_rg2"]
    labels1 = ["RG_Flx"]
    labels2 = ["RG_Ext"]

elif plot_mnp_only==1:
    name_patterns1 = ["output_mnp1"]  
    name_patterns2 = ["output_mnp2"]
    labels1 = ["MNP_Flx"]
    labels2 = ["MNP_Ext"]

else:
    name_patterns1 = ["output_rg1","output_v2b","output_v2a1","output_v0c1","output_1a1","output_rc1","output_mnp1"]  
    name_patterns2 = ["output_rg2","output_v1","output_v2a2","output_v0c2","output_1a2","output_rc2","output_mnp2"]
    labels1 = ["RG_Flx","V2b","V2a_Flx","V0c_Flx","1a_Flx","RC_Flx","MNP_Flx"]
    labels2 = ["RG_Ext","V1","V2a_Ext","V0c_Ext","1a_Ext","RC_Ext","MNP_Ext"]
    
# Dictionary to store data for each pattern pair
data_pairs = {f"{pattern1}_{pattern2}": {"data1": [], "data2": [], "labels": []} 
              for pattern1, pattern2 in zip(name_patterns1, name_patterns2)}

# Iterate over all subdirectories and load CSV files matching the patterns
for subdir, _, files in os.walk(parent_directory):
    for file in files:
        for pattern1, pattern2 in zip(name_patterns1, name_patterns2):
            if file.endswith(".csv"):
                if pattern1 in file:  # Match files for pattern1
                    file_path = os.path.join(subdir, file)
                    relative_path = os.path.relpath(subdir, parent_directory)
                    subfolder_name = relative_path.split(os.sep)[0] if relative_path != "." else ""
                    #subfolder_name = os.path.basename(os.path.dirname(subdir))
                    if drive_to_compare in subfolder_name:
                        df = pd.read_csv(file_path)
                        data_pairs[f"{pattern1}_{pattern2}"]["data1"].append(df.values.flatten())
                        data_pairs[f"{pattern1}_{pattern2}"]["labels"].append(subfolder_name)
                elif pattern2 in file:  # Match files for pattern2
                    file_path = os.path.join(subdir, file)
                    relative_path = os.path.relpath(subdir, parent_directory)
                    subfolder_name = relative_path.split(os.sep)[0] if relative_path != "." else ""
                    #subfolder_name = os.path.basename(os.path.dirname(subdir))
                    if drive_to_compare in subfolder_name:
                        df = pd.read_csv(file_path)
                        data_pairs[f"{pattern1}_{pattern2}"]["data2"].append(df.values.flatten())

# Define the order of subfolder names for sorting
if timepoints_to_plot=='all':
    desired_order = ["P0", "P45", "P63", "P112"]
elif timepoints_to_plot=='disease':    
    desired_order = ["P45", "P63", "P112"]
elif timepoints_to_plot=='P0':
    desired_order = ["P0"]
elif timepoints_to_plot=='P45':
    desired_order = ["P45"]  
elif timepoints_to_plot=='P63':
    desired_order = ["P63"]      
elif timepoints_to_plot=='P112':
    desired_order = ["P112"]
    

# Plot each pair in its own figure across all timepoints
for i, (pair_key, pair_data) in enumerate(data_pairs.items()):
    # Sort data for each pair according to the desired order
    sorted_data = sorted(
        zip(pair_data["labels"], pair_data["data1"], pair_data["data2"]),
        key=lambda x: next((desired_order.index(order) for order in desired_order if order in x[0]), float('inf'))
    )
    
    # Unzip sorted data
    sorted_labels, sorted_data1, sorted_data2 = zip(*sorted_data)
    
    # Create a figure for the current pattern pair
    fig, axes = plt.subplots(len(desired_order), 1, figsize=(35, 6 * len(desired_order)), sharex=True, sharey=True)
    #fig.suptitle(f"{labels1[i]} vs {labels2[i]} - Comparison Across Timepoints", fontsize=36)

    if len(desired_order) == 1:
        axes = [axes]  # Ensure axes is always a list for consistency

    # Plot data for each timepoint in a separate subplot
    for ax, timepoint in zip(axes, desired_order):
        for label, data1, data2 in zip(sorted_labels, sorted_data1, sorted_data2):
            if timepoint in label:  # Plot only if the label matches the timepoint
                time = np.arange(0, len(data1), 1)
                ax.plot(time, data1, label=f'{labels1[i]}', color='blue', alpha=0.7, linewidth=4)
                ax.plot(time, data2, label=f'{labels2[i]}', color='orange', alpha=0.7, linewidth=4)   
                if plot_complete_simulation==1:
                    ax.set_xlim(0, 100000)
                else:
                    ax.set_xlim(0, 50000)
                ax.set_ylabel('APs per bin')
                if plot_isolated_rgs==1:
                    title = "Isolated RG Output ("+drive_to_compare+")"
                    ax.set_ylim(0, 1400)
                elif plot_mnp_only==1 and timepoints_to_plot=="P0":
                    title = "Healthy Network MNP Output ("+drive_to_compare+")"
                    ax.set_ylim(0, 120)
                else:    
                    title = "Healthy "+drive_to_compare if timepoint == "P0" else timepoint+' '+drive_to_compare
                    ax.set_ylim(0, 120)
                ax.set_title(title, pad=20)
                ax.legend()
                for axis in ['top','bottom','left','right']:
                    ax.spines[axis].set_linewidth(2)
                ax.legend(loc='upper right')    

    # Label the x-axis on the bottom subplot only
    axes[-1].set_xlabel('Time (ms)')
    if plot_complete_simulation==1:
        axes[-1].set_xticks([0,10000,20000,30000,40000,50000,60000,70000,80000,90000])
        axes[-1].set_xticklabels([0,1000,2000,3000,4000,5000,6000,7000,8000,9000])
    else:    
        axes[-1].set_xticks([0,10000,20000,30000,40000,50000])
        axes[-1].set_xticklabels([0,1000,2000,3000,4000,5000])

    # Adjust layout for each figure
    plt.tight_layout()
    fig.subplots_adjust(hspace=0.2)  # Add more vertical space between subplots
    fig.subplots_adjust(top=0.9)
    if save_as_svg == 0:
        plt.savefig(parent_directory + '/' + 'population_output_' + labels1[i] + '_vs_' + labels2[i] + '_' + drive_to_compare +'.pdf',bbox_inches="tight")
    elif save_as_svg == 1:
        plt.savefig(parent_directory + '/' + 'population_output_' + labels1[i] + '_vs_' + labels2[i] + '_' + drive_to_compare +'.svg',bbox_inches="tight", format='svg')
    
#plt.show()