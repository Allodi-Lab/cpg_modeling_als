# cpg_als_model
This is a spiking CPG model containing the interneuron populations affected during ALS degeneration.

<strong>Running the analysis:</strong><br>
<strong>compare_metrics_across_timepoints.py:</strong> This script compares metric variation across timepoints or drives. It takes two arguments, the first is the folder path where the data is located. The second is the comparison to be run. D1 or D2 will compare the selected drive across timepoints. P0, P45, P63, or P112 will compare drives within the selected timepoint (Example arguments: path/to/my_new_test D1). Note, this plots the variation for each metric and expects multiple trials per timepoint / drive. This is not for comparing single trials across timepoints. It imports several functions from "calculate_stability_metrics_all_trials.py" so this script must be in the same folder.

<strong>calculate_stability_metrics_all_trials.py:</strong> This returns the metrics for individual trials to a CSV file. This is useful for identifying outlier trials. The script takes one argument, the path the folder containing the random trials (Example argument: path/to/my_new_test/P63_D2). The folder will use the name pattern Pxx_Dx and individual folders within will be named with their respective seedvalue.

<strong>calculate_stability_metrics.py:</strong> This returns the metrics for an individual trial to the terminal and a plot showing where peaks and threshold crossing was detected for that trial. This is useful if the automated calculation returns an unexpected value. The trial can be identified using the "calculate_stability_metrics_all_trials.py" and then this script can be run to see values used for calculation. It takes one argument, the folder for the individual trial, so it must point to not only the Pxx_Dx folder and the subfolder within which is the seed value for the targeted trial (Example argument: path/to/my_new_test/P63_D2/6780264).

<strong>compare_stability_metrics_indiv_trials.py</strong> This returns the coefficient of variance (CV) as calculated within single trials of a test and plots each as a data point on the box and whisker plot. This differs from the "compare_metrics_across_timepoints.py" because it is calcuating the CV within an individual trial as opposed to comparing trials. The script takes one argument, the path the folder containing the random trials (Example argument: path/to/my_new_test/P63_D2).

<strong>plot_timepoint_comparison.py:</strong> This script plots the output of the MNPs at each timepoint. It takes two arguments, the first is the folder path where the data is located. The second is the drive to be compared, D1 or D2 (Example arguments: path/to/my_new_test D1). This is for plotting a single trial per timepoint.

<strong>Running the simulation:</strong><br>
<strong>configuration_run_nest.yaml:</strong> This is the configuration file for the simulation and is where test type is defined, desired plots are selected, etc.
NOTE: When you run a simulation, the results will be automatically be saved in a subfolder called "saved_simulations". The analysis scripts must also be located in the "saved_simulations" folder or the 

There are several bash scripts included to run the network in different configurations:<br>
<strong>run_rg_layer_provide_seed.sh:</strong> This script can be used to run the RGs in isolation, providing output of the RG flexor and RG extensor separately. It can also run the complete RG layer, providing the output from both of the RGs and the V1 and V2b populations. The mode is set in the "configuration_run_nest.yaml" file by toggling the parameter "rgs_connected".<br>
<strong>run_trials_provide_seeds.sh:</strong> This script can be used to run a single test type (ex. P63 at drive 1) with a set amount of random seeds. The seeds to be used are defined in the script. The number of days after onset (P63) and the desired drive (D1) are both set in the "configuration_run_nest.yaml" file by setting the parameters "days_after_onset" and "freq_test" respectively.<br>
<strong>run_all_timepoints.sh:</strong> This script can be used to loop through all desired tests, this is defined in the script. Ex. "P45_D1" "P45_D2" "P63_D1" "P63_D2" would run four test types for all seeds provided in the script. This script overrides the "configuration_run_nest.yaml" file parameters of "days_after_onset" and "freq_test".

<strong>Suggested way of working:</strong><br>
When you are running a new test and would like to compare the output of a single trial across all timepoints:<br>
In the "configuration_run_nest.yaml" file, update the parameters you are testing, and turn on "save_all_pops" using a 1.<br>
In the "run_all_timepoints.sh" bash script, ensure the seed value array only contains the single seed you are testing. Update the test name array to include all timepoints and drives you would like to test.<br>
Run the "run_all_timepoints.sh" bash script.<br>
Create a new folder with a relevant test name and move the newly created trial folders into it.<br>
Run the "plot_timepoint_comparison.py" python script, this will create a figure comparing timepoints for each neural population. The figures will be automatically saved to the test folder.<br>

When you are running a new test and would like to compare the output of random trials across all timepoints to see the variance for network metrics (i.e. max spikes, frequency, burst duration, and phase):<br>
In the "configuration_run_nest.yaml" file, update the parameters you are testing, and make sure "save_all_pops" is OFF using a 0 (otherwise the saved data will be too large).<br>
In the "run_all_timepoints.sh" bash script, ensure the seed value array contains the all seeds you are testing. Update the test name array to include all timepoints and drives you would like to test.<br>
Run the "run_all_timepoints.sh" bash script.<br>
Create a new folder with a relevant test name and move the newly created trial folders into it.<br>
Run the "compare_metrics_across_timepoints.py" python script, this will a figure with the variance per metric.<br>
If some timepoints have outliers that you would not like to include, run the "calculate_stability_metrics_all_trials.py" python script. This will save a csv file with all trial data. You can review this data to identify the outlier trials. You can then move or delete the folder of the outlier trial, it can easily be identified by navigating to the folder of the identified timepoint and the corresponding seed value. Ex. .../P63_D2/1487307 .<br>
Once the outlier trials are removed, re-run the "compare_metrics_across_timepoints.py" python script.

In order to run this code, you need to install NEST and NESTML. This simulation software can be installed on your OS using a conda environment. Follow these steps:<br>
Install miniconda - https://docs.anaconda.com/miniconda/<br>
Install NEST within the conda environment - https://nest-simulator.readthedocs.io/en/stable/installation/conda_forge.html#conda-forge-install/<br>
Install NESTML within the same conda environment - https://nestml.readthedocs.io/en/latest/installation.html<br>
Install the Adaptive Exponential Integrate and Fire neuron with a beta function synaptic response (aeif_cond_beta) within your NESTML installation (instructions below).

From the "installer_files" git folder:<br>
Copy the "aeif_cond_beta.nestml" file into the path that was created to the NESTML models within your environment. The path will look similar to: .../.../miniconda3/envs/MY_ENVIRONMENT_NAME/models/neurons<br>
Copy the "create_aeif_cond_beta.py" to a relevant folder on your computer.<br>
Create the aeif_cond_beta neuron by running the python script. Before running the script, update the "input_path" and "target_path" in the Python file. The input path should be the path where you copied the "aeif_cond_beta.nestml" file.
# cpg_modeling_als
