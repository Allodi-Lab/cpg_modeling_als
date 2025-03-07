import nest
import numpy as np
from skopt import gp_minimize
from skopt.space import Real
import matplotlib.pyplot as plt
import create_cpg_rg_layer_optimize as sim
from set_network_params import neural_network
netparams = neural_network()

# Define the search space for optimal parameters
search_space = [
    Real(0.01, 1.),  # RG-V1/V2b conn% 
    Real(0.01, 1.),  # V1/V2b-RG conn%
    Real(0.01, 0.1),  # RG-RG conn%
    Real(0.1, 5.),  # RG-V1/V2b weight 
    Real(-5., -0.1),  # V1/V2b-RG weight
    Real(0.1, 5.)  # RG-RG weight
]

def wrapped_simulation(*params):
    # Run the simulation and get the result
    diff_value, other_metrics = sim.run_rg_simulation(*params)
    
    # Return only the value to minimize (diff_value) to gp_minimize
    return diff_value    
    
# Run the Bayesian optimization
result = gp_minimize(
    wrapped_simulation,                 # The NEST simulation function
    dimensions=search_space,            # The search space for optimization
    n_calls=10,                         # Number of optimization iterations
    random_state=netparams.rng_seed           # Seed for reproducibility
)

# Get the optimal parameters
optimal_parameters = result.x
print(f"Optimal parameters: {optimal_parameters}")

#Parameters producing in-phase firing but oscillating: [0.5004594740175283, 0.1983019415026673, 0.24488500501967486, 5.543463669607473, -9.78375261263654, 8.349787635345562]

#Parameters producing oscillating extensor, flexor suppressed: [0.9322587346144438, 0.5936931215552462, 0.14493780552146834, 5.0447443485022445, -3.285145046279567, 6.950208998142421]

#Parameters producing oscillating output of flx/ext with new tau: [0.6310278917988495, 0.44986192483783627, 0.5717694860153688, 2.3264911927501615, -1.7514375639676412, 1.328878589951758]

# Plot the convergence of the optimization process
from skopt.plots import plot_convergence
plot_convergence(result)

# Check how close the optimal weights got to the target
final_difference = result.fun
print(f"Final difference from frequency: {final_difference}")

diff_value, other_metrics = sim.run_rg_simulation(optimal_parameters)
spike_bins_rg1_true, spike_bins_rg2_true, spike_bins_inh_inter1_true, spike_bins_inh_inter2_true = other_metrics

# Plot results of the optimal solution
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
#if nn.args['save_results']: plt.savefig(nn.pathFigures + '/' + 'rate_coded_output.pdf',bbox_inches="tight")

plt.show()
