import orbitfit.orbitfit as orb
from orbitfit.utils import (oe2ee, oe2rv, ee2oe, rv2oe)

import orbitfit.utils as utils
import dateutil.parser
import matplotlib.pyplot as plt
from datetime import datetime
import copy
import pandas as pd
import numpy as np

import BATCH_MIO.AngularBatchEst as ang

# constants

R_mean_earth = 6371.0  # Earth's mean radius in km
mu_earth = 398600.4418  # Earth's gravitational parameter in km^3/s^2


#####################################################################################
#                                                                                   #
#                     DATA SIMULATION TO FEED THE BATCH                             #
#                                                                                   #
#####################################################################################

# Define initial state and epoch:

# position and velocity of the servicer in ECI frame (in meters and m/s)
posvel_servicer_ECI_m = np.array([-2528776.634917358, -637910.2273496351, -6454161.018921344, 5660.809351470285, 4262.999013701585, -2623.9097735226724])
epoch = dateutil.parser.parse("2021-03-09T16:08:14.991000Z")
duration = 0.5   # days of propagation

# initial OE for the servicer in ECI frame
oe_servicer_ECI = np.array(rv2oe(posvel_servicer_ECI_m[0:3]*1e-3, posvel_servicer_ECI_m[3:6]*1e-3))

# initial OE for the client in ECI frame (same as servicer, but with small perturbations)
oe_client_ECI = oe_servicer_ECI
oe_client_ECI[1] = oe_client_ECI[1]+0.01
oe_client_ECI[2] = oe_client_ECI[2]+np.deg2rad(0.1)
oe_client_ECI[5] = oe_client_ECI[5]+np.deg2rad(0.01)

# convert OE back to RV for the client in ECI frame
posvel_client_ECI_km = np.array(oe2rv(*oe_client_ECI))
posvel_client_ECI_m = posvel_client_ECI_km.flatten()*1e3


## INITIAL ORBIT PROPAGATION

# Define the configuration for the orbit propagation
servicer_config = copy.deepcopy(orb.STK_CONFIG)
propagation_config = dict()
propagation_config["InitialState"] = posvel_servicer_ECI_m.tolist()
propagation_config["Step"] = 60
propagation_config['Start'] = epoch.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
propagation_config['End'] = (epoch+datetime.timedelta(days= duration)).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
servicer_config["Propagation"] = propagation_config

client_config = copy.deepcopy(orb.STK_CONFIG)
propagation_config = dict()
propagation_config["InitialState"] = posvel_client_ECI_m.tolist()
propagation_config["Step"] = 60
propagation_config['Start'] = epoch.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
propagation_config['End'] = (epoch+datetime.timedelta(days= duration)).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
client_config["Propagation"] = propagation_config

# Propagation
index, data = orb.propagate_orbits_wrapper(servicer_config)
df_servicer_ECI_m= pd.DataFrame(data=np.array(data), index=pd.DatetimeIndex(index), columns=["randv_mks_{}".format(i) for i in range(6)])

index, data = orb.propagate_orbits_wrapper(client_config)
df_client_ECI_m= pd.DataFrame(data=np.array(data), index=pd.DatetimeIndex(index), columns=["randv_mks_{}".format(i) for i in range(6)])


## REAL DATA

# Now we have to retrieve the relative position between client and servicer (versor as seen  from the servicer, no range)
# this will serve as "real" relative position that we want to retrieve. We will add noise to get the "measured data" that we'll use for the orbit fitting

df_relative_ECI_real = df_client_ECI_m.copy()
df_relative_ECI_real.iloc[:, :3] -= df_servicer_ECI_m.iloc[:, :3].values  # Subtract servicer position from client position
df_relative_ECI_real["range_m"] = np.linalg.norm(df_relative_ECI_real.iloc[:, :3].values, axis=1)

# compute unit direction vectors and store components separately
versor_arr_real = df_relative_ECI_real.iloc[:, :3].values / df_relative_ECI_real["range_m"].values.reshape(-1, 1)


## MEASURED DATA

# Perturb the client state to obtain the noisy measurements
# Number of measurements
n_measurements = len(df_relative_ECI_real)

# Add noise to the relative position and velocity measurements
# Columns 0-2 are position; columns 3-5 are velocity components

df_client_ECI_perturbed = df_client_ECI_m.copy()
noise_std_dev_pos = 0.1  # Standard deviation of the noise in km
noise_std_dev_vel = 0.01  # Standard deviation of the noise in km/s
# add position noise
df_client_ECI_perturbed.iloc[:, :3] += np.random.normal(
    0, noise_std_dev_pos, (n_measurements, 3))
# add velocity noise to all three components at once
df_client_ECI_perturbed.iloc[:, 3:6] += np.random.normal(
    0, noise_std_dev_vel, (n_measurements, 3))


# Find the measured relative direction (versor)
df_relative_ECI_measured = df_client_ECI_perturbed.copy()
df_relative_ECI_measured.iloc[:, :3] -= df_servicer_ECI_m.iloc[:, :3].values  # Subtract servicer position from client position
df_relative_ECI_measured["range_m"] = np.linalg.norm(df_relative_ECI_measured.iloc[:, :3].values, axis=1)

# compute measured versor components
versor_arr_meas = df_relative_ECI_measured.iloc[:, :3].values / df_relative_ECI_measured["range_m"].values.reshape(-1, 1)

#################################################################################################################################################################
# We have simulated the measurements, now we can proceed with the orbit fitting using the measured versor and range (with noise) as input for the fitting process.
# we need an initial guess for the servicer orbit (since we don't have the range) to find the computed versor and compare it with the measured one, 
# and minimize the error with a LS algorithm. We want to retrieve the best approx for the unperturbed original orbit.
#################################################################################################################################################################

# initial guess: perturbed original initial OE for the client in ECI frame (same as servicer, but with small perturbations)
oe_initial_guess = oe_servicer_ECI.copy()
#oe_initial_guess[0] += np.random.normal(0, 1e-2)  # Add noise to semi-major axis                    [km]
oe_initial_guess[1] += np.random.normal(0, 1e-5)  # Add noise to eccentricity                       [-]
oe_initial_guess[2] += np.random.normal(0, 1e-3)  # Add noise to inclination                        [deg]
oe_initial_guess[3] += np.random.normal(0, 1e-3)  # Add noise to argument of periapsis              [deg]
oe_initial_guess[4] += np.random.normal(0, 1e-3)  # Add noise to right ascension of ascending node  [deg]
oe_initial_guess[5] += np.random.normal(0, 1e-3)  # Add noise to true anomaly                       [deg]

pos, vel = oe2rv(*oe_initial_guess)
rv_initial_guess = np.concatenate((pos, vel))*1e3
ee_initial_guess = np.array(oe2ee(*oe_initial_guess))


#####################################################################################
#                                                                                   #
#                        HERE STARTS RUN_OREKIT_ANG                                 #
#                                                                                   #
#####################################################################################

import time

start_wall = datetime.now()  # Record computer time at start

# guessed orbit propagation

# Configuration
client_config = copy.deepcopy(orb.STK_CONFIG)
propagation_config = dict()
propagation_config["InitialState"] = rv_initial_guess.tolist()
propagation_config["Step"] = 60
propagation_config['Start'] = epoch.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
propagation_config['End'] = (epoch+datetime.timedelta(days=duration)).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
client_config["Propagation"] = propagation_config

# Propagation for initial guess
index, data = orb.propagate_orbits_wrapper(client_config)
df_client_ECI_fit= pd.DataFrame(data=np.array(data), index=pd.DatetimeIndex(index), columns=["randv_mks_{}".format(i) for i in range(6)])

# Initialize the class

estimator = ang.Optimizer(
    ee_initialguess=ee_initial_guess,      # Array degli elementi equinoziali (6,)
    df_client=df_client_ECI_fit,           # DataFrame (N,6) della prima propagazione guess
    df_servicer=df_servicer_ECI_m,         # DataFrame (N,6) della posizione del servicer (reale)
    versor_arr_meas=versor_arr_meas,       # Array (N,3) dei versori misurati (osservazioni)
    config=propagation_config,             # Solo il dizionario della propagazione (Step, Start, End)
    config_0=client_config,                # L'intero dizionario STK_CONFIG completo
    damping_lambda=0.001,                  # (Opzionale) Valore iniziale per Levenberg-Marquardt
    max_loops=40                           # (Opzionale) Numero massimo di iterazioni
)

##Call to Batch estimator

df_state_final, ee_final, n_loops = estimator.LSLoop()

# Return optimized state df_state_final, ee_final

end_wall = datetime.now()    # Record computer time at end
execution_time = end_wall - start_wall
print()
print(f"Start time: {start_wall:.2f} ")
print(f"End time: {end_wall:.2f} ")
print(f"Total processing time: {execution_time:.2f} seconds")
print()
#
#
#
    ########################################
    #                                      #
    #            Plot results              #
    #                                      #
    ########################################
#
# Propagating results

# prop_config_final = copy.deepcopy(orb.STK_CONFIG)
# prop_config_final["Propagation"] = propagation_config
# index_mod, data_mod = orb.propagate_orbits_wrapper(prop_config_loop)
# data_mod = np.array([list(item) for item in data_mod])
# df_state_final = pd.DataFrame(data_mod, index=index_mod, columns=['randv_mks_0', 'randv_mks_1', 'randv_mks_2', 'randv_mks_3', 'randv_mks_4', 'randv_mks_5'])

# 3D plot of original client orbit df_client_ECI_m and the perturbated initial guess df_client_ECI_fit:

fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Plot original client orbit
ax.scatter(df_client_ECI_m["randv_mks_0"],
           df_client_ECI_m["randv_mks_1"],
           df_client_ECI_m["randv_mks_2"],
           label="Original Client Orbit", color="red", s=5)

# Plot final fitted orbit
ax.scatter(df_client_ECI_fit["randv_mks_0"],
           df_client_ECI_fit["randv_mks_1"],
           df_client_ECI_fit["randv_mks_2"],
           label="Initial guess Orbit", color="green", s=5)

ax.set_xlabel("X (km)")
ax.set_ylabel("Y (km)")
ax.set_zlabel("Z (km)")
ax.set_title("Original vs Initial guess Orbit")
ax.legend()
plt.show()

# 3D plot of original client orbit df_client_ECI_m and the final, fitted df_state_final:

fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Plot original client orbit
ax.scatter(df_client_ECI_m["randv_mks_0"],
           df_client_ECI_m["randv_mks_1"],
           df_client_ECI_m["randv_mks_2"],
           label="Original Client Orbit", color="red", s=5)

# Plot final fitted orbit
ax.scatter(df_state_final["randv_mks_0"],
           df_state_final["randv_mks_1"],
           df_state_final["randv_mks_2"],
           label="Final Fitted Orbit", color="green", s=5)

ax.set_xlabel("X (km)")
ax.set_ylabel("Y (km)")
ax.set_zlabel("Z (km)")
ax.set_title("Original vs Final Fitted Orbit")
ax.legend()
plt.show()

# 3D plot of perturbated initial guess df_client_ECI_fit and final interpolated:

fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Plot initial guess (perturbed) orbit
ax.scatter(df_client_ECI_fit["randv_mks_0"],
           df_client_ECI_fit["randv_mks_1"],
           df_client_ECI_fit["randv_mks_2"],
           label="Initial Guess (Perturbed)", color="blue", s=5)

# Plot final fitted orbit
ax.scatter(df_state_final["randv_mks_0"],
           df_state_final["randv_mks_1"],
           df_state_final["randv_mks_2"],
           label="Final Fitted Orbit", color="green", s=5)

ax.set_xlabel("X (km)")
ax.set_ylabel("Y (km)")
ax.set_zlabel("Z (km)")
ax.set_title("Initial Guess vs Final Fitted Orbit")
ax.legend()
plt.show()


# compute and plot: 1) difference between original and initial guessed orbit\\ 2) difference between original and final fitted orbit\\ 3) difference between initial guess and final fitted orbit

import matplotlib.pyplot as plt
import numpy as np

# Funzione per calcolare la norma della differenza (errore di posizione e velocità)
def compute_errors(df_a, df_b):
    pos_cols = ['randv_mks_0', 'randv_mks_1', 'randv_mks_2']
    vel_cols = ['randv_mks_3', 'randv_mks_4', 'randv_mks_5']
    
    pos_error = np.linalg.norm(df_a[pos_cols].values - df_b[pos_cols].values, axis=1)
    vel_error = np.linalg.norm(df_a[vel_cols].values - df_b[vel_cols].values, axis=1)
    
    return pos_error, vel_error

df_true = df_client_ECI_m
df_initial = df_client_ECI_fit
df_final = df_state_final

# 1) Original vs Initial Guess
err_pos_orig_init, err_vel_orig_init = compute_errors(df_true, df_initial)

# 2) Original vs Final Fit
err_pos_orig_final, err_vel_orig_final = compute_errors(df_true, df_final)

# 3) Initial Guess vs Final Fit
err_pos_init_final, err_vel_init_final = compute_errors(df_initial, df_final)


fig, ax = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

# Sottoplot 1: Posizione
ax[0].plot(df_true.index, err_pos_orig_init, label='Original vs Initial Guess', color='red')
ax[0].plot(df_true.index, err_pos_init_final, label='Initial Guess vs Final Fit', color='orange', linestyle=':')
ax[0].plot(df_true.index, err_pos_orig_final, label='Original vs Final Fit (Residual)', color='green', linewidth=2, linestyle='--')

ax[0].set_ylabel('Position Error [m]')
ax[0].set_title('Comparison of Orbital Position Differences')
ax[0].legend()
ax[0].grid(True, which='both', linestyle='--', alpha=0.5)
#ax[0].set_yscale('log') # Usiamo scala logaritmica per vedere il miglioramento

# Sottoplot 2: Velocità
ax[1].plot(df_true.index, err_vel_orig_init, label='Original vs Initial Guess', color='red')
ax[1].plot(df_true.index, err_vel_init_final, label='Initial Guess vs Final Fit', color='orange', linestyle=':')
ax[1].plot(df_true.index, err_vel_orig_final, label='Original vs Final Fit (Residual)', color='green', linewidth=2, linestyle='--')

ax[1].set_ylabel('Velocity Error [m/s]')
ax[1].set_xlabel('Time')
ax[1].legend()
ax[1].grid(True, which='both', linestyle='--', alpha=0.5)
#ax[1].set_yscale('log')

plt.tight_layout()
plt.show()

print()
print("ratio between final and original error (hopefully <<1):")
print((np.linalg.norm(err_pos_orig_final)/np.linalg.norm(err_pos_orig_init)))
print()
print("norm of error between original and initial guess:")
print(np.linalg.norm(err_pos_orig_init))
print()
print("norm of error between original and final guess:")
print(np.linalg.norm(err_pos_orig_final))
print()
print("Max error between final and real positions:")
print(np.max(err_pos_orig_final))
print()
print("Iterations:")
print({n_loops})


