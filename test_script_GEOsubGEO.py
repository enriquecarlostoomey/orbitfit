import orbitfit.orbitfit as orb
from orbitfit.utils import (oe2ee, oe2rv, rv2oe, ee2oe)
import dateutil.parser
import datetime
import copy
import pandas as pd
import numpy as np
import os
import BATCH_MIO.AngularBatchEst as ang
from BATCH_MIO.PlotResultsFunc import PLOTRESULTS, plot_detectability

#####################################################################################
#                                                                                   #
#                     DATA SIMULATION TO FEED THE BATCH                             #
#                                                                                   #
#####################################################################################

# comments to besaved in the final output:
comments = "\nincreased eccentricity, but the encounter is not at perigee (magMax increased)\n"

# The perturbation for measured versors are now applied directly to the versors to mantain physical consistency
# the perturbation on initial guess has been implemented in the cartesian state vector
# DEFINE PARAMETERS FOR THE SIMULATION:

propstep = 20             # time step for propagation [s]
duration = 20.5           # propagation time [h]
MaxLoop = 20                 # Max Loop in LS algorithm [-]
noise_std_dev_pos = 1000    # Standard deviation of the client position noise for the initial guess perturbation [m]
noise_std_dev_vel = 10      # Standard deviation of the client velocity noise for the initial guess perturbation [m/s]
sigma_rad = 1.7453e-05      # Angular error for the measured versors (bot azimuth and elevation)(small) [rad]
Epsilon = 1e-9              # condition to exit the LS loop [-]
FOV = 0                     # FOV semi-aperture to filter out-of-sight measurements [deg] - if 0 the filter is not activated
alphaMax = 0                # Maximum sun phase angle to see the target [deg] - if 0 the filter is not activated
magnitudeMax = 16           # 13 suggested (see comment in the related function) - if 1e6 the filter is not activated
fov_offset = 30
seed = 100         


epoch = dateutil.parser.parse("2021-03-09T09:40:14.991000") 
oe_client_ECI = [42164.14, 1e-06, 1e-06, 1e-06, 0, 2.5]  #[a, e, i, Omega, omega, M]
oe_servicer_ECI = [41864.14, 0.1, 1e-06, 1e-06, 0, 2.5-np.deg2rad(1)]  #[a, e, i, Omega, omega, M]

# position and velocity of the client in ECI frame (in meters and m/s) (GEO orbit)
pos, vel=  np.array(oe2rv(*oe_client_ECI))
posvel_client_ECI_m = np.concatenate([pos, vel]) * 1e3 

# The servicer is in sub-GEO (-300km radius)
pos, vel=  np.array(oe2rv(*oe_servicer_ECI))
posvel_servicer_ECI_m = np.concatenate([pos, vel]) * 1e3                                          # GEO coordinates for servicer [m, m/s]


## INITIAL ORBIT PROPAGATION

# Define the configuration for the orbit propagation
servicer_config = copy.deepcopy(orb.STK_CONFIG)
propagation_config = dict()
propagation_config["InitialState"] = posvel_servicer_ECI_m.tolist()
propagation_config["Step"] = propstep
propagation_config['Start'] = epoch.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
propagation_config['End'] = (epoch+datetime.timedelta(hours= duration)).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
servicer_config["Propagation"] = propagation_config

client_config = copy.deepcopy(orb.STK_CONFIG)
propagation_config = dict()
propagation_config["InitialState"] = posvel_client_ECI_m.tolist()
propagation_config["Step"] = propstep
propagation_config['Start'] = epoch.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
propagation_config['End'] = (epoch+datetime.timedelta(hours= duration)).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
client_config["Propagation"] = propagation_config

# Propagation
print("\nInital orbits propagation...")
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

# Perturb the real versors to obtain the measured ones.

versor_arr_meas = copy.deepcopy(versor_arr_real)
rng = np.random.default_rng(seed=seed)

# 1. Genera rumore gaussiano 3D per ogni versore
# Usiamo sigma_rad come deviazione standard per le componenti trasversali
noise = rng.normal(0, sigma_rad, versor_arr_meas.shape)

# 2. Rendi il rumore perpendicolare al versore originale
# Proiezione: n_perp = n - (n . v) * v
dot_products = np.sum(noise * versor_arr_meas, axis=1, keepdims=True)
noise_perp = noise - dot_products * versor_arr_meas

# 3. Aggiungi il rumore perpendicolare al versore originale
versor_arr_meas = versor_arr_meas + noise_perp

# 4. Rinormalizza per garantire che siano ancora versori unitari
norms = np.linalg.norm(versor_arr_meas, axis=1, keepdims=True)
versor_arr_meas = versor_arr_meas / norms




####################################################################################################################################################################
# We have simulated the measurements, now we can proceed with the orbit fitting using the measured versor and range (with noise) as input for the fitting process. # 
# we need an initial guess for the servicer orbit, (r,v) state vector. (since we don't have the range) to find the computed versor and compare it with the         #
# measured one, and minimize the error with a LS algorithm. We want to retrieve the best approx for the unperturbed original orbit.                                #
####################################################################################################################################################################


rng = np.random.default_rng(seed=seed)

# initial guess: perturbed original initial OE for the client in ECI frame (same as nominal, but with small perturbations)
rv_initial_guess = df_client_ECI_m.iloc[0].to_numpy(copy=True)
rv_initial_guess[:3] += rng.normal(0, noise_std_dev_pos, 3)
rv_initial_guess[3:] += rng.normal(0, noise_std_dev_vel, 3)

# obtain oe and ee for the initial guess
oe_initial_guess = rv2oe(rv_initial_guess[:3] * 1e-3, rv_initial_guess[3:] * 1e-3)
ee_initial_guess = np.array(oe2ee(*oe_initial_guess))


#####################################################################################
#                                                                                   #
#                        HERE STARTS RUN_OREKIT_ANG                                 #
#                                                                                   #
#####################################################################################



start_wall = datetime.datetime.now()  # Record computer time at start

# guessed orbit propagation

# Configuration
client_config = copy.deepcopy(orb.STK_CONFIG)
propagation_config = dict()
propagation_config["InitialState"] = rv_initial_guess.tolist()
propagation_config["Step"] = propstep
propagation_config['Start'] = epoch.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
propagation_config['End'] = (epoch+datetime.timedelta(hours=duration)).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
client_config["Propagation"] = propagation_config

# Propagation for initial guess
index, data = orb.propagate_orbits_wrapper(client_config)
df_client_ECI_fit= pd.DataFrame(data=np.array(data), index=pd.DatetimeIndex(index), columns=["randv_mks_{}".format(i) for i in range(6)])

# Initialize the class

print("\nInitializing Optimizer...\n")
estimator = ang.Optimizer(
    df_client=df_client_ECI_fit,            # DataFrame (N,6) della prima propagazione guess
    df_client_real=df_client_ECI_m,         # 
    df_servicer=df_servicer_ECI_m,          # DataFrame (N,6) della posizione del servicer (reale)
    versor_arr_meas=versor_arr_meas,        # Array (N,3) dei versori misurati (osservazioni)
    config=propagation_config,              # Solo il dizionario della propagazione (Step, Start, End)
    max_loops=MaxLoop,                      # (Opzionale) Numero massimo di iterazioni
    epsilon=Epsilon,
    fov = FOV,
    fov_offset=fov_offset,
    alphamax = alphaMax,
    m_v_threshold = magnitudeMax           
    )

##Call to Batch estimator

df_state_final, ee_final, n_loops = estimator.LSLoop()


# Return optimized state df_state_final, ee_final

end_wall = datetime.datetime.now()    # Record computer time at end
execution_time = end_wall - start_wall
print()
print(f"Start time: {start_wall} ")
print(f"End time: {end_wall} ")
print(f"Total processing time: {execution_time} [hh:mm:ss]")
print()


# ==============================================================================
# --- OUTPUT FOLDER CREATION                                                 ---
# ==============================================================================
if n_loops > 2:
    # Creating the folder with dynamic name
    time_suffix = start_wall.strftime("%H%M%S")
    folder_name = f"test_{duration}h_{propstep}sStep_{magnitudeMax}maxMag_{time_suffix}"
    folder_name = os.path.join("tests", folder_name)

    # additional folder if the previous one was already existing
    os.makedirs(folder_name, exist_ok=True)

    # number of points for saving
    total_points = len(versor_arr_meas)
    if hasattr(estimator, 'mask') and estimator.mask is not None:
        valid_points = np.sum(estimator.mask)
    else:
        valid_points = total_points

    if n_loops == 0:
        exit_reason = "All points filtered out (Optimization skipped)"
    elif n_loops >= MaxLoop:
        exit_reason = "Maximum number of iterations reached (MaxLoop)"
    else:
        exit_reason = f"Convergence reached (Residual variation < Epsilon: {Epsilon})"

    # ==============================================================================
    # --- WRITE NOTE.TXT                                                         ---
    # ==============================================================================
    note_path = os.path.join(folder_name, "note.txt")

    with open(note_path, 'w') as f:
        f.write(f"Start time: {start_wall}\n")
        f.write(f"End time: {end_wall}\n")
        f.write(f"Total processing time: {execution_time} [hh:mm:ss]\n\n")

        f.write("\n--- COMMENTS ---\n")
        f.write(comments)
        
        f.write("\n--- PARAMETERS ---\n")
        f.write(f"propstep = {propstep} [s]\n")
        f.write(f"duration = {duration} [h]\n")
        f.write(f"MaxLoop = {MaxLoop}\n")
        f.write(f"noise_std_dev_pos = {noise_std_dev_pos} [m]\n")
        f.write(f"noise_std_dev_vel = {noise_std_dev_vel} [m/s]\n")
        f.write(f"Epsilon = {Epsilon}\n")
        f.write(f"FOV = {FOV} [deg]\n")
        f.write(f"alphaMax = {alphaMax} [deg]\n")
        f.write(f"magnitudeMax = {magnitudeMax}\n\n")

        f.write("\n--- INITIAL ORBITAL ELEMENTS (Guess & True) ---\n")
        f.write(f"epoch = {epoch}\n")
        # Stampiamo gli array convertendoli in liste per renderli leggibili
        f.write(f"oe_client_ECI = {oe_client_ECI.tolist()} [a, e, i, Omega, omega, M]\n")
        f.write(f"oe_servicer_ECI = {oe_servicer_ECI.tolist()} [a, e, i, Omega, omega, M]\n\n")

        f.write("\n--- RESULTS ---\n")
        f.write(f"Total measurements before filtering: {total_points}\n")
        f.write(f"Valid measurements after filtering: {valid_points}\n")
        f.write(f"Iterations (Loops): {n_loops}\n")
        f.write(f"Optimization exit reason: {exit_reason}\n")
        
        # Salva la matrice di covarianza formattata
        if hasattr(estimator, 'covariance_matrix'):
            f.write("\n\nFinal Covariance Matrix:\n")
            for row in estimator.covariance_matrix:
                f.write(" ".join(f"{val:12.4e}" for val in row) + "\n")
                
            f.write("\nFinal ST deviation on unknowns [af, ag, a, L, pe, qe]:\n")
            f.write(" ".join(f"{val:12.4e}" for val in np.sqrt(np.diag(estimator.covariance_matrix))) + "\n")


# ==============================================================================
# --- PLOTTING AND SAVING IMAGES                                             ---
# ==============================================================================
if n_loops > 2:
    versor_arr_real_filt = estimator.applyMask(versor_arr_real)
    versor_arr_meas_filt = estimator.applyMask(versor_arr_meas)
    df_client_ECI_m_filt = estimator.applyMask(df_client_ECI_m)
    df_client_ECI_fit_filt = estimator.applyMask(df_client_ECI_fit)

    # Passiamo folder_name alla classe PLOTRESULTS
    Plot = PLOTRESULTS(versor_arr_real, versor_arr_real_filt, versor_arr_meas_filt, 
                       estimator.versor_arr_comp, df_client_ECI_m, df_client_ECI_m_filt, 
                       df_client_ECI_fit_filt, df_state_final, df_servicer_ECI_m, 
                       estimator.b_history, n_loops, save_dir=folder_name)
    
    Plot.plotResiduals()
    Plot.plotResidualsEvolution()
    Plot.plotFinalFit() # I risultati del fit verranno stampati a terminale e salvati
    
    if estimator.detectability_filter != 1e6:
        plot_detectability(estimator.phi, estimator.d, estimator.m_v, estimator.m_v_threshold, save_dir=folder_name)

    print("\n\nresults saved!\n")

