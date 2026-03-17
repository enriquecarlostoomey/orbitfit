import orbitfit.orbitfit as orb
from orbitfit.utils import (oe2ee, oe2rv)
import dateutil.parser
import datetime
import copy
import pandas as pd
import numpy as np
import BATCH_MIO.AngularBatchEst as ang
from BATCH_MIO.PlotResultsFunc import PLOTRESULTS, plot_detectability

#####################################################################################
#                                                                                   #
#                     DATA SIMULATION TO FEED THE BATCH                             #
#                                                                                   #
#####################################################################################



# DEFINE PARAMETERS FOR THE SIMULATION:

propstep = 60              # time step for propagation [s]
duration = 20           # propagation time [h]
MaxLoop = 20                # Max Loop in LS algorithm [-]
noise_std_dev_pos = 100     # Standard deviation of the client position noise for the measured versors computation [m]
noise_std_dev_vel = 10      # Standard deviation of the client velocity noise for the measured versors computation [m/s]
Epsilon = 1e-9              # condition to exit the LS loop [-]
FOV = 0                    # FOV semi-aperture to filter out-of-sight measurements [deg] - if 0 the filter is not activated
alphaMax = 0               # Maximum sun phase angle to see the target [deg] - if 0 the filter is not activated
magnitudeMax = 1e6          # 13 suggested (see comment in the related function) - if 1e6 the filter is not activated         

# position and velocity of the client in ECI frame (in meters and m/s) (GEO orbit)
epoch = dateutil.parser.parse("2021-03-09T09:40:14.991000Z")
oe_client_ECI = np.array([42164.140, 1e-6, 1e-6, 1e-6, 1e-6, 0.39])
pos, vel=  np.array(oe2rv(*oe_client_ECI))
posvel_client_ECI_m = np.concatenate([pos, vel]) * 1e3 

# The servicer is in sub-GEO (-300km radius)
oe_servicer_ECI = np.array([42164.140-300, 1e-6, 1e-6, 1e-6, 1e-6, 0.39-np.deg2rad(1.8)])                         # GEO oe for servicer [km]
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

# Perturb the client state to obtain the noisy measurements
# Number of measurements
n_measurements = len(df_relative_ECI_real)

# Add noise to the relative position and velocity measurements
# Columns 0-2 are position; columns 3-5 are velocity components

df_client_ECI_perturbed = df_client_ECI_m.copy()

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




####################################################################################################################################################################
# We have simulated the measurements, now we can proceed with the orbit fitting using the measured versor and range (with noise) as input for the fitting process. # 
# we need an initial guess for the servicer orbit (since we don't have the range) to find the computed versor and compare it with the measured one,                #
# and minimize the error with a LS algorithm. We want to retrieve the best approx for the unperturbed original orbit.                                              #
####################################################################################################################################################################


rng = np.random.default_rng(seed=666)

# initial guess: perturbed original initial OE for the client in ECI frame (same as nominal, but with small perturbations)
oe_initial_guess = oe_client_ECI.copy()
#oe_initial_guess[0] += np.random.normal(0, 0.1)       # Add noise to semi-major axis                    [km]
oe_initial_guess[1] += np.random.normal(0, 1e-2)    # Add noise to eccentricity                       [-]
oe_initial_guess[2] += np.random.normal(0, 0.001)     # Add noise to inclination                        [rad]
oe_initial_guess[3] += np.random.normal(0, 0.1)     # Add noise to argument of periapsis              [rad]
oe_initial_guess[4] += np.random.normal(0, 0.1)     # Add noise to right ascension of ascending node  [rad]
oe_initial_guess[5] += np.random.normal(0, 0.01)    # Add noise to true anomaly                       [rad]

pos, vel = oe2rv(*oe_initial_guess)
rv_initial_guess = np.concatenate((pos, vel))*1e3
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



#    ########################################
#    #                                      #
#    #            Plot results              #
#    #                                      #
#    ########################################

if magnitudeMax != 1e6:
    print("\n\nSimulated magnitude:\n")
    print({estimator.m_v.shape})
    print (estimator.m_v)

    print("\n\nIncoming light angle [deg]:\n")
    print({estimator.phi.shape})
    print (np.rad2deg(estimator.phi))

    print("\n\nDistance [m]:\n")
    print({estimator.d.shape})
    print (estimator.d)

    print("\n\nBoolean logic mask\n")
    print(estimator.mask)

# apply the mask to the real vector (if valid)

if n_loops >1:
    versor_arr_real_filt = estimator.applyMask(versor_arr_real)
    versor_arr_meas_filt = estimator.applyMask(versor_arr_meas)
    df_client_ECI_m_filt = estimator.applyMask(df_client_ECI_m)
    df_client_ECI_fit_filt = estimator.applyMask(df_client_ECI_fit)

    Plot = PLOTRESULTS(versor_arr_real, versor_arr_real_filt, versor_arr_meas_filt, estimator.versor_arr_comp, df_client_ECI_m, df_client_ECI_m_filt, 
                       df_client_ECI_fit_filt, df_state_final, df_servicer_ECI_m, estimator.b_history, n_loops)
    Plot.plotResiduals()
    Plot.plotResidualsEvolution()
    Plot.plotFinalFit()
    if estimator.detectability_filter != 1e6:
        plot_detectability(estimator.phi, estimator.d, estimator.m_v, estimator.m_v_threshold)
    



