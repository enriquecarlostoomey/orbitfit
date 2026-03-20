import orbitfit.orbitfit as orb
from orbitfit.utils import (oe2ee, oe2rv, rv2oe, ee2oe)
import dateutil.parser
import argparse
import datetime
import copy
import pandas as pd
import numpy as np
import os
import BATCH_MIO.AngularBatchEst as ang
from BATCH_MIO.PlotResultsFunc import PLOTRESULTS, plot_detectability

def run_orbit_estimation(
    propstep=60,              
    duration=2,           
    MaxLoop=3,                
    noise_std_dev_pos=1000,     
    noise_std_dev_vel=10,
    sigma_rad = 1.7453e-05,      
    Epsilon=1e-10,              
    FOV=0,
    FOV_offset = 0,                    
    alphaMax = 0,               
    magnitudeMax=1e6,
    epoch_str="2021-03-09T09:40:14.991000Z",
    oe_client_ECI=np.array([42164.140, 1e-6, 1e-6, 1e-6, 1e-6, 0.2]),
    oe_servicer_ECI=np.array([42164.140-300, 1e-6, 1e-6, 1e-6, 1e-6, 0.2-np.deg2rad(1.8)]),
    comments = "Nothing new?",
    seed = 100,
    block_plot = False
    ):
    """
    Runs a single Angle-Only Initial Orbit Determination and Batch Least Squares 
    estimation simulation.

    Inputs:
        propstep (int): Time step for the orbital propagation [s].
        duration (float): Total duration of the simulation arc [h].
        MaxLoop (int): Maximum number of iterations for the Least Squares optimizer.
        noise_std_dev_pos (float): Standard deviation of the initial guess position error [m].
        noise_std_dev_vel (float): Standard deviation of the initial guess velocity error [m/s].
        sigma_rad (float): Standard deviation of the angular measurement noise from the image processing [rad].
        Epsilon (float): Relative error variation threshold to exit the optimization loop.
        FOV (float): Camera Field of View semi-aperture [deg]. 0 means filter disabled.
        fov_offset: Pointing direction. if positive pointing forward, otherwise pointin backward (if <=1e-6 or not specified: zenithal)
        alphaMax (float): Maximum allowed Sun phase angle [deg]. 0 means filter disabled.
        magnitudeMax (float): Maximum visual magnitude threshold for detectability. 1e6 means disabled.
        epoch_str (str): Simulation start epoch in ISO 8601 format.
        oe_client_ECI (np.ndarray): Target ground truth orbital elements [a, e, i, Omega, omega, M].
        oe_servicer_ECI (np.ndarray): Observer true orbital elements [a, e, i, Omega, omega, M].
        comments (str): Custom string to append to the saved note.txt file.
        seed (int): Random seed for reproducibility of the measurement noise.
        block_plot (boolean): Stop the run at each plot? (default False)

    Outputs:
        bool: Returns True upon successful completion. (Also generates and saves 
              a dedicated folder containing the note.txt summary and output plots).
    """
    
    print(f"\n===========================================================================")
    print(f"--- STARTING SIMULATION: {duration}h duration, {noise_std_dev_pos}m noise ---")
    print(f"===========================================================================\n")

    # Start the timer
    start_wall = datetime.datetime.now()
    
    # Parse the epoch string
    epoch = dateutil.parser.parse(epoch_str)
    
    # ==============================================================================
    # --- 1. INITIALIZATION & PROPAGATION (Paste your existing code here)        ---
    # ==============================================================================
         
    # position and velocity of the client in ECI frame (in meters and m/s) (GEO orbit)
    pos, vel=  np.array(oe2rv(*oe_client_ECI))
    rv_client_groundTruth = np.concatenate([pos, vel]) * 1e3 

    # The servicer is in sub-GEO (-300km radius)
    pos, vel=  np.array(oe2rv(*oe_servicer_ECI))
    rv_servicer_groundTruth = np.concatenate([pos, vel]) * 1e3                                          # GEO coordinates for servicer [m, m/s]


    ## INITIAL ORBIT PROPAGATION

    # Define the configuration for the orbit propagation
    servicer_config = copy.deepcopy(orb.STK_CONFIG)
    propagation_config = dict()
    propagation_config["InitialState"] = rv_servicer_groundTruth.tolist()
    propagation_config["Step"] = propstep
    propagation_config['Start'] = epoch.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    propagation_config['End'] = (epoch+datetime.timedelta(hours= duration)).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    servicer_config["Propagation"] = propagation_config

    client_config = copy.deepcopy(orb.STK_CONFIG)
    propagation_config = dict()
    propagation_config["InitialState"] = rv_client_groundTruth.tolist()
    propagation_config["Step"] = propstep
    propagation_config['Start'] = epoch.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    propagation_config['End'] = (epoch+datetime.timedelta(hours= duration)).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    client_config["Propagation"] = propagation_config

    # Propagation
    print("\nInital orbits propagation...")
    index, data = orb.propagate_orbits_wrapper(servicer_config)
    df_servicer_groundTruth= pd.DataFrame(data=np.array(data), index=pd.DatetimeIndex(index), columns=["randv_mks_{}".format(i) for i in range(6)])

    index, data = orb.propagate_orbits_wrapper(client_config)
    df_client_groundTruth= pd.DataFrame(data=np.array(data), index=pd.DatetimeIndex(index), columns=["randv_mks_{}".format(i) for i in range(6)])


    ## REAL DATA

    # Now we have to retrieve the relative position between client and servicer (versor as seen  from the servicer, no range)
    # this will serve as "real" relative position that we want to retrieve. We will add noise to get the "measured data" that we'll use for the orbit fitting

    df_relative_groundTruth = df_client_groundTruth.copy()
    df_relative_groundTruth.iloc[:, :3] -= df_servicer_groundTruth.iloc[:, :3].values  # Subtract servicer position from client position
    df_relative_groundTruth["range_m"] = np.linalg.norm(df_relative_groundTruth.iloc[:, :3].values, axis=1)

    # compute unit direction vectors and store components separately
    versor_arr_groundTruth = df_relative_groundTruth.iloc[:, :3].values / df_relative_groundTruth["range_m"].values.reshape(-1, 1)


    ## MEASURED DATA

    # Perturb the real versors to obtain the measured ones.

    versor_arr_measured = copy.deepcopy(versor_arr_groundTruth)
    rng = np.random.default_rng(seed=seed)

    # 1. Gaussian noise vector
    noise = rng.normal(0, sigma_rad, versor_arr_measured.shape)

    # 2. Creating normal perturbation vectors from noise
    dot_products = np.sum(noise * versor_arr_measured, axis=1, keepdims=True)
    noise_perp = noise - dot_products * versor_arr_measured

    # 3. Aggiungi il rumore perpendicolare al versore originale
    versor_arr_measured = versor_arr_measured + noise_perp

    # 4. Rinormalizza per garantire che siano ancora versori unitari
    norms = np.linalg.norm(versor_arr_measured, axis=1, keepdims=True)
    versor_arr_measured = versor_arr_measured / norms




    ####################################################################################################################################################################
    # We have simulated the measurements, now we can proceed with the orbit fitting using the measured versor and range (with noise) as input for the fitting process. # 
    # we need an initial guess for the servicer orbit, (r,v) state vector. (since we don't have the range) to find the computed versor and compare it with the         #
    # measured one, and minimize the error with a LS algorithm. We want to retrieve the best approx for the unperturbed original orbit.                                #
    ####################################################################################################################################################################


    rng = np.random.default_rng(seed=seed)

    # initial guess: perturbed original initial OE for the client in ECI frame (same as nominal, but with small perturbations)
    rv_initial_guess = df_client_groundTruth.iloc[0].to_numpy(copy=True)
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
    df_initial_guess= pd.DataFrame(data=np.array(data), index=pd.DatetimeIndex(index), columns=["randv_mks_{}".format(i) for i in range(6)])

    # Initialize the class

    print("\nInitializing Optimizer...\n")
    estimator = ang.Optimizer(
        df_initial_guess=df_initial_guess,            # DataFrame (N,6) della prima propagazione guess
        df_client_groundTruth=df_client_groundTruth,         # 
        df_servicer=df_servicer_groundTruth,          # DataFrame (N,6) della posizione del servicer (reale)
        versor_arr_meas=versor_arr_measured,        # Array (N,3) dei versori misurati (osservazioni)
        config=propagation_config,              # Solo il dizionario della propagazione (Step, Start, End)
        max_loops=MaxLoop,                      # (Opzionale) Numero massimo di iterazioni
        epsilon=Epsilon,
        fov = FOV,
        fov_offset= FOV_offset,
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
    if estimator.errorFlag == False:
        # Creating the folder with dynamic name
        time_suffix = start_wall.strftime("%H%M%S")
        folder_name = f"test_{duration}h_{propstep}sStep_{magnitudeMax}maxMag_{time_suffix}"
        folder_name = os.path.join("tests", folder_name)

        # additional folder if the previous one was already existing
        os.makedirs(folder_name, exist_ok=True)

        # number of points for saving
        total_points = len(versor_arr_measured)
        if hasattr(estimator, 'mask') and estimator.mask is not None:
            valid_points = np.sum(estimator.mask)
        else:
            valid_points = total_points


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
            f.write(f"angular error on measured versors = {sigma_rad} [rad]\n")
            f.write(f"Epsilon (convergency criteria) = {Epsilon}\n")
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
            f.write(f"Optimization exit reason: {estimator.exitReason}\n\n")
            f.write(f"Original OE elements: \n{oe_client_ECI}  [a e i RAAN w v]\n")
            f.write(f"final, interpolated OE elements: \n{ee2oe(*ee_final)}  [a e i RAAN w v]")            
            
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
    
        versor_arr_groundTruth_filt = estimator.applyMask(versor_arr_groundTruth)
        versor_arr_measured_filt = estimator.applyMask(versor_arr_measured)
        df_client_groundTruth_filt = estimator.applyMask(df_client_groundTruth)
        df_initial_guess_filt = estimator.applyMask(df_initial_guess)

        # Passiamo folder_name alla classe PLOTRESULTS
        Plot = PLOTRESULTS(versor_arr_groundTruth, versor_arr_groundTruth_filt, versor_arr_measured_filt, 
                        estimator.versor_arr_comp, df_client_groundTruth, df_client_groundTruth_filt, 
                        df_initial_guess_filt, df_state_final, df_servicer_groundTruth, 
                        estimator.b_history, n_loops, save_dir=folder_name, Block=block_plot)
        
        Plot.plotResiduals()
        Plot.plotResidualsEvolution()
        Plot.plotFinalFit() # I risultati del fit verranno stampati a terminale e salvati
        
        if estimator.m_v_threshold != 1e6:
            plot_detectability(estimator.phi, estimator.d, estimator.m_v, estimator.m_v_threshold, save_dir=folder_name)

        print("\n\nresults saved in:")
        print(folder_name, "\n\n")


########################################################################################################################

if __name__ == "__main__":
    # Initialize the argument parser
    parser = argparse.ArgumentParser(description="Run Batch Least Squares Orbit Estimation")
    
    # Define all available command-line arguments
    parser.add_argument("--propstep", type=int, default=60, help="Time step for propagation [s]")
    parser.add_argument("--duration", type=float, default=2.0, help="Propagation time [h]")
    parser.add_argument("--MaxLoop", type=int, default=3, help="Maximum number of LS iterations")
    
    # Using dest to match your function's parameter names internally
    parser.add_argument("--noise_pos", type=float, default=1000.0, dest="noise_std_dev_pos", help="Initial guess position error [m]")
    parser.add_argument("--noise_vel", type=float, default=10.0, dest="noise_std_dev_vel", help="Initial guess velocity error [m/s]")
    
    parser.add_argument("--sigma_rad", type=float, default=1.7453e-05, help="Angular measurement noise [rad]")
    parser.add_argument("--Epsilon", type=float, default=1e-8, help="Exit condition threshold")
    parser.add_argument("--FOV", type=float, default=0.0, help="FOV semi-aperture [deg]")
    parser.add_argument("--alphaMax", type=float, default=0.0, help="Maximum sun phase angle [deg]")
    parser.add_argument("--magnitudeMax", type=float, default=1e6, help="Maximum visual magnitude threshold")
    parser.add_argument("--epoch_str", type=str, default="2021-03-09T09:40:14.991000Z", help="Start epoch string")
    parser.add_argument("--block_plot", type=bool, default=False, help="Stop the run at each plot (True-False)")
    
    # nargs=6 allows passing exactly 6 floats from the terminal
    parser.add_argument("--oe_client", type=float, nargs=6, help="Target orbital elements (6 values)")
    parser.add_argument("--oe_servicer", type=float, nargs=6, help="Observer orbital elements (6 values)")
    
    parser.add_argument("--comments", type=str, default="\nNothing new?\n", help="Notes to append in the output text file")
    parser.add_argument("--seed", type=int, default=100, help="Random seed for noise generation")

    # Parse the arguments from the terminal
    args = parser.parse_args()

    # Build a dictionary of arguments to pass to the function
    kwargs = {
        "propstep": args.propstep,
        "duration": args.duration,
        "MaxLoop": args.MaxLoop,
        "noise_std_dev_pos": args.noise_std_dev_pos,
        "noise_std_dev_vel": args.noise_std_dev_vel,
        "sigma_rad": args.sigma_rad,
        "Epsilon": args.Epsilon,
        "FOV": args.FOV,
        "alphaMax": args.alphaMax,
        "magnitudeMax": args.magnitudeMax,
        "epoch_str": args.epoch_str,
        "comments": args.comments,
        "seed": args.seed,
        "block_plot": args.block_plot
    }

    # Only override the numpy arrays if the user specifically provided them via command line
    if args.oe_client is not None:
        kwargs["oe_client_ECI"] = np.array(args.oe_client)
    if args.oe_servicer is not None:
        kwargs["oe_servicer_ECI"] = np.array(args.oe_servicer)

    # Execute the main function unpacking the dictionary
    run_orbit_estimation(**kwargs)

