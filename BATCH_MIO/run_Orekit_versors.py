#!/usr/bin/env python3
"""
NOT TESTED YET!!!!!

Command-line interface tool for Angles-Only Batch Least Squares Orbit Determination. 
It parses user configurations, propagates an initial guess orbit from initial guess state using Orekit, and runs 
the Levenberg-Marquardt optimizer to refine the satellite's orbital elements based on 
relative line-of-sight measurements (versors). Finally, it exports the optimized 
trajectory and elements to a specified output directory.

Input (Command Line Arguments):
    --output (str): Directory name to store the results (default: "output_ang").
    --step (float): Orekit propagator time step in seconds (default: 30.0).
    --duration (float): Days of propagation (default: 1/24, i.e., 1 hour).
    --loops (int): Maximum loops in the LS algorithm (default: 30).
    --damping (float): Initial Levenberg-Marquardt damping lambda (default: 0.001).
    --epsilon (float): Relative tolerance to stop the iterations (default: 1e-12).
    --percentchg (float): Percentage change for the Jacobian finite differencing (default: 1e-6).
    --deltaamtchg (float): Minimum absolute delta amount for finite differencing (default: 1e-7).    
Output (Files saved to disk in the --output directory):
    df_state_final.csv: A CSV file containing the propagated state (RV) of the optimized orbit.
    optimized_elements.json: A JSON file containing the final Equinoctial Elements, 
                             number of iterations, and total processing time.

Call in the terminal:

    python run_Orekit_ang.py \
  --servicer data/df_servicer.pkl \
  --meas data/versors_meas.npy \
  --real data/versors_real.npy \
  --epoch "2021-03-09T16:08:14.991Z" \
  --rv-guess "-2528776.63,-637910.22,-6454161.01,5660.80,4262.99,-2623.90" \
  --step 60.0 \
  --loops 30
"""

import argparse
import json
import copy
import datetime
import os
import pandas as pd
import numpy as np
import datetime
import dateutil.parser

# Importa i tuoi moduli
import orbitfit.orbitfit as orb
from orbitfit.utils import rv2oe, oe2ee
import AngularBatchEst as ang 

def get_parser():
    parser = argparse.ArgumentParser(description="Tool for Angles-Only Batch Least Squares Orbit Determination")

    # --- Data Input Group ---
    data_group = parser.add_argument_group("Data Input Options")
    data_group.add_argument('--servicer', required=True, help="Path to .pkl Servicer DataFrame")
    data_group.add_argument('--meas', required=True, help="Path to .npy measured versors")
    data_group.add_argument('--epoch', required=True, help="Start epoch in ISO (es. 2021-03-09T16:08:14Z)")
    data_group.add_argument('--rv-guess', required=True, help="initial RV state [m, m/s] (es. 'x,y,z,vx,vy,vz')")
    
    # --- Output Group ---
    parser.add_argument('-o', '--output', default="output_ang", help="Directory name in which to store results")
    
    # --- Propagation Group ---
    prop_group = parser.add_argument_group("Propagation options")
    prop_group.add_argument('--step', type=float, default=30.0, help="Orekit propagator time step [s] (default: 30)")
    prop_group.add_argument('--duration', type=float, default=1/24, help="Days of propagation (default: 1/24, i.e., 1 hour)")
    
    # --- Optimizer (LSQR) Group ---
    lstsqr_group = parser.add_argument_group("Least Squares algorithm options")
    lstsqr_group.add_argument('--loops', type=int, default=30, help="Max loops in LS algorithm (default: 30)")
    lstsqr_group.add_argument('--damping', type=float, default=0.001, help="Initial Levenberg-Marquardt damping lambda (default: 0.001)")
    lstsqr_group.add_argument('--epsilon', type=float, default=1e-12, help="Relative tolerance to stop the iterations (default: 1e-12)")
    lstsqr_group.add_argument('--percentchg', type=float, default=1e-6, help="Percentage change for finite differencing (default: 1e-6)")
    lstsqr_group.add_argument('--deltaamtchg', type=float, default=1e-7, help="Minimum delta amount for finite differencing (default: 1e-7)")
    
    # NOTA: Per caricare i dati reali da terminale, dovresti aggiungere argomenti come:
    # parser.add_argument('--meas-file', required=True, help="Pickle file with versor measurements")
    # parser.add_argument('--servicer-file', required=True, help="Pickle file with servicer ephemeris")
    # parser.add_argument('--epoch', required=True, type=dateutil.parser.parse, help="Start epoch")
    
    return parser

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    
    # Output folder creation (if any)
    os.makedirs(args.output, exist_ok=True)
    
    print(f"--- Starting Angles-Only Batch Estimator ---")
    print(f"Output directory: {args.output}")
    print(f"Propagation: duration={args.duration} days, step={args.step}s")
    print(f"Optimizer: max_loops={args.loops}, damping={args.damping}, epsilon={args.epsilon}")

    #####################################################################################
    #                                  DATA LOADING                                     #
    #####################################################################################
    
    print("\nUploading data...")
    
    # 1. Parsing dell'epoca
    epoch = dateutil.parser.parse(args.epoch)
    
    # 2. Lettura del DataFrame del servicer (.pkl)
    df_servicer_ECI_m = pd.read_pickle(args.servicer)
    
    # 3. Lettura degli array dei versori 
    # (Usa np.load se li salvi con np.save, altrimenti adatta se usi pd.read_pickle)
    versor_arr_meas = np.load(args.meas)
    versor_arr_real = np.load(args.real)
    
    # 4. Parsing del RV guess iniziale (da stringa ad array numpy)
    rv_list = [float(x) for x in args.rv_guess.split(',')]
    rv_initial_guess = np.array(rv_list)
    
    # 5. Conversione da RV (metri) -> OE (km) -> EE (radianti)
    # Ricorda che rv2oe si aspetta i km, quindi dividiamo per 1000
    oe_initial_guess = np.array(rv2oe(rv_initial_guess[0:3]*1e-3, rv_initial_guess[3:6]*1e-3))
    ee_initial_guess = np.array(oe2ee(*oe_initial_guess))
    
    
    #####################################################################################
    #                        HERE STARTS RUN_OREKIT_ANG                                 #
    #####################################################################################

    start_wall = datetime.datetime.now()  # Record computer time at start

    # --- 1. Guessed orbit propagation ---
    client_config = copy.deepcopy(orb.STK_CONFIG)
    propagation_config = dict()
    propagation_config["InitialState"] = rv_initial_guess.tolist()
    propagation_config["Step"] = args.step  # Usa l'argomento da terminale
    propagation_config['Start'] = epoch.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    propagation_config['End'] = (epoch + datetime.timedelta(days=args.duration)).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    
    client_config["Propagation"] = propagation_config

    print("\nPropagating initial guess...")
    index, data = orb.propagate_orbits_wrapper(client_config)
    df_client_ECI_fit = pd.DataFrame(data=np.array(data), index=pd.DatetimeIndex(index), columns=[f"randv_mks_{i}" for i in range(6)])

    # --- 2. Initialize the Optimizer class ---
    print("Initializing Optimizer...")
    estimator = ang.Optimizer(
        ee_initialguess=ee_initial_guess,      
        df_client=df_client_ECI_fit,           
        df_servicer=df_servicer_ECI_m,         
        versor_arr_meas=versor_arr_meas,
        config=propagation_config,             
        config_0=client_config,                
        damping_lambda=args.damping,         
        max_loops=args.loops,                  
        epsilon=args.epsilon,              
        deltaamtchg=args.deltaamtchg,           
        percentchg=args.percentchg             
    )

    # --- 3. Call to Batch estimator ---
    print("Starting optimization loop...")
    df_state_final, ee_final, n_loops_done = estimator.LSLoop()

    # --- 4. Save Outputs ---
    print(f"\nOptimization finished after {n_loops_done} loops. Saving results to '{args.output}'...")
    
    # Salva il dataframe finale come CSV (o Pickle)
    df_state_final.to_csv(os.path.join(args.output, "df_state_final.csv"))
    # df_state_final.to_pickle(os.path.join(args.output, "df_state_final.pkl"))
    
    # Salva gli elementi equinoziali finali in un file JSON
    final_results = {
        "ee_final": ee_final.tolist() if isinstance(ee_final, np.ndarray) else list(ee_final),
        "total_loops_executed": n_loops_done,
        "processing_time_seconds": (datetime.datetime.now() - start_wall).total_seconds()
    }
    with open(os.path.join(args.output, "optimized_elements.json"), "w") as f:
        json.dump(final_results, f, indent=4)

    # --- 5. Timing ---
    end_wall = datetime.datetime.now()    # Record computer time at end
    execution_time = end_wall - start_wall
    
    print("\n=======================================================")
    print(f"Start time: {start_wall} ")
    print(f"End time: {end_wall} ")
    print(f"Total processing time: {execution_time} seconds")
    print("=======================================================\n")