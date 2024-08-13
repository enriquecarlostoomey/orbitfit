#!/usr/bin/python3

from orbitfit.orbitfit import TLEFit
from orbitfit.orbitfit import logger as orbitfit_logger
from orbitfit.rotate import rotate_gps
from orbitfit.io import parse_gps_data
from orbitfit.utils import interp, filter_with_tle, filter_with_mean_oe, get_mean_oe, plot_error_aar
from orbitfit.tle_utils import load_tle, write_tle, print_states_from_TLE, plot_gps_against_tle

import dateutil.parser
import matplotlib.pyplot as plt
import argparse
import logging
import json
import copy
import datetime
import os
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


def set_logger(verbose, output_dir):
    level = {0: logging.WARN, 1: logging.INFO, 2: logging.DEBUG}[verbose]
    ch = logging.StreamHandler()
    ch.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    ch_2 = logging.StreamHandler()
    ch_2.setLevel(level)
    formatter_2 = logging.Formatter('%(asctime)s - ORBITFIT - %(levelname)s - %(message)s')
    ch_2.setFormatter(formatter_2)
    orbitfit_logger.addHandler(ch_2)

    logger.info(f"Creating output file {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    ch_file = logging.FileHandler(os.path.join(output_dir, "log"))
    ch_file.setLevel(logging.DEBUG)
    ch_file.setFormatter(formatter)
    logger.addHandler(ch_file)

    ch_file_2 = logging.FileHandler(os.path.join(output_dir, "log"))
    ch_file_2.setLevel(logging.DEBUG)
    ch_file_2.setFormatter(formatter_2)
    orbitfit_logger.addHandler(ch_file_2)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Tool to fit a TLE to an input set of ephemerides data")
    group = parser.add_argument_group("Input values")
    group.add_argument("gps_filename", help="csv filename with gps data")  # TODO: Specify format
    group.add_argument("tle_filename", help="TLE filename")  # TODO: Specify format
    parser.add_argument('-v', '--verbose', action='count', default=0, help='be more verbose')
    parser.add_argument('-s', '--save', action='store_true', help="save plots")
    parser.add_argument('-o', '--output', default="output", help="Directory name in which to store results")
    parser.add_argument('-p', '--plot', action='store_true', help="show plots")
    parser.add_argument('--statesize', type=int, default=7, help="Number of states: 6 to keep bstart fix,"
                                                                 " 7 to optimize for bstart")
    filter_group = parser.add_argument_group("filter options options")
    filter_group.add_argument("--filter-mean-oe", type=json.loads, nargs="?", default=None,
                              const='{"mean_sma": 1, "mean_ecc": 0.001, "mean_inc": 0.008}',
                              help='Apply mean Orbital Elements filter. Optionally add dict with threshold as '
                                   'in example: {"mean_sma": [km], "mean_ecc": [-], "mean_inc": [deg]}')
    lstsqr_group = parser.add_argument_group("lstsqr method options")
    lstsqr_group.add_argument('--loops', type=int, default=5, help="Number of loops")
    lstsqr_group.add_argument('--percentchg', type=float, default=0.01, help="the percentage change for the finite"
                                                                             " differencing")
    lstsqr_group.add_argument('--deltaamtchg', type=float, default=1e-6, help="the amount in finite differencing below"
                                                                              " which to set a value to avoid divide by"
                                                                              " zeros")
    lstsqr_group.add_argument('--epsilon', type=float, default=1e-8, help="the rms tolerance to stop the iterations")
    return parser.parse_args()


if __name__ ==  "__main__":
    args = parse_arguments()
    set_logger(args.verbose, args.output)

    logger.info(f"Load gps data from {args.gps_filename}")
    df_gps = parse_gps_data(args.gps_filename)
    logger.info(f"GPS data available from {df_gps.index[0]} to {df_gps.index[-1]}")

    logger.info(f"Load initial TLE from {args.tle_filename}")
    initial_tle = load_tle(args.tle_filename)
    logger.info(f"TLE epoch: {initial_tle.epoch}")
    logger.info("TLE output states:")
    logger.info(print_states_from_TLE(initial_tle))

    logger.info(f"Rotate GPS data to ECI")
    df_gps_teme = rotate_gps(df_gps, method="E2T")
    df_gps_teme.to_pickle(os.path.join(args.output, "df_gps_teme.pkl"))

    if args.filter_mean_oe is not None:
        logger.info("Apply mean element filter")
        df_gps_teme_filtered = filter_with_mean_oe(df_gps_teme, thresholds=args.filter_mean_oe,
                                                   plot=args.plot, output=args.output)
        df_gps_teme_filtered.to_pickle(os.path.join(args.output, "df_gps_teme_filtered.pkl"))
        percentage_after_filter = df_gps_teme_filtered.shape[0] / df_gps_teme.shape[0] * 100
        logger.info(f"{100-percentage_after_filter:.2f}% points filtered")
    else:
        df_gps_teme_filtered = copy.deepcopy(df_gps_teme)

    logger.info(f"Setting up optimizer")
    optimizer = TLEFit(df_gps_teme_filtered, initial_TLE=initial_tle, statesize=args.statesize)
    df_in = optimizer.propagate_state(optimizer.initial_state)

    logger.info(f"Plot initial error")
    fig, _ = plot_gps_against_tle(df_gps_teme_filtered, initial_tle,
                                  title="TLE(initial)-GPS(filtered) diff",
                                  orbits=True)
    if args.save:
        fig.savefig(os.path.join(args.output, "initial_error_arr_filtered.png"))
    if args.plot:
        plt.show()

    logger.info(f"Running orbitfit optmization")
    tle_out = optimizer.run_fit(max_loops=args.loops, percentchg=args.percentchg,
                                deltaamtchg=args.deltaamtchg, epsilon=args.epsilon)

    logger.info("TLE output states:")
    logger.info(print_states_from_TLE(tle_out))

    logger.info(f"Write TLE to file {os.path.join(args.output,'output_TLE.txt')}")
    write_tle(tle_out, os.path.join(args.output, 'output.TLE'))

    logger.info("Plotting fit (using filtered data)")
    fig, _ = plot_gps_against_tle(df_gps_teme_filtered, tle_out, title="TLE-GPS(filtered) diff", orbits=True)
    if args.save:
        fig.savefig(os.path.join(args.output, "error_arr_filtered.png"))
    if args.plot:
        plt.show()

    logger.info("Plotting fit")
    fig, _ = plot_gps_against_tle(df_gps_teme, tle_out, title="TLE-GPS diff", orbits=True)
    if args.save:
        fig.savefig(os.path.join(args.output, "error_arr.png"))
    if args.plot:
        plt.show()