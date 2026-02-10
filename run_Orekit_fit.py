#!/usr/bin/python3

from orbitfit.orbitfit import OrekitFit, add_maneuver_to_config_dict, inclination_change_get_directions, STK_CONFIG, PRECISE_CONFIG
from orbitfit.orbitfit import logger as orbitfit_logger
from orbitfit.rotate import rotate_gps
from orbitfit.io import parse_gps_data
from orbitfit.utils import (interp, filter_with_tle, filter_with_mean_oe, get_mean_oe, plot_error_aar,
                            get_gps_error_percentiles)
from orbitfit.tle_utils import load_tle

import dateutil.parser
import matplotlib.pyplot as plt
import argparse
import logging
import json
import copy
import datetime
import os
import pandas as pd
import numpy as np

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


def get_parser():
    parser = argparse.ArgumentParser(description="Tool to fit and OREKIT propagated orbit to an input set of ephemerides data")
    group = parser.add_argument_group("Input values")
    group.add_argument("-f", "--filename", help="pickle filename")  # TODO: Specify format
    group.add_argument("--start", type=dateutil.parser.parse, help="start date string for telemetry")
    group.add_argument("--end", type=dateutil.parser.parse, help="end date string for telemetry")
    parser.add_argument('-v', '--verbose', action='count', default=0, help='be more verbose')
    parser.add_argument('-s', '--save', action='store_true', help="save plots")
    parser.add_argument('-o', '--output', default="output", help="Directory name in which to store results")
    parser.add_argument('-p', '--plot', action='store_true', help="show plots")
    parser.add_argument('--step', type=float, default=10.0, help="Orekit propagator step")
    parser.add_argument('--config', default='stk', help="Chose propagator config between stk and precise, or give"
                                                        "a json filename with the configuration. If input json file"
                                                        "contains InitialState, then this will be use intead of first"
                                                        "GPS measurement")
    parser.add_argument('--optimize-area', action='store_true',
                        help='Include area in the optimization loop. Area might also account for drag coefficient and'
                             'mass errors in the config file definition. Recommended for fitting long periods of time')
    filter_group = parser.add_argument_group("filter options options")
    filter_group.add_argument("--filter-tle", type=str, default=None, nargs=1,
                              help="Apply TLE filter. Requires TLE filename")
    filter_group.add_argument("--tle-threshold", type=float, default=3,
                              help="TLE filtering threshold in [km]")
    filter_group.add_argument("--filter-mean-oe", type=json.loads, nargs="?", default=None,
                              const='{"mean_sma": 1.0, "mean_ecc": 0.001, "mean_inc": 0.017}',
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
    maneuver_group = parser.add_argument_group("maneuver optimization options")
    maneuver_group.add_argument("--propulsion-date", type=lambda x: dateutil.parser.parse(x),
                                help="Propulsion date in dateutil.parse compatible format")
    maneuver_group.add_argument("--propulsion-time", type=float, default=None, help="propulsion duration in [s]")
    maneuver_group.add_argument("--propulsion-mode", type=float, default=None, choices=(17, 18, 19, 20, 21, 22, 23, 24),
                                help="Refer to ADCS propulsion modes for meaning of each number")
    maneuver_group.add_argument("--initial-thrust", type=float, default=17e-3, help="Initial guess Thrust in [N]")

    return parser


if __name__ ==  "__main__":
    parser = get_parser()
    args = parser.parse_args()
    set_logger(args.verbose, args.output)

    logger.info(f"Loading orekit configuration file from {args.config}")
    if args.config == "stk":
        logger.info("Using STK config")
        config_dict = STK_CONFIG
    elif args.config == "precise":
        logger.info("Using precise config")
        config_dict = PRECISE_CONFIG
    else:
        with open(args.config, 'r') as fp:
            config_dict = json.load(fp)

    if args.filename:
        logger.info(f"Load gps data from {args.filename}")
        df_gps_raw = pd.read_pickle(args.filename)
    else:
        parser.error('--filename is required')
    df_gps = parse_gps_data(df_gps_raw, args.save, args.plot)

    logger.info(f"Rotate GPS data to ECI")
    df_gps_eci = rotate_gps(df_gps, method="E2I")
    df_gps_eci.to_pickle(os.path.join(args.output, "df_gps_eci.pkl"))

    if args.filter_tle is not None:
        if args.propulsion_time is not None:
            logger.warning("TLE filter is not recommended for a propulsion maneuver fit")
        logger.info(f"load TLE file {args.filter_tle}")
        tle = load_tle(args.filter_tle)
        logger.info(f"Rotate GPS data from ECI to TEME")
        df_gps_teme = rotate_gps(df_gps_eci, method="I2T")
        logger.info(f"Apply TLE filtering")
        df_gps_teme_filtered = filter_with_tle(df_gps_teme, tle, threshold_km=args.filter_tle, plot=True,
                                               output=args.output)
        df_gps_eci_tle_filtered = copy.deepcopy(df_gps_eci.loc[df_gps_teme_filtered.index])
        percentage_after_filter = df_gps_eci_tle_filtered.shape[0] / df_gps_eci.shape[0] * 100
        logger.info(f"{100-percentage_after_filter:.2f}% points filtered")
        df_gps_eci_tle_filtered.to_pickle(os.path.join(args.output, "df_gps_eci_tle_filtered.pkl"))
    else:
        df_gps_eci_tle_filtered = copy.deepcopy(df_gps_eci)

    if args.filter_mean_oe is not None:
        logger.info("Apply mean element filter")
        if args.propulsion_date:
            df_gps_eci_before = df_gps_eci_tle_filtered[df_gps_eci_tle_filtered.index < args.propulsion_date]
            df_gps_eci_after = df_gps_eci_tle_filtered[
                df_gps_eci_tle_filtered.index > args.propulsion_date + datetime.timedelta(seconds=args.propulsion_time)]
            logger.info("Apply mean element filter to before data")
            df_gps_eci_before_filtered = filter_with_mean_oe(df_gps_eci_before, thresholds=args.filter_mean_oe,
                                                             plot=args.plot, output=args.output, suffix="pre")
            logger.info("Apply mean element filter to after data")
            df_gps_eci_after_filtered = filter_with_mean_oe(df_gps_eci_after, thresholds=args.filter_mean_oe,
                                                            plot=args.plot, output=args.output, suffix="post")
            df_gps_eci_oe_filtered = df_gps_eci_before_filtered.append(df_gps_eci_after_filtered)
        else:
            df_gps_eci_oe_filtered = filter_with_mean_oe(df_gps_eci_tle_filtered, thresholds=args.filter_mean_oe,
                                                         plot=args.plot, output=args.output)
        df_gps_eci_oe_filtered.to_pickle(os.path.join(args.output, "df_gps_eci_oe_filtered.pkl"))
        percentage_after_filter = df_gps_eci_oe_filtered.shape[0] / df_gps_eci_tle_filtered.shape[0] * 100
        logger.info(f"{100-percentage_after_filter:.2f}% points filtered")
    else:
        df_gps_eci_oe_filtered = copy.deepcopy(df_gps_eci_tle_filtered)

    if args.propulsion_date:
        assert args.propulsion_time is not None, "missing propulsion_time argument"
        assert args.propulsion_mode is not None, "missing propulsion_mode argument"
        logger.info("Adding maneuver group at time {} with total duration of {} s and mode {}".format(
            args.propulsion_date, args.propulsion_time, args.propulsion_mode
        ))
        if (args.propulsion_mode == 17 or args.propulsion_mode == 19):
            manoeuvres = [{"direction": np.array([1,0,0]), "date": args.propulsion_date, "duration":args.propulsion_time}]
        elif (args.propulsion_mode == 18 or args.propulsion_mode == 20):
            manoeuvres = [{"direction": np.array([-1,0,0]), "date": args.propulsion_date, "duration":args.propulsion_time}]
        elif (args.propulsion_mode == 21 or args.propulsion_mode == 23):
            manoeuvres = inclination_change_get_directions(df_gps_eci, args.propulsion_date, args.propulsion_time)
        elif (args.propulsion_mode == 25 or args.propulsion_mode == 24):
            manoeuvres = inclination_change_get_directions(df_gps_eci, args.propulsion_date, args.propulsion_time, inclination=-1)
        else:
            raise RuntimeError(f"mode {args.propulsion_mode} is not available")
        for manoeuvre in manoeuvres:
            logger.info("Adding individual maneuver at time {} with duration of {} s and direction {}".format(
                manoeuvre["date"], manoeuvre["duration"],manoeuvre["direction"]))
            add_maneuver_to_config_dict(config_dict, manoeuvre["date"], manoeuvre["duration"],
                                        manoeuvre["direction"], args.initial_thrust)

    with open(os.path.join(args.output, "config_dict_in.json"), "w") as fid:
        json.dump(config_dict, fid, indent=2)

    logger.info(f"Setting up optimizer")
    optimizer = OrekitFit(df_gps_eci_oe_filtered,
                          baseconfig=config_dict,
                          step=args.step,
                          optimize_area=args.optimize_area,
                          optimize_maneuver=bool(args.propulsion_date))
    logger.info(f"Plot initial error")
    df_out = optimizer.propagate_state(optimizer.initial_state)
    fig = plot_error_aar(df_gps_eci_oe_filtered, df_out)
    if args.save:
        fig.savefig(os.path.join(args.output, "initial_error_arr_filtered.png"))
    if args.plot:
        plt.show()

    logger.info(f"Running orbitfit optmization")
    df_out, config_dict_out = optimizer.run_fit(max_loops=args.loops,
                                                percentchg=args.percentchg,
                                                deltaamtchg=args.deltaamtchg,
                                                epsilon=args.epsilon)
    logger.info("output orekit state:")
    logger.info("inital date:{}".format(config_dict_out["Propagation"]["Start"]))
    logger.info("inital state:{}".format(config_dict_out["Propagation"]["InitialState"]))

    with open(os.path.join(args.output, "config_dict_out.json"), "w") as fid:
        json.dump(config_dict_out, fid, indent=2)

    if False:#args.propulsion_date:
        spacetrack_maneuver_file = """CCSDS_OPM_VERS = 2.0
CREATION_DATE  = {}
ORIGINATOR     = SATELLOGIC
USER_DEFINED_RELEASABILITY = PRIVATE
USER_DEFINED_CLASSIFICATION = unclassified

OBJECT_NAME       = NUSAT-123
OBJECT_ID         = 2020-003B
USER_DEFINED_NORAD_CAT_ID = 99123
CENTER_NAME       = EARTH
TIME_SYSTEM       = UTC

""".format(datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S"))
        for i, maneuver in enumerate(config_dict_out["Maneuvers"]):
            direction = maneuver["ManeuverParams"][:3]
            thrust_N = maneuver["ManeuverParams"][4]
            logger.info("Thust: {:.3f} mN".format(thrust_N*1000))
            deltaV_ms = propulsion_mode_mapper[args.propulsion_mode] * thrust_N * args.propulsion_time\
                        / config_dict_out["SpaceObject"]["Mass"]
            logger.info("DeltaV: {:.8f} m/s".format(deltaV_ms))
            deltaV_ms_x = direction[0] * deltaV_ms
            deltaV_ms_y = direction[1] * deltaV_ms
            maneuver_file = """
COMMENT Maneuver {:02d}
MAN_EPOCH_IGNITION = {}
MAN_DURATION      =   {} [s]
MAN_REF_FRAME     = RTN
MAN_DV_1          =  0.00000000 [km/s]
MAN_DV_2          =  {} [km/s]
MAN_DV_3          =  {} [km/s]
USER_DEFINED_MAN_PURPOSE = OTHER
USER_DEFINED_MAN_STATUS = DETERMINED

""".format(i+1, maneuver["Time"], maneuver["ManeuverParams"][3], deltaV_ms_x / 1000, deltaV_ms_y / 1000)
            spacetrack_maneuver_file += maneuver_file
        logger.info("Creating maneuver file")
        filename = "maneuver_file.txt"
        with open(os.path.join(args.output, filename), "w") as fid:
            fid.write(spacetrack_maneuver_file)

    df_out_filtered_interpolated = interp(df_out, df_gps_eci_oe_filtered.index)
    logger.info("Plotting fit (using filtered data)")
    fig = plot_error_aar(df_gps_eci_oe_filtered, df_out_filtered_interpolated)
    if args.save:
        fig.savefig(os.path.join(args.output, "error_arr_filtered.png"))
    if args.plot:
        plt.show()

    df_out_interpolated = interp(df_out, df_gps_eci.index)
    df_out_interpolated.to_pickle(os.path.join(args.output, "df_out_interpolated.pkl"))
    logger.info("Plotting fit")
    fig = plot_error_aar(df_gps_eci, df_out_interpolated)
    if args.save:
        fig.savefig(os.path.join(args.output, "error_arr.png"))
    if args.plot:
        plt.show()

    logger.info("Get GPS error quantiles")
    df_quantiles = get_gps_error_percentiles(df_gps_eci, df_out_interpolated,
                                             plot=args.plot, output=args.output)
    logger.info(f"{df_quantiles}")

    if args.propulsion_date:
        logger.info("Getting mean orbital elements for gps filtered and fitted data")
        df_gps_eci_oe = get_mean_oe(df_gps_eci_oe_filtered)
        df_fit_oe = get_mean_oe(df_out_interpolated)
        logger.info("plotting mean orbital elements")
        fig, axis = plt.subplots(3, 1, figsize=(14, 20))
        for ax, key in zip(axis, ['mean_sma', 'mean_ecc', 'mean_inc']):
            df_gps_eci_oe[key].plot(ax=ax, style='*', label="gps")
            df_fit_oe[key].plot(ax=ax, style='-', label="fit")
            ax.set_title(key)
        if args.save:
            fig.savefig(os.path.join(args.output, "error_oe_filtered.png"))
        if args.plot:
            plt.show()

    logger.info("Rotating results back to ECEF")
    df_out_ecef = rotate_gps(df_out_interpolated, method="I2E")
    logger.info(f"saving results to {args.output}")
    df_out_ecef.to_csv(os.path.join(args.output, "fitted_points.csv"))
