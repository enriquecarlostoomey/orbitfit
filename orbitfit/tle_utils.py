from spacetrack import SpaceTrackClient
from sgp4.io import twoline2rv, rv2twoline
from sgp4.earth_gravity import wgs84
from sgp4.propagation import sgp4, sgp4init
from sgp4.ext import jday

import datetime
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .utils import distance_between_position


def load_tle(tlefile):
    with open(tlefile, "r") as fid:
        line1 = fid.readline()
        line2 = fid.readline()
    return twoline2rv(line1, line2, wgs84)


def print_tle(tle):
    line1, line2 = rv2twoline(tle)
    print(line1)
    print(line2)


def write_tle(tle, filename):
    with open(filename, "w") as fid:
        line1, line2 = rv2twoline(tle)
        fid.write(line1)
        fid.write(line2)


def print_states_from_TLE(tle):
    if tle.mo < 0:
        tle.mo = 2*np.pi + tle.mo
    else:
        tle.mo = np.mod(tle.mo, 2*np.pi)
    if tle.argpo < 0:
        tle.argpo = 2*np.pi + tle.argpo
    else:
        tle.argpo = np.mod(tle.argpo, 2 * np.pi)
    return f"a: {tle.a*tle.whichconst.radiusearthkm:.3f} [km], T: {tle.no*24*60:.8f} [s], ecco: {tle.ecco:.8f}," \
           f" inclo: {np.degrees(tle.inclo):.4f} [deg], nodeo: {np.degrees(tle.nodeo):.4f}," \
           f" argpo: {np.degrees(tle.argpo):.4f}, mo: {np.degrees(tle.mo):.4f} [deg], bstar: {tle.bstar:.8f}"


def get_tle_lines(satellite_id, date, username=None, password=None):
    assert (username is not None and password is not None), "user as password are required to access spacetrack"
    st = SpaceTrackClient(identity=username, password=password)
    lines = st.tle_latest(norad_cat_id=satellite_id, ordinal=1, format='tle',
                          epoch=">{}".format(date.strftime("%Y-%m-%d"))).split("\n")
    return lines


def tle_list_to_tle_data(tle_list):
    """
    Takes as input a list of tle elements, and returns a list of dict with the following keys:
    tle: tle element
    start: starting time for period in which the tle is applicable
    end: ending time for period in which the tle is applicable
    """
    default_delta_tle_epoch = datetime.timedelta(hours=72)
    if len(tle_list) == 1:
        tle = tle_list[0]
        start = tle.epoch.replace(tzinfo=datetime.timezone.utc) - default_delta_tle_epoch
        end = tle.epoch.replace(tzinfo=datetime.timezone.utc) + default_delta_tle_epoch
        tle_list_with_data = [{"tle": tle, "start": start, "end": end}]
    else:
        tle_list_with_data = []
        for i, tle in enumerate(tle_list):
            if i == 0:
                start = tle.epoch.replace(tzinfo=datetime.timezone.utc) - default_delta_tle_epoch
                end = (tle.epoch + 0.5 * (tle_list[i + 1].epoch - tle.epoch)).replace(tzinfo=datetime.timezone.utc)
            elif i == len(tle_list) - 1:
                start = (tle_list[i - 1].epoch + 0.5 * (tle.epoch - tle_list[i - 1].epoch)).replace(
                    tzinfo=datetime.timezone.utc)
                end = tle.epoch.replace(tzinfo=datetime.timezone.utc) + default_delta_tle_epoch
            else:
                start = (tle_list[i - 1].epoch + 0.5 * (tle.epoch - tle_list[i - 1].epoch)).replace(
                    tzinfo=datetime.timezone.utc)
                end = (tle.epoch + 0.5 * (tle_list[i + 1].epoch - tle.epoch)).replace(tzinfo=datetime.timezone.utc)
            tle_list_with_data.append({"tle": tle, "start": start, "end": end})
    return tle_list_with_data


def plot_tle_againt_tle(tle1, tle2, periods=2, steps_per_period=72):
    start = tle1.epoch if tle1.epoch < tle2.epoch else tle2.epoch
    period_min = 2 * np.pi / tle2.no
    dt_min = np.arange(0, stop=periods * period_min,
                       step=period_min / steps_per_period)
    tle1_dt_min_0 = (start - tle1.epoch).total_seconds() / 60.0
    r_teme, _ = sgp4(tle1, dt_min + tle1_dt_min_0)
    tle1_rteme_km = np.vstack(r_teme).T

    tle2_dt_min_0 = (start - tle2.epoch).total_seconds() / 60.0
    r_teme, _ = sgp4(tle2, dt_min + tle2_dt_min_0)
    tle2_rteme_km = np.vstack(r_teme).T
    errors_aar = distance_between_position(tle1_rteme_km, tle2_rteme_km)

    plt.figure()
    plt.plot(dt_min / period_min, errors_aar, '-')
    plt.grid()
    plt.title("TLE1 - TLE2  diff")
    plt.ylabel("diff [km]")
    plt.xlabel("orbits ")
    plt.legend(["along-track", "cross-track", "radial"])
    plt.tight_layout()
    plt.show()


def plot_gps_against_tle(df_gps_teme, tle, subsampling=1, title="TLE-GPS diff", orbits=False):
    df_gps_teme_slice = df_gps_teme.iloc[::subsampling]
    orbital_period_min = (2 * np.pi / tle.no_kozai)  # min/rev
    dt_min = np.asarray((df_gps_teme_slice.index
                         - tle.epoch.replace(tzinfo=df_gps_teme_slice.index.tz)).total_seconds()/60)
    r_teme, _ = sgp4(tle, dt_min)
    tle_rteme_km = np.vstack(r_teme).T
    gps_rteme_km = df_gps_teme_slice[["randv_mks_{}".format(i) for i in range(3)]].values.astype('double') / 1000.0

    errors_aar = distance_between_position(tle_rteme_km, gps_rteme_km)
    height = np.linalg.norm(gps_rteme_km, axis=1) - wgs84.radiusearthkm
    antena_error_deg = np.degrees(
        np.arccos((2 * height ** 2 - np.linalg.norm(errors_aar, axis=1) ** 2) / (2 * height ** 2)))

    if orbits:
        x = dt_min / orbital_period_min
        xlabel = "[orbits]"
    else:
        x = dt_min
        xlabel = "[min]"
    fig, ax = plt.subplots(2, 1, sharex=True)
    ax[0].plot(x, errors_aar, '-*')
    ax[0].grid()
    ax[0].set_title(title)
    ax[0].set_ylabel("diff [km]")
    ax[0].legend(["along-track", "cross-track", "radial", "New TLE"])
    ax[1].plot(x, antena_error_deg, '--r')
    ax[1].set_ylabel("[deg]")
    ax[1].legend(["antena error"])
    ax[1].grid()
    ax[1].set_xlabel(xlabel)
    plt.tight_layout()
    return fig, ax


def modify_tle_epoch(tle, new_epoch):
    dt = new_epoch - tle.epoch
    dt_min = dt.total_seconds()/60
    tle.epoch = new_epoch
    tle.no_kozai = tle.no_kozai + tle.nddot * dt_min
    tle.mo = (tle.mo + tle.mdot * dt_min) % (2 * np.pi)
    tle.argpo = (tle.argpo + tle.argpdot * dt_min) % (2 * np.pi)
    tle.nodeo = (tle.nodeo + tle.nodedot * dt_min) % (2 * np.pi)
    tle.jdsatepoch = jday(new_epoch.year, new_epoch.month, new_epoch.day,
                             new_epoch.hour, new_epoch.minute, new_epoch.second+new_epoch.microsecond*1e-6)
    sgp4init(wgs84, False, tle.satnum, tle.jdsatepoch - 2433281.5,
             tle.bstar, tle.ecco, tle.argpo, tle.inclo, tle.mo, tle.no_kozai, tle.nodeo, tle)