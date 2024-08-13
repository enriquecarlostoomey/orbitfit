import datetime
import numpy as np
from quaternions import Quaternion
from .utils import angle2dcm
import os.path
import csv
import logging

logger = logging.getLogger(__name__)

UTC_SECONDS_SINCE_UNIX_EPOCH = 946728000.0
SECONDS_IN_CENTURY = 60 * 60 * 24 * 36525.0
MEEUS_DEG = [280.46061837, 36525.0 * 360.98564736629, 0.000387933, -1.0 / 38710000.0]
# The date of 2017/03/15 is arbitrary and is used only because it is the same currently used in AOCS flight software.

ECI2ECEF_reference_datetime = datetime.datetime(2021, 3, 9, 16, 8, 14, 991000, tzinfo=datetime.timezone.utc)
ECI2ECEF_reference_matrix = np.array([[ 0.647689994635446,   0.761903977446722,                   0],
                                      [-0.761903977446722,   0.647689994635446,                   0],
                                      [                 0,                   0,   1.000000000000000]])

GREENWICH_DOT_RADSEC = 7.292115855377001e-05
DELTAAT = -37  # obtained from https://datacenter.iers.org/data/latestVersion/bulletinC.txt


def load_aeroCIP2006():
    with open(os.path.join(os.path.dirname(__file__), "I2E_data", "aeroCIP2006_X.csv"), 'r') as fid:
        aeroX = np.loadtxt(fid, delimiter=',')

    with open(os.path.join(os.path.dirname(__file__), "I2E_data", "aeroCIP2006_Y.csv"), 'r') as fid:
        aeroY = np.loadtxt(fid, delimiter=',')

    with open(os.path.join(os.path.dirname(__file__), "I2E_data", "aeroCIP2006_S.csv"), 'r') as fid:
        aeroS = np.loadtxt(fid, delimiter=',')
    return aeroX, aeroY, aeroS


aeroX, aeroY, aeroS = load_aeroCIP2006()


def load_eop(filename=None):
    """
    Loads EOP data and returns scipy interp1 functions to interpolate thought time.
    Interpolators maintain the value constant if a date outside the data range is asked.
    :param filename: If None, default iers data value is used (extracted from
    https://datacenter.iers.org/data/latestVersion/223_EOP_C04_14.62-NOW.IAU1980223.txt).
    :return: dictionary with structure [year][month][day]{"DELTAUT1":float, "POLARMOTION":tuple,
    "dCIP":tuple}
    """
    eop_data = {}  # type: dict
    if filename is None:
        filename = os.path.join(os.path.dirname(__file__), "I2E_data", "eopc04_IAU2000.txt")
    with open(filename, 'r') as fid:
        spamreader = csv.reader(fid, delimiter=' ', quotechar='|', skipinitialspace=True)
        i = 0
        while True:
            try:
                data = spamreader.__next__()
                if i > 15:
                    year = int(data[0])
                    month = int(data[1])
                    day = int(data[2])
                    DELTAUT1 = float(data[6])
                    POLAR_MOTION_X = float(data[4])/3600*np.pi/180
                    POLAR_MOTION_Y = float(data[5])/3600*np.pi/180
                    LOD = float(data[7])
                    dCIP_X = float(data[8]) / 3600 * np.pi / 180
                    dCIP_Y = float(data[9]) / 3600 * np.pi / 180
                    if year not in eop_data.keys():
                        eop_data[year] = {}
                    if month not in eop_data[year].keys():
                        eop_data[year][month] = {}
                    if day in eop_data[year][month].keys():
                        logging.error("repeated day: %s - %s - %s", year, month, day)

                    eop_data[year][month][day] = {"DELTAUT1": DELTAUT1,
                                                  "POLARMOTION": (POLAR_MOTION_X, POLAR_MOTION_Y),
                                                  "LOD": LOD,
                                                  "dCIP": (dCIP_X, dCIP_Y)}
                i += 1
            except Exception:
                break

    return eop_data


eop_data = load_eop()


def get_eop_values(year, month, day):
    date = datetime.date(year, month, day)
    min_year = min(list(eop_data.keys()))
    min_month = min(list(eop_data[min_year].keys()))
    min_day = min(list(eop_data[min_year][min_month].keys()))
    min_date = datetime.date(min_year, min_month, min_day)
    max_year = max(list(eop_data.keys()))
    max_month = max(list(eop_data[max_year].keys()))
    max_day = max(list(eop_data[max_year][max_month].keys()))
    max_date = datetime.date(max_year, max_month, max_day)
    if date > max_date:
        return eop_data[max_year][max_month][max_day]
    elif date < min_date:
        return eop_data[min_year][min_month][min_day]
    else:
        return eop_data[year][month][day]


def get_I2E_theta_from_time(utc_timestamp_sec):
    """Formula 11.4 of Meeus "Astronomical Algorithms" for the GMST (Greenwich Mean Sidereal Time)
    Expects UTC timestamp in seconds.
    Returns theta angle in rads"""
    J2000_2_Now_JC = (utc_timestamp_sec - UTC_SECONDS_SINCE_UNIX_EPOCH) / SECONDS_IN_CENTURY  # julian years
    return np.deg2rad(MEEUS_DEG[0] + MEEUS_DEG[1] * J2000_2_Now_JC + MEEUS_DEG[2] * J2000_2_Now_JC ** 2 +
                      MEEUS_DEG[3] * J2000_2_Now_JC ** 3)


def get_MEEUS_I2E_matrix(utc_timestamp_sec):
    """Expects UTC timestamp in seconds"""
    theta = get_I2E_theta_from_time(utc_timestamp_sec)
    return np.array([[np.cos(theta), -np.sin(theta), 0.0],
                     [np.sin(theta), np.cos(theta), 0.0],
                     [0.0, 0.0, 1.0]]).T


def get_MEEUS_I2E_quaternion(utc_timestamp_sec):
    """Expects UTC timestamp in seconds.
    Returns a quaternion object from matias quaternion library"""
    theta = get_I2E_theta_from_time(utc_timestamp_sec)
    return Quaternion(np.cos(theta / 2.0), 0, 0, np.sin(theta / 2.0))


def get_reference_I2E_matrix(utc_timestamp_sec):
    """Expects UTC timestamp in seconds. Uses I2E matrix calculated with matlab dcmeci2ecef() function to get nutation
    and precession for a reference date. The date of 2017/03/15 is arbitrary and is used only because it is the same
    currently used in AOCS flight software. Then, it uses Meeus's polynomial formula to extrapolate from
    that reference date"""
    ECI2ECEF_reference_timestamp = ECI2ECEF_reference_datetime.timestamp()
    theta_reference = get_I2E_theta_from_time(ECI2ECEF_reference_timestamp)
    theta_now = get_I2E_theta_from_time(utc_timestamp_sec)
    delta_theta = theta_now - theta_reference
    ECEF_reference2ECEF = np.array([[np.cos(delta_theta), -np.sin(delta_theta), 0.0],
                                    [np.sin(delta_theta), np.cos(delta_theta), 0.0],
                                    [0.0, 0.0, 1.0]]).T
    ECI2ECEF = ECEF_reference2ECEF.dot(ECI2ECEF_reference_matrix)
    return ECI2ECEF


def get_reference_I2E_quaternion(utc_timestamp_sec):
    """Expects UTC timestamp in seconds.
    Returns a quaternion object from matias quaternion library"""
    ECI2ECEF = get_reference_I2E_matrix(utc_timestamp_sec)
    return Quaternion.from_matrix(ECI2ECEF)


def get_reference_I2E_matrix_constant_omega(utc_timestamp_sec):
    """Expects UTC timestamp in seconds. Uses I2E matrix calculated with matlab dcmeci2ecef() function to get nutation
        and precession for a reference date. Then, it uses constant I2E omega to rotate along Z"""
    ECI2ECEF_reference_timestamp = ECI2ECEF_reference_datetime.timestamp()
    delta_time_seconds = utc_timestamp_sec - ECI2ECEF_reference_timestamp
    delta_theta = GREENWICH_DOT_RADSEC * delta_time_seconds
    ECEF_reference2ECEF = np.array([[np.cos(delta_theta), -np.sin(delta_theta), 0.0],
                                    [np.sin(delta_theta), np.cos(delta_theta), 0.0],
                                    [0.0, 0.0, 1.0]]).T
    ECI2ECEF = ECEF_reference2ECEF.dot(ECI2ECEF_reference_matrix)
    return ECI2ECEF


def get_reference_I2E_quaternion_constant_omega(utc_timestamp_sec):
    """Expects UTC timestamp in seconds.
    Returns a quaternion object from matias quaternion library"""
    ECI2ECEF = get_reference_I2E_matrix_constant_omega(utc_timestamp_sec)
    return Quaternion.from_matrix(ECI2ECEF)


def get_omega_I2E_E_rad_sec(utc_timestamp_sec=None):
    """ If timestamp available, calculates omega_I2E using derivated Meeus expresion.
    Else, returns GREENWICH_DOT value.
    :param utc_timestamp_sec: UTC timestamp in secods
    :return: omega_I2E expressed in ECEF, in rad/sec
    """
    if utc_timestamp_sec:
        J2000_2_Now_JC = (utc_timestamp_sec - UTC_SECONDS_SINCE_UNIX_EPOCH) / SECONDS_IN_CENTURY  # julian years
        omega_z = np.deg2rad(((MEEUS_DEG[1] + 2 * MEEUS_DEG[2] * J2000_2_Now_JC +
                               3 * MEEUS_DEG[3] * J2000_2_Now_JC ** 2) / SECONDS_IN_CENTURY))
        I2E = get_reference_I2E_matrix(utc_timestamp_sec)
        omega_I2E_E = I2E.dot(np.array([0, 0, omega_z]))
    else:
        omega_I2E_E = np.array([0, 0, GREENWICH_DOT_RADSEC])

    return omega_I2E_E


def juliandate(date):
    year = date.year
    month = date.month
    day = date.day
    hour = date.hour
    minute = date.minute
    sec = date.second + date.microsecond*1e-6
    if month <= 2:  # January & February
        year = year - 1.0
        month = month + 12.0
    dayFraction = (hour + minute / 60 + sec / 3600) / 24
    day = (np.floor(365.25 * (year + 4716.0)) + np.floor(30.6001 * (month + 1.0)) + 2.0 -
           np.floor(year / 100.0) + np.floor(np.floor(year / 100.0) / 4.0) + day - 1524.5)
    return dayFraction + day


def mjuliandate(date):
    year = date.year
    month = date.month
    day = date.day
    hour = date.hour
    minute = date.minute
    sec = date.second + date.microsecond*1e-6
    if month <= 2:  # January & February
        year = year - 1.0
        month = month + 12.0
    dayFraction = (hour + minute / 60 + sec / 3600) / 24
    day = (np.floor(365.25 * year) + np.floor(30.6001 * (month + 1.0)) + 2.0 -
           np.floor(year / 100.0) + np.floor(np.floor(year / 100.0) / 4.0) + day - 679006.0)
    return dayFraction + day


def dcmeci2ecef(UTC, DELTAAT, DELTAUT1, POLARMOTION, dCIP=(0, 0)):
    """
    :param UTC:
    :param DELTAAT:
    :param DELTAUT1:
    :param POLARMOTION:
    :param dCIP: 2 element array for the adjustment in radians to the location of the Celestial Intermediate Pole (CIP)
     along the x (dDeltaX) and y (dDeltaY) axis in rad.
    :return: DCM I2E
    """
    # Julian date for terrestrial time
    jdTT = mjuliandate(UTC + datetime.timedelta(seconds=(DELTAAT + 32.184)))
    # Number of Julian centuries since J2000 for terrestrial time.
    tTT = (jdTT - 51544.5) / 36525
    tTT2 = tTT * tTT
    tTT3 = tTT2 * tTT
    tTT4 = tTT3 * tTT
    tTT5 = tTT4 * tTT
    # Julian date for UT1
    jdUT1 = mjuliandate(UTC + datetime.timedelta(seconds=DELTAUT1))
    # Elapsed Julian days since J2000
    jdElapsed = jdUT1 - 51544.5
    jdFraction = jdElapsed % 1

    # Polar motion
    # TIO locator
    sp = np.deg2rad(-0.000047 * tTT / 3600)
    # Transformation matrix for polar motion
    W = angle2dcm(sp, -POLARMOTION[0], -POLARMOTION[1], 'ZYX')

    # Earth rotation
    # Earth rotation angle
    thetaERA = (2 * np.pi * (jdFraction + 0.7790572732640 + 0.00273781191135448 * jdElapsed)) % (2 * np.pi)
    R = angle2dcm(thetaERA, 0, 0, 'ZYX')

    # Celestial Motion of the CIP
    # Arguments for lunisolar nutation
    mMoon = 485868.249036 + 1717915923.2178 * tTT + 31.8792 * tTT2 + 0.051635 * tTT3 - 0.00024470 * tTT4
    mSun = 1287104.793048 + 129596581.0481 * tTT - 0.5532 * tTT2 + 0.000136 * tTT3 - 0.00001149 * tTT4
    umMoon = 335779.526232 + 1739527262.8478 * tTT - 12.7512 * tTT2 - 0.001037 * tTT3 + 0.00000417 * tTT4
    dSun = 1072260.703692 + 1602961601.2090 * tTT - 6.3706 * tTT2 + 0.006593 * tTT3 - 0.00003169 * tTT4
    omegaMoon = 450160.398036 - 6962890.5431 * tTT + 7.4722 * tTT2 + 0.007702 * tTT3 - 0.00005939 * tTT4

    # Arguments for planetary nutation
    lMercury = 4.402608842 + 2608.7903141574 * tTT
    lVenus = 3.176146697 + 1021.3285546211 * tTT
    lEarth = 1.753470314 + 628.3075849991 * tTT
    lMars = 6.203480913 + 334.06124267 * tTT
    lJupiter = 0.599546497 + 52.9690962641 * tTT
    lSaturn = 0.874016757 + 21.329910496 * tTT
    lUranus = 5.481293872 + 7.4781598567 * tTT
    lNeptune = 5.311886287 + 3.8133035638 * tTT
    pa = 0.02438175 * tTT + 0.00000538691 * tTT2

    # Vector arrangement for series evaluation
    nutationV = (np.hstack((np.deg2rad(np.array([mMoon, mSun, umMoon, dSun, omegaMoon]) / 3600),
                            np.array([lMercury, lVenus, lEarth, lMars, lJupiter, lSaturn, lUranus, lNeptune, pa])))
                 % (2 * np.pi))

    # Polynomial part of X and Y
    X0 = -16617 + 2004191898 * tTT - 429782.9 * tTT2 - 198618.34 * tTT3 + 7.578 * tTT4 + 5.9285 * tTT5
    Y0 = -6951 - 25896 * tTT - 22407274.7 * tTT2 + 1900.59 * tTT3 + 1112.526 * tTT4 + 0.1358 * tTT5
    # Polynomial part of S
    S0 = 94 + 3808.65 * tTT - 122.68 * tTT2 - 72574.11 * tTT3 + 27.98 * tTT4 + 15.62 * tTT5

    # Series evaluation
    # For X:
    FX = np.zeros(len(aeroX))
    FX[1 - 1:1306] = np.ones(1306)
    FX[1307 - 1:1559] = tTT
    FX[1560 - 1:1595] = tTT2
    FX[1596 - 1:1599] = tTT3
    FX[1600 - 1:] = tTT4
    argX = np.dot(aeroX[:, 4 - 1:17], nutationV)
    X = np.sum((aeroX[:, 1] * np.sin(argX) + aeroX[:, 2] * np.cos(argX)) * FX)

    # For Y:
    FY = np.zeros(len(aeroY))
    FY[1 - 1:962] = np.ones(962)
    FY[963 - 1:1239] = tTT
    FY[1240 - 1:1269] = tTT2
    FY[1270 - 1:1274] = tTT3
    FY[1275 - 1:] = tTT4
    argY = np.dot(aeroY[:, 4 - 1:17], nutationV)
    Y = np.sum((aeroY[:, 1] * np.sin(argY) + aeroY[:, 2] * np.cos(argY)) * FY)

    # For S:
    FS = np.zeros(len(aeroS))
    FS[1 - 1:33] = np.ones(33)
    FS[34 - 1:36] = tTT
    FS[37 - 1:61] = tTT2
    FS[62 - 1:65] = tTT3
    FS[66 - 1:] = tTT4
    argS = np.dot(aeroS[:, 4 - 1:11], np.hstack((nutationV[1 - 1:5], nutationV[7 - 1:8], nutationV[14 - 1])))
    S = np.sum((aeroS[:, 1] * np.sin(argS) + aeroS[:, 2] * np.cos(argS)) * FS)

    X = X + X0
    Y = Y + Y0
    S = S + S0
    # Convert from microarcseconds to radians

    X = np.deg2rad(X * 1e-6 / 3600) + dCIP[0]
    Y = np.deg2rad(Y * 1e-6 / 3600) + dCIP[1]
    S = np.deg2rad(S * 1e-6 / 3600) - X * Y / 2
    # Coordinates of the CIP
    E = np.arctan2(Y, X)
    d = np.arctan(np.sqrt((X ** 2 + Y ** 2) / (1 - X ** 2 - Y ** 2)))
    # Transformation matrix for celestial motion of the CIP
    Q = angle2dcm(E, d, -E - S, 'ZYZ')

    return np.dot(W, np.dot(R, Q))


def dcmeci2ecef_full(UTC):
    eop_data_UTC = get_eop_values(UTC.year, UTC.month, UTC.day)
    DELTAUT1 = eop_data_UTC["DELTAUT1"]
    logger.debug(DELTAUT1)
    POLARMOTION = eop_data_UTC["POLARMOTION"]
    logger.debug(POLARMOTION)
    dCIP = eop_data_UTC["dCIP"]
    logger.debug(dCIP)
    return dcmeci2ecef(UTC, DELTAAT, DELTAUT1, POLARMOTION)
