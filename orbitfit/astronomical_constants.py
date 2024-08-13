import numpy as np

# WGS-84 parameters
f_earth = 1 / 298.257223563         # flattening
R_eq_earth = 6378.137               # [km] Ellipsoid (equatorial) semimajor axis
e_earth2 = f_earth * (2 - f_earth)  # Eccentricity squared
R_mean_earth = 6371.0088            # [km] mean Earth radius
mu_earth = 3.986004418e5            # [km^3/sec^2]


# EGM-96
J2_earth = 0.0010826267


# Time
tropical_year = 365.2421897                                     # mean solar days
sidereal_year = 365.256363004                                   # mean solar days
secs_per_hour = 60 * 60                                         # [sec]
secs_per_day = 24 * secs_per_hour                               # [sec]
sun_mean_longitude_rate = 2 * np.pi / (secs_per_day * tropical_year)   # [rad/sec]