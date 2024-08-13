import numpy as np
from .I2E import get_eop_values, juliandate, DELTAAT
import datetime
import logging

logger = logging.getLogger(__name__)


def teme2ecef_dcm_full(UTC):
    eop_data_UTC = get_eop_values(UTC.year, UTC.month, UTC.day)
    DELTAUT1 = eop_data_UTC["DELTAUT1"]
    logger.debug(DELTAUT1)
    POLARMOTION = eop_data_UTC["POLARMOTION"]
    logger.debug(POLARMOTION)

    # Julian date for terrestrial time
    jdTT = juliandate(UTC + datetime.timedelta(seconds=(DELTAAT + 32.184)))
    # Number of Julian centuries since J2000 for terrestrial time.
    ttt = (jdTT - 51544.5) / 36525

    # Julian date for UT1
    jdut1 = juliandate(UTC + datetime.timedelta(seconds=DELTAUT1))

    logger.debug("input: [%r]", [ttt, jdut1, POLARMOTION[0], POLARMOTION[1]])

    return teme2ecef_dcm(ttt, jdut1, xp=POLARMOTION[0], yp=POLARMOTION[1])


def teme2ecef_full(rteme, vteme, UTC):
    eop_data_UTC = get_eop_values(UTC.year, UTC.month, UTC.day)
    DELTAUT1 = eop_data_UTC["DELTAUT1"]
    logger.debug(DELTAUT1)
    POLARMOTION = eop_data_UTC["POLARMOTION"]
    logger.debug(POLARMOTION)
    lod = eop_data_UTC["LOD"]
    logger.debug(lod)

    # Julian date for terrestrial time
    jdTT = juliandate(UTC + datetime.timedelta(seconds=(DELTAAT + 32.184)))
    # Number of Julian centuries since J2000 for terrestrial time.
    ttt = (jdTT - 51544.5) / 36525

    # Julian date for UT1
    jdut1 = juliandate(UTC + datetime.timedelta(seconds=DELTAUT1))

    logger.debug("input: [%r]", [rteme, vteme, ttt, jdut1, lod, POLARMOTION[0], POLARMOTION[1]])

    return teme2ecef(rteme, vteme, ttt, jdut1, lod=lod, xp=POLARMOTION[0], yp=POLARMOTION[1])


def teme2ecef(rteme, vteme, ttt, jdut1, lod=0, xp=0, yp=0):
    # ----------------------------------------------------------------------------
    #
    #                           function teme2ecef
    #
    #  this function trsnforms a vector from the true equator mean equniox frame
    #    (teme), to an earth fixed (ITRF) frame.  the results take into account
    #    the effects of sidereal time, and polar motion.
    #
    #  author        : david vallado                  719-573-2600   10 dec 2007
    #
    #  revisions
    #
    #  inputs          description                    range / units
    #    rteme       - position vector teme           km
    #    vteme       - velocity vector teme           km/s
    #    ateme       - acceleration vector teme       km/s2
    #    lod         - excess length of day           sec
    #    ttt         - julian centuries of tt         centuries
    #    jdut1       - julian date of ut1             days from 4713 bc
    #    xp          - polar motion coefficient       rad
    #    yp          - polar motion coefficient       rad
    #
    #  outputs       :
    #    recef       - position vector earth fixed    km
    #    vecef       - velocity vector earth fixed    km/s
    #    aecef       - acceleration vector earth fixedkm/s2
    #
    #  locals        :
    #    st          - matrix for pef - tod
    #    pm          - matrix for ecef - pef
    #
    #  coupling      :
    #   gstime       - greenwich mean sidereal time   rad
    #   polarm       - rotation for polar motion      pef - ecef
    #
    #  references    :
    #    vallado       2007, 219-228
    #
    # ----------------------------------------------------------------------------

    # ------------------------ find gmst --------------------------
    gmst = gstime(jdut1)

    st = np.array([[np.cos(gmst), -np.sin(gmst), 0.0],
                   [np.sin(gmst), np.cos(gmst), 0.0],
                   [0.0, 0.0, 1.0]])
    pm = polarm(xp, yp, ttt, '80')

    thetasa = 7.29211514670698e-05 * (1.0 - lod / 86400.0)
    omegaearth = np.array([0, 0, thetasa])

    rpef = np.dot(st.T, rteme)
    recef = np.dot(pm.T, rpef)

    vpef = np.dot(st.T, vteme) - np.cross(omegaearth, rpef)
    vecef = np.dot(pm.T, vpef)

    return recef, vecef


def teme2ecef_dcm(ttt, jdut1, xp=0, yp=0):
    gmst = gstime(jdut1)

    st = np.array([[np.cos(gmst), -np.sin(gmst), 0.0],
                   [np.sin(gmst), np.cos(gmst), 0.0],
                   [0.0, 0.0, 1.0]])
    pm = polarm(xp, yp, ttt, '80')
    return np.dot(pm.T, st.T)


def polarm(xp, yp, ttt, opt="80"):
    # ----------------------------------------------------------------------------
    #
    #                           function polarm
    #
    #  this function calulates the transformation matrix that accounts for polar
    #    motion. both the 1980 and 2000 theories are handled. note that the rotation
    #    order is different between 1980 and 2000 .
    #
    #  author        : david vallado                  719-573-2600   25 jun 2002
    #
    #  revisions
    #    vallado     - consolidate with iau 2000                     14 feb 2005
    #
    #  inputs          description                    range / units
    #    xp          - polar motion coefficient       rad
    #    yp          - polar motion coefficient       rad
    #    ttt         - julian centuries of tt (00 theory only)
    #    opt         - method option                  '01', '02', '80'
    #
    #  outputs       :
    #    pm          - transformation matrix for ecef - pef
    #
    #  locals        :
    #    convrt      - conversion from arcsec to rad
    #    sp          - s prime value
    #
    #  coupling      :
    #    none.
    #
    #  references    :
    #    vallado       2004, 207-209, 211, 223-224
    #
    # ----------------------------------------------------------------------------

    cosxp = np.cos(xp)
    sinxp = np.sin(xp)
    cosyp = np.cos(yp)
    sinyp = np.sin(yp)

    if opt == '80':
        pm = np.array([[cosxp, 0.0, -sinxp],
                       [sinxp * sinyp, cosyp, cosxp * sinyp],
                       [sinxp * cosyp, -sinyp, cosxp * cosyp]])
        # a1 = rot2mat(xp)
        # a2 = rot1mat(yp)
        # pm = a2*a1
        # Approximate matrix using small angle approximations
        # pm(1,1) =  1.0
        # pm(2,1) =  0.0
        # pm(3,1) =  xp
        # pm(1,2) =  0.0
        # pm(2,2) =  1.0
        # pm(3,2) = -yp
        # pm(1,3) = -xp
        # pm(2,3) =  yp
        # pm(3,3) =  1.0
    else:
        convrt = np.pi / (3600.0 * 180.0)
        # approximate sp value in rad
        sp = -47.0e-6 * ttt * convrt
        cossp = np.cos(sp)
        sinsp = np.sin(sp)

        # form the matrix
        pm = np.array([[cosxp * cossp, -cosyp * sinsp + sinyp * sinxp * cossp,
                        -sinyp * sinsp - cosyp * sinxp * cossp],
                       [cosxp * sinsp, cosyp * cossp + sinyp * sinxp * sinsp,
                        sinyp * cossp - cosyp * sinxp * sinsp],
                       [sinxp, -sinyp * cosxp, cosyp * cosxp]])
        # a1 = rot1mat(yp)
        # a2 = rot2mat(xp)
        # a3 = rot3mat(-sp)
        # pm = a3*a2*a1
    return pm


def gstime(jdut1):
    # -----------------------------------------------------------------------------
    #
    #                           function gstime
    #
    #  this function finds the greenwich sidereal time (iau-82).
    #
    #  author        : david vallado                  719-573-2600    7 jun 2002
    #
    #  revisions
    #                -
    #
    #  inputs          description                    range / units
    #    jdut1       - julian date of ut1             days from 4713 bc
    #
    #  outputs       :
    #    gst         - greenwich sidereal time        0 to 2pi rad
    #
    #  locals        :
    #    temp        - temporary variable for reals   rad
    #    tut1        - julian centuries from the
    #                  jan 1, 2000 12 h epoch (ut1)
    #
    #  coupling      :
    #
    #  references    :
    #    vallado       2007, 193, Eq 3-43
    #
    # gst = gstime(jdut1)
    # -----------------------------------------------------------------------------
    twopi = 2.0 * np.pi
    deg2rad = np.pi / 180.0

    # ------------------------  implementation   ------------------
    tut1 = (jdut1 - 2451545.0) / 36525.0

    temp = (- 6.2e-6 * tut1 * tut1 * tut1 + 0.093104 * tut1 * tut1 +
            (876600.0 * 3600.0 + 8640184.812866) * tut1 + 67310.54841)

    # 360/86400 = 1/240, to deg, to rad
    temp = temp * deg2rad / 240.0 % twopi

    # ------------------------ check quadrants --------------------
    if temp < 0.0:
        temp = temp + twopi

    gst = temp
    return gst


def ecef2teme_full(recef, vecef, UTC):
    eop_data_UTC = get_eop_values(UTC.year, UTC.month, UTC.day)
    DELTAUT1 = eop_data_UTC["DELTAUT1"]
    logger.debug(DELTAUT1)
    POLARMOTION = eop_data_UTC["POLARMOTION"]
    logger.debug(POLARMOTION)
    lod = eop_data_UTC["LOD"]
    logger.debug(lod)

    # Julian date for terrestrial time
    jdTT = juliandate(UTC + datetime.timedelta(seconds=(DELTAAT + 32.184)))
    # Number of Julian centuries since J2000 for terrestrial time.
    ttt = (jdTT - 51544.5) / 36525

    # Julian date for UT1
    jdut1 = juliandate(UTC + datetime.timedelta(seconds=DELTAUT1))

    logger.debug("input: [%r]", [recef, vecef, ttt, jdut1, lod, POLARMOTION[0], POLARMOTION[1]])

    return ecef2teme(recef, vecef, ttt, jdut1, lod=lod, xp=POLARMOTION[0], yp=POLARMOTION[1])


def ecef2teme(recef, vecef, ttt, jdut1, lod=0, xp=0, yp=0):
    # ----------------------------------------------------------------------------
    #
    #                           function ecef2teme
    #
    #  this function trsnforms a vector from the earth fixed (ITRF) frame, to the true equator mean equniox frame
    #   (teme). The results take into account the effects of sidereal time, and polar motion.
    #
    #  author        : david vallado                  719-573-2600   10 dec 2007
    #
    #  revisions
    #
    #  inputs          description                    range / units
    #    rteme       - position vector teme           km
    #    vteme       - velocity vector teme           km/s
    #    ateme       - acceleration vector teme       km/s2
    #    lod         - excess length of day           sec
    #    ttt         - julian centuries of tt         centuries
    #    jdut1       - julian date of ut1             days from 4713 bc
    #    xp          - polar motion coefficient       rad
    #    yp          - polar motion coefficient       rad
    #
    #  outputs       :
    #    recef       - position vector earth fixed    km
    #    vecef       - velocity vector earth fixed    km/s
    #    aecef       - acceleration vector earth fixedkm/s2
    #
    #  locals        :
    #    st          - matrix for pef - tod
    #    pm          - matrix for ecef - pef
    #
    #  coupling      :
    #   gstime       - greenwich mean sidereal time   rad
    #   polarm       - rotation for polar motion      pef - ecef
    #
    #  references    :
    #    vallado       2007, 219-228
    #
    # ----------------------------------------------------------------------------

    # ------------------------ find gmst --------------------------
    gmst = gstime(jdut1)

    st = np.array([[np.cos(gmst), -np.sin(gmst), 0.0],
                   [np.sin(gmst), np.cos(gmst), 0.0],
                   [0.0, 0.0, 1.0]])
    pm = polarm(xp, yp, ttt, '80')

    thetasa = 7.29211514670698e-05 * (1.0 - lod / 86400.0)
    omegaearth = np.array([0, 0, thetasa])

    rpef = np.dot(pm, recef)
    rteme = np.dot(st, rpef)

    vpef = np.dot(pm, vecef) + np.cross(omegaearth, rpef)
    vteme = np.dot(st, vpef)

    return rteme, vteme