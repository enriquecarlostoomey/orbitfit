import textwrap
import numpy as np
from collections import namedtuple

mu = 398600.4418
# from https://nssdc.gsfc.nasa.gov/planetary/factsheet/earthfact.html:
MAJOR_EARTH_RADIUS = 6378137  # [m], semi-major axis
MINOR_EARTH_RADIUS = 6356752  # [m], semi-minor axis
Rq = MAJOR_EARTH_RADIUS/1000  # major earth radius in Km
J2 = +1.08262668355E-003

MeanOrbitalELements = namedtuple('MeanOrbitalElements', ['mean_semimajor_axis', 'mean_eccentricity', 'mean_inclination',
                                                         'mean_RAAN', 'mean_arg_perigee', 'mean_mean_anomaly'])


class OrbitalElements:
    """
    This object is responsible for handling orbital elements
    (mainly, of an Earth satellite).

    The orbital elements can be found in https://en.wikipedia.org/wiki/Orbital_elements
    Since most of the orbits we work with have an almost 0 eccentricity,
    we allow for other elements to be stated.

    Angles are measured in rad, distances in km.
    """

    def __init__(self,
                 semimajor_axis, eccentricity,
                 inclination, raan,
                 argument_of_perigee, true_anomaly):

        # set first Keplerian elements
        self.semimajor_axis = semimajor_axis
        self.eccentricity = eccentricity
        self.inclination = inclination
        self.raan = raan
        self.argument_of_perigee = argument_of_perigee
        self.true_anomaly = true_anomaly

    def __str__(self):
        fmt = textwrap.dedent('''
        semimajor axis:      {s.semimajor_axis:f}
        eccentricity:        {s.eccentricity:g}
        inclination:         {s.inclination:f}
        RAAN:                {s.raan:f}
        argument of perigee: {s.argument_of_perigee:f}
        true anomaly:        {s.true_anomaly:f}
        ''')
        return fmt.format(s=self)

    def todict(self):
        return {'semimajor_axis': self.semimajor_axis,
                'eccentricity': self.eccentricity,
                'inclination': np.degrees(self.inclination),
                'RAAN': np.degrees(self.raan),
                'argument_of_perigee': np.degrees(self.argument_of_perigee),
                'true anomaly': np.degrees(self.true_anomaly),
                'r': self.r,
                'argument_of_latitude': np.degrees(self.argument_of_latitude)
                }

    def compute_mean_elements(self):
        """
        This function maps osculating elements (input) into mean elements (output)
        The input angles must be in rad
        alpha = [a_osc,e_osc,i_osc,w_osc,f_osc,OM_osc]
        The output angles are in deg
        Based on Appendix F - Book of Schaub - Junkins
        """

        two_pi = 2 * np.pi

        a = self.semimajor_axis
        e = self.eccentricity
        i = self.inclination
        w = self.argument_of_perigee
        f = self.true_anomaly
        u = self.argument_of_latitude % two_pi
        OM = self.raan

        ci = np.cos(i)
        E = (2 * np.arctan(np.sqrt((1-e)/(1+e))*np.tan(f/2))) % two_pi
        M = E - e * np.sin(E)
        gamma2 = -J2/2*(Rq/a)**2  # the minus means :    Osc ---> Mean
        eta = np.sqrt(1-e**2)
        gamma2_tag = gamma2/(eta**4)
        a_r = (1 + e * np.cos(f)) / (eta**2)
        a_ave = a + a * gamma2 * ((3*ci**2-1) * (a_r**3 - 1/(eta**3)) + 3*(1-ci**2) * (a_r**3) * np.cos(2*u))
        de1 = gamma2_tag / 8 * e * (eta**2) * (1-11*ci**2-40*(ci**4)/(1-5*ci**2)) * np.cos(2*w)
        de = de1 + eta**2 / 2 * (gamma2 * ((3*ci**2-1) / (eta**6) *
                                           (e*eta + e/(1+eta) + 3*np.cos(f) + 3*e*(np.cos(f)**2) +
                                            (e**2)*np.cos(f)**3) +
                                           3 * (1-ci**2) / (eta**6) * (e + 3*np.cos(f) + 3*e*(np.cos(f)**2) +
                                                                       (e**2)*np.cos(f)**3) * np.cos(2*u)) -
                                 gamma2_tag * (1-ci**2) * (3*np.cos(2*w+f) + np.cos(2*w+3*f)))
        di = (-e * de1 / ((eta**2)*np.tan(i)) + gamma2_tag / 2*ci * np.sqrt(1-ci**2) *
              (3*np.cos(2*w+2*f) + 3*e*np.cos(2*w+f) + e*np.cos(2*w+3*f)))

        MWO_ave = (M + w + OM +
                   gamma2_tag / 8 * (eta**3) * (1-11*ci**2-40*(ci**4)/(1-5*ci**2)) -
                   gamma2_tag / 16 * (2 + e**2-11*(2+3*e**2)*ci**2 -
                                      40*(2+5*e**2)*(ci**4)/(1-5*ci**2) -
                                      400*(e**2)*ci**6/(1-5*ci**2)**2) +
                   gamma2_tag / 4 * (- 6 * (1-5*ci**2) * (f-M+e*np.sin(f)) +
                                     (3-5*ci**2) * (3*np.sin(2*u)+3*e*np.sin(2*w+f)+e*np.sin(2*w+3*f))) -
                   gamma2_tag / 8 * (e**2) * ci * (11 + 80*(ci**2)/(1-5*ci**2) + 200*(ci**4)/(1-5*ci**2)**2) -
                   gamma2_tag / 2 * ci * (6 * (f-M+e*np.sin(f)) - 3*np.sin(2*u) - 3*e*np.sin(2*w+f) -
                                          e*np.sin(2*w+3*f)))

        edM = (gamma2_tag/8*e*(eta**3)*(1-11*ci**2-40*(ci**4)/(1-5*ci**2)) -
               gamma2_tag/4*(eta**3)*(2*(3*ci**2-1)*((a_r*eta)**2+a_r+1)*np.sin(f) +
                                      3*(1-ci**2)*((-(a_r*eta)**2-a_r+1)*np.sin(2*w+f)+((a_r*eta)**2+a_r+1/3) *
                                                   np.sin(2*w+3*f))))

        dOM = (-gamma2_tag/8*(e**2)*ci*(11+80*(ci**2)/(1-5*ci**2)+200*(ci**4)/(1-5*ci**2)**2) -
               gamma2_tag/2*ci*(6*(f-M+e*np.sin(f))-3*np.sin(2*u)-3*e*np.sin(2*w+f)-e*np.sin(2*w+3*f)))

        d1 = (e+de)*np.sin(M) + edM*np.cos(M)
        d2 = (e+de)*np.cos(M) - edM*np.sin(M)
        M_ave = np.arctan2(d1, d2) % two_pi
        e_ave = np.sqrt(d1**2 + d2**2)

        d3 = (np.sin(i/2)+np.cos(i/2)*di/2) * np.sin(OM) + np.sin(i/2) * dOM * np.cos(OM)
        d4 = (np.sin(i/2)+np.cos(i/2)*di/2) * np.cos(OM) - np.sin(i/2) * dOM * np.sin(OM)
        OM_ave = np.arctan2(d3, d4) % two_pi

        i_ave = 2 * np.arcsin(np.sqrt(d3**2+d4**2))

        w_ave = MWO_ave - M_ave - OM_ave

        return MeanOrbitalELements(a_ave, e_ave, np.rad2deg(i_ave), np.rad2deg(OM_ave), np.rad2deg(w_ave), np.rad2deg(M_ave))

    @property
    def argument_of_latitude(self):
        return self.true_anomaly + self.argument_of_perigee

    @property
    def r(self):
        return self.semimajor_axis * (1 - self.eccentricity**2) / (1 + self.eccentricity * np.cos(self.true_anomaly))

    @staticmethod
    def from_cartesian_coordinates(position_J2000_km, velocity_J2000_km_s):
        """
        returns an OrbitalElements object computed from cartesian coordinates.
        Input position and velocity are in ECI (inertial reference frame).
        """
        r = np.linalg.norm(position_J2000_km, 2)
        v = np.linalg.norm(velocity_J2000_km_s, 2)
        osc_a = -mu / (2 * (np.power(v, 2) / 2 - mu / r))
        osc_r = r
        h = np.cross(position_J2000_km, velocity_J2000_km_s)
        h_norm = np.linalg.norm(h)

        e_vec = 1 / mu * np.cross(velocity_J2000_km_s, h) - position_J2000_km / r
        osc_e = np.linalg.norm(e_vec)
        osc_i = np.arccos(h[2] / h_norm)
        if osc_i != 0:
            line_of_nodes = np.cross(np.array([0, 0, 1]), h)
            osc_right_ascension_AN = np.arctan2(line_of_nodes[1], line_of_nodes[0])
            osc_right_ascension_AN = osc_right_ascension_AN % (2 * np.pi)
            if osc_e > 0:
                cos_w = np.inner(line_of_nodes, e_vec) / (osc_e * np.linalg.norm(line_of_nodes))
                if e_vec[2] >= 0:
                    osc_arg_perigee = np.arccos(cos_w)
                else:
                    osc_arg_perigee = 2 * np.pi - np.arccos(cos_w)
                osc_arg_perigee = osc_arg_perigee % (2 * np.pi)

                cos_f = np.inner(position_J2000_km, e_vec) / (osc_e * r)
                if np.inner(position_J2000_km, velocity_J2000_km_s) >= 0:
                    osc_true_anomaly = np.arccos(cos_f)
                else:
                    osc_true_anomaly = 2 * np.pi - np.arccos(cos_f)
                osc_true_anomaly = osc_true_anomaly % (2 * np.pi)

            else:
                osc_arg_perigee = np.NaN
                osc_true_anomaly = np.NaN

            cos_arg_lat = np.inner(position_J2000_km, line_of_nodes) / (np.linalg.norm(line_of_nodes) * r)
            if position_J2000_km[2] >= 0:
                osc_arg_lat = np.arccos(cos_arg_lat)
            else:
                osc_arg_lat = 2 * np.pi - np.arccos(cos_arg_lat)
            osc_arg_lat = osc_arg_lat % (2 * np.pi)

        else:
            osc_right_ascension_AN = np.NaN
            osc_arg_perigee = np.NaN
            osc_arg_lat = np.NaN
            if osc_e > 0:
                cos_f = np.inner(position_J2000_km, e_vec) / (np.linalg.norm(e_vec) * r)
                if np.inner(position_J2000_km, e_vec) >= 0:
                    osc_true_anomaly = np.arccos(cos_f)
                else:
                    osc_true_anomaly = 2 * np.pi - np.arccos(cos_f)

        osc = OrbitalElements(osc_a,
                              osc_e,
                              osc_i,
                              osc_right_ascension_AN,
                              osc_arg_perigee,
                              osc_true_anomaly)
        return osc
