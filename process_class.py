import abc

from characteristic_funs import (
    heston_log_st_chf,
    vg_log_st_chf,

)

from moment_generating_funs import (
    heston_log_st_mgf,
    vg_log_st_mgf,

)


def chf_and_mgf_switch(chf, mgf, type):
    if type == 'chf':
        return chf
    elif type == 'mgf':
        return mgf
    else:
        raise ValueError('type only accept chf or mgf')


class LogSt(abc.ABC):
    def __init__(self, type=None):
        self._type = type

    def set_type(self, type):
        self._type = type
        return self

    @property
    def type(self):
        return self._type

class Heston(LogSt):
    def __init__(self, V0, theta, k, sigma, rho):
        self.V0 = V0
        self.theta = theta
        self.k = k
        self.sigma = sigma
        self.rho = rho
        super(Heston, self).__init__()

    def __call__(self, u, t, r, q, S0):
        return chf_and_mgf_switch(heston_log_st_chf, heston_log_st_mgf, self.type)(u, t, r, q, S0, self.V0,
                                                                                   self.theta,
                                                                                   self.k,
                                                                                   self.sigma,
                                                                                self.rho)

class VarianceGamma(LogSt):
    def __init__(self, theta, v, sigma):
        self.theta = theta
        self.v = v
        self.sigma = sigma
        super(VarianceGamma, self).__init__()

    def __call__(self, u, t, r, q, S0):
        return chf_and_mgf_switch(vg_log_st_chf, vg_log_st_mgf, self.type)(u, t, r, q, S0, self.theta, self.v, self.sigma)
