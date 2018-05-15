# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 20:03:17 2018

@author: user
"""

import numpy as np

def heston_log_st_chf(u, t, r, q, S0, V0, theta, k, sigma, rho):
    dt = np.sqrt((sigma ** 2) * (1j * u + u ** 2) + (k - 1j * rho * sigma * u) ** 2)
    beta = k - 1j * u * rho * sigma
    g = (beta - dt) / (beta + dt)
    D_t = (beta - dt) / (sigma ** 2) * ((1 - np.exp(-dt * t)) / (1 - g * np.exp(-dt * t)))
    C_t = 1j * u * (r - q) * t + k * theta / (sigma ** 2) * (
        (beta - dt) * t - 2 * np.log((1 - g * np.exp(-dt * t)) / (1 - g)))
    return np.exp(C_t + D_t * V0 + 1j * u * np.log(S0))


def vg_chf(u, t, theta, v, sigma):
    return (1 - 1j * u * theta * v + 0.5 * (sigma ** 2) * (u ** 2) * v) ** (-t / v)


def general_ln_st_chf(u, t, r, q, S0, chf_xt, *args, **kwargs):
    martingale_adjust = -(1 / t) * np.log(chf_xt(-1j, t, *args, **kwargs))
    normal_term = 1j * (np.log(S0) + (r - q + martingale_adjust) * t) * u
    ln_st_chf = np.exp(normal_term) * chf_xt(u, t, *args, **kwargs)
    return ln_st_chf


def vg_log_st_chf(u, t, r, q, S0, theta, v, sigma):
    chf_xt = vg_chf
    return general_ln_st_chf(u, t, r, q, S0, chf_xt, theta=theta, v=v, sigma=sigma)
