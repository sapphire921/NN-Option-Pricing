import numpy as np


def heston_log_st_mgf(u, t, r, q, S0, V0, theta, k, sigma, rho):
    dt = np.sqrt((sigma ** 2) * (u - u ** 2) + (k - rho * sigma * u) ** 2)
    beta = k - u * rho * sigma
    g = (beta - dt) / (beta + dt)
    D_t = (beta - dt) / (sigma ** 2) * ((1 - np.exp(-dt * t)) / (1 - g * np.exp(-dt * t)))
    C_t = u * (r - q) * t + k * theta / (sigma ** 2) * (
        (beta - dt) * t - 2 * np.log((1 - g * np.exp(-dt * t)) / (1 - g)))
    return np.exp(C_t + D_t * V0 + u * np.log(S0))


def vg_mgf(u, t, theta, v, sigma):
    return (1 - u * theta * v + 0.5 * (sigma ** 2) * (-u ** 2) * v) ** (-t / v)

def general_ln_st_mgf(u, t, r, q, S0, mgf_xt, *args, **kwargs):
    martingale_adjust = -(1 / t) * np.log(mgf_xt(1, t, *args, **kwargs))
    normal_term = (np.log(S0) + (r - q + martingale_adjust) * t) * u
    ln_st_mgf = np.exp(normal_term) * mgf_xt(u, t, *args, **kwargs)
    return ln_st_mgf


def vg_log_st_mgf(u, t, r, q, S0, theta, v, sigma):
    mgf_xt = vg_mgf
    return general_ln_st_mgf(u, t, r, q, S0, mgf_xt, theta=theta, v=v, sigma=sigma)
