# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 20:05:27 2018

@author: user
"""

from fourier_pricer import (
    carr_madan_fft_call_pricer,
)
from helper import spline_fitting


class FFTEngine(object):
    def __init__(self, N, d_u, alpha, spline_order):
        self.N = N
        self.d_u = d_u
        self.alpha = alpha
        self.spline_order = spline_order

    def __call__(self, strike, t, r, q, S0, chf_ln_st):
        sim_strikes, call_prices = carr_madan_fft_call_pricer(self.N, self.d_u, self.alpha, r, t, S0, q, chf_ln_st.set_type('chf'))
        ffn_prices = spline_fitting(sim_strikes, call_prices, self.spline_order)(strike)
        return ffn_prices