#!/usr/bin/python
# Author: Tyler Gamvrelis
# Pricing functions
#
# Paper: "Online Combinatorial Auctions for Resource Allocation With Supply
# Costs and Capacity Limits" by X. Tan et al. is referenced throughout

# Standard library imports
from abc import ABC, abstractmethod
from functools import partial
import logging
import warnings

# Third party imports
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import AccuracyWarning
from scipy.optimize import brentq
from scipy.integrate import romberg as integrate

# Local application imports
from cost_func import *

# Globals
logger = logging.getLogger(__name__)

# Suppress integration accuracy warnings
warnings.filterwarnings(
    action='ignore', category=AccuracyWarning, module='scipy'
)

class PriceFunc(ABC):
    """Base class for all price functions."""

    def __init__(self, val_func):
        """
        Args:
            val_func : ValuationFunc
                Valuation function instance we want optimal pricing for
        """
        pass

    @abstractmethod
    def __call__(self, x):
        pass

    @abstractmethod
    def __str__(self):
        pass

class MyopicPowerPriceFunc(PriceFunc):
    """Myopic pricing for a power cost function."""

    def __init__(self, val_func, pbar):
        """
        Args:
            val_func : ValuationFunc
                Valuation function instance we want optimal pricing for
            pbar : float
                The maximum possible bidder valuation
        """
        self.pbar = pbar

        # Copy params from cost function
        self.s = val_func.s
        self.a = val_func.a
        logger.debug(
            f'Creating myopic power pricing func '
            f'(a = {self.a}, s = {self.s}, pbar = {pbar})'
        )

    def _cost_func_derivative(self, x):
        """
        Returns the derivative of the power cost function, evaluated at x.
        """
        return self.s * self.a * (x ** (self.s - 1))

    def __call__(self, y):
        return self._cost_func_derivative(y)

    def __str__(self):
        sb = ''
        sb += 'Myopic power pricing function '
        sb = f'(a = {self.a}, s = {self.s}, pbar = {pbar})\n'
        return sb

class OptPowerPriceFunc(PriceFunc):
    """Optimal pricing for a power cost function."""

    def __init__(self, val_func, pbar):
        """
        Args:
            val_func : ValuationFunc
                Valuation function instance we want optimal pricing for
            pbar : float
                The maximum possible bidder valuation
        """
        self.pbar = pbar

        # Copy params from cost function
        self.s = val_func.s
        self.a = val_func.a
        logger.debug(
            f'Creating optimal power pricing func '
            f'(a = {self.a}, s = {self.s}, pbar = {pbar})'
        )

        # Initialize params we'll need later
        self.c_lb = val_func.get_min_marginal_cost()
        self.c_ub = val_func.get_max_marginal_cost()
        assert self.pbar > self.c_lb, f'Need pbar={self.pbar} > {self.c_lb}'
        self._is_luc = (pbar > self.c_lb) and (pbar <= self.c_ub)
        self.alpha_min = self.s ** (self.s / (self.s - 1))
        logger.debug(f'\tLUC = {self._is_luc}, alpha_min = {self.alpha_min}')
        logger.debug(f'\tc_lb = {self.c_lb}, c_ub = {self.c_ub}')

        # Compute Cs
        int_val = integrate(
            lambda nu: (nu ** (self.s - 1)) * np.exp(-nu),
            self.s,
            self.alpha_min
        )
        self.Cs = self.c_ub
        self.Cs *= (1 / np.exp(self.s)) - (int_val / (self.s ** self.s))
        self.Cs *= np.exp(self.alpha_min)
        logger.debug(f'\tCs = {self.Cs}')

        # Detect operating regime (LUC or HUC) + one-time setup actions
        if self._is_luc:
            self.w = val_func.inverse(pbar / val_func.s)
            self.v = val_func.inverse(pbar)
            assert self.w > 0, 'Error: w > 0 must hold'
            assert self.w < self.v, 'Error: w < v must hold'
            assert self.v <= 1, 'Error: v <= 1 must hold'
            # TODO: what's a good init for m?
            self.set_m((self.v + self.w) / 2)
            logger.debug(
                '\tDone LUC init. (w, v) = ({self.w}, {self.v})'
            )
        else:
            # Determine whether HUC1 or HUC2
            if self.pbar > self.Cs:
                self._huc_mode = 2
            else:
                self._huc_mode = 1
            logger.debug(f'\tHUC mode {self._huc_mode}')

            # Compute CDT.
            # The following computations are prone to numerical instabilities
            # near the open end points...so try to avoid them for now and
            # assume it won't be a problem...
            #
            # WARNING: if brentq is complaining about the lb and ub not having
            # different signs, it is likely that the root we are searching for
            # is very close to the open end point (and not within the selected
            # epsilon tolerance). To compensate for this, you could do two
            # things:
            #  1. Decrease epsilon (not recommended, for numerical reasons)
            #  2. Perturb pbar. Likely, pbar is too close to Cs if you are
            #     experiencing this
            self.u_s = (1 / self.s) ** (1 / (self.s - 1))
            EPS = 5e-3
            if abs(self.Cs - self.pbar) < self.Cs / 10:
                # If we detect that Cs and pbar are "very close", try
                # decreasing EPS to avoid the breqnt complained explained
                # above. This will NOT work flawlessly for all cases; ideally
                # we need to set EPS adaptively; this is, however, beyond the
                # current scope
                EPS /= 10
            logger.debug(f'\tStarting u_cdt search. u_s = {self.u_s}')
            if self._huc_mode == 1:
                lb = self.u_s
                ub = 1 - EPS
                logger.debug(f'\t\tlb = {lb}, ub = {ub}')
                self.u_cdt, r = brentq(
                    self._u_cdt_huc1_eqn, lb, ub, full_output=True
                )
                assert r.converged, 'Root finder failed to converge'
                logger.debug(f'\t\tu_cdt = {self.u_cdt}')
            else:
                lb = EPS
                ub = self.u_s - EPS
                logger.debug(f'\t\tlb = {lb}, ub = {ub}')
                self.u_cdt, r = brentq(
                    self._u_cdt_huc2_eqn, lb, ub, full_output=True
                )
                assert r.converged, 'Root finder failed to converge'
                logger.debug(f'\t\tu_cdt = {self.u_cdt}')

            # Find rho_s if HUC1. rho_s is a resource utilization level, so it
            # lies in [0, 1] (or maybe the open interval?)
            if self._huc_mode == 1:
                lb = 0
                ub = 1
                logger.debug(f'\tStarting rho_s search. lb = {lb}, ub = {ub}')
                self.rho_s, r = brentq(
                    self._rho_s_eqn, lb, ub, full_output=True
                )
                assert r.converged, 'Root finder failed to converge'
                logger.debug(f'\t\trho_s = {self.rho_s}')

            # Set u parameter if HUC1
            if self._huc_mode == 1:
                # TODO: what's a good init for u?
                self.set_u((self.u_s + self.u_cdt) / 2)

    def _rho_s_eqn(self, rho_s):
        """
        Returns the value of the rho_s equation (25) with all terms on the LHS,
        used to find rho_s via root-finding.

        Args:
            rho_s : float
                The resource utilization level such that phi_ivp = pbar;
                parameter we search for via root-finding
        """
        lb = self.s
        ub = self.alpha_min * rho_s
        int_val = integrate(
            lambda nu: (nu ** (self.s - 1)) * np.exp(-nu), lb, ub
        )
        rat1 = (self.s ** self.s) / np.exp(self.s)
        rat2 = self.pbar * (self.s ** self.s) / (self.c_ub * np.exp(self.alpha_min * rho_s))
        val = int_val - (rat1 - rat2)
        return val

    def _alpha_s(self, u):
        """Returns the evaluation of alpha_s(u) as per eq. (17)."""
        return (self.s - 1) / (u - (u ** self.s))

    def _u_cdt_huc1_eqn(self, u_cdt):
        """
        Returns the value of the HUC1 equation with all terms on the LHS, used
        to find u_cdt via root-finding.

        Args:
            u_cdt : float
                The critical dividing threshold; parameter we search for via
                root-finding
        """
        lb = u_cdt * self.alpha_min
        ub = self.alpha_min
        int_val = integrate(
            lambda nu: (nu ** (self.s - 1)) * np.exp(-nu), lb, ub
        )
        rat1 = (self.s ** self.s) / np.exp(u_cdt * self.alpha_min)
        rat2 = self.pbar * (self.s ** self.s) / (self.c_ub * np.exp(self.alpha_min))
        val = int_val - (rat1 - rat2)
        return val

    def _u_cdt_huc2_eqn(self, u_cdt):
        """
        Returns the value of the HUC2 equation with all terms on the LHS, used
        to find u_cdt via root-finding.

        Args:
            u_cdt : float
                The critical dividing threshold; parameter we search for via
                root-finding
        """
        asucdt = self._alpha_s(u_cdt)
        lb = u_cdt * asucdt
        ub = asucdt
        int_val = integrate(
            lambda nu: (nu ** (self.s - 1)) * np.exp(-nu), lb, ub
        )
        rat1 = (asucdt ** (self.s - 1)) / np.exp(u_cdt * asucdt)
        rat2 = self.pbar * (asucdt ** (self.s - 1)) / (self.c_ub * np.exp(asucdt))
        val = int_val - (rat1 - rat2)
        return val

    def _phi_ivp(self, y, u):
        """
        Returns the evaluation of eq. (24).

        Args:
            y : float
                Utilization level
            u : float
                Location of dividing threshold
        """
        assert y >= u and y <= 1, f'y={y} needs to be in [u={u}, 1]'
        def a1_of_u(u):
            assert u > 0 and u < 1, f'u={u} needs to be in (0, 1)'
            if u > 0 and u < self.u_s:
                return self._alpha_s(u)
            else:
                return self.alpha_min
        a1u = a1_of_u(u)
        lb = y * a1u
        ub = a1u * u
        int_val = integrate(
            lambda nu: (nu ** (self.s - 1)) * np.exp(-nu), lb, ub
        )
        term1 = (self.c_ub * np.exp(y * a1u) / (a1u ** (self.s - 1))) * int_val
        term2 = self.c_ub * np.exp((y - u) * a1u)
        val = term1 + term2
        return val
    
    def is_luc(self):
        """
        Returns True if the low-uncertainty case (LUC) is detected; else, the
        high-uncertainty case (HUC) is detected and False is returned.
        """
        return self._is_luc

    def get_huc_mode(self):
        """
        Returns 1 if operating in HUC1 mode, 2 if operating in HUC2 mode, and
        None if operating in LUC mode.
        """
        mode = None
        if not self.is_luc():
            mode = self._huc_mode
        return mode

    def get_Cs(self):
        """Gets the value for Cs."""
        return self.Cs
    
    def get_theoretical_competitive_ratio(self):
        """
        Returns the theoretical competitive ratio for the current operating
        parameters.
        """
        if self.is_luc() or (not self.is_luc() and self._huc_mode == 1):
            return self.alpha_min
        else:
            assert self._huc_mode == 2, 'Invalid HUC case'
            return (self.s - 1) / (self.u_cdt - self.u_cdt ** self.s)

    def get_w_and_v(self):
        """Gets the values for w and v, if applicable"""
        retval = None
        if self.is_luc():
            retval = (self.w, self.v)
        return retval

    def get_m(self):
        """Gets the value for m, if applicable."""
        retval = None
        if self.is_luc():
            retval = self.m
        return retval

    def set_m(self, m):
        """
        Sets the function parameter m, if applicable (LUC). This m parameter
        controls the resource utilization level for which the maximum valuation
        occurs.

        Args:
            m : float
                The value to use for the m parameter. Must lie in the closed
                interval [w, v].

        Returns: True if success, False otherwise
        """
        success = False
        if self.is_luc():
            if (m >= self.w) and (m <= self.v):
                self.m = m
                success = True
                logger.debug(f'Set m = {self.m}')
        return success

    def get_u(self):
        """Gets the value for u, if applicable."""
        retval = None
        if self.get_huc_mode() == 1:
            retval = self.u
        return retval

    def set_u(self, u):
        """
        Sets the function parameter u, if applicable (HUC1). This u parameter
        controls the resource utilization level for which the maximum valuation
        occurs.

        Args:
            u : float
                The value to use for the u parameter. Must lie in the closed
                interval [u_s, u_cdt].

        Returns: True if success, False otherwise
        """
        success = False
        if self.get_huc_mode() == 1:
            if (u >= self.u_s) and (u <= self.u_cdt):
                self.u = u
                success = True
                logger.debug(f'Set u = {self.u}')
        if success:
            # Compute new value for rho
            if self.u == self.u_s:
                self.rho = self.rho_s
            elif self.u == self.u_cdt:
                self.rho = 1
            else:
                # General case
                lb = max(self.rho_s, self.u)
                ub = 1
                self.rho, r = brentq(
                    lambda rho: self._phi_ivp(rho, self.u) - self.pbar,
                    lb, ub, full_output=True
                )
                assert r.converged, 'Root finder failed to converge'
                logger.debug(f'\tUpdated rho = {self.rho}')
            assert self.rho >= self.rho_s and self.rho <= 1, \
                f'rho={self.rho} needs to be in [rho_s={self.rho_s}, 1]'
        return success

    def get_rho(self):
        """Returns the value for rho, if applicable."""
        retval = None
        if not self.is_luc():
            retval = self.rho
        return retval

    def _cost_func_derivative(self, x):
        """
        Returns the derivative of the power cost function, evaluated at x.
        """
        return self.s * self.a * (x ** (self.s - 1))

    def _phi_eqn(self, y, phi):
        """
        Returns the value of equation (27)/(30) with all terms on the LHS.

        Args:
            y : float
                Utilization level
            phi : float
                Corresponds to phi_luc or phi_huc in the paper; parameter we
                search for via root-finding
        """
        # Input validation
        if self.is_luc():
            assert phi > 0 and phi <= 1, \
                f'Error: phi is {phi} but must be in (0, 1]'
            assert y > 0, \
                f'Error: y is {y} but must be in (0, 1] if we reach here'
            param_val = self.m
        else:
            assert phi > 0 and phi < 1, \
                f'Error: phi is {phi} but must be in (0, 1)'
            assert y > 0, \
                f'Error: y={y} but must be in (0, u={self.u}) if we reach here'
            param_val = self.u
        # Define integrand
        def func(nu):
            """
            The integrand for eq. (27)/(30). Note that the form of this is a
            bit different than in the paper, but it is equivalent.

            Args:
                nu : float
                    The value of the variable of integration
            """
            coeff = (self.alpha_min / (self.s - 1))
            denom = nu + coeff * (nu ** (1 - self.s) - 1)
            try:
                res = 1 / denom
            except ZeroDivisionError:
                # Note: this happens if the numerical integration hits a pole
                # of the rational function. This kept happening when I used the
                # quad integrator, and I spent a while trying to deal with it
                # by providing the points argument, since quad started
                # returning some garbage results. However, adding points only
                # seemed to worsen the situation. Switching to the Romberg
                # integrator fixed this issue; it seems to provide useful
                # results
                res = np.inf
            return res
        # Evaluate function
        lb = 1 / param_val
        ub = phi / y
        int_val = integrate(func, lb, ub)
        val = int_val - np.log(param_val / y)
        return val

    def __call__(self, y):
        # Input validation. Note that each case has its own range of valid
        # inputs, but these could be exceeded in practise. For example, suppose
        # we are operating in the LUC case with miny = 0.78. Furthermore,
        # suppose the current usage of the resource is 0.77999...If we then
        # receive a resource request with a sufficient price, we should accept
        # it as long as it does not exceed the supply. But then, the current
        # usage of the resource would increase beyond 0.78, therefore causing
        # the next pricing call to fail due to an out of bounds error. In this
        # case, the correct action is to return a price of pbar
        assert y >= 0 and y <= 1, f'Input y = {y} must be in [0, 1]'
        if self.is_luc():
            miny = min(1, self.m)
            if y > miny:
                return self.pbar
        elif self._huc_mode == 1:
            if y > self.rho:
                return self.pbar
        else:
            # HUC 2
            if y == 1:
                return self.pbar
        retval = 0
        # Check for trivial case with early return. Pretty sure y == 0 is a
        # sufficient check, but stick to what paper says to implement just in
        # case...
        ret_0 = (y == 0) and \
            (self.is_luc() or (not self.is_luc() and (self._huc_mode == 1)))
        if ret_0:
            return retval
        # Handle non-trivial cases
        if self.is_luc():
            if self.m == self.w:
                # Edge case with simpler form. Should be equivalent to the
                # general case, as long as m = w
                retval = self.s * self._cost_func_derivative(y)
            else:
                # General LUC case
                phi_luc_eqn_partial = partial(self._phi_eqn, y)
                # For the lb, could do np.nextafter(0,1) to get the smallest
                # positive float. However, if phi is any smaller than y/m, it
                # would result in a negative value for the integral. This
                # cannot be a solution since the log term on the RHS of eq.
                # (27) is always non-negative
                #TODO: is above note necessarily true? Maybe there are weird
                # cases where integrand becomes negative (it IS possible to hit
                # vertical asymptotes, so...)
                lb = y / self.m
                ub = 1
                # Invoke root finder!
                try:
                    phi_of_y, r = brentq(phi_luc_eqn_partial, lb, ub, full_output=True)
                except ValueError:
                    lb = np.nextafter(0,1) # Just in case??
                    phi_of_y, r = brentq(phi_luc_eqn_partial, lb, ub, full_output=True)
                assert r.converged, 'Root finder failed to converge'
                # Note: there's a typo in eq. (27). In the original equation,
                # we would have multiplied by cbar (c_ub in the code) rather
                # than pbar, but looking at Figure 3 in the paper show us that
                # pbar is actually the correct multiplicative factor. The
                # reason is that there's no point having a price greater than
                # pbar, since no customer would ever be willing to pay it
                retval = self.pbar * (phi_of_y) ** (self.s - 1)
        else:
            if self._huc_mode == 1:
                if self.u == self.u_s:
                    # Edge case with simpler form. Should be equivalent to
                    # general case as long as u = u_s
                    if y < self.u_s:
                        retval = self.s * self._cost_func_derivative(y)
                    else:
                        retval = self._phi_ivp(y, self.u_s)
                else:
                    # General case
                    if y < self.u:
                        # We use EPS here to approximate the notion of the open
                        # interval (0, 1). Could also use, e.g., 
                        # np.nextafter(0, 1), but this is prone to numerical
                        # issues such as underflow. Assume for now that this
                        # value of EPS won't cause problems
                        EPS = 1e-15
                        phi_huc_eqn_partial = partial(self._phi_eqn, y)
                        lb = EPS
                        ub = 1 - EPS
                        # Invoke root finder!
                        phi_of_y, r = brentq(phi_huc_eqn_partial, lb, ub, full_output=True)
                        assert r.converged, 'Root finder failed to converge'
                        retval = self.c_ub * (phi_of_y ** (self.s - 1))
                    else:
                        retval = self._phi_ivp(y, self.u)
            else:
                if y <= self.u_cdt:
                    retval = self._cost_func_derivative(y / self.u_cdt)
                else:
                    retval = self._phi_ivp(y, self.u_cdt)
        return retval

    def __str__(self):
        # TODO: consider adding more details for each case
        sb = ''
        sb += 'Optimal power price func ('
        if self.is_luc():
            sb += f'LUC, m = {self.m}'
        else:
            if self._huc_mode == 1:
                sb += 'HUC1, '
            else:
                sb += 'HUC2, '
            sb += f'Cs = {self.Cs}, u_cdt = {self.u_cdt}'
        sb += ', a = {self.a}, s = {self.s}, pbar = {pbar})\n'
        return sb

class PriceFuncFactory:
    """Builds optimal price function for (various) valuation functions."""

    @classmethod
    def get_price_func(self, val_func, **kwargs):
        """
        Creates an optimal price function for the given valuation function.

        Args:
            val_func : ValuationFunc
                Valuation function instance we want optimal pricing for
        
        Keyword args:
            pbar : float
                    The maximum possible bidder valuation
        
        Returns: optimal PriceFunc instance for the given val_func
        """
        assert isinstance(val_func, CostFunc), 'Incorrect argument class'
        if isinstance(val_func, PowerFunc):
            pbar = kwargs['pbar']
            return OptPowerPriceFunc(val_func, pbar)
        else:
            raise NotImplementedError(f'Oops...no pricing implementation for {val_func}')


# Playground!
if __name__ == "__main__":
    vals = [(0.223, 3), (8.38e-6, 1.2)] # (a, s) pairs

    # Case 1: LUC
    for i in range(len(vals)):
        a, s = vals[i]
        val_func = PowerFunc(a, s)
        c_ub = val_func.get_max_marginal_cost()
        c_lb = val_func.get_min_marginal_cost()

        # Different pbar values for the different cases...
        # Note: t defines a convex combination between c_lb and c_ub for the LUC
        # case. You will notice that the t-value GREATLY affects what the price
        # functions look like. For example, when s=1.2 and t=0.95, you see heavy
        # utilization compared to when s=1.2 and t=0.5. Another example is that
        # with s=3 and t=0.95, the v-curve has an inflection point whereas this
        # does not occur with s=3 and t=0.5
        t = 0.95
        pbar_luc = t * c_ub + (1 - t) * c_lb

        print(f'Starting LUC case...(pbar = {pbar_luc})')
        price_func = PriceFuncFactory.get_price_func(val_func, pbar=pbar_luc)
        print(price_func.is_luc())
        w, v = price_func.get_w_and_v()
        print(f'w = {w}, v = {v}')
        fig = plt.figure()
        ms = [w, w*(1 + 1e-8), (w + v) / 2, v]
        for mm in ms:
            assert price_func.set_m(mm), f'Failed to set m={mm}'
            y = np.linspace(0, mm)
            phi = []
            for yy in y:
                phi.append(price_func(yy))
            plt.plot(y, phi, label=f'm = {mm}')
        plt.title(
            f'LUC case (a={a}, s={s}, '
            f'pbar={np.format_float_scientific(pbar_luc, precision=3)})'
        )
        plt.xlabel('Utilization')
        plt.xlim([0, 1])
        plt.ylabel('Price')
        plt.legend()
        fname = f'price_test_luc_a{a}_s{s}'
        plt.savefig(f'images/test/{fname}.eps', format='eps')
        plt.savefig(f'images/test/{fname}.png', format='png')

    # Case 1: HUC1
    for i in range(len(vals)):
        a, s = vals[i]
        val_func = PowerFunc(a, s)
        c_ub = val_func.get_max_marginal_cost()
        c_lb = val_func.get_min_marginal_cost()

        done = False
        Cs = 0.9 * c_ub # 1. Ensure not HUC the first time
        # t defines a convex combination between c_ub and Cs. The t-value
        # greatly affects what the price function looks like
        t = 0.5
        while not done:
            pbar_huc1 = t * Cs + (1 - t) * c_ub # 3. Guarantee HUC1
            price_func = PriceFuncFactory.get_price_func(val_func, pbar=pbar_huc1)
            Cs = price_func.get_Cs() # 2. Get actual Cs
            done = (price_func.get_huc_mode() == 1)
        print(f'Starting HUC case 1...(pbar = {pbar_huc1})')

        fig = plt.figure()
        u_s = price_func.u_s
        u_cdt = price_func.u_cdt
        us = [u_s, (u_s + u_cdt) / 2, u_cdt]
        for uu in us:
            assert price_func.set_u(uu), f'Failed to set u={u}'
            rho = price_func.get_rho()
            y = np.linspace(0, rho)
            phi = []
            for yy in y:
                phi.append(price_func(yy))
            plt.plot(y, phi, label=f'u = {uu}')
        plt.title(
            f'HUC1 case (a={a}, s={s}, '
            f'pbar={np.format_float_scientific(pbar_huc1, precision=3)})'
        )
        plt.xlabel('Utilization')
        plt.xlim([0, 1])
        plt.ylabel('Price')
        plt.legend()
        fname = f'price_test_huc1_a{a}_s{s}'
        plt.savefig(f'images/test/{fname}.eps', format='eps')
        plt.savefig(f'images/test/{fname}.png', format='png')

    # Case 3: HUC2
    for i in range(len(vals)):
        a, s = vals[i]
        val_func = PowerFunc(a, s)
        c_ub = val_func.get_max_marginal_cost()
        c_lb = val_func.get_min_marginal_cost()
        
        done = False
        Cs = 0.9 * c_ub # 1. Ensure not HUC the first time
        while not done:
            pbar_huc2 = 2 * Cs # 3. Guarantee HUC2
            price_func = PriceFuncFactory.get_price_func(val_func, pbar=pbar_huc2)
            Cs = price_func.get_Cs() # 2. Get actual Cs
            done = (price_func.get_huc_mode() == 2)
        print(f'Starting HUC case 2...(pbar = {pbar_huc2})')

        fig = plt.figure()
        y = np.linspace(0, 0.99)
        phi = []
        for yy in y:
            phi.append(price_func(yy))
        plt.plot(y, phi)
        plt.title(
            f'HUC2 case (a={a}, s={s}, '
            f'pbar={np.format_float_scientific(pbar_huc2, precision=3)})'
        )
        plt.xlabel('Utilization')
        plt.xlim([0, 1])
        plt.ylabel('Price')
        fname = f'price_test_huc2_a{a}_s{s}'
        plt.savefig(f'images/test/{fname}.eps', format='eps')
        plt.savefig(f'images/test/{fname}.png', format='png')
