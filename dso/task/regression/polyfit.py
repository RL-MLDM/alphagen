"""Polynomial optimizer used for deep symbolic optimization."""

import numpy as np
import scipy
from scipy import linalg, optimize, stats
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from itertools import compress

from dso.library import Polynomial, StateChecker


class PolyRegressorMixin:
    """
    Defines auxiliary functions to be used by DSO's specialized regressors
    """
    def np_array_signature(self, X):
        """
        Computes simplified hash of matrix X (m rows, n columns, m > n) for polynomial fitting purposes.
        Parameters
        ==========
        X : ndarray
            X data
        
        Returns
        =======
        result : int
            Simplified hash of X.
        """
        return hash((X.shape,                                               # array shape
                     X.diagonal().tobytes(),                                # main (top) diagonal
                     X.diagonal(offset=X.shape[1]-X.shape[0]).tobytes()))   # lowest diagonal
        
    def delete_oldest_pair(self, dictionary):
        """
        Deletes oldest (key, value) pair from dictionary.
        Takes advantage of ordered dictionaries in Python 3.6 and newer.
        """
        dictionary.pop(next(iter(dictionary)))

    def zero_out_ls_terms(self, cLS, XtX_inv, zero_out_indices):
        """
        Fast recomputation of least-squares fit when zeroing-out terms in the regression
        Solves:  [ XtX_inv   indexing^T ] [ c ] == [ Xt * y ]
                 [ indexing      0      ] [ z ]    [    0   ]
        which corresponds to the optimality conditions of:
                max_c || X c - y || : indexing * c = 0
        """
        # 1. form D = XtX_inv * indexing^T = (indexing * XtX_inv)^T = XtX_inv[indexing,:]^T
        D = np.ascontiguousarray(XtX_inv[zero_out_indices,:].transpose())
        # 2. form E = indexing * D = D[indexing,:]
        E = D[zero_out_indices,:]
        # NOTE: E is just a minor of XtX_inv, hence it is symmetric and positive definite
        # 3. solve linear system E * z = cLS[indexing]
        z = scipy.linalg.solve(E, cLS[zero_out_indices], assume_a="pos")    # take advantage D is PD
        # 4. compute solution with zero-ed out components and return
        return cLS - np.matmul(D, z)
    
    def regression_p_values(self, X, XtX_inv, y, c):
        """
        Computes p-values using t-Test (null hyphotesis: c_i == 0)
        """
        yhat = np.matmul(X, c)
        df = len(X) - X.shape[1]
        mse = sum((y - yhat)**2)/df
        sd_err = np.sqrt(mse * XtX_inv.diagonal())
        t_vals = c/sd_err
        return 2 * (1 - stats.t.cdf(np.abs(t_vals), df))
    

class DSOLeastSquaresData:
    """
    Holds Gram inverse and pseudo-inverse
    """
    def __init__(self, X, compute_inv=False):
        if X.shape[0] < X.shape[1]:
            raise AssertionError("X should have more rows than columns.")
        self.X_pinv = scipy.linalg.pinv(X)
        if compute_inv:
            XtX = np.matmul(X.transpose(), X)
            if not np.isfinite(np.linalg.cond(XtX)):
                raise AssertionError("X^t * X should always be invertible.")
            self.XtX_inv = scipy.linalg.inv(XtX)
        else:
            self.XtX_inv = None

class DSOLeastSquaresRegressor(PolyRegressorMixin):
    """
    Solve the problem min_{c} || X*c - y || by applying the psuedo-inverse
            c = (X^T*X)^{-1} * X^T * y
    """
    def __init__(self, cutoff_p_value=1.0, n_max_terms=None, coef_tol=1E-12):
        # include intercept_ just to match with sklearn regressors
        self.intercept_ = 0.0
        self.coef_ = None
        self.n_max_records = 10
        self.data_dict = {}
        if isinstance(cutoff_p_value, float) and \
           cutoff_p_value > 0.0 and cutoff_p_value <= 1.0:
            self.cutoff_p_value_ = cutoff_p_value
        else:
            raise ValueError("cutoff p-value should be in (0., 1.]")
        if (isinstance(n_max_terms, int) and n_max_terms >= 2) or n_max_terms is None:
           # 2 terms: constant + term
            self.n_max_terms_ = n_max_terms
        elif isinstance(n_max_terms, int):
            raise ValueError("maximum number of terms should be >= 2")
        else:
            raise TypeError("n_max_terms should be int or None")
        self.coef_tol_ = coef_tol
    
    def fit(self, X, y, X_signature=None):
        """
        Linear fit between X (data) and y (observations)
        """
        # check if data is cached, if not add it to the data_dict
        if X_signature is None:
            X_signature = self.np_array_signature(X)
        if X_signature not in self.data_dict.keys():
            while len(self.data_dict) >= self.n_max_records:
                self.delete_oldest_pair(self.data_dict)
            self.data_dict[X_signature] = DSOLeastSquaresData(X, self.cutoff_p_value_ < 1.0 or
                                                                 self.n_max_terms_ is not None)
        # perform regression
        lsd = self.data_dict[X_signature]
        self.coef_ = np.matmul(lsd.X_pinv, y)
        # if necessary, enforce p-value cutoff and maximum number of terms equal to zero
        if self.cutoff_p_value_ < 1.0 or \
           (self.n_max_terms_ is not None and \
            np.count_nonzero(np.abs(self.coef_) > self.coef_tol_) > self.n_max_terms_):
            # compute p-values for all coefficients
            p_values = self.regression_p_values(X, lsd.XtX_inv, y, self.coef_)
            # sort coefficients from smallets to largest p-value
            perm = np.argsort(p_values)
            # compute number of terms to keep
            n_terms = next((x[0] for x in enumerate(perm) if p_values[x[1]] > self.cutoff_p_value_),
                           len(p_values))
            if self.n_max_terms_ is not None:
                n_terms = np.minimum(n_terms, self.n_max_terms_)
            # zero-out coefficients with largest p-values
            if n_terms < len(self.coef_):
                zero_out_indices = np.sort(perm[n_terms:])
                self.coef_ = self.zero_out_ls_terms(self.coef_, lsd.XtX_inv, zero_out_indices)
    
    def clear(self):
        """
        Reset memory allocated to pseudo-inverses
        """
        self.data_dict.clear()
    

class DSOLassoRegressorData:
    """
    Holds information useful for multiple calls to DSOLassoRegressor
    """
    def __init__(self, X):
        self.XtX_inv = scipy.linalg.inv(np.matmul(X.transpose(), X))
        self.X_pinv = np.matmul(self.XtX_inv, X.transpose())
        self.n_obs = X.shape[0]
        self.n_params = X.shape[1]
    

class DSOLassoRegressor(PolyRegressorMixin):
    """
    Computes Lasso for X, y with gamma weighted L1 regularization, i.e. finds optimum beta for
        min_{beta} (1/2 * 1/var(y) * 1/n_obs * || y - X * beta ||^2_2 + gamma * 1/n_params * || beta ||_1)
    
    Implementation solves dual Lasso problem.
    """
    def __init__(self, gamma=0.1, comp_tol=1E-4, rtrn_constrnd_ls=True):
        # include intercept_ just to match with sklearn regressors        
        self.intercept_ = 0.0
        self.coef_ = None
        self.gamma_ = gamma         # L1 weight -- standarized
        self.comp_tol_ = comp_tol   # tolerance for complementarity slackness
        self.rtrn_constrnd_ls_ = rtrn_constrnd_ls       # return re-optimized sparse least-squares
        self.data_dict = {}
        self.n_max_records = 10
    
    def fit(self, X, y, X_signature=None):
        # check if signature is provided, compute signature otherwise
        if X_signature is None:
            X_signature = self.np_array_signature(X)
        # if signature does not appear in dict, compute and store lasso regressor data
        if X_signature not in self.data_dict.keys():
            while len(self.data_dict) >= self.n_max_records:
                self.delete_oldest_pair(self.data_dict)
            self.data_dict[X_signature] = DSOLassoRegressorData(X)
        # perform lasso fit
        ldata = self.data_dict[X_signature]
        self.coef_ = self.dual_lasso(ldata.XtX_inv, ldata.X_pinv,
                                     ldata.n_obs, ldata.n_params, y)
    
    def dual_lasso(self, XtX_inv, X_pinv, n_obs, n_params, y):
        # compute program parameters
        beta_LS = np.matmul(X_pinv, y)  # least squares solution
        rho_bnd = n_obs/n_params * np.var(y) * self.gamma_

        # currently only scipy.minimize is supported as the solver
        # objective function and derivatives
        f_obj = lambda rho : 1/2 * np.dot(rho, np.matmul(XtX_inv, rho)) - np.dot(beta_LS, rho)
        g_obj = lambda rho : np.matmul(XtX_inv, rho) - beta_LS
        # define initial guess
        rho_init = rho_bnd * np.ones(n_params)
        rho_init[beta_LS > 0.0] *= -1.0
        # call scipy minimize
        bnds = scipy.optimize.Bounds(-rho_bnd * np.ones(n_params), rho_bnd * np.ones(n_params))
        res = scipy.optimize.minimize(f_obj, rho_init, jac=g_obj, bounds=bnds)
        if not res.success:
            raise Exception("failed to solve dual lasso problem.")
        rho_opt = res.x

        if self.rtrn_constrnd_ls_:
            # determine indexes to zero-out
            zero_out_indices = [i for i in range(n_params) if 
                                0.25 * (1 + rho_opt[i]/rho_bnd) * (1 - rho_opt[i]/rho_bnd) > self.comp_tol_]
            # recompute least squares with zero-ed out coefficients
            beta_cLS = self.zero_out_ls_terms(beta_LS, XtX_inv, zero_out_indices)
            beta_cLS[zero_out_indices] = 0.0    # these will be zero up to floating point precision
            return beta_cLS
        else:
            # compute lasso regressor parameters
            beta_Lasso = beta_LS - np.matmul(XtX_inv, rho_opt)
            # crash (dual) interior interior point solution
            for i in range(n_params):
                if 0.25 * (1 + rho_opt[i]/rho_bnd) * (1 - rho_opt[i]/rho_bnd) > self.comp_tol_:
                    beta_Lasso[i] = 0.0
            # return solution
            return beta_Lasso
        
    def clear(self):
        """
        Reset memory allocated to Gram inverse and pseudo inverse
        """
        self.data_dict.clear()
    

regressors = {
        "linear_regression": LinearRegression,
        "lasso": Lasso,
        "ridge": Ridge,
        "dso_least_squares" : DSOLeastSquaresRegressor,
        "dso_lasso" : DSOLassoRegressor,
    }

inverse_function_map = {
    "add" : np.subtract,
    "sub" : np.add,
    "mul" : np.divide,
    "div" : np.multiply,
    "sin" : np.arcsin,
    "cos" : np.arccos,
    "tan" : np.arctan,
    "exp" : np.log,
    "log" : np.exp,
    "sqrt" : np.square,
    "n2" : np.sqrt,
    "n3" : np.cbrt,
    "abs" : np.abs,
    "tanh" : np.arctanh,
    "inv" : np.reciprocal
}


def partial_execute(traversal, X):
    """
    Evaluate from terminal nodes all the branches that has no 'poly' token.
    If some (function) value in the partial execution is not finite, None is returned.
    """
    apply_stack = []
    for node in traversal:
        apply_stack.append([node])
        while len(apply_stack[-1]) == apply_stack[-1][0].arity + 1:
            token = apply_stack[-1][0]
            terminals = apply_stack[-1][1:]
            if isinstance(token, Polynomial):
                intermediate_result = "poly"
            elif token.input_var is not None:
                intermediate_result = X[:, token.input_var]
            else:
                if all(isinstance(t, np.ndarray) for t in terminals):
                    if isinstance(token, StateChecker):
                        token.set_state_value(X[:, token.state_index])
                    intermediate_result = token(*terminals)

                    if not np.isfinite(intermediate_result).all():
                        return None
                else:
                    intermediate_result = [token, *terminals]

            if len(apply_stack) != 1:
                apply_stack.pop()
                apply_stack[-1].append(intermediate_result)
            else:
                return intermediate_result


def recursive_inversion(intermediate_result, y):
    """
    Obtain the 'y data' for 'poly' token by inverting tokens starting from root.
    For tokens of arity 2, find out the child that has been evaluated (there must be
    one and only one), and get the value of the other child, until 'poly' is reached.

    If some entry of y is not finite, None is returned.
    """
    if not np.isfinite(y).all():
        return None
    if intermediate_result == "poly":
        return y

    assert len(intermediate_result) < 4
    func = intermediate_result[0]
    if func.arity == 1:
        out = inverse_function_map[func.name](y)
        return recursive_inversion(intermediate_result[1], out)
    else:
        if isinstance(intermediate_result[1], np.ndarray):
            if func.name == "sub" or func.name == "div":
                out = func(intermediate_result[1], y)
            else:
                out = inverse_function_map[func.name](y, intermediate_result[1])
            return recursive_inversion(intermediate_result[2], out)
        else:
            out = inverse_function_map[func.name](y, intermediate_result[2])
            return recursive_inversion(intermediate_result[1], out)


def make_poly_data(traversal, X, y):
    """
    Obtain the 'y data' for 'poly' token in two steps. First is a bottom-up pass of the
    expression tree starting from terminal nodes, all the branches that can be evaluated
    will be evaluated. Effectively this creates a single chain of unary functions with the
    terminal token being the 'poly' token. The second step is a top-down pass inverting
    all the unary functions in partial_results starting from the root.

    If some (function) value in the partial execution or recursive inversion is not finite,
    None is returned.
    """
    partial_results = partial_execute(traversal, X)
    return None if partial_results is None else recursive_inversion(partial_results, y)


def nonnegative_int_tuples_to_sum(length, given_sum):
    """
    generate all tuples of nonnegative integers that are of size length such that sum of entries equals given_sum
    https://stackoverflow.com/questions/7748442/generate-all-possible-lists-of-length-n-that-sum-to-s-in-python
    """
    if length == 1:
        yield (given_sum,)
    else:
        for value in range(given_sum + 1):
            for permutation in nonnegative_int_tuples_to_sum(length - 1, given_sum - value):
                yield (value,) + permutation


def generate_all_exponents(n_input_var, degree):
    """
    Generate a list of tuples of exponents corresponding to all monomials of n_input_var
    variables of degree at most degree.
    """
    out = []
    for monomial_degree in range(degree + 1):
        out.extend(list(nonnegative_int_tuples_to_sum(n_input_var, monomial_degree)))
    return out


class PolyOptimizerData(PolyRegressorMixin):
    """
    Helper class to process and hold data passed to the polynomial optimizer
    """
    def __init__(self, X, degree, X_signature_=None):
        """
        Generate and store the data for all the monomials (basis for poly).
        This allows dso to skip repeated generation of monomials' data for
        the same X data during training.
        
        Parameters
        ==========
        X : ndarray
            X data
        degree: int
            The (maximal) degree of the polynomials used to fit the data.
        """
        self.all_exponents = generate_all_exponents(X.shape[1], degree)
        self.all_monomials_data = Polynomial.eval_monomials(X, self.all_exponents)
        if X_signature_ is None:
            self.X_signature = self.np_array_signature(X)
        else:
            self.X_signature = X_signature_
    

class PolyOptimizer(PolyRegressorMixin):
    def __init__(self, degree, coef_tol, regressor, regressor_params):
        """
        Optimizer for fitting a polynomial in traversals to given datasets.

        Parameters
        ==========
        degree : int
            The (maximal) degree of the polynomials used to fit the data.

        coef_tol : float
            Cutoff value for the coefficients of polynomials. Coefficients
            with magnitude less than this value will be regarded as 0.

        regressor : str
            Key to dictionary regressors. Supported options are 'lasso',
            'ridge', and 'linear_regression'.

        regressor_params : dict
            Parameters for the regressor. See sklearn for more information.
        """
        self.degree = degree
        self.coef_tol = coef_tol
        self.regressor = regressors[regressor](**regressor_params)
        self.data_dict = dict()
        self.n_max_records = 10

    def fit(self, X, y):
        """
        Fit a polnomial to the dataset (X, y) based on the regressor.
        Parameters
        ==========
        X : ndarray
            X data
        y : ndarray
            y data
        
        Returns
        =======
        result : Polynomial(Token)
            The polynomial token of which the underlying polynomial best fits the dataset (X, y)
        """
        X_signature = self.np_array_signature(X)
        if X_signature not in self.data_dict.keys():
            while len(self.data_dict) >= self.n_max_records:
                self.delete_oldest_pair(self.data_dict)
            self.data_dict[X_signature] = PolyOptimizerData(X, self.degree, X_signature)
        
        # reference to PolyOptimizerData object (to avoid multiple lookups)
        pod = self.data_dict[X_signature]
        
        try:
            # perform fit; pass monomial data signature if using custom DSO optimizers
            if isinstance(self.regressor, (DSOLeastSquaresRegressor,)):
                self.regressor.fit(pod.all_monomials_data, y, pod.X_signature)
            else:
                self.regressor.fit(pod.all_monomials_data, y)
        except: # the only thing we have seen is ValueError
            return Polynomial([(0,)*X.shape[1]], np.ones(1))
        
        # Correct the coefficient of the constant term when regressor.intercept_ is nonzero.
        # This can happen when fit_intercept in regressor_params is True.
        if self.regressor.intercept_ != 0.0: 
            self.regressor.coef_[0] += self.regressor.intercept_

        if np.isfinite(self.regressor.coef_).all():
            mask = np.abs(self.regressor.coef_) >= self.coef_tol
            if np.count_nonzero(mask) == 0:
                # fit succesful, but all coefficients are zero
                return Polynomial([(0,)*X.shape[1]], np.ones(0))
            return Polynomial(list(compress(pod.all_exponents, mask)), self.regressor.coef_[mask])
        
        return Polynomial([(0,)*X.shape[1]], np.ones(1))
    
    def clear(self):
        """
        Reset memory allocated to exponents and monomials data, and to cached regressor data
        """
        self.data_dict.clear()
        if isinstance(self.regressor, (DSOLeastSquaresRegressor, DSOLassoRegressor)):
            self.regressor.clear()
        
    

class PolyGenerator(object):
    def __init__(self, degree, n_input_var):
        """
        Parameters
        ----------
        degree : int
            Maximal degree of the polynomials to be generated.
        coef : int
            Number of input (independent) variables.
        """
        self.all_exponents = generate_all_exponents(n_input_var, degree)

    def generate(self, n_terms_mean=2, n_terms_sd=1,
                 coef_mean=0, coef_sd=10, coef_precision=2):
        """
        Generate a Polynomial token. The number of terms and the coefficients of the
        terms are sampled from normal distributions based on the input parameters.
        Parameters
        ----------
        n_terms_mean : int
            Mean of the normal distribution from which number of terms is sampled.
        n_terms_sd : int
            Standard deviation of the normal distribution from which number of terms is sampled.
        coef_mean : float
            Mean of the normal distribution from which the coefficents are sampled.
        coef_sd : float
            Standard deviation of the normal distribution from which the coefficents are sampled.
        coef_precision : int
            Number of decimal places of the coefficients in the generated polynomial.

        Returns
        =======
        result : Polynomial(Token)
            The generated polynomial token
        """
        n_terms = int(max(1, np.random.normal(n_terms_mean, n_terms_sd)))
        n_terms = min(n_terms, len(self.all_exponents))
        coefs = np.random.normal(coef_mean, coef_sd, n_terms)
        coefs = np.around(coefs, decimals=coef_precision)
        coef_pos = np.random.choice(len(self.all_exponents), n_terms, replace=False)
        return Polynomial([self.all_exponents[pos] for pos in coef_pos], coefs)
