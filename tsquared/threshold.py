import numpy as np
from scipy.linalg import sqrtm
from scipy.optimize import minimize


class ThresholdCalculator:
    """Class providing different methods for T-squared threshold calculation.

    This class implements two main approaches for determining T-squared thresholds:
    1. Statistical approach based on F-distribution (default method)
    2. Optimization-based approach using convex optimization

    The statistical approach is based on classical Hotelling's T-squared distribution
    theory, while the optimization approach uses convex optimization to find the
    maximum T-squared value under box constraints.
    """

    @staticmethod
    def statistical_ucl(n_samples, n_features, alpha):
        """Calculate threshold using classical statistical UCL approach.

        Args:
            n_samples (int): Number of samples in training set
            n_features (int): Number of features
            alpha (float): Significance level (between 0 and 1)

        Returns:
            float: Upper Control Limit (UCL) threshold value

        Raises:
            ValueError: If alpha is not between 0 and 1
            ValueError: If n_samples <= n_features
        """
        if not 0 <= alpha <= 1:
            raise ValueError("The significance level alpha must be between 0 and 1.")

        if n_samples <= n_features:
            raise ValueError("The number of samples must be greater than the number of features.")

        from scipy import stats
        critical_val = stats.f.ppf(q=1 - alpha, dfn=n_features,
                                   dfd=n_samples - n_features)

        return n_features * (n_samples + 1) * (n_samples - 1) / \
            (n_samples * (n_samples - n_features)) * critical_val

    @staticmethod
    def optimization_based(mu, Sigma):
        """Calculate threshold using optimization-based approach.

        This method maximizes the T-squared value under box constraints using
        convex optimization techniques.

        Args:
            mu (numpy.ndarray): Mean vector of shape (n_features,)
            Sigma (numpy.ndarray): Covariance matrix of shape (n_features, n_features)

        Returns:
            float: Optimized T-squared threshold value

        Raises:
            ValueError: If Sigma is not positive semidefinite
            ValueError: If mu and Sigma dimensions don't match
            ValueError: If mu is not 1D or Sigma is not 2D
        """
        # Input validation
        if not isinstance(mu, np.ndarray) or not isinstance(Sigma, np.ndarray):
            raise TypeError("mu and Sigma must be numpy arrays")

        if mu.ndim != 1:
            raise ValueError("mu must be a 1D array")

        if Sigma.ndim != 2:
            raise ValueError("Sigma must be a 2D array")

        if mu.shape[0] != Sigma.shape[0] or Sigma.shape[0] != Sigma.shape[1]:
            raise ValueError("Dimension mismatch between mu and Sigma")

        # Check if Sigma is positive semidefinite and handle near-singular cases
        eigvals = np.linalg.eigvals(Sigma)
        if not np.all(eigvals >= -1e-10):
            raise ValueError("Sigma must be positive semidefinite")

        # Handle near-singular matrices by regularization if needed
        if np.any(eigvals < 1e-10):
            # Add small regularization term to diagonal
            Sigma = Sigma + np.eye(len(mu)) * 1e-10

        # Extract diagonal elements (variances)
        sigma = np.diag(Sigma)

        # Handle zero or near-zero variances
        sigma = np.where(sigma < 1e-10, 1e-10, sigma)

        # Extract principal square root of covariance matrix with error handling
        try:
            sqrt_Sigma = sqrtm(Sigma)
        except ValueError:
            # If sqrtm fails, use regularized version
            sqrt_Sigma = sqrtm(Sigma + np.eye(len(mu)) * 1e-10)

        # Construct right-hand side constraints (3-sigma rule)
        rhs = np.array([3 * sigma[i] / np.linalg.norm(sqrt_Sigma[:, i])
                        for i in range(len(mu))])

        # Define optimization problem
        obj_fun = lambda y: -y
        constraints = {
            'type': 'ineq',
            'fun': lambda y: rhs - y * np.ones(len(rhs))
        }

        # Solve optimization problem
        result = minimize(obj_fun, 1.0, constraints=constraints,
                          bounds=[(0, None)], method='SLSQP')

        if not result.success:
            raise RuntimeError("Optimization failed to converge")

        return float(result.x[0])