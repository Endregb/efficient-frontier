import numpy as np
import pandas as pd
import random
from scipy.optimize import minimize

def monte_carlo(N, tickers, annualized_mean_returns, annualized_covariance):
    """
    Generates N random portfolios using Monte Carlo simulation for portfolio optimization analysis.
    
    This function creates random weight allocations and calculates the corresponding portfolio
    returns and volatilities to visualize the risk-return landscape of possible portfolios.
    
    Parameters:
    -----------
    N : int
        Number of random portfolios to generate
    tickers : list
        List of ticker symbols for the assets in the portfolio
    annualized_mean_returns : array-like
        Annualized expected returns for each asset
    annualized_covariance : array-like
        Annualized covariance matrix of asset returns
    
    Returns:
    --------
    tuple : (random_weights_df, portfolio_returns, portfolio_volatilities)
        random_weights_df : pandas.DataFrame
            DataFrame containing the weights for each random portfolio
        portfolio_returns : numpy.ndarray
            Array of portfolio returns for each random portfolio
        portfolio_volatilities : numpy.ndarray
            Array of portfolio volatilities (risk) for each random portfolio
    """
    random_weights = np.zeros((N, len(tickers)))
    portfolio_returns = []
    portfolio_volatilities = []

    for i in range(N):
        for j in range(len(tickers)):
            random_weights[i][j] = random.uniform(0,1)

        # Normalise weights
        weight_sum = sum(random_weights[i])
        random_weights[i] = random_weights[i] / weight_sum

        # Calculate portfolio return
        portfolio_return = np.sum(random_weights[i] * annualized_mean_returns)
        portfolio_returns.append(portfolio_return)

        # Calculate portfolio volatility
        portfolio_volatility = np.sqrt(np.dot(random_weights[i].T, np.dot(annualized_covariance, random_weights[i])))
        portfolio_volatilities.append(portfolio_volatility)
    
    portfolio_returns = np.array(portfolio_returns)
    portfolio_volatilities = np.array(portfolio_volatilities)
        
    random_weights_df = pd.DataFrame(random_weights, columns=tickers)
    return random_weights_df, portfolio_returns, portfolio_volatilities

def markowitz_mean_variance(covariance, mean_returns = None, desired_return = 0, total_weight = 1):
    """
    Calculates optimal portfolio weights using the analytical Markowitz mean-variance method.
    
    This function uses Lagrange multipliers to solve the quadratic optimization problem
    analytically. It can find either the minimum variance portfolio (when desired_return=0)
    or the portfolio that minimizes risk for a specific target return.
    
    Parameters:
    -----------
    covariance : array-like
        Covariance matrix of asset returns
    mean_returns : array-like, optional
        Expected returns for each asset (required when desired_return > 0)
    desired_return : float, default=0
        Target portfolio return. If 0, finds minimum variance portfolio
    total_weight : float, default=1
        Total portfolio weight constraint (typically 1 for fully invested)
    
    Returns:
    --------
    tuple : (weights, lagrange_multipliers)
        weights : numpy.ndarray
            Optimal portfolio weights
        lagrange_multipliers : numpy.ndarray
            Lagrange multipliers from the optimization (for analysis)
    
    Notes:
    ------
    This method assumes no bounds on weights (allows negative weights/short selling).
    For constrained optimization, use portfolio_minimize() or portfolio_maximize_sharpe().
    """
    n = covariance.shape[0]
    one = np.ones((n, 1)) # n x 1
    zero_scalar = 0.0

    if desired_return == 0:
        A = np.block([[2 * covariance, one],
                    [one.T, zero_scalar]])
        
        b = np.array([0] * n + [total_weight])
    
    else:
        # Reshape mean_returns:
        mu = mean_returns.values.reshape(-1, 1)

        A = np.block([[2 * covariance, mu, one],
                      [mu.T, zero_scalar, zero_scalar],
                      [one.T, zero_scalar, zero_scalar]])
        b = np.array([0] * n + [desired_return] + [total_weight])

    solution = np.linalg.solve(A, b)

    weights = solution[:n]
    lagrange_multiplier = solution[n:] # return restriction, budget/weight restriction

    return weights, lagrange_multiplier

def portfolio_minimize(mean_returns, covariance, desired_return, min_weight = 0.0, max_weight = 1.0):
    """
    Minimizes portfolio variance for a target return using numerical optimization (SLSQP).
    
    This function uses Sequential Least Squares Programming to find the portfolio with
    minimum risk (variance) that achieves a specific target return, subject to weight
    constraints. This is useful for constructing the efficient frontier.
    
    Parameters:
    -----------
    mean_returns : array-like
        Expected returns for each asset
    covariance : array-like
        Covariance matrix of asset returns
    desired_return : float
        Target portfolio return to achieve
    min_weight : float, default=0.0
        Minimum weight for any individual asset (0.0 = no short selling)
    max_weight : float, default=1.0
        Maximum weight for any individual asset (1.0 = no leverage)
    
    Returns:
    --------
    scipy.optimize.OptimizeResult
        Optimization result containing:
        - x: optimal portfolio weights
        - success: whether optimization converged
        - fun: final objective value (portfolio variance)
        - message: optimization status message
    
    Notes:
    ------
    Uses equality constraints for target return and weight sum = 1.
    Bounds prevent short selling and leverage when using default parameters.
    """

    # Initial guess with equal weights
    n_assets = len(mean_returns)
    initial_weights = [1 / n_assets] * n_assets

    # The weight of any asset must be between 0 and 1 inclusive
    bounds = tuple((min_weight, max_weight) for i in range(n_assets))
    constraints = [
        # Sum of weights equals to 1
        {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
        # Ensures that the difference between calculated expected returns and target return is 0 
        {"type": "eq", "fun": lambda w : np.dot(w, mean_returns.values if hasattr(mean_returns, 'values') else mean_returns) - desired_return}]
    
    # Objective function: minimize portfolio variance (ensure it returns a scalar)
    def objective(w):
        # Convert pandas data to numpy arrays if needed
        cov_matrix = covariance.values if hasattr(covariance, 'values') else covariance
        portfolio_variance = np.dot(w, np.dot(cov_matrix, w))
        return float(portfolio_variance)
    
    result = minimize(objective, x0 = initial_weights, bounds = bounds, constraints = constraints, options = {"ftol": 1e-10})
    return result

def portfolio_maximize_sharpe(mean_returns, covariance, risk_free_rate, min_weight = 0.0, max_weight = 1.0):
    """
    Maximizes the Sharpe ratio to find the optimal risk-adjusted portfolio.
    
    This function finds the portfolio that provides the highest risk-adjusted return
    (Sharpe ratio) by minimizing the negative Sharpe ratio using numerical optimization.
    This represents the tangency portfolio on the efficient frontier.
    
    Parameters:
    -----------
    mean_returns : array-like
        Expected returns for each asset
    covariance : array-like
        Covariance matrix of asset returns
    risk_free_rate : float
        Risk-free rate of return (e.g., government bond yield)
    min_weight : float, default=0.0
        Minimum weight for any individual asset (0.0 = no short selling)
    max_weight : float, default=1.0
        Maximum weight for any individual asset (1.0 = no leverage)
    
    Returns:
    --------
    scipy.optimize.OptimizeResult
        Optimization result containing:
        - x: optimal portfolio weights for maximum Sharpe ratio
        - success: whether optimization converged
        - fun: negative Sharpe ratio (multiply by -1 to get actual Sharpe ratio)
        - message: optimization status message
    
    Notes:
    ------
    The Sharpe ratio is calculated as (portfolio_return - risk_free_rate) / portfolio_volatility.
    This portfolio represents the optimal point for mean-variance investors.
    """

    # Initial guess with equal weights
    n_assets = len(mean_returns)
    initial_weights = [1 / n_assets] * n_assets

    # The weight of any asset must be between 0 and 1 inclusive
    bounds = tuple((min_weight, max_weight) for i in range(n_assets))
    
    constraints = [
        # Sum of weights equals to 1
        {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    
    # Objective function: maximize Sharpe ratio (minimize negative Sharpe ratio)
    def objective(w):
        # Convert pandas data to numpy arrays if needed
        returns_array = mean_returns.values if hasattr(mean_returns, 'values') else mean_returns
        cov_matrix = covariance.values if hasattr(covariance, 'values') else covariance
        
        # Calculate portfolio return
        portfolio_return = np.dot(w, returns_array)
        
        # Calculate portfolio volatility
        portfolio_volatility = np.sqrt(np.dot(w, np.dot(cov_matrix, w)))
        
        # Calculate Sharpe ratio (return negative for minimization)
        if portfolio_volatility == 0:
            return -np.inf  # Avoid division by zero
        
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
        return -float(sharpe_ratio)  # Negative because we want to maximize
    
    result = minimize(objective, x0 = initial_weights, bounds = bounds, constraints = constraints)
    return result

def create_sector_constraints(tickers, sectors_data, sector_limits=None):
    """
    Create sector allocation constraints for portfolio optimization.
    
    Parameters:
    -----------
    tickers : list
        List of ticker symbols in the portfolio
    sectors_data : dict
        Dictionary with ticker symbols as keys and sector info as values
        Format: {ticker: {'sector': 'Technology', 'industry': '...', 'company_name': '...'}}
    sector_limits : dict, optional
        Dictionary specifying min/max allocation per sector
        Format: {'Technology': {'min': 0.1, 'max': 0.4}, 'Healthcare': {'min': 0.05, 'max': 0.3}}
        If None, uses default limits (5-40% per sector)
    
    Returns:
    --------
    list : List of constraint dictionaries for scipy.optimize.minimize
    """
    
    # Get unique sectors
    sectors = set()
    for ticker in tickers:
        if ticker in sectors_data:
            sectors.add(sectors_data[ticker]['sector'])
    
    # Create sector mapping: which assets belong to which sector
    sector_mapping = {}
    for sector in sectors:
        sector_mapping[sector] = []
        for i, ticker in enumerate(tickers):
            if ticker in sectors_data and sectors_data[ticker]['sector'] == sector:
                sector_mapping[sector].append(i)
    
    # Default sector limits if not provided
    if sector_limits is None:
        sector_limits = {}
        for sector in sectors:
            sector_limits[sector] = {'min': 0.00, 'max': 1.00}  # 0-100% per sector
    
    # Create constraints
    constraints = []
    
    for sector, asset_indices in sector_mapping.items():
        if sector in sector_limits:
            min_alloc = sector_limits[sector].get('min', 0.0)
            max_alloc = sector_limits[sector].get('max', 1.0)
            
            # Minimum sector allocation constraint
            if min_alloc > 0:
                constraints.append({
                    "type": "ineq",
                    "fun": lambda w, indices=asset_indices, min_val=min_alloc: 
                           np.sum([w[i] for i in indices]) - min_val
                })
            
            # Maximum sector allocation constraint
            if max_alloc < 1.0:
                constraints.append({
                    "type": "ineq", 
                    "fun": lambda w, indices=asset_indices, max_val=max_alloc:
                           max_val - np.sum([w[i] for i in indices])
                })
    
    return constraints, sector_mapping

def portfolio_minimize_with_sectors(mean_returns, covariance, desired_return, tickers, sectors_data, 
                                   min_weight=0.0, max_weight=1.0, sector_limits=None):
    """
    Portfolio variance minimization with sector allocation constraints.
    
    Parameters:
    -----------
    mean_returns : array-like
        Expected returns for each asset
    covariance : array-like
        Covariance matrix of asset returns
    desired_return : float
        Target portfolio return
    tickers : list
        List of ticker symbols
    sectors_data : dict
        Sector information for each ticker
    min_weight : float
        Minimum weight for any individual asset
    max_weight : float  
        Maximum weight for any individual asset
    sector_limits : dict, optional
        Sector allocation limits. If None, uses default 5-40% per sector
    
    Returns:
    --------
    result : OptimizeResult
        Optimization result from scipy.optimize.minimize
    """
    
    # Initial guess with equal weights
    n_assets = len(mean_returns)
    initial_weights = [1 / n_assets] * n_assets

    # Individual asset bounds
    bounds = tuple((min_weight, max_weight) for i in range(n_assets))
    
    # Basic constraints
    constraints = [
        # Sum of weights equals to 1
        {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
        # Target return constraint
        {"type": "eq", "fun": lambda w: np.dot(w, mean_returns.values if hasattr(mean_returns, 'values') else mean_returns) - desired_return}
    ]
    
    # Add sector constraints
    sector_constraints, sector_mapping = create_sector_constraints(tickers, sectors_data, sector_limits)
    constraints.extend(sector_constraints)
    
    # Objective function: minimize portfolio variance
    def objective(w):
        cov_matrix = covariance.values if hasattr(covariance, 'values') else covariance
        portfolio_variance = np.dot(w, np.dot(cov_matrix, w))
        return float(portfolio_variance)
    
    result = minimize(objective, x0=initial_weights, bounds=bounds, constraints=constraints, 
                     method='SLSQP', options={"ftol": 1e-10})
    
    # Add sector mapping to result for analysis
    result.sector_mapping = sector_mapping
    return result

def portfolio_maximize_sharpe_with_sectors(mean_returns, covariance, risk_free_rate, tickers, sectors_data,
                                          min_weight=0.0, max_weight=1.0, sector_limits=None):
    """
    Sharpe ratio maximization with sector allocation constraints.
    
    Parameters:
    -----------
    mean_returns : array-like
        Expected returns for each asset
    covariance : array-like
        Covariance matrix of asset returns  
    risk_free_rate : float
        Risk-free rate of return
    tickers : list
        List of ticker symbols
    sectors_data : dict
        Sector information for each ticker
    min_weight : float
        Minimum weight for any individual asset
    max_weight : float
        Maximum weight for any individual asset
    sector_limits : dict, optional
        Sector allocation limits. If None, uses default 5-40% per sector
    
    Returns:
    --------
    result : OptimizeResult
        Optimization result from scipy.optimize.minimize
    """
    
    # Initial guess with equal weights
    n_assets = len(mean_returns)
    initial_weights = [1 / n_assets] * n_assets

    # Individual asset bounds
    bounds = tuple((min_weight, max_weight) for i in range(n_assets))
    
    # Basic constraints
    constraints = [
        # Sum of weights equals to 1
        {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
    ]
    
    # Add sector constraints
    sector_constraints, sector_mapping = create_sector_constraints(tickers, sectors_data, sector_limits)
    constraints.extend(sector_constraints)
    
    # Objective function: maximize Sharpe ratio (minimize negative Sharpe ratio)
    def objective(w):
        returns_array = mean_returns.values if hasattr(mean_returns, 'values') else mean_returns
        cov_matrix = covariance.values if hasattr(covariance, 'values') else covariance
        
        portfolio_return = np.dot(w, returns_array)
        portfolio_volatility = np.sqrt(np.dot(w, np.dot(cov_matrix, w)))
        
        if portfolio_volatility == 0:
            return -np.inf
        
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
        return -float(sharpe_ratio)  # Negative because we want to maximize
    
    result = minimize(objective, x0=initial_weights, bounds=bounds, constraints=constraints, 
                     method='SLSQP', options={"ftol": 1e-10})
    
    # Add sector mapping to result for analysis
    result.sector_mapping = sector_mapping
    return result