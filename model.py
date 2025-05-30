import numpy as np
import pandas as pd
from scipy.optimize import curve_fit, minimize
from scipy.stats import ks_2samp, wasserstein_distance
from data import inflation_datasets
import seaborn as sns # type: ignore
import matplotlib.pyplot as plt


# Returns inflation data for the specified dataset.
def get_inflation_data(dataset_id):
    if dataset_id not in inflation_datasets:
        raise ValueError(f"Dataset with id '{dataset_id}' not found.")
    
    inflation_data = inflation_datasets[dataset_id]
    df = pd.DataFrame(inflation_data)
    df["Date"] = pd.to_datetime(df["Date"], format="%m.%Y")
    return df


# Calculating consumption activity
def consumer_acticity(x):
    """Calculates euclidian norm ||s,s,sr|| in value space"""
    return np.linalg.norm(x)


# Models definition
# Classical Maxwell-Bolzman distribution
def clas_maxwell_boltzmann(x, B, beta):
    x_adj = x
    epsilon = 1e-10
    x_adj = np.clip(x_adj, epsilon, None)
    return B * (x_adj**2) * np.exp(-beta * x_adj**2)


# Modified Maxwell-Bolzman distribution
def mod_maxwell_boltzmann(x, B, beta, x0, alpha):
    x_adj = x - x0
    epsilon = 1e-10
    x_adj = np.clip(x_adj, epsilon, None)
    return B * (x_adj**alpha) * np.exp(-beta * x_adj**2)


# Computes the negative log-likelihood for a given set of parameters (params) and observed data (x_data, y_data)
def neg_log_likelihood(params, x_data, y_data):
    B, beta, x0, alpha = params
    y_model = mod_maxwell_boltzmann(x_data, B, beta, x0, alpha)
    y_model = np.clip(y_model, 1e-10, None) 
    log_likelihood = -np.sum(y_data * np.log(y_model / np.sum(y_model)))
    return log_likelihood


# Normalizes the fitted curve (y_fit) to match the scale of the observed data (y_data).
def normalize_curve(y_fit, y_data):
    y_fit *= max(y_data) / max(y_fit)
    return y_fit


# Computes quality metrics for comparing empirical data and the model
def compute_metrics(x_empirical, y_empirical, y_model):
    """
    Parameters:
    x_empirical (array-like): Empirical values ​​on the X-axis (e.g., histogram bin edges).
    y_empirical (array-like): Empirical values ​​on the Y-axis (e.g., histogram normalized frequencies).
    y_model (array-like): Values ​​predicted by the model, corresponding to the same X-bins.

    Returns:
    dict: A dictionary with the computed metrics: Root Mean Squared Error, Kolmogorov-Smirnov statistic,
    p-value for the test Kolmogorov-Smirnov, Wasserstein distance.
    """
    rmse = np.sqrt(np.mean((y_empirical - y_model) ** 2))
    ks_stat, ks_p = ks_2samp(x_empirical, y_model)
    wass = wasserstein_distance(x_empirical, y_model)
    return {'rmse': rmse, 'ks_stat': ks_stat, 'ks_p': ks_p, 'wass': wass}


# Calculation of theoretical distribution
def evaluate_fit(x_data, y_data, model_type, params):
    if model_type=='modified':
        y_fit = mod_maxwell_boltzmann(x_data, *params)
        y_fit = normalize_curve(y_fit, y_data)
    elif model_type=='classical':
        y_fit = clas_maxwell_boltzmann(x_data, *params)
        y_fit = normalize_curve(y_fit, y_data)
    ks_stat, ks_pvalue = ks_2samp(y_data, y_fit)
    wasserstein = wasserstein_distance(y_data, y_fit)
    return ks_stat, ks_pvalue, wasserstein


# Fits a model to data using one of two methods: MLE or MLS
# It returns the optimal parameters for the model based on the chosen method.
def fit_methods(x_data, y_data, method):
    initial_params = [0.1, 0.1, 0, 2]
    bounds = [(1e-6, None), (1e-6, None), (None, None), (0, None)]
    if method == 'mle':
        result = minimize(neg_log_likelihood, initial_params, args=(x_data, y_data),
                          method='L-BFGS-B', bounds=bounds)
        params = result.x
    elif method == 'mls':
        try:
            params, _ = curve_fit(mod_maxwell_boltzmann, x_data, y_data, p0=initial_params, bounds=(
                [1e-6, 1e-6, -np.inf, 0], [np.inf, np.inf, np.inf, np.inf]), maxfev=5000)
        except RuntimeError:
            params = [np.nan, np.nan, np.nan, np.nan] 
    return params


# Computes RMSE
def compute_rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

# Process data for a single date
def calc_distr_params(individual, date, model_type):
    """
    Parameters:
    individual (pd.DataFrame): The original data with a multi-index ['date', 'client'].
    date (str): The date to process.
    model_type (str): The model type ('modified' or 'classical').

    Returns:
    pd.DataFrame: A DataFrame with the results for the given date.
    """
    try:
        data = individual.loc[date]
    except KeyError:
        print(f"Date {date} is missing.")
        return pd.DataFrame()

    # Calculating consumption activity
    b = data.apply(lambda x: consumer_acticity(x), axis=1)

    if b.empty or b.isna().all():
        print(f"The data is empty or contains only NaN for {date}")
        return pd.DataFrame()

    try:
        b = b.value_counts(bins=30, normalize=True)
    except ValueError as e:
        print(f"Error creating histogram for {date}: {e}")
        return pd.DataFrame()

    # Extract x_data and y_data
    x_data = np.array([(edge.left + edge.right) / 2 for edge in b.index])
    y_data = np.array(b.values)
    y_data = y_data / np.sum(y_data)

    results = []

    # Maximum Likelihood Estimation
    try:
        mle_params = fit_methods(x_data, y_data, method='mle')
        if not np.isnan(mle_params).any():
            mle_ks, mle_p, mle_wd = evaluate_fit(x_data, y_data, model_type, mle_params)
            y_pred_mle = mod_maxwell_boltzmann(x_data, *mle_params)
            mle_rmse = compute_rmse(y_data, y_pred_mle)
            results.append([date, *mle_params, 'mle', mle_ks, mle_p, mle_wd, mle_rmse])
    except Exception as e:
        print(f"Error while fitting MLE for {date}: {e}")

    # Ordinary Least Squares (MLS)
    try:
        mls_params = fit_methods(x_data, y_data, method='mls')
        if not np.isnan(mls_params).any():
            mls_ks, mls_p, mls_wd = evaluate_fit(x_data, y_data, model_type, mls_params)
            y_pred_mls = mod_maxwell_boltzmann(x_data, *mls_params)
            mls_rmse = compute_rmse(y_data, y_pred_mls)
            results.append([date, *mls_params, 'mls', mls_ks, mls_p, mls_wd, mls_rmse])
    except Exception as e:
        print(f"Error while fitting MLS for {date}: {e}")

    # Creating DataFrame with results
    columns = ['date', 'B', 'beta', 'x0', 'alpha', 'method', 'ks_statistic', 'p_value', 'wasserstein_distance', 'rmse']
    results_df = pd.DataFrame(results, columns=columns)
    return results_df


# Creates graphs for each metric
def plot_metric_distributions(results_df):
    required_columns = {'p_value', 'rmse', 'wasserstein_distance'}
    if not required_columns.issubset(results_df.columns):
        raise ValueError(f"DataFrame must contain the following columns: {required_columns}")
    
    grouped_p_value = results_df.groupby('method')['p_value']
    grouped_rmse = results_df.groupby('method')['rmse']
    grouped_wasserstein = results_df.groupby('method')['wasserstein_distance']
    
    metrics = {
        'p_value': grouped_p_value,
        'rmse': grouped_rmse,
        'wasserstein_distance': grouped_wasserstein
    }
    
    for metric_name, grouped_data in metrics.items():
        fig, axes = plt.subplots(len(grouped_data), 1, figsize=(10, 6 * len(grouped_data)), sharex=True)
        
        if len(grouped_data) == 1: 
            axes = [axes]
        
        for ax, (method, data) in zip(axes, grouped_data):
            mean_value = data.mean()
            std_dev = data.std()
            
            sns.histplot(data, kde=True, bins=30, color='blue', alpha=0.6, ax=ax, label=f'{method} distribution')
            
            ax.axvline(mean_value, color='red', linestyle='--', linewidth=2, label=f'Mean = {mean_value:.4f}')
            ax.tick_params(axis='both', which='major', labelsize=18)
            ax.fill_between(
                np.linspace(data.min(), data.max(), 100),
                0,
                ax.get_ylim()[1],
                where=(np.linspace(data.min(), data.max(), 100) >= mean_value - std_dev) &
                      (np.linspace(data.min(), data.max(), 100) <= mean_value + std_dev),
                color='green', alpha=0.2, label=f'Mean ± StdDev'
            )
            
            ax.set_title(f'Distribution of {metric_name.replace("_", " ").capitalize()} ({method.capitalize()})', fontsize=20)
            ax.set_xlabel(metric_name.replace("_", " ").capitalize(), fontsize=20)
            ax.set_ylabel('Frequency', fontsize=20)
            ax.legend(fontsize=20)
            ax.grid(True, linestyle='--', alpha=0.6)
            
            
            if metric_name == 'rmse':
                ax.set_xlim(0, 0.15)
        
        plt.tight_layout()
        plt.show()


# Analyses the results
def analyze_results(results_df):
    summary = results_df.groupby('method').agg({
        'ks_statistic': ['mean', 'std'],
        'p_value': ['mean', 'std'],
        'wasserstein_distance': ['mean', 'std'],
        'rmse': ['mean', 'std']
    }).reset_index()

    summary.columns = [
        'method',
        'ks_statistic_mean', 'ks_statistic_std',
        'p_value_mean', 'p_value_std',
        'wasserstein_distance_mean', 'wasserstein_distance_std',
        'rmse_mean', 'rmse_std'
    ]
    summary['ks_statistic_variation'] = summary['ks_statistic_std'] / summary['ks_statistic_mean']
    summary['p_value_variation'] = summary['p_value_std'] / summary['p_value_mean']
    summary['wasserstein_distance_variation'] = summary['wasserstein_distance_std'] / summary['wasserstein_distance_mean']
    summary['rmse_variation'] = summary['rmse_std'] / summary['rmse_mean']
    print(summary)
    return summary