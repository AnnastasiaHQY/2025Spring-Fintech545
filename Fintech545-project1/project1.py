#!/usr/bin/env python
# coding: utf-8

# # Fintech 545 - Project1

# ## Problem 1
# Given the dataset in **problem1.csv**, answer the following:
# 
# ### **A. Calculate the Mean, Variance, Skewness, and Kurtosis of the Data**

# In[23]:


import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
from scipy.stats import norm, t, skew, kurtosis, kstest
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_four_moments(data_array):
    n = len(data_array)
    mean_val = np.mean(data_array)
    centered_data = data_array - mean_val
    second_moment = np.mean(centered_data**2)
    variance_val = np.var(data_array, ddof=1)
    skew_val = np.mean(centered_data**3) / (second_moment**1.5)
    kurt_val = np.mean(centered_data**4) / (second_moment**2)
    excess_kurt_val = kurt_val - 3
    return mean_val, variance_val, skew_val, excess_kurt_val

data = pd.read_csv("downloads/problem1.csv")
data_array = data['X'][1:].to_numpy()
mean, variance, skew, kurt = calculate_four_moments(data_array)

print(f"Mean: {mean}")
print(f"Variance: {variance}")
print(f"Skewness: {skew}")
print(f"Kurtosis (excess): {kurt}")


# **Answer:**
# - **Mean** 0.050366360906888966
# - **Variance** 0.010314441602725719
# - **Skewness** 0.12046167291008684
# - **Kurtosis** 0.22908332509377516

# ### **B. Choose Between Normal and T-Distribution**
# Given the statistical characteristics of the dataset:
# - Would you model the data using a **Normal Distribution** or a **T-Distribution**?
# - Justify your choice based on the data properties.

# In[24]:


# B. Distribution Fitting
# Fit Normal Distribution
mu, sigma = norm.fit(data_array)
# Fit T-Distribution
df, loc, scale = t.fit(data_array)

# Perform Goodness-of-Fit Tests
ks_norm = kstest(data_array, 'norm', args=(mu, sigma))
ks_t = kstest(data_array, 't', args=(df, loc, scale))

print("\nB. Distribution Fitting and Hypothesis Testing:")
print("Normal Distribution:")
print(f"1. Parameters: μ = {mu}, σ = {sigma}")
print(f"2. KS Test Statistic: {ks_norm.statistic}")
print(f"3. KS Test p-value: {ks_norm.pvalue}")
print("\nT Distribution:")
print(f"1. Parameters: df = {df}, loc = {loc}, scale = {scale}")
print(f"2. KS Test Statistic: {ks_t.statistic}")
print(f"3. KS Test p-value: {ks_t.pvalue}")



# **Answer:**
# 
# Based on the statistical properties:
# 
# - **Normal Distribution** is appropriate if the data is symmetric with kurtosis close to 3.
# - **T-Distribution** is preferred if the data has heavier tails (kurtosis > 3) or extreme outliers.
# 
# 
# Since both distributions have very similar KS test statistics and high p-values (>> 0.05), there is no strong evidence to reject either model. However, the T-distribution has an extremely high degrees of freedom (~5.1 million), making it nearly identical to the normal distribution. This suggests that the **normal distribution is an appropriate choice** for modeling the data.
# 

# ### **C. Fit Both Distributions and Evaluate the Choice**
# 1. Fit both a **Normal Distribution** and a **T-Distribution** to the dataset.
# 2. Use statistical methods presented in class to verify which model fits better.
# 3. Compare results using appropriate goodness-of-fit tests.

# In[25]:


# C. Visualization and Detailed Analysis
# Create a figure with two subplots
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Histogram with distribution fits
sns.histplot(data_array, kde=True, bins=30, stat="density", ax=axes[0], label="Observed Data")
xmin, xmax = axes[0].get_xlim()
x = np.linspace(xmin, xmax, 100)

# Normal Distribution Fit
axes[0].plot(x, norm.pdf(x, mu, sigma), 
             label=f"Normal Fit (μ={mu:.2f}, σ={sigma:.2f})", 
             linestyle="dashed", color='red')

# T Distribution Fit
axes[0].plot(x, t.pdf(x, df, loc, scale), 
             label=f"T Fit (df={df:.2f}, loc={loc:.2f})", 
             linestyle="dotted", color='green')

axes[0].set_title("Histogram with Distribution Fits")
axes[0].legend()

# QQ Plot for Normal Distribution
stats.probplot(data_array, dist="norm", plot=axes[1])
axes[1].set_title("Q-Q Plot (Normal Distribution)")

plt.tight_layout()
plt.show()

# C. Selection
print("\nC. Distribution Selection:")
print("Selection Criteria:")
print(f"1. Normal Distribution p-value: {ks_norm.pvalue:.4f}")
print(f"2. T Distribution p-value: {ks_t.pvalue:.4f}")

# Determine the best-fitting distribution
if ks_norm.pvalue > ks_t.pvalue and ks_norm.pvalue > 0.05:
    recommendation = "Normal Distribution"
elif ks_t.pvalue > ks_norm.pvalue and ks_t.pvalue > 0.05:
    recommendation = "T Distribution"
else:
    recommendation = "Neither distribution provides a good fit"

print(f"Selected Distribution: {recommendation}")


# In[26]:


n = len(data_array)
mu_norm, std_norm = norm.fit(data_array)
# Compute the log-likelihood for the Normal model
log_likelihood_norm = np.sum(norm.logpdf(data_array, loc=mu_norm, scale=std_norm))
# Number of parameters for Normal is 2
k_norm = 2

# Calculate AIC and BIC for the Normal model
aic_norm = 2 * k_norm - 2 * log_likelihood_norm
bic_norm = k_norm * np.log(n) - 2 * log_likelihood_norm

df_t, loc_t, scale_t = t.fit(data_array)
# Compute the log-likelihood for the T-distribution model
log_likelihood_t = np.sum(t.logpdf(data_array, df=df_t, loc=loc_t, scale=scale_t))
# Number of parameters for the T-distribution is 3
k_t = 3

# Calculate AIC and BIC for the T-distribution model
aic_t = 2 * k_t - 2 * log_likelihood_t
bic_t = k_t * np.log(n) - 2 * log_likelihood_t

print("Normal Distribution Fit:")
print("----------------------------")
print(f"Parameters: mu = {mu_norm:.4f}, std = {std_norm:.4f}")
print(f"Log-Likelihood: {log_likelihood_norm:.4f}")
print(f"AIC: {aic_norm:.4f}")
print(f"BIC: {bic_norm:.4f}\n")

print("T-Distribution Fit:")
print("----------------------------")
print(f"Parameters: df = {df_t:.4f}, loc = {loc_t:.4f}, scale = {scale_t:.4f}")
print(f"Log-Likelihood: {log_likelihood_t:.4f}")
print(f"AIC: {aic_t:.4f}")
print(f"BIC: {bic_t:.4f}\n")

if aic_t < aic_norm and bic_t < bic_norm:
    print("Based on AIC and BIC, the T-distribution provides a better fit to the data.")
else:
    print("Based on AIC and BIC, the Normal distribution provides a better fit to the data.")


# **Answer**
# 
# Based on the Kolmogorov-Smirnov (KS) test results, both the **Normal** and **T-Distributions** exhibit high p-values, indicating that neither model is strongly rejected. However, the **T-Distribution has a slightly higher p-value (0.7610 vs. 0.7608)**, suggesting a marginally better fit.
# 
# Additionally, from the **histogram with fitted distributions**, we can observe that both the normal and T-distributions follow the data closely. The **Q-Q plot** shows that the empirical quantiles align well with the theoretical quantiles of a normal distribution, though slight deviations exist in the tails.
# 
# Since the T-distribution with **very high degrees of freedom (df ≈ 5.1 million)** essentially behaves like a normal distribution, the difference is negligible. However, based on the slight p-value advantage, we select the **T-Distribution** as the final model.

# ## Problem 2

# Given the data in problem2.csv

# ### A. Calculate the pairwise covariance matrix of the data.

# In[27]:


import numpy as np
import pandas as pd

# A. Calculate the pairwise covariance matrix
df = pd.read_csv('downloads/problem2.csv')
cov_matrix = df.cov()
print("A. Covariance Matrix:")
print(cov_matrix)


# **Pairwise Covariance Matrix:**
# 
# |     | x1       | x2       | x3       | x4       | x5       |
# |-----|---------|---------|---------|---------|---------|
# | x1  | 1.4705  | 1.4542  | 0.8773  | 1.9032  | 1.4444  |
# | x2  | 1.4542  | 1.2521  | 0.5395  | 1.6219  | 1.2379  |
# | x3  | 0.8773  | 0.5395  | 1.2724  | 1.1720  | 1.0919  |
# | x4  | 1.9032  | 1.6219  | 1.1720  | 1.8145  | 1.5897  |
# | x5  | 1.4444  | 1.2379  | 1.0919  | 1.5897  | 1.3962  |
# 

# ### B. Is the Matrix at least positive semi-definite? Why?

# In[28]:


# B. Check if matrix is positive semi-definite
eigenvalues = np.linalg.eigvals(cov_matrix)
is_psd = np.all(eigenvalues >= 0)

print("\nB. PSD Check:")
print(f"Is PSD: {is_psd}")
print(f"Eigenvalues: {eigenvalues}")
print(f"Minimum Eigenvalue: {min(eigenvalues)}")


# **Answer:**
# ***No***. Here are the eigenvalues: [ 6.78670573  0.83443367 -0.31024286  0.02797828 -0.13323183]. The matrix is not positive semi-definite because it has negative eigenvalues (-0.310 and -0.133), despite pairwise covariances being positive. 

# ### C. If not, find the nearest positive semi-definite matrix using Higham’s method and the near-psd method of Rebenato and Jackel.
# 
# **Answer:**
# 

# In[30]:


# C. Find the nearest positive semi-definite matrix
def higham_psd(matrix):
    sym_matrix = (matrix + matrix.T) / 2
    eigenvalues, eigenvectors = np.linalg.eigh(sym_matrix)
    eigenvalues[eigenvalues < 1e-8] = 1e-8
    return eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

def near_psd_rebonato_jackel(cov_matrix):
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    eigenvalues[eigenvalues < 0] = 1e-8
    adjusted_cov_matrix = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    original_variances = np.diag(cov_matrix)
    scaling_factors = np.sqrt(original_variances / np.diag(adjusted_cov_matrix))
    scaling_matrix = np.diag(scaling_factors)
    return scaling_matrix @ adjusted_cov_matrix @ scaling_matrix

# Apply both methods
higham_matrix = higham_psd(cov_matrix)
rebonato_matrix = near_psd_rebonato_jackel(cov_matrix)

print("Nearest PSD Matrix (Higham):")
print(higham_matrix)
print("\nNearest PSD Matrix (Rebonato-Jackel):")
print(rebonato_matrix)

# Verify PSD property for projected matrices
higham_eigenvals = np.linalg.eigvals(higham_matrix)
rebonato_eigenvals = np.linalg.eigvals(rebonato_matrix)

print("\nVerifying PSD for Projected Matrices:")
print("Higham's Method:")
print(f"Is PSD: {np.all(higham_eigenvals >= 0)}")
print(f"Min Eigenvalue: {min(higham_eigenvals):.6f}")
print("\nNear-PSD Method:")
print(f"Is PSD: {np.all(rebonato_eigenvals >= 0)}")
print(f"Min Eigenvalue: {min(rebonato_eigenvals):.6f}")
    


# ### D. Calculate the covariance matrix using only overlapping data

# In[ ]:


# D. Calculate covariance using overlapping data
overlap_cov = df.dropna().cov()
print("\nD. Overlapping Covariance Matrix:")
print(overlap_cov)


# ### E. Compare the results of the covariance matrices in C and D. Explain the differences.
# Note: the generating process is a covariance matrix with 1 on the diagonals and 0.99
# elsewhere.
# 

# In[ ]:


# E. Compare results
difference = np.abs(higham_matrix - overlap_cov)
print("\nE. Comparison:")
print(f"Maximum difference: {np.max(difference)}")
print(f"Average difference: {np.mean(difference)}")


# ## Problem 3
# Given the data in problem3.csv
# 

# ### A. Fit a multivariate normal to the data.
# 

# In[ ]:


from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from scipy.stats import norm
from numpy.linalg import inv, cholesky
import statsmodels.api as sm

# Load data
data1 = pd.read_csv('downloads/problem3.csv')
x1 = data1['x1'].values
x2 = data1['x2'].values

mu = data1.mean().values       # mu = [mu1, mu2]
Sigma = data1.cov().values
print("\nMean vector (mu):", mu)
print("Covariance matrix (Sigma):\n", Sigma)

# Extract individual parameters for clarity
mu1, mu2 = mu[0], mu[1]
sigma11 = Sigma[0, 0]
sigma12 = Sigma[0, 1]  
sigma21 = Sigma[1, 0]
sigma22 = Sigma[1, 1]


# ### B. Given that fit, what is the distribution of X2 given X1=0.6. Use the 2 methods described in class.

# In[ ]:


# Extract individual parameters for clarity
mu1, mu2 = mu[0], mu[1]
sigma11 = Sigma[0, 0]
sigma12 = Sigma[0, 1]  
sigma21 = Sigma[1, 0]
sigma22 = Sigma[1, 1]

x_value = 0.6

# Method 1: Using the conditional distribution formula
cond_mean_formula = mu2 + (sigma12 / sigma11) * (x_value - mu1)
cond_var_formula  = sigma22 - (sigma12**2 / sigma11)

print("\n--- Conditional Distribution using the Conditional Formula ---")
print("Conditional Mean =", cond_mean_formula)
print("Conditional Variance =", cond_var_formula)

# Method 2: Using the OLS method

X = sm.add_constant(data1['x1'])
y = data1['x2']
model = sm.OLS(y, X).fit()

beta = sigma12/sigma11 
alpha = mu2 - beta*mu1 
ols_cond_mean = alpha + beta * x_value

ols_estimated_var = np.var(model.resid)

print("\n--- Method 2: OLS Method ---")
print(f"Beta (slope) = {beta}")
print(f"Alpha (intercept) = {alpha}")
print(f"Predicted Mean = {ols_cond_mean}")
print(f"Estimated Variance = {ols_estimated_var}")


# ### C. Given the properties of the Cholesky Root, create a simulation that proves your distribution of X2 | X1=0.6 is correct

# In[ ]:


# Part C
def generate_conditional_samples(target_x1, mu_vector, covariance, sim_count=10000, delta=0.05):
    # Generate correlated normal variables using Cholesky
    decomp_matrix = cholesky(covariance)
    random_vars = np.random.standard_normal((sim_count, 2))
    generated_data = random_vars @ decomp_matrix.T + mu_vector
    
    # Extract conditional values using boolean indexing
    mask = (generated_data[:, 0] >= target_x1 - delta) & (generated_data[:, 0] <= target_x1 + delta)
    conditional_vals = generated_data[mask, 1]
    
    if conditional_vals.size == 0:
        raise ValueError(f"Unable to find samples near X1 = {target_x1} with window {delta}")
    
    return conditional_vals

# Run simulation and compute statistics
try:
    target_value = 0.6
    conditional_samples = generate_conditional_samples(target_value, mu, Sigma, sim_count=100000000)
    
    # Compute descriptive statistics
    empirical_mu = np.mean(conditional_samples)
    empirical_sigma2 = np.var(conditional_samples)
    
    print("\nSimulation Statistics:")
    print(f"Conditional Mean (X2|X1={target_value}): {empirical_mu}")
    print(f"Conditional Variance (X2|X1={target_value}): {empirical_sigma2}")
    
except ValueError as e:
    print(f"Error in simulation: {e}")

# Visualize distribution
sample_size = 10000
decomp_matrix = cholesky(Sigma)
base_samples = np.random.standard_normal((sample_size, 2))
full_samples = base_samples @ decomp_matrix.T + mu

# Create distribution plot
plt.figure(figsize=(10, 6))
plt.hist(full_samples[:, 0], bins=50, color='navy', alpha=0.6, 
         edgecolor='white', density=False)
plt.axvline(x=target_value, color='crimson', linestyle='--', 
            label=f'X1 = {target_value}')
plt.title('X1 Distribution from Simulated Data')
plt.xlabel('X1')
plt.ylabel('Count')
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()


# ## Problem 4
# Given the data in problem4.csv
# ### A. Simulate an MA(1), MA(2), and MA(3) process and graph the ACF and PACF of each. What do you notice?

# In[ ]:


from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Set random seed for reproducibility
np.random.seed(123)

def simulate_ma(q, n=1000, coef=0.5):
    """
    Simulate MA(q) process
    q: order of MA process
    n: number of observations
    coef: MA coefficient (same for all lags)
    """
    ma_params = np.zeros(q + 1)
    ma_params[0] = 1  # MA(0) = 1 for identification
    ma_params[1:] = coef  # Set MA coefficients
    ma_process = ArmaProcess(ar=[1], ma=ma_params)
    simulated_data = ma_process.generate_sample(nsample=n)
    return simulated_data

def plot_acf_pacf(data, title):
    """
    Plot ACF and PACF for given time series
    data: time series data
    title: title for the plots
    """
    plt.figure(figsize=(12, 6))
    
    # Plot ACF
    plt.subplot(1, 2, 1)
    plot_acf(data, lags=20, ax=plt.gca(), title=f"{title} ACF")
    
    # Plot PACF
    plt.subplot(1, 2, 2)
    plot_pacf(data, lags=20, ax=plt.gca(), title=f"{title} PACF")
    
    plt.tight_layout()
    plt.show()

# Simulate and plot MA(1), MA(2), MA(3)
for q in [1, 2, 3]:
    ma_data = simulate_ma(q, n=1000, coef=0.5)
    plot_acf_pacf(ma_data, title=f"MA({q})")


# 1. MA(1) Process:
# - ACF: The ACF shows a significant spike at lag 1 and then quickly drops to near zero for higher lags. This is characteristic of an MA(1) process, where the correlation only persists for one lag.
# - PACF: The PACF has a single significant spike at lag 1, and the rest are within the confidence bounds, confirming the order of the moving average process.
# 2. MA(2) Process:
# - ACF: The ACF exhibits significant spikes at lags 1 and 2, after which it rapidly drops to near zero. This aligns with the expected pattern for an MA(2) process.
# - PACF: The PACF shows a more dispersed pattern, with some values outside the confidence bounds at lower lags, but generally, it does not cut off sharply.
# 3. MA(3) Process:
# - ACF: The ACF shows significant spikes at lags 1, 2, and 3, then decays rapidly to zero, which matches the theoretical expectation for an MA(3) process.
# - PACF: The PACF has multiple significant spikes at lower lags (especially around lag 3), with no clear cutoff.
# 
# Conclusion:
# ACF for an MA(q) process cuts off after q lags, meaning that correlations exist up to lag q and are zero afterward.
# PACF for an MA(q) process does not cut off but instead exhibits a trailing decay. This is because each lag is indirectly correlated through earlier terms.

# ### B. Simulate an AR(1), AR(2), and AR(3) process and graph the ACF and PACF of each. What do you notice?

# In[31]:


import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Set random seed for reproducibility
np.random.seed(123)

def simulate_ar(p, n=1000, coef=0.5):
    """
    Simulate AR(p) process
    p: order of AR process
    n: number of observations
    coef: AR coefficient (same for all lags)
    """
    ar_params = np.zeros(p + 1)
    ar_params[0] = 1  # AR(0) = 1 for identification
    ar_params[1:] = coef  # Set AR coefficients
    ar_process = ArmaProcess(ar=ar_params, ma=[1])
    simulated_data = ar_process.generate_sample(nsample=n)
    return simulated_data

def plot_acf_pacf(data, title):
    """
    Plot ACF and PACF for given time series
    data: time series data
    title: title for the plots
    """
    plt.figure(figsize=(12, 6))
    
    # Plot ACF
    plt.subplot(1, 2, 1)
    plot_acf(data, lags=20, ax=plt.gca(), title=f"{title} ACF")
    
    # Plot PACF
    plt.subplot(1, 2, 2)
    plot_pacf(data, lags=20, ax=plt.gca(), title=f"{title} PACF")
    
    plt.tight_layout()
    plt.show()

# Simulate and plot AR(1), AR(2), AR(3)
for p in [1, 2, 3]:
    ar_data = simulate_ar(p, n=1000, coef=0.5)
    plot_acf_pacf(ar_data, title=f"AR({p})")


# ### C. Examine the data in problem4.csv. What AR/MA process would you use to model thedata? Why?

# In[32]:


data4 = pd.read_csv('downloads/problem4.csv')
y = data4['y']

# Plot the time series, ACF, and PACF for the actual data
plt.figure(figsize=(12,4))
plt.plot(y)
plt.title('Time Series Data from problem4.csv')
plt.show()

plot_acf(y, lags=30)
plt.title('ACF of y')
plt.show()

plot_pacf(y, lags=30)
plt.title('PACF of y')
plt.show()


# 1. Time Series Plot: The series appears to be stationary with fluctuations around a mean level, without obvious trends or seasonality.
# 2. ACF: The ACF shows a gradual decay, indicating a potential moving average (MA) component.
# 3. PACF: The PACF cuts off sharply after lag 2, suggesting an autoregressive (AR) process with an order around 2.
# 
# 
# Conclusion：
# The PACF behavior (significant lags at 1 and 2, followed by a cutoff) suggests an AR(2) process. Since both AR and MA components are present, an ARMA(2,q) model may be appropriate.

# ### D. Fit the model of your choice in C along with other AR/MA models. Compare the AICc ofeach. What is the best fit?

# In[ ]:


def compute_aicc(model_result):
    """
    Compute the AICc (AIC with correction for small sample sizes)
    k: number of parameters (including constant)
    n: number of observations
    """
    aic = model_result.aic
    n = model_result.nobs
    k = model_result.df_model + 1  # df_model does not include the intercept, so add 1
    return aic + (2 * k * (k + 1)) / (n - k - 1)

# List of candidate (p, d, q) models; d=0 since we assume stationarity.
candidate_orders = [(1,0,0), (2,0,0), (3,0,0),    # AR models
                    (0,0,1), (0,0,2), (0,0,3),    # MA models
                    (2,0,2), (3,0,3),]            # ARMA model

results = {}
print("Fitting candidate models and their AICc:")
for order in candidate_orders:
    try:
        model = SARIMAX(y, order=order,
                        enforce_stationarity=False,
                        enforce_invertibility=False)
        res = model.fit(disp=False)
        aicc = compute_aicc(res)
        results[order] = aicc
        print("Order {}: AICc = {:.2f}".format(order, aicc))
    except Exception as e:
        print("Order {} failed: {}".format(order, e))

# Identify the best model (lowest AICc)
best_order = min(results, key=results.get)
print("\nBest model order based on AICc is:", best_order)


# Based on AICc, the best model is **(2, 0, 2) → ARMA(2,2)**.

# ## Problem 5
# Given the stock return data in DailyReturns.csv.
# ### A. Create a routine for calculating an exponentially weighted covariance matrix. If you have a package that calculates it for you, verify it produces the expected results from the testdata folder.

# In[ ]:


import pandas as pd
import numpy as np

# Load the data
data = pd.read_csv('downloads/DailyReturn.csv', index_col='Date')

# Define the lambda (decay factor)
lambda_value = 0.97  # You can adjust this value

# Calculate the exponentially weighted covariance matrix
ew_cov_matrix = data.ewm(alpha=1 - lambda_value).cov()

# Display the result
print(ew_cov_matrix)


# ### B. Vary λ. Use PCA and plot the cumulative variance explained of λ in (0,1) by each eigenvalue for each λ chosen.

# In[ ]:


from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Define a range of lambda values
lambda_values = np.linspace(0.01, 0.99, 50)  # 50 values between 0.01 and 0.99
cumulative_variance_explained = []

for lambda_value in lambda_values:
    # Calculate the exponentially weighted covariance matrix
    ew_cov_matrix = data.ewm(alpha=1 - lambda_value).cov()
    ew_cov_matrix = ew_cov_matrix.groupby(level=1).mean()  # Reshape into 2D
    
    # Perform PCA
    pca = PCA()
    pca.fit(ew_cov_matrix)
    
    # Calculate cumulative variance explained
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    cumulative_variance_explained.append(cumulative_variance)

# Plot the results
plt.figure(figsize=(10, 6))
for i, lambda_value in enumerate(lambda_values):
    plt.plot(cumulative_variance_explained[i], label=f'λ = {lambda_value:.2f}')

plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Variance Explained')
plt.title('Cumulative Variance Explained by PCA for Different λ Values')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid()
plt.show()


# ### C. What does this tell us about the values of λ and the effect it has on the covariancematrix?

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Load the dataset
file_path = "downloads/DailyReturn.csv"
data = pd.read_csv(file_path)

def exponential_weights(n, lambda_value):
    weights = (1 - lambda_value) * (lambda_value ** np.arange(n - 1, -1, -1))
    weights /= weights.sum()  # Normalize weights
    return weights

def exponentially_weighted_covariance_matrix(data, lambda_value):
    """
    Computes the exponentially weighted covariance matrix of the given data.
    
    Parameters:
    data (pd.DataFrame): DataFrame containing asset returns (excluding the date column).
    lambda_value (float): Decay factor (0 < lambda_value < 1).
    
    Returns:
    np.ndarray: Exponentially weighted covariance matrix.
    """
    returns = data.iloc[:, 1:].values  # Exclude the Date column
    T, N = returns.shape  # T: time periods, N: number of assets

    weights = exponential_weights(T, lambda_value)
    
    # Use unweighted mean to match your friend's approach
    mean_returns = np.mean(returns, axis=0)
    centered_returns = returns - mean_returns

    ew_cov_matrix = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            ew_cov_matrix[i, j] = np.sum(weights * centered_returns[:, i] * centered_returns[:, j])

    return ew_cov_matrix

# Define a range of lambda values
lambda_values = np.linspace(0.1, 0.99, 10)

# Store cumulative variance explained for each lambda
cumulative_variance_results = {}

for lambda_val in lambda_values:
    ew_cov_matrix = exponentially_weighted_covariance_matrix(data, lambda_val)
    
    # Perform PCA on the covariance matrix
    pca = PCA()
    pca.fit(ew_cov_matrix)
    
    # Compute cumulative explained variance
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    cumulative_variance_results[lambda_val] = cumulative_variance

# Print Exponentially Weighted Covariance Matrix for lambda = 0.97
lambda_97 = 0.97
ew_cov_matrix_97 = exponentially_weighted_covariance_matrix(data, lambda_97)
print("Exponentially Weighted Covariance Matrix (λ=0.97):\n", ew_cov_matrix_97)

# Plot cumulative variance explained for each lambda
plt.figure(figsize=(10, 6))
for lambda_val, cum_var in cumulative_variance_results.items():
    plt.plot(range(1, len(cum_var) + 1), cum_var, label=f"\u03bb={lambda_val:.2f}")

plt.xlabel("Number of Principal Components")
plt.ylabel("Cumulative Variance Explained")
plt.title("Cumulative Variance Explained by PCA for Different \u03bb Values")
plt.legend()
plt.grid(True)
plt.show()

# Perform PCA on the covariance matrix
pca_97 = PCA()
pca_97.fit(ew_cov_matrix_97)

# Compute cumulative explained variance
cumulative_variance_97 = np.cumsum(pca_97.explained_variance_ratio_)

# Plot cumulative variance explained for λ = 0.97
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(cumulative_variance_97) + 1), cumulative_variance_97, label="\u03bb=0.97", color="red")

plt.xlabel("Number of Principal Components")
plt.ylabel("Cumulative Variance Explained")
plt.title("Cumulative Variance Explained by PCA for \u03bb = 0.97")
plt.legend()
plt.grid(True)
plt.show()


# A higher λ (e.g., 0.97) places more weight on recent data while still incorporating past observations, leading to a smoother and more stable covariance matrix. This results in lower variability in covariance estimates, making the model less sensitive to short-term fluctuations. In contrast, a lower λ would emphasize recent changes more strongly, causing the covariance structure to shift more dynamically over time.
# 
# From the PCA results, we see that for high λ, only a few principal components explain most of the variance, indicating a well-structured covariance matrix with strong correlations among assets. Lower λ values would distribute variance across more components, making the model more reactive to recent market shifts. Choosing λ depends on the balance between stability and responsiveness—higher values are useful for long-term trends, while lower values are better for short-term adaptations.

# ## Problem 6
#     Implement a multivariate normal simulation using the Cholesky root of a covariance matrix. 
#     Implement a multivariate normal simulation using PCA with percent explained as an input. 
#     Using the covariance matrix found in problem6.csv
# ### A. Simulate 10,000 draws using the Cholesky Root method.

# In[18]:


import numpy as np
import pandas as pd
from scipy.linalg import cholesky, eigh

# **Step 1: Adjust covariance matrix to be positive definite**
def make_positive_definite(matrix, epsilon=1e-10):
    """
    Adjusts a covariance matrix to be the nearest positive definite matrix.
    
    - **Only** replaces negative eigenvalues with `epsilon`
    - Keeps positive eigenvalues unchanged
    """
    eigenvalues, eigenvectors = eigh(matrix)
    eigenvalues[eigenvalues < 0] = epsilon  # Set only negative eigenvalues to epsilon
    return eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T  # Reconstruct matrix

# **Step 2: Load the covariance matrix**
file_path = "downloads/problem6.csv"
cov_matrix_df = pd.read_csv(file_path)  # Load without setting index
cov_matrix = cov_matrix_df.values  # Convert to NumPy array

# Ensure the matrix is square
assert cov_matrix.shape[0] == cov_matrix.shape[1], "Covariance matrix must be square!"

# **Step 3: Check if the matrix is positive definite**
eigenvalues = np.linalg.eigvals(cov_matrix)
is_positive_definite = np.all(eigenvalues > 0)

# Adjust if not positive definite
if not is_positive_definite:
    print("Matrix is not positive definite. Adjusting...")
    cov_matrix = make_positive_definite(cov_matrix)

# Verify the adjusted matrix
adjusted_eigenvalues = np.linalg.eigvals(cov_matrix)
is_adjusted_psd = np.all(adjusted_eigenvalues > 0)  # Should be True after adjustment

# **Step 4: Perform Cholesky decomposition**
L = cholesky(cov_matrix, lower=True)

# **Step 5: Simulate 10,000 draws using Cholesky method**
n_samples = 10000
np.random.seed(42)
Z = np.random.randn(n_samples, cov_matrix.shape[0])  # Generate standard normal draws
X_cholesky = Z @ L.T  # Transform using Cholesky root

# **Step 6: Output results**
print("\n=== Part A: Cholesky Simulation ===")
print("Adjusted Matrix Eigenvalues (first 10):", adjusted_eigenvalues[:10])  # Show first 10 eigenvalues
print("Is Adjusted Matrix SPD?", is_adjusted_psd)  # Should print True
print("Cholesky Simulation Shape:", X_cholesky.shape)  # Should print (10000, 500)


# ### B. Simulate 10,000 draws using PCA with 75% variance

# In[19]:


# **Part B: Simulate 10,000 draws using PCA (retain 75% variance) using manual eigen decomposition**

# **Step 1: Perform PCA using Eigen Decomposition**
eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)  # Compute eigenvalues & eigenvectors

# **Step 2: Sort eigenvalues and eigenvectors in descending order**
sorted_indices = np.argsort(eigenvalues)[::-1]  # Indices to sort from largest to smallest
eigenvalues = eigenvalues[sorted_indices]  # Sorted eigenvalues
eigenvectors = eigenvectors[:, sorted_indices]  # Corresponding eigenvectors

# **Step 3: Compute cumulative variance explained**
cumulative_variance = np.cumsum(eigenvalues) / np.sum(eigenvalues)

# **Step 4: Determine the number of components needed to explain 75% variance**
n_components = np.argmax(cumulative_variance >= 0.75) + 1

# **Step 5: Select the top `n_components` eigenvalues and eigenvectors**
eigenvalues_k = eigenvalues[:n_components]  # Selected eigenvalues
eigenvectors_k = eigenvectors[:, :n_components]  # Corresponding eigenvectors

# **Step 6: Generate 10,000 standard normal samples**
Z_pca = np.random.normal(size=(n_components, n_samples))  # Standard normal draws

# **Step 7: Transform the samples using the selected principal components**
X_pca = np.dot(eigenvectors_k, np.sqrt(np.diag(eigenvalues_k)) @ Z_pca).T

# **Step 8: Output results**
print("\n=== Part B: PCA Simulation (Manual Eigen Decomposition) ===")
print(f"Number of Principal Components Used: {n_components}")
print("PCA Simulation Shape:", X_pca.shape)  # Should print (10000, 500) if full dimensionality is retained


# ### C. Take the covariance of each simulation. Compare the Frobenius norm of these matrices to the original covariance matrix. What do you notice?

# In[20]:


# **Part C: Compare the Frobenius Norm of the Simulated Covariance Matrices**

# **Step 1: Compute covariance matrices of the simulated samples**
cov_cholesky = np.cov(X_cholesky, rowvar=False)  # Covariance from Cholesky-based simulation
cov_pca = np.cov(X_pca, rowvar=False)  # Covariance from PCA-based simulation

# **Step 2: Compute Frobenius norms**
frobenius_cholesky = np.linalg.norm(cov_cholesky - cov_matrix, ord="fro")  # Cholesky norm difference
frobenius_pca = np.linalg.norm(cov_pca - cov_matrix, ord="fro")  # PCA norm difference

# **Step 3: Output Frobenius norm comparison**
print("\n=== Part C: Frobenius Norm Comparison ===")
print(f"Frobenius Norm (Cholesky): {frobenius_cholesky:.6f}")
print(f"Frobenius Norm (PCA): {frobenius_pca:.6f}")


# ### D. Compare the cumulative variance explained by each eigenvalue of the 2 simulated covariance matrices along with the input matrix. What do you notice?

# In[21]:


import matplotlib.pyplot as plt

# **Step 1: Compute cumulative variance for each method**
original_cumulative_variance = np.cumsum(np.linalg.eigvals(cov_matrix)) / np.sum(np.linalg.eigvals(cov_matrix))
cholesky_cumulative_variance = np.cumsum(np.linalg.eigvals(cov_cholesky)) / np.sum(np.linalg.eigvals(cov_cholesky))
pca_cumulative_variance = np.cumsum(np.linalg.eigvals(cov_pca)) / np.sum(np.linalg.eigvals(cov_pca))

# **Step 2: Plot cumulative variance**
plt.figure(figsize=(10, 6))
plt.plot(original_cumulative_variance, label="Original Covariance Matrix", linestyle="dashed")
plt.plot(cholesky_cumulative_variance, label="Cholesky Simulated Covariance")
plt.plot(pca_cumulative_variance, label="PCA Simulated Covariance")
plt.xlabel("Eigenvalue Index")
plt.ylabel("Cumulative Variance Explained")
plt.legend()
plt.grid()
plt.title("Cumulative Variance Explained Comparison")
plt.show()



# - Cholesky closely matches the original covariance, preserving most of the structure.
# - PCA deviates after a certain number of components, reflecting its loss of variance due to dimensionality reduction.

# ### E. Compare the time it took to run both simulations.

# In[17]:


import time

# **Step 1: Time Cholesky Simulation**
start_time_cholesky = time.time()
X_cholesky_timed = np.dot(L, np.random.normal(size=(cov_matrix.shape[0], n_samples))).T
time_cholesky = time.time() - start_time_cholesky

# **Step 2: Time PCA Simulation**
start_time_pca = time.time()
Z_pca = np.random.normal(size=(n_components, n_samples))
X_pca_timed = np.dot(top_eigenvectors, np.sqrt(np.diag(top_eigenvalues)) @ Z_pca).T
time_pca = time.time() - start_time_pca

# **Step 3: Output results**
print("\n=== Part E: Computation Time Comparison ===")
print(f"Cholesky Simulation Time: {time_cholesky:.6f} seconds")
print(f"PCA Simulation Time: {time_pca:.6f} seconds")



# - **PCA is significantly faster than Cholesky**, taking only **~14% of the time** required for Cholesky.
# - This difference is expected because **PCA reduces the number of dimensions**, while Cholesky operates on the **full covariance matrix**.
# - **Cholesky requires full matrix factorization**, which is computationally more expensive than the **eigenvalue-based transformation in PCA**.

# ### F. Discuss the tradeoffs between the two methods.

# Both **Cholesky decomposition** and **PCA-based simulation** are useful methods for generating multivariate normal samples, but they have different strengths and weaknesses.
# 
# #### **1️⃣ Cholesky Simulation: High Accuracy, Higher Cost**
# The **Cholesky method** maintains the **exact covariance structure** of the original data. Since it factorizes the covariance matrix directly, it produces simulations that **perfectly preserve correlations** between variables. However, it **requires more computational effort**, especially when dealing with very large datasets, and it **only works if the covariance matrix is positive semi-definite (PSD)**. 
# 
# ##### ✅ **Pros**
# - **Exact covariance structure is preserved**
# - **More reliable for financial modeling and risk analysis**
# - **No variance loss, all information is retained**
# 
# ##### ❌ **Cons**
# - **Computationally expensive for large matrices**
# - **Requires a PSD matrix, otherwise adjustments are needed**
# - **Not ideal if dimensionality reduction is needed**
# 
# #### **2️⃣ PCA Simulation: Faster, but Less Accurate**
# The **PCA method** reduces the number of dimensions while keeping the **most important variance**. This makes it **faster and more memory-efficient**, but it also **loses some covariance structure** in the process. PCA is particularly useful when dealing with **high-dimensional data** where keeping all variables is unnecessary. However, the **simulated covariance matrix is only an approximation** of the original.
# 
# ##### ✅ **Pros**
# - **Faster computation, especially for high-dimensional datasets**
# - **Does not require a PSD matrix (handles ill-conditioned data well)**
# - **Can reduce noise by removing less important components**
# 
# ##### ❌ **Cons**
# - **Loses some covariance structure due to dimensionality reduction**
# - **Not ideal if exact correlations must be maintained**
# - **Requires choosing a variance threshold (e.g., 75%), which affects accuracy**
# 
# ---
# 
# #### **3️⃣ Side-by-Side Comparison**
# | Feature | **Cholesky Simulation** | **PCA-Based Simulation** |
# |---------|------------------|------------------|
# | **Speed** | ❌ Slower for large matrices | ✅ Faster due to dimensionality reduction |
# | **Accuracy** | ✅ More accurate, exact covariance | ❌ Less accurate, variance loss occurs |
# | **Handles Non-PSD Matrix?** | ❌ No, requires adjustment | ✅ Yes, works naturally |
# | **Dimensionality Reduction** | ❌ No, keeps all variables | ✅ Yes, reduces dataset size |
# | **Best for** | When accuracy matters most (finance, risk) | When speed and efficiency matter (ML, high-dim data) |
# 
# ---
# 
# #### **4️⃣ Final Thoughts**
# - **Use Cholesky** when we need **precise covariance replication**, and computational cost is not a major issue.
# - **Use PCA** when **reducing dimensionality** is important, even if it means sacrificing some accuracy.
# - If **storage and speed are the priority**, PCA is a better choice.
# - If **financial modeling or risk assessment** is the goal, Cholesky is preferred.
# 
# ---
# 

# In[ ]:




