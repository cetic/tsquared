import numpy as np

from tsquared import clean_samples

seed = 1

# Set seed.
np.random.seed(seed)

# Generate data.
n_samples = 100
n_noisy_samples = 20
n_features = 4

true_mean = np.array([4, -1.3, 8.7, -5.4])
true_cov = np.array([
	[1, 0.4, -0.4, 0.1],
	[0.4, 1, 0.6, -0.2],
	[-0.4, 0.6, 1, 0.02],
	[0.1, -0.2, 0.02, 1]
])
normal_samples = np.random.multivariate_normal(true_mean, true_cov, size=n_samples)

low = 100
high = 199
noise = np.random.uniform(low=low, high=high, size=(n_noisy_samples, n_features))

samples = np.concatenate((normal_samples, noise), axis=0)

# Inputs.
print("--- Inputs ---\n")

print(f"True mean vector used for generating normal samples: {true_mean}")
print(f"True covariance matrix used for generating normal samples:\n{true_cov}")
print(f"Normal samples (n={normal_samples.shape[0]}):\n{normal_samples}")
print(f"Noise (n={noise.shape[0]}):\n{noise}")
print(f"All samples, i.e. normal samples + noise (n={samples.shape[0]}):\n"
	f"{samples}")

# Clean samples.
print("\n--- Clean samples ---\n")

cleaned_samples = clean_samples(samples,
	n_iters=0,
	perc_samples=0.5,
	n_samples_per_iter=1,
	hz_test=True,
	alpha=0.05)
removed_samples = np.array(list(
	{tuple(row) for row in samples} - {tuple(row) for row in cleaned_samples}
)).reshape(-1, 4)

print(f"Cleaned samples (n={cleaned_samples.shape[0]}):\n{cleaned_samples}")
print(f"Removed samples (n={removed_samples.shape[0]}):\n{removed_samples}")
