import matplotlib.pyplot as plt
import numpy as np

from tsquared import HotellingT2

seed = 1

# Set seed.
np.random.seed(seed)

# Generate data.
n_train = 1000
n_test = 100

true_mean = np.array([4, -1.3, 8.7, -5.4])
true_cov = np.array([
	[1, 0.4, -0.4, 0.1],
	[0.4, 1, 0.6, -0.2],
	[-0.4, 0.6, 1, 0.02],
	[0.1, -0.2, 0.02, 1]
])

train = np.random.multivariate_normal(true_mean, true_cov, size=n_train)
test = np.random.multivariate_normal(true_mean, true_cov, size=n_test)

# Inputs.
print("--- Inputs ---\n")

print(f"True mean vector: {true_mean}")
print(f"True covariance matrix:\n{true_cov}")

# Fit and print some attributes.
print("\n--- Hotelling's T-squared fitting on the training set---\n")

hotelling = HotellingT2()
hotelling.fit(train)

print(f"Computed mean vector: {hotelling.mean_}")
print(f"Computed covariance matrix:\n{hotelling.cov_}")
print(f"Hotelling's T-squared UCL: {hotelling.ucl(test)}")

# Compute Hotelling's T-squared score for each sample in the test set.
print("\n--- Hotelling's T-squared scores on the test set ---\n")

ucl_baseline = 0.1
t2_scores = hotelling.score_samples(test)
scaled_t2_scores = hotelling.scaled_score_samples(test,
	ucl_baseline=ucl_baseline)

print(f"Hotelling's T-squared score for each sample:\n{t2_scores}")
print(f"Scaled Hotelling's T-squared score for each sample:"
	f"\n{scaled_t2_scores}")

# Classify each sample.
print("\n--- Outlier detection ---\n")

preds = hotelling.predict(test)
outliers = test[preds == -1]

print(f"Prediction for each sample:\n{preds}")
print(f"Detected outliers:\n{outliers}")

# Compute Hotelling's T-squared score for the entire test set.
print("\n--- Hotelling's T-squared score on the test set ---\n")

t2_score = hotelling.score(test)
ucl = n_train / (n_train + 1) * hotelling.ucl_indep_

print(f"Hotelling's T-squared score for the entire test set: {t2_score}")
print(f"Do the training set and the test set come from the same "
	f"distribution? {t2_score <= ucl}")

# Plot scaled Hotelling's T-squared scores and the UCL.
fig, ax = plt.subplots(figsize=(14, 8))

plt.scatter(range(scaled_t2_scores.size), scaled_t2_scores)
ucl_line = plt.axhline(y=ucl_baseline, color='r', linestyle='-')

ax.set_title('Scaled Hotelling\'s T2 scores')
ax.set_xlabel('Index')
ax.set_ylabel('Scaled Hotelling\'s T2 score')
ucl_line.set_label('UCL')
plt.legend()

fig.tight_layout()
plt.show()
