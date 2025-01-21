import matplotlib.pyplot as plt
import numpy as np

from tsquared import HotellingT2, THRESHOLD_STATISTICAL, THRESHOLD_OPTIMIZATION

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

# Print input information
print("--- Inputs ---\n")
print(f"True mean vector: {true_mean}")
print(f"True covariance matrix:\n{true_cov}")

# Initialize both methods
hotelling_stat = HotellingT2(threshold_method=THRESHOLD_STATISTICAL)
hotelling_opt = HotellingT2(threshold_method=THRESHOLD_OPTIMIZATION)

# Fit both methods
print("\n--- Hotelling's T-squared fitting with both methods ---\n")

hotelling_stat.fit(train)
hotelling_opt.fit(train)

print("Statistical method:")
print(f"UCL: {hotelling_stat.ucl(test)}")
print("\nOptimization method:")
print(f"UCL: {hotelling_opt.ucl(test)}")

# Compute scores for both methods
print("\n--- Comparing T-squared scores between methods ---\n")

ucl_baseline = 0.1
scores_stat = hotelling_stat.score_samples(test)
scores_opt = hotelling_opt.score_samples(test)

scaled_scores_stat = hotelling_stat.scaled_score_samples(test, ucl_baseline=ucl_baseline)
scaled_scores_opt = hotelling_opt.scaled_score_samples(test, ucl_baseline=ucl_baseline)

print("Statistical method scores (first 5 samples):")
print(f"Raw scores: {scores_stat[:5]}")
print(f"Scaled scores: {scaled_scores_stat[:5]}")

print("\nOptimization method scores (first 5 samples):")
print(f"Raw scores: {scores_opt[:5]}")
print(f"Scaled scores: {scaled_scores_opt[:5]}")

# Outlier detection comparison
print("\n--- Outlier Detection Comparison ---\n")

preds_stat = hotelling_stat.predict(test)
preds_opt = hotelling_opt.predict(test)

n_outliers_stat = np.sum(preds_stat == -1)
n_outliers_opt = np.sum(preds_opt == -1)

print(f"Statistical method detected {n_outliers_stat} outliers")
print(f"Optimization method detected {n_outliers_opt} outliers")

# Agreement between methods
agreement = np.sum(preds_stat == preds_opt)
print(f"\nMethods agree on {agreement}/{len(test)} samples ({agreement / len(test) * 100:.1f}%)")

# Visualization
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))

# Statistical method plot
ax1.scatter(range(len(scaled_scores_stat)), scaled_scores_stat, label='Samples')
ucl_line = ax1.axhline(y=ucl_baseline, color='r', linestyle='-', label='UCL')
ax1.set_title('Statistical Method: Scaled Hotelling\'s T2 scores')
ax1.set_xlabel('Index')
ax1.set_ylabel('Scaled T2 score')
ax1.legend()

# Optimization method plot
ax2.scatter(range(len(scaled_scores_opt)), scaled_scores_opt, label='Samples')
ucl_line = ax2.axhline(y=ucl_baseline, color='r', linestyle='-', label='UCL')
ax2.set_title('Optimization Method: Scaled Hotelling\'s T2 scores')
ax2.set_xlabel('Index')
ax2.set_ylabel('Scaled T2 score')
ax2.legend()

fig.tight_layout()
plt.show()

# Compare method performance on specific examples
print("\n--- Method Comparison on Specific Cases ---\n")

# Generate some specific test cases with correct dimensions
specific_cases = {
    'Mean point': true_mean.reshape(1, -1),
    'Far point': (true_mean + np.array([3, 3, 3, 3])).reshape(1, -1),
    'Edge point': (true_mean + np.array([2, 0, 0, 0])).reshape(1, -1)
}

print("Comparing methods on specific test cases:")
for name, point in specific_cases.items():
    print(f"\n{name}:")
    stat_score = hotelling_stat.score_samples(point)[0]
    opt_score = hotelling_opt.score_samples(point)[0]

    is_stat_outlier = stat_score > hotelling_stat.ucl(point)
    is_opt_outlier = opt_score > hotelling_opt.ucl(point)

    print(f"Statistical method:")
    print(f"  Score: {stat_score:.2f}")
    print(f"  Classification: {'outlier' if is_stat_outlier else 'inlier'}")

    print(f"Optimization method:")
    print(f"  Score: {opt_score:.2f}")
    print(f"  Classification: {'outlier' if is_opt_outlier else 'inlier'}")