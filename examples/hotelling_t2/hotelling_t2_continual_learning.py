import matplotlib.pyplot as plt
import numpy as np

from tsquared import HotellingT2

seed = 1

# Set seed.
np.random.seed(seed)

# Generate data.
n1_train = 1000
n1_test = 100

n2_train = 2000
n2_test = 100

true_mean = np.array([4, -1.3, 8.7, -5.4])
true_cov = np.array([
	[1, 0.4, -0.4, 0.1],
	[0.4, 1, 0.6, -0.2],
	[-0.4, 0.6, 1, 0.02],
	[0.1, -0.2, 0.02, 1]
])

train1 = np.random.multivariate_normal(true_mean, true_cov, size=n1_train)
test1 = np.random.multivariate_normal(true_mean, true_cov, size=n1_test)
train2 = np.random.multivariate_normal(true_mean, true_cov, size=n2_train)
test2 = np.random.multivariate_normal(true_mean, true_cov, size=n2_test)
# Inputs.
print("--- Inputs ---\n")

print(f"True mean vector: {true_mean}")
print(f"True covariance matrix:\n{true_cov}")

# Fit and print some attributes.
## DATA set 1

print("\n--- Hotelling's T-squared fitting on the training set 1---\n")

hotelling = HotellingT2()
hotelling.fit(train1)

print(f"Computed mean vector: {hotelling.mean_}")
print(f"Computed covariance matrix:\n{hotelling.cov_}")
print(f"Hotelling's T-squared UCL: {hotelling.ucl(test1)}")

# Compute Hotelling's T-squared score for each sample in the test set.
print("\n--- Hotelling's T-squared scores on the test set 1 ---\n")

ucl_baseline = 0.1
t2_scores = hotelling.score_samples(test1)
scaled_t2_scores = hotelling.scaled_score_samples(test1,
	ucl_baseline=ucl_baseline)

print(f"Hotelling's T-squared score for each sample:\n{t2_scores}")
print(f"Scaled Hotelling's T-squared score for each sample:"
	f"\n{scaled_t2_scores}")

## DATA set 2

print("\n--- Hotelling's T-squared fitting on the training set 2---\n")

hotelling2 = HotellingT2()
hotelling2.fit(train2)

print(f"Computed mean vector: {hotelling2.mean_}")
print(f"Computed covariance matrix:\n{hotelling2.cov_}")
print(f"Hotelling's T-squared UCL: {hotelling2.ucl(test2)}")

# Compute Hotelling's T-squared score for each sample in the test set.
print("\n the test set 2 ---\n")

ucl_baseline = 0.1
t2_scores = hotelling2.score_samples(test2)
scaled_t2_scores = hotelling2.scaled_score_samples(test2,
	ucl_baseline=ucl_baseline)

print(f"Hotelling's T-squared score for each sample:\n{t2_scores}")
print(f"Scaled Hotelling's T-squared score for each sample:"
	f"\n{scaled_t2_scores}")

## DATA set 1 + continuation d'entrainement on the data set 2

print("\n--- Hotelling's T-squared with continual training on the set 2---\n")

hotelling3 = HotellingT2()
hotelling3.fit(train2,old_model=hotelling)

print(f"Computed mean vector: {hotelling3.mean_}")
print(f"Computed covariance matrix:\n{hotelling3.cov_}")
print(f"Hotelling's T-squared UCL: {hotelling3.ucl(test2)}")

# Compute Hotelling's T-squared score for each sample in the test set.
print("\n  test set 2 ---\n")

ucl_baseline = 0.1
t2_scores = hotelling3.score_samples(test2)
scaled_t2_scores = hotelling3.scaled_score_samples(test2,
	ucl_baseline=ucl_baseline)

print(f"Hotelling's T-squared score for each sample:\n{t2_scores}")
print(f"Scaled Hotelling's T-squared score for each sample:"
	f"\n{scaled_t2_scores}")

## DATA set 1 + 2

print("\n--- Hotelling's T-squared training on 1 and 2 --\n")

hotelling4 = HotellingT2()
train3=np.concatenate((train1, train2), axis=0)
test3=np.concatenate((test1, test2), axis=0)
hotelling4.fit(train3,old_model=hotelling)

print(f"Computed mean vector: {hotelling3.mean_}")
print(f"Computed covariance matrix:\n{hotelling3.cov_}")
print(f"Hotelling's T-squared UCL: {hotelling3.ucl(test3)}")

# Compute Hotelling's T-squared score for each sample in the test set.
print("\n  test set 2 ---\n")

ucl_baseline = 0.1
t2_scores = hotelling3.score_samples(test3)
scaled_t2_scores = hotelling3.scaled_score_samples(test3,
	ucl_baseline=ucl_baseline)

print(f"Hotelling's T-squared score for each sample:\n{t2_scores}")
print(f"Scaled Hotelling's T-squared score for each sample:"
	f"\n{scaled_t2_scores}")
