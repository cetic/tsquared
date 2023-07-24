import numpy as np

from tsquared import HotellingT2, MYTDecomposition

# Create data.
train = np.array([[10. , 10.7],
                  [10.4,  9.8],
                  [ 9.7, 10. ],
                  [ 9.7, 10.1],
                  [11.7, 11.5],
                  [11. , 10.8],
                  [ 8.7,  8.8],
                  [ 9.5,  9.3],
                  [10.1,  9.4],
                  [ 9.6,  9.6],
                  [10.5, 10.4],
                  [ 9.2,  9. ],
                  [11.3, 11.6],
                  [10.1,  9.8],
                  [ 8.5,  9.2]])
test = np.array([[12.3, 12.5],
                 [ 7. ,  7.3],
                 [11. ,  9. ],
                 [ 7.3,  9.1]])

# Inputs.
print("--- Inputs ---\n")

print(f"Training set:\n{train}")
print(f"Test set:\n{test}")

# Fit and print some attributes.
print("\n--- Hotelling's T-squared fitting on the training set---\n")

hotelling = HotellingT2().fit(train)

print(f"Computed mean vector: {hotelling.mean_}")
print(f"Computed covariance matrix:\n{hotelling.cov_}")

# Compute Hotelling's T-squared score for each sample in the test set.
print("\n--- Hotelling's T-squared scores on the test set ---\n")

t2_scores = hotelling.score_samples(test)

print(f"Hotelling's T-squared score for each sample:\n{t2_scores}")

# MYT Decomposition.
print("\n--- MYT Decomposition of the Hotelling's T-squared statistic ---\n")

myt_dec = MYTDecomposition(hotelling)

uncond_t2_terms = myt_dec.unconditional_t2_terms(test)
uncond_ucl = myt_dec.ucl_unconditional_terms()
cond_t2_terms = myt_dec.conditional_t2_terms(test)
cond_ucl = myt_dec.ucl_conditional_terms()

print(f"Unconditional T-squared terms:\n{uncond_t2_terms}")
print(f"UCL of unconditional T-squared terms: {uncond_ucl}")
print(f"p conditional T-squared terms:\n{cond_t2_terms}")
print(f"UCL of the p conditional T-squared terms: {cond_ucl}")
