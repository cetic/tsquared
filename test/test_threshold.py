import numpy as np
import pytest
from numpy.testing import assert_almost_equal
from scipy import stats

from tsquared import (
    HotellingT2,
    ThresholdCalculator,
    THRESHOLD_STATISTICAL,
    THRESHOLD_OPTIMIZATION
)


# ============= Fixtures ============= #

@pytest.fixture
def sample_data_2d():
    """Generate sample 2D data for testing"""
    np.random.seed(42)
    X = np.random.multivariate_normal(
        mean=[0, 0],
        cov=[[1, 0.5], [0.5, 1]],
        size=100
    )
    return X


@pytest.fixture
def high_dim_data():
    """Generate high-dimensional data"""
    np.random.seed(42)
    n_features = 20
    n_samples = 1000
    X = np.random.multivariate_normal(
        mean=np.zeros(n_features),
        cov=np.eye(n_features),
        size=n_samples
    )
    return X


@pytest.fixture
def ill_conditioned_data():
    """Generate ill-conditioned data"""
    np.random.seed(42)
    # Create highly correlated variables
    x = np.random.normal(0, 1, 100)
    y = x * 0.99999 + np.random.normal(0, 0.0001, 100)
    return np.column_stack([x, y])


@pytest.fixture
def extreme_variance_data():
    """Generate data with extreme variance differences"""
    np.random.seed(42)
    x = np.random.normal(0, 1, 100)
    y = np.random.normal(0, 1e6, 100)
    return np.column_stack([x, y])


# ============= Statistical Method Tests ============= #

@pytest.mark.parametrize("alpha", [0.01, 0.05, 0.1])
def test_statistical_different_alphas(alpha, sample_data_2d):
    """Test statistical method with different alpha values"""
    clf = HotellingT2(alpha=alpha, threshold_method=THRESHOLD_STATISTICAL)
    clf.fit(sample_data_2d)
    assert clf.ucl_indep_ > 0
    # Higher alpha should result in lower threshold
    if alpha == 0.1:
        clf_lower = HotellingT2(alpha=0.01, threshold_method=THRESHOLD_STATISTICAL)
        clf_lower.fit(sample_data_2d)
        assert clf.ucl_indep_ < clf_lower.ucl_indep_


def test_statistical_known_values():
    """Test statistical method against pre-calculated values"""
    # These values were calculated manually using the F-distribution formula
    n_samples, n_features = 100, 2
    alpha = 0.05
    threshold = ThresholdCalculator.statistical_ucl(n_samples, n_features, alpha)

    # Calculate expected value
    f_crit = stats.f.ppf(1 - alpha, n_features, n_samples - n_features)
    expected = n_features * (n_samples + 1) * (n_samples - 1) / \
               (n_samples * (n_samples - n_features)) * f_crit

    assert_almost_equal(threshold, expected, decimal=6)


@pytest.mark.parametrize("n_features,n_samples", [
    (2, 3),  # Minimal valid case
    (10, 11),  # Borderline case
    (50, 1000),  # High dimensional case
])
def test_statistical_sample_size_effects(n_features, n_samples):
    """Test statistical method with different sample and feature sizes"""
    threshold = ThresholdCalculator.statistical_ucl(n_samples, n_features, 0.05)
    assert threshold > 0


# ============= Optimization Method Tests ============= #

def test_optimization_ill_conditioned(ill_conditioned_data):
    """Test optimization method with ill-conditioned data"""
    clf = HotellingT2(threshold_method=THRESHOLD_OPTIMIZATION)
    clf.fit(ill_conditioned_data)
    assert np.isfinite(clf.ucl_indep_)
    assert clf.ucl_indep_ > 0


def test_optimization_extreme_variance(extreme_variance_data):
    """Test optimization method with extreme variance differences"""
    clf = HotellingT2(threshold_method=THRESHOLD_OPTIMIZATION)
    clf.fit(extreme_variance_data)
    assert np.isfinite(clf.ucl_indep_)
    assert clf.ucl_indep_ > 0


def test_optimization_high_dim(high_dim_data):
    """Test optimization method in high dimensions"""
    clf = HotellingT2(threshold_method=THRESHOLD_OPTIMIZATION)
    clf.fit(high_dim_data)
    assert np.isfinite(clf.ucl_indep_)
    assert clf.ucl_indep_ > 0


# ============= Edge Cases and Error Handling ============= #

def test_edge_cases():
    """Test various edge cases"""
    # Near-singular covariance (but still valid)
    X = np.array([[1, 1], [1.001, 1], [0.999, 1], [1.002, 0.998]])
    clf = HotellingT2(threshold_method=THRESHOLD_OPTIMIZATION)
    clf.fit(X)
    assert np.isfinite(clf.ucl_indep_)

    # Very small variance in one direction
    X = np.array([[1, 1], [1, 2], [1, 3], [1.0001, 4]])
    clf = HotellingT2(threshold_method=THRESHOLD_OPTIMIZATION)
    clf.fit(X)
    assert np.isfinite(clf.ucl_indep_)

    # Single feature case
    X = np.array([[1], [2], [3]])
    clf = HotellingT2(threshold_method=THRESHOLD_OPTIMIZATION)
    clf.fit(X)
    assert np.isfinite(clf.ucl_indep_)


def test_input_validation():
    """Test comprehensive input validation"""
    # Invalid alpha values
    with pytest.raises(ValueError):
        HotellingT2(alpha=-0.1)
    with pytest.raises(ValueError):
        HotellingT2(alpha=1.1)

    # Invalid threshold method
    with pytest.raises(ValueError):
        HotellingT2(threshold_method='invalid')

    # Invalid data shapes
    X = np.array([[1, 2]])  # Single sample
    clf = HotellingT2()
    with pytest.raises(ValueError):
        clf.fit(X)

    # Non-numeric data
    X = np.array([['a', 'b'], ['c', 'd']])
    with pytest.raises(ValueError):
        clf.fit(X)


# ============= Comparison Tests ============= #

def test_method_comparison(sample_data_2d):
    """Compare statistical and optimization methods"""
    clf_stat = HotellingT2(threshold_method=THRESHOLD_STATISTICAL)
    clf_opt = HotellingT2(threshold_method=THRESHOLD_OPTIMIZATION)

    clf_stat.fit(sample_data_2d)
    clf_opt.fit(sample_data_2d)

    # Test points
    inlier = np.array([[0, 0]])
    mild_outlier = np.array([[2, 2]])
    extreme_outlier = np.array([[10, 10]])

    # Both methods should agree on clear cases
    assert clf_stat.predict(inlier) == clf_opt.predict(inlier)
    assert clf_stat.predict(extreme_outlier) == clf_opt.predict(extreme_outlier)

    # Score ordering should be consistent
    stat_scores = clf_stat.score_samples(np.vstack([inlier, mild_outlier, extreme_outlier]))
    opt_scores = clf_opt.score_samples(np.vstack([inlier, mild_outlier, extreme_outlier]))

    assert np.all(np.diff(stat_scores) > 0)  # Strictly increasing
    assert np.all(np.diff(opt_scores) > 0)  # Strictly increasing


# ============= Performance Tests ============= #

@pytest.mark.parametrize("n_features", [2, 10, 50])
def test_performance_scaling(n_features):
    """Test performance with increasing dimensions"""
    np.random.seed(42)
    n_samples = 1000
    X = np.random.multivariate_normal(
        mean=np.zeros(n_features),
        cov=np.eye(n_features),
        size=n_samples
    )

    clf_stat = HotellingT2(threshold_method=THRESHOLD_STATISTICAL)
    clf_opt = HotellingT2(threshold_method=THRESHOLD_OPTIMIZATION)

    clf_stat.fit(X)
    clf_opt.fit(X)

    assert np.isfinite(clf_stat.ucl_indep_)
    assert np.isfinite(clf_opt.ucl_indep_)