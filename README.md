# TSquared

Python implementation of Hotelling's T-squared (T2) for process monitoring and MYT decomposition.

## Table of Contents:

1. [Features](#features)
2. [Specific Implementation](#impl)
3. [Questions](#quest)
	- [How TSquared is related to T-Test?](#ttest)
	- [How TSquared is related to Mahalanobis Distance?](#maha)
	- [How TSquared is related to MCD?](#mcd)
	- [Should I use PCA with TSquared?](#pca)
	- [Can I apply T-Squared to any kind of process? What are the conditions on parameters to use T-Squared?](#conditions)
	- [Should I clean dataset before training? Is there a procedure to clean the data?](#cleaning)
	- [What variables cause the outlier? What is MYT decomposition?](#MYT)
	- [How deviation types impact T-Squared?](#types)
	- [Is a TSquared monitoring sufficient? Or do I still need univariate monitoring?](#sufficient)
	- [UCL, what does that mean in multivariate context? How to compute UCL?](#UCL)

4. [References](#ref) 

## <a name="features">Features</a>

1. Classical multivariate T2 chart in which Hotelling's T2 statistic is computed as a distance of a multivariate observation from the multivariate mean scaled by the covariance matrix of the variables
2. Python scikit-learn -like implementation
3. Efficient with large datasets
4. MYT decomposition

## <a name="impl">Specific Implementation</a>

Hotelling's T2 is initially for sampled distribution comparison to a reference distribution,
known as a generalization of the t-statistic for multivariate hypothesis testing.

For monitoring, only a single multivariate observation is compared to a reference distribution.

```python
from tsquared import HotellingT2

clf = HotellingT2()

# Hotelling's T2 without dataset cleaning.
clf.fit(X_train)
test_scores = clf.score_samples(X_test)
	
# Hotelling's T2 with one-pass cleaning.
clf.cleanfit(X_train, iter=1)
respg = multivariate_normality(X_train, alpha=.05)
test_scores = clf.score_samples(X_test_norm)
	
# Hotelling's T2 with two-pass cleaning.
clf.cleanfit(X_train, iter=2)
respg=multivariate_normality(X_train, alpha=.05)
test_scores = clf.score_samples(X_test_norm)
	
# Hotelling's T2 with infinite-pass cleaning (set a high number, size criteria will stop iterations).
clf.cleanfit(X_train, iter=100)
test_scores = clf.score_samples(X_test_norm)
	
# Hotelling's T2 with smart cleaning (decision based on multivariate normality coefficient).
clf.cleanfit(X_train)
test_scores = clf.score_samples(X_test, alpha=.05)
```

## <a name="quest"> Questions</a>

#### <a name="ttest">How TSquared is related to T-Test?</a>

Hotelling's T2 is a generalization of the t-statistic for multivariate hypothesis testing
When a single multivariate observation is compared to a reference distribution, it can be viewed as a generalization of the z-score.
The difference is the nature of the entities (point >< distribution) that are considered in the distance computation and in the denominator of the equation also.

<img src="https://raw.githubusercontent.com/cetic/tsquared/master/figures/z-score.jpg">

What's the relationship with z-score then?

<a href="figures/equ_zscore.png"><img src="https://raw.githubusercontent.com/cetic/tsquared/master/figures/equ_zscore.png" width="300"></a>

<img src="https://raw.githubusercontent.com/cetic/tsquared/master/figures/equ_T2.png" width="450">

X is in this case the observation (point) in the multivariate space.

The covariance matrix of the reference multivariate distribution is formed by covariance terms between each dimensions and by variance terms (square of standard deviations) on the diagonal.

#### <a name="maha">How TSquared is related to Mahalanobis Distance?</a>

<img src="https://raw.githubusercontent.com/cetic/tsquared/master/figures/equ_mahalanobis.PNG" width="550">

#### <a name="mcd">How TSquared is related to MCD?</a>

MCD = minimum covariance determinant is an algorithm available in the Outlier Detection framework pyOD
MCD is based on Mahalanobis Squared Distance (MSD =~ Hotelling's T2)
Based on the distribution of MSD, the training consists to find the subgroup of points ($h < n$) that minimizes the covariance determinant. This subgroup of points can be thought of as the minimum number of points which must not be outlying (with $h > n/2$ to keep a sufficient number of point)

⟹ It is equivalent to the cleaning operation in TSquared.

#### <a name="pca">Should I use PCA with TSquared?</a>

Yes, you can!

But this should be done cautiously

1. PCA defines new coordonates for each points
2. PCA is often used to reduce dimensionality by selecting the strongest "principal" components defining the underlying relation between variables
3. T2 score on all PCA components = T2 on all original variables 

Can we apply T2 on a reduced number of (principal) components?
Let's try a 2D example. In the following picture, the relation between Var1 and Var2 is mostly linear, these variables are strongly correlated. Let's suppose that the 1st component of the PCA is sufficient to define the relation, component 2 being the noisy part of the relation.

<img src="https://raw.githubusercontent.com/cetic/tsquared/master/figures/Capture_pca.PNG" width="500">

In this case, monitoring any futur observation is like applying a z-score (1 dimension) to this observation compared to the distribution of all past observations projected on the first component axis.

If a loss of correlation happened between Var1 and Var2, it won't be seen on this univariate monitoring because it is the second component that will be impacted. This can happened if the senor capturing Var2 is defectuous. 

By extension to more dimensions, we understand that reducing "blindly" the number of components before a TSquared monitoring is not advised. It is certainly not a thing to do in case of sensors validation.

Instead, if PCA is used to reduce the dimensionnality, it is advised to monitor as well the residual group of components in a separated monitoring.

#### <a name="conditions">Can I apply T-Squared to any kind of process? What are the conditions on parameters to use T-Squared?</a>

The basic assumption is that all variables should be normally distributed. 
However, the algorithm is tolerant to some extent if the distributions are not perfectly normal.

#### <a name="cleaning">Should I clean dataset before training? Is there a procedure to clean the data?</a>

Yes, the cleaner the better

The TSquared procedure can be applied 1 or 2 times to the training set and outliers can be filtered at each round.

The risk to work with a training set not clean is to have an univariate outlier which is an inlier in multivariate, the multivariate UCL being too large (Observation n°78).

<img src="https://raw.githubusercontent.com/cetic/tsquared/master/figures/Image_inlier.png">

#### <a name="MYT"> What variables cause the outlier? What is MYT decomposition?

#### <a name="types"> How deviation types impact TSquared?

#### <a name="sufficient"> Is a TSquared monitoring sufficient? Or do I still need univariate monitoring?

#### <a name="UCL"> UCL, what does that mean in multivariate context? How to compute UCL?

#### <a name="boxcox"> My data are not normally distributed. Does it help to apply a boxcox transformation on each variables?

The experiment was done using TSquared autocleaning function and boxcox transformation on each variables.
	
## <a name="cleaning">References</a>

- <cite> Decomposition of T2 for Multivariate Control Chart Interpretation, ROBERT L. MASON, NOLA D. TRACY and JOHN C. YOUNG</cite>

	[Decomposition of T2 for Multivariate Control Chart Interpretation]: ../Références/Decomposition%20of%20T2%20for%20Multivariate%20Control%20Chart%20Interpretation.pdf

- <cite> Application of Multivariate Statistical Quality Control In Pharmaceutical Industry, Mesut ULEN, Ibrahim DEMIR</cite>

	[Application of Multivariate Statistical Quality Control In Pharmaceutical Industry]: ../Références/Application%20of%20Multivariate%20Statistical%20Quality%20Control%20In%20Pharmaceutical%20Industry.pdf

- <cite> Identifying Variables Contributing to Outliers in Phase I, ROBERT L. MASON, YOUN-MIN CHOU, AND JOHN C. YOUNG</cite>

	[Identifying Variables Contributing to Outliers in Phase I]: ../Références/Identifying%20variables%20contributing%20to%20outliers%20in%20Phase%20I.pdf

- <cite> Multivariate Control Charts for Individual Observations, NOLA D. TRACY, JOHN C. YOUNG, ROBERT L. MASON</cite>

 	[Multivariate Control Charts for Individual Observations]: ../Références/Multivariate%20control%20charts%20for%20individual%20observations.pdf

<!---
![](figures/z-score.jpg)
![](figures/equ_zscore.png)
![](figures/equ_T2.png)
--->
<!---
<a href="https://github.com/cetic/TSquared/tree/master/figures/z-score.jpg"><img class="fig" src="https://raw.githubusercontent.com/cetic/tsquared/master/figures/z-score.jpg" style="width:100%; height:auto;"/></a>
--->
