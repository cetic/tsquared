# TSquared
Python implementation of Hotelling's T2 for process monitoring + MYT decomposition


## Features

1. Classical multivariate T2 chart in which Hotelling’s T 2 statistic is computed as a distance of a multivariate observation from the multivariate mean scaled by the covariance matrix of the variables
2. Python scikit-learn -like implementation
3. Efficient with large datasets
4. MYT decomposition


## Specific Implementation

Hotelling’s T2 is initially for sampled distribution comparison to a reference distribution,
known as a generalization of the t-statistic for multivariate hypothesis testing.

For monitoring, a single multivariate observation is compared to a reference distribution.
This is more a generalization of the z-score.


<img src="pictures/z-score.jpg" width="300" >

### Relationship between z-score and TSquared 


<a href="pictures/equ_zscore.png"><img src="pictures/equ_zscore.png" width="300" ></a>

<img src="pictures/equ_T2.png" width="450" >

X is in this case the observation (point) in the multivariate space.

The covariance matrix of the reference multivariate distribution is formed by covariance terms between each dimensions and by variance terms (square of standard deviations) on the diagonal.

## Questions
How T-Squared is related to T-Test?

How T-Squared is related to Mahalanobis Distance?

<img src="pictures/equ_mahalanobis.PNG" width="300" >

Should I use PCA with T-Squared?

Can I apply T-Squared to any kind of process?

What are the conditions on parameters to use T-Squared?

Should I clean dataset before training?


Is there a procedure to clean the data?

What variables cause the outlier? 

What is MYT decomposition?

How deviation types impact T-Squared?

Is a T-Squared monitoring sufficient? Or do I still need univariate monitoring?

UCL, what does that mean in multivariate context?

How to compute UCL?



<!---
![](pictures/z-score.jpg)
![](pictures/equ_zscore.png)
![](pictures/equ_T2.png)
--->
<!---
<a href="https://github.com/cetic/TSquared/tree/master/pictures/z-score.jpg"><img class="fig" src="https://github.com/cetic/TSquared/tree/master/pictures/z-score.jpg" style="width:100%; height:auto;"/></a>
--->

