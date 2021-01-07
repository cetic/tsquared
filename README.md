# TSquared
Python implementation of Hotelling's T2 for process monitoring + MYT decomposition

## Specific Implementation

Hotelling’s T2 is initially for sampled distribution comparison to a reference distribution,
known as a generalization of the t-statistic for multivariate hypothesis testing.

For monitoring, a single multivariate observation is compared to a reference distribution
This is more a generalization of the z-score.


<img src="pictures/z-score.jpg" width="300" >

### Relationship between z-score and TSquared 


<a href="pictures/equ_zscore.png"><img src="pictures/equ_zscore.png" width="300" ></a>

<img src="pictures/equ_T2.png" width="450" >

X is in this case the observation (point) in the multivariate space.

The covariance matrix of the reference multivariate distribution is formed by covariance terms between each dimensions and by variance terms (square of standard deviations) on the diagonal.

<!---
![](pictures/z-score.jpg)
![](pictures/equ_zscore.png)
![](pictures/equ_T2.png)
--->
<!---
<a href="https://github.com/cetic/TSquared/tree/master/pictures/z-score.jpg"><img class="fig" src="https://github.com/cetic/TSquared/tree/master/pictures/z-score.jpg" style="width:100%; height:auto;"/></a>
--->

