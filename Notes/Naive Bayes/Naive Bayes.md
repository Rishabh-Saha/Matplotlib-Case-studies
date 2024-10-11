# Naive Bayes

Text based classifier

P(E) = Number of favourable outcomes / Total number of outcomes

P(A) = email being spam

P(B) = email being spam given that a word say "FREE TRIP" has occured

P(A) is called as Prior probability

P(B) is called as Posterioir probability


## Bayes Theorem

Building blocks: conditional probability, joint probability


**Joint probability,** **P(A,B)** refers to the chances that both outcome (A) and outcome (B) occur.

**Conditional probability,** **P(A|B)** refers to the chances that some outcome (A) occurs given that another event (B) has already occurred.

Relation between them:

**P(A|B) = P(A,B)/P(B)  OR   P(A,B) = P(A|B).P(B)   OR   P(A,B) = P(B|A).P(A)**


Bayes Theorem:

**P(A|B) = P(B|A).P(A)/P(B)   OR   P(A|B) = P(A,B)/P(B)**


Prior Probability = P(Ci)

Posterior Probability = P(Ci/x) = (P(x/Ci) * P(Ci)) / P(x)


* P(![C_{i}](https://latex.codecogs.com/gif.latex?C_%7Bi%7D)) is known as the prior probability. It is the probability of an event occurring before the collection of new data.  Prior plays an important role while classifying, when using Naïve Bayes, as it highly influences the class of the new test point.
* P(X/![C_{i}](https://latex.codecogs.com/gif.latex?C_%7Bi%7D)) represents the likelihood function. It tells the likelihood of a data point occurring in a category. The conditional independence assumption is leveraged while computing the likelihood probability.
* The effect of the denominator P(x) is not incorporated while calculating probabilities as it is the same for both the classes and hence, can be ignored without affecting the final outcome.
* P(![C_{i}](https://latex.codecogs.com/gif.latex?C_%7Bi%7D)/X) is called the posterior probability, which is finally compared for the classes, and the test point is assigned the class whose Posterior probability is greater.


## Prior, Posterior and Likelihood

Let’s understand the terminology of Bayes theorem.

You have been using 3 terms: P(Class = edible / poisonous), P(X | Class) and P(Class | X). Bayesian classification is based on the principle that ‘you combine your **prior knowledge or beliefs about a population** with the **case specific information** to get the actual (posterior) probability’.

* P(Class = edible) or P(Class = poisonous) is called the **prior probability**

This incorporates our ‘ **prior beliefs** ’ before you collect specific information. If 90% of mushrooms are edible, then the prior probability is 0.90. Prior gets multiplied with the likelihood to give the posterior. In many cases, the prior has a tremendous effect on the classification. If the prior is neutral (50% are edible), then the likelihood may largely decide the outcome.

* P(X|Class) is the **likelihood**

After agreeing upon the prior, you collect new, case-specific data (like plucking mushrooms randomly from a farm and observing the cap colours). Likelihood updates our prior beliefs with the new information. If you find a CONVEX mushroom, then you’d want to know how likely you were to find a convex one if you had only plucked edible mushrooms.

If  P(CONVEX| edible) is high, say 80%, implying that there was an 80% chance of getting a convex mushroom if you only took from edible mushrooms, this will reflect in increased chances of the mushroom being edible.

If the likelihood is neutral (e.g. 50%), then the prior probability may largely decide the outcome. If the prior is way too powerful, then likelihood often barely affects the result.

* P(Class = edible | X) is the **posterior probability**

It is the outcome which **combines prior beliefs and case-specific information** . It is a balanced outcome of the prior and the likelihood.

If Zimbabwe takes 3 Australian wickets in the first over in a world cup, would you predict Australia to lose? Probably not, because the prior odds are way too strong in favour of Australia. They’ve never lost to Zimbabwe in a world cup! The likelihood, though it may be high, gets balanced by the prior odds (Australia’s prior odds may even be 99%!) to give you the correct posterior.


## **Laplace Smoothing**

Laplace Smoothing helps us deal with - 'zero probability problem’ - the probability of a word which has never appeared in a class (though it may have appeared in the dataset in another class) is 0

You just add 1 to each of the words in the dictionary so that they no longer have 0s in them

Please note that - If there are words occurring in a test sentence which are not a part of the dictionary, then they will not be considered as part of the feature vector since it only considers the words that are part of the dictionary. These new words will be completely ignored.


### Bernoulli Naive Bayes Classifier

The most fundamental difference in Bernoulli Naive Bayes Classifier is the way we build the bag of words representation, which in this case is just 0 or 1. Simply put, Bernoulli Naive Bayes is concerned only with whether the word is present or not in a document, whereas Multinomial Naive Bayes counts the no. of occurrences of the words as well.
