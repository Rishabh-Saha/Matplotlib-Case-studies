# Logistic Regression

## Pre-requisites

### Exponentials

Few properties of Exponentials:

* x^a⋅x^b = x^(a+b)
* (xy)^a = x^a.y^a
* x^a/x^b = X^(a-b) ; x != 0

### Logarithms

Few properties of Log:

* log(a)^b = b.log(a)
* log(a.b) = log(a) + log(b)
* log(a/b) = log(a) - log(b)
* (x)^3 = y is same as 3 = log (y) with base (x)

## Binary Classification

Outcomes which can be classified into two:

Eg: Whether the mails are spam or ham

But in some cases it is difficult to classify into 2 segments just based on the data we have. Eg:

Eg: ![1722698134790](image/LogisticRegression/1722698134790.png)

As you can see, person below 200 are non-diabetic and person above 200 are diabetic. But we also notice people with 210 are non-diabetic. So we saw how using a **simple boundary decision method** would not work in this case.

So we have to use sigmoid curve which has all the properties you would want — extremely low values in the start, extremely high values in the end, and intermediate values in the middle — it’s a good choice for modelling the value of the **probability of diabetes**.

This is the Sigmoid curve equation:

![1722698295994](image/LogisticRegression/1722698295994.png)

We have to take B0 and B1 and plot the sigmoid curve which looks like below:

![1722698380443](image/LogisticRegression/1722698380443.png)

The sigmoid curve actually has the properties we need, i.e. extremely low values in the start, extremely high values in the end, and intermediate values in the middle.

So the question we have to answer is how to find the best B0 and B1

### Finding the best fit sigmoid curve

So, if you notice the expression above "**e**", it is minus of linear expression

If we recall, in linear regression we used "**Least sum of Squares**" to find the best fit line.

But in logistic regression we cannot do that, because the values are either 1 or 0.

We have to use something called "**Maximum Likelihood**" method to find the best fitted line.

If you, recall the Simoid curve equation, it is not very intuitive. So we try to linearize the equation.

![1722727389048](image/LogisticRegression/1722727389048.png)

### Interpreting P/(1-P)

So the terms are

- Odds: P/(1-P) = e^(B0+B1.x)
- Log Odds: log(P/(1-P)) = B0+ B1.x

Eg: Say P is the probability of someone being diabetic and say P/(1-P) = 4

Therefore, we can say, P(Diabetics)/P(Non-Diabetics) = 4

P(Diatetic) = 4* P(Non-Diabetic); P = 4(1-P); P = 4-4P; 5P= 4; P = 4/5; P = 8/10 = 0.8

P(Diabetic) = 0.8 (80%)

### Relation between Odds and Probability

![1722727935097](image/LogisticRegression/1722727935097.png)

## Model Evaluation

The most commonly used metrics to evaluate a classification model:

1. Sensitivity = No of actual Yeses correctly predicted / Total no. of actual Yeses
2. Specificity = No. of actual Nos correctly predicted / Total no. of actual Nos

### ROC Curve (Receiver Operating Characteristic)

2 important metrics to remember

1. True Positive Rate (TPR)  = True Positive / Actual Positive, or Sensitivity
2. False Positive Rate (FPR) = False Positive / Actual Negatives. or 1 - Specificity

So, a good model is where TPR is high and FPR is low.

The following is the ROC curve where X- Axis: FPR, Y-Axis: TPR. 

- A good model will have points closer to Y-axis as it approaches 1
- A bad model will have points away from Y-axis as it approaches 1

![1723002857624](image/LogisticRegression/1723002857624.png)


The Area Under the Curve (AUC) for the Receiver Operating Characteristic (ROC) curve is a performance measurement for classification problems, the higher the AUC, the more accurate the model

### Precision and Recall

**Precision**: Probability that a predicted 'Yes' is actually 'Yes'

* Jitna tune 'Yes' bola usme se actually kitna 'Yes' hai
* same as the 'Positive Predictive Value'

**Recall**: Probability that a 'Yes' case is predicted as such

* Jitna Yes hai, usme se tune kitno ko 'Yes' bola
* Same as sensitivity
