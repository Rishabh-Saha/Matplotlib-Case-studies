# Model Selection

Occam's razor is perhaps the most important thumb rule in machine learning and is incredibly 'simple' at the same time.  **When in dilemma, choose the simpler model** . The question then is 'how do we define simplicity?'.

* Which model do I pick?
* How do I make the decision?

IMP: Do not use the training data for evaluation

## Importance of simpler models

1. Simpler models are usually more ’generic’ and are more widely applicable (are generalizable).
2. Simpler models require fewer training samples
3. Simpler models are more robust
4. Simpler models make more errors in the training set

### Overfitting

Overfitting is a phenomenon wherein a model becomes highly specific to the data on which it is trained and fails to generalise to other unseen data points in a larger domain. A model that has become highly specific to a training data set has ‘learnt’ not only the hidden patterns in the data but also the noise and the inconsistencies in it. In a typical case of overfitting, a model performs quite well on the training data but fails miserably on the test data.

## Bias-Variance Tradeoff

**VARIANCE:** How sensitive is the model to input data

**BIAS:** How much error the model likely to make in train data

The expected error that a model will make is **Variance + Bias**

We considered the example of a model memorising the entire training data set. If you change the data set slightly, this model will also need to change drastically. The model is, therefore,  **unstable and sensitive to changes in training data** , and this is called  **high variance** .

**Bias** quantifies how **accurate the model is likely to be** on future (test) data. Extremely simple models are likely to fail in predicting complex real-world phenomena. Simplicity has its own disadvantages.

## Regularization

Regularization is the process of deliberately simplifying models to achieve the correct balance between keeping the model simple and not too naive. Recall that a few objective ways of measuring simplicity are as follows: choice of simpler functions, fewer model parameters and usage of lower degree polynomials.

### Hyperparameters

**Hyperparameters** are parameters that we pass on to the learning algorithm to control the complexity of the final model. They are choices that the algorithm designer makes to ‘tune’ the behaviour of the learning algorithm. Therefore, the choice of hyperparameters has a lot of bearing on the final model produced by the learning algorithm.

* Hyperparameters are used to 'fine-tune' or regularize the model to keep it optimally complex.
* The learning algorithm is given the hyperparameters as the input, and it returns the model parameters as the output.
* Hyperparameters are not part of the final model output.

#### Model Evaluation and Cross Validation

The key point to remember here is that a model should never be evaluated on data that it has already seen before. With that in mind, you will have either one of the following two cases: 1) the training data is abundant and 2) the training data is limited.

When training data is abundant, you keep a set of records untouched which is called as validation data. You use this validation data to tune your hyperparameters and improve the model and keep it from becoming very complex. 

When the training data is limited, keeping a seperate store of validation data from training data will actually eat into the training set.

**CrossValidation** - Instead of eating into training data by keeping out validation data, you train mulitple models by sampling the train set.  Finally, you only use the test set to test the hyperparameter once

The idea for both the methodologies —Hold-Out Strategy and Cross Validation — the basic idea is the same: to keep aside some data
that will not in any way influence the model building. The part of the data that is kept aside is then used as a ’proxy’ for the unknown (as far as the model we have built is concerned) test data on which we want to estimate the performance of the model.

##### Types of cross-validation

* K-fold cross-validation
* Leave one out (LOO) cross-validation
* Leave P-out (LPO) cross-validation
* Stratified K-Fold cross-validation
