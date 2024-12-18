# Random Forest

## Ensembles

An ensemble refers to a group of things viewed as a whole rather than individually. In an ensemble, a **collection of models** is used to make predictions, rather than individual models. Arguably, the most popular in the family of ensemble models is the random forest, which is an ensemble made by the **combination of a large number of decision trees**

### Diversity and Acceptability

**Diversity** ensures that the models serve **complementary** purposes, which means that the individual models make predictions  **independent of each other** .

**Acceptability** implies that each model is at least  **better than a random model** . This is a pretty lenient criterion for each model to be accepted into the ensemble, i.e., it has to be at least better than a random guesser.

#### Ways to achieve diversity

There are a number of ways in which you can bring diversity among your models you plan to include in your ensemble.

1. Use different subsets of training data
2. Use different training hyperparameters
3. Use different types of classifiers
4. Use different features

---

Also, the ensembles cannot be misled by the  **assumptions made by individual models** . For example, ensembles (particularly random forests) successfully reduce the problem of overfitting. If a decision tree in an ensemble overfits, you let it. Chances are extremely low that more than 50% of the models are overfitted. Ensembles ensure that you do not put all your eggs in one basket.

### Approaches to ensembles

* **Voting**

  * Voting combines the output of different algorithms by taking a vote.
  * For classification, it is majority and for regression it is average of all predictions
  * **Check out example in Housing Case Study #Ensembles**
* **Stacking and Blending**

  * Pass the outputs of the individual models to a level-2 classifier/regressor as derived meta features, which will decide what weights should be given to each of the model outputs in the final prediction
  * **Check out example in Housing Case Study #Stacking after #Ensembles**
* **Boosting**

  * Boosting can be used with any technique and combines the weak learners into strong learners by creating sequential models such that the final model has higher accuracy than the individual models
  * It is one of the most popular approaches to ensembling
  * There are 2 major approaches to boosting: Adaptive and Gradient
* **Bagging (Bootstrap aggregation)**

  * Bagging creates different training subsets from the sample training data with replacement, and an algorithm with the same set of hyperparameters is built on these different subsets of data. In this way, the same algorithm with a similar set of hyperparameters is exposed to different parts of data, resulting in a slight difference between the individual models. The predictions of these individual models are combined by taking the average of all the values for regression or a majority vote for a classification problem.
  * Bagging works well with high variance algorithms and is easy to parallelise.

## Random Forest

1. Creates an ensemble (bagging) of decision trees
2. Builds decision trees on different samples
3. Takes majority or average

a random forest selects a random sample of data points (bootstrap sample) to build each tree and a random sample of features  **while splitting a node** . Randomly selecting features ensures that each tree is **diverse **and that some prominent features are not dominating in all the trees making them somewhat similar

### Advantages of Blackbox Models Over Tree and Linear Models

* **Diversity**: Diversity arises because each tree is created with a subset of the attributes/features/variables
* **Stability:** Stability arises because the answers given by a large number of trees average out
* **Immunity to the curse of dimensionality** : Since each tree does not consider all the features, the feature space (the number of features that a model has to consider) reduces.
* **Parallelization** : You need a number of trees to make a forest. Since two trees are independently built on different data and attributes, they can be built separately at the same time
* **Testing/training data and the OOB (out-of-bag) error** : Similar to cross-validation error, OOB is the mean prediction error on each training sample xᵢ, using only the trees that do not have xᵢ in their bootstrap sample used for building the model.

### Importance of Features in Random Forest

The importance of features in random forests, sometimes called ‘ **Gini importance** ’ or ‘ **mean decrease impurity** ’, is defined as the **total decrease in node impurity** (it is weighted by the probability of reaching that node (which is approximated by the proportion of samples reaching that node)) **averaged** over all the trees of the ensemble.

For each variable, the sum of the Gini decreases across every tree of the forest and is accumulated every time that variable is chosen to split a node. The sum is divided by the number of trees in the forest to give an average.


### Hyperparameter Tuning

The following hyperparameters are present in a random forest classifier. Note that most of these hyperparameters are actually of the decision trees that are in the forest.
- n_estimators: integer, optional (default=10): The number of trees in the forest.
- criterion: string, optional (default= “gini”)The function to measure the quality of a split. Supported criteria
are “gini” for the Gini impurity and “entropy” for the information gain. Note: this parameter is tree-specific.
- max_features : int, float, string or None, optional (default=”auto”)The number of features to consider when
looking for the best split:
  - If int, then consider max_features features at each split.
  - If float, then max_features is a percentage and int(max_features * n_features) features are considered at each split.
  - If “auto”, then max_features=sqrt(n_features).
  - If “sqrt”, then max_features=sqrt(n_features) (same as “auto”).
  - If “log2”, then max_features=log2(n_features).
  - If None, then max_features=n_features.
- max_depth : integer or None, optional (default=None)The maximum depth of the tree. If None, then nodes
are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.
- min_samples_split : int, float, optional (default=2)The minimum number of samples required to split an
internal node:
  - If int, then consider min_samples_split as the minimum number.
  - If float, then min_samples_split is a percentage and ceil(min_samples_split, n_samples) are the minimum number of samples for each split.
- min_samples_leaf : int, float, optional (default=1)The minimum number of samples required to be at a leaf
node:
  - If int, then consider min_samples_leaf as the minimum number.
  - If float, then min_samples_leaf is a percentage and ceil(min_samples_leaf * n_samples) are the minimum number of samples for each node.
- min_weight_fraction_leaf : float, optional (default=0.)The minimum weighted fraction of the sum total of
weights (of all the input samples) required to be at a leaf node. Samples have equal weight when
sample_weight is not provided.
- max_leaf_nodes : int or None, optional (default=None)Grow trees with max_leaf_nodes in best-first fashion.
Best nodes are defined as relative reduction in impurity. If None then unlimited number of leaf nodes.

- min_impurity_split : float,Threshold for early stopping in tree growth. A node will split if its impurity is above
the threshold, otherwise it is a leaf.