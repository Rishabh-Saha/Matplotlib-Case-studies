## Pros

1. **Logistic regression**
   1. It is convenient for generating probability scores.
   2. Efficient implementation is available across different tools.
   3. The issue of multicollinearity can be countered with regularisation.
   4. It has widespread industry use.
2. **Decision trees**
   1. Intuitive decision rules make it easy to interpret.
   2. Trees handle nonlinear features well.
   3. The variable interaction is taken into account.
3. **Support vector machines**
   1. SVMs can handle large feature space.
   2. These can handle nonlinear feature interaction.
   3. They do not rely on the entire dimensionality of the data for the transformation.

## Cons

1. **Logistic regression**
   1. It does not perform well when the features space is too large.
   2. It does not perform well when there are a lot of categorical variables in the data.
   3. The nonlinear features have to be transformed to linear features in order to efficiently use them for a logistic model.
   4. It relies on entire data i.e. if there is even a small change in the data, the logistic model can change significantly.
2. **Decision trees**
   1. Trees are highly biased towards the training set and overfit it more often than not.
   2. There is no meaningful probability score as the output.
3. **Support vector machines**
   1. SVMs are not efficient in terms of computational cost when the number of observations is large.
   2. It is tricky and time-consuming to find the appropriate kernel for a given data.

## End to end modelling

1. **Start with logistic regression** . Using a logistic regression model serves two purposes:
   1. It acts as a **baseline** (benchmark) model.
   2. It gives you an idea about the important variables.
2. Then, go for **decision trees** and compare their performance with the logistic regression model. If there is no significant improvement in their performance, then just use the important variables drawn from the logistic regression model.
3. Finally, if you still do not meet the performance requirements, use  **support vector machines** . But, keep in mind the  **time and resource constraints** , because it takes time to find an appropriate kernel for SVM. Also, they are computationally expensive.


## CART and CHAID Trees

**CART (Classification and Regression Trees)**

**CHAID (Chi-square Automatic Interaction Detection)**

A **chi-square test** is a statistical hypothesis test where the test statistic is chi-squared distribution. This test is used to compare the interaction of independent variables with the dependent variable.

You are already familiar with  **CART** , which creates a  **binary tree-** a tree with a maximum of two child nodes for any node in the tree. Sometimes CART is not appropriate to visualise the important features in a dataset because binary trees tend to be much **deeper** and more **complex** than a **non-binary tree-** a tree which can have more than two child nodes for any node in the tree. This is where **CHAID** comes in. CHAID can create non-binary trees which tend to be shallower than the binary trees. This makes CHAID trees easier to look at and understand the important drivers (features) in a business problem. The process of finding out important features is also referred to as  **driver analysis** .

**CART**

1. It performs binary split at each node
2. It can do regression and classification both
3. It grows a large tree first and then prunes it back to a smaller tree
4. It may easily overfit unless it's pruned back

**CHAID**

1. It uses multiway splits by default
2. It is intended to work with categorical/discretized targets
3. It uses truncation by making a node split only if a signifance criterion is fulfilled
4. It tries to prevent overfitting from the start


## Summary

To summarise, you should start with a **logistic regression** model. Then, build a **decision tree** model. While building a decision tree, you should choose the appropriate method: **CART** for predicting and **CHAID** for driver analysis. If you are not satisfied with the model performance mentioned so far, and you have sufficient time and resources in hand, then go ahead and build more complex models like **random forests** and  **support vector machines** .
