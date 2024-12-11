# Artificial Nueral Networks

Artificial neural networks (ANN) are a collection of many simple devices called artificial neurons . The network ‘learns’ to conduct certain tasks, such as recognising a cat, by training the neurons  to ‘fire’ in a certain way when given a particular input, such as a cat. In other words, the network learns to inhibit or amplify the input signals to perform a certain task, such as recognising a cat, speaking a word or identifying a tree.

### Perceptron

A perceptron acts as a tool that enables you to make a decision based on multiple factors. Each decision factor holds a different ‘weight’. It takes a weighted sum of multiple inputs (with bias) as the cumulative input and applies an output function on the cumulative input to get the output, which then assists in making a decision.

**CumulativeInput = w1.x1 + w2.x2 + w3.x3 + b**

Where, **xi’s** represent the inputs, **wi’s** represent the weights associated with inputs and **b** represents bias.

We then apply the step function to the cumulative input. According to the step function, if this cumulative sum of inputs is greater than 0, then the output is 1/yes; or else, it is 0/no

### Artificial Nuerons

A neuron is quite similar to a perceptron. However, in perceptrons, the commonly used activation/output is the step function, whereas in the case of ANNs, the activation functions are non-linear functions.

![Artificial Neuron](https://images.upgrad.com/e649f6a7-35e6-4485-9ad3-a6d614147628-unnamed%20(28).png)

multiple artificial neurons in a neural network are arranged in different layers. The first layer is known as the **input layer** , and the last layer is called the **output layer** . The layers in between these two are the **hidden layers**

![Classification and Regression Neural Network](https://images.upgrad.com/9b6d3636-cef3-4f5d-b54b-36aff3d53391-unnamed%20(29).png)

There are six main elements that must be specified for any neural network. They are as follows:

1. Input layer
2. Output layer
3. Hidden layers
4. Network topology or structure
5. Weights and biases
6. Activation functions

### Inputs and Outputs of Nueral Network

**IMPORTANT: The inputs can only be numeric**

For different types of input data, you need to use different ways to convert the inputs into a numeric form. The most commonly used inputs for ANNs are as follows:

* **Structured data** : The type of data that we use in standard machine learning algorithms with multiple features and available in two dimensions, such that the data can be represented in a tabular format, can be used as input for training ANNs. Such data can be stored in **CSV files, MAT files, Excel files, etc** . This is highly convenient because the input to an ANN is usually given as a numeric feature vector. Such structured data eases the process of feeding the input into the ANN.
* **Text data** : For text data, you can use a **one-hot vector** **or** **word embeddings** corresponding to a certain word. For example, in one hot vector encoding, if the vocabulary size is |V|, then you can represent the word w**n** as a one-hot vector of size |V| with '1' at the nth element with all other elements being zero. The problem with one-hot representation is that, usually, the vocabulary size |V| is huge, in tens of thousands at least; hence, it is often better to use word embeddings that are a lower-dimensional representation of each word.
* **Images** : Images are naturally represented as arrays of numbers and can thus be fed into the network directly. These numbers are the **raw pixels of an image** . ‘Pixel’ is short for ‘picture element’. In images, pixels are arranged in rows and columns (an array of pixel elements).
* **Speech:** In the case of a speech/voice input, the basic input unit is in the form of **phonemes** . These are the distinct units of speech in any language. The speech signal is in the form of waves, and to convert these waves into numeric inputs, you need to use Fourier Transforms. Note that the input after conversion should be numeric, so you are able to feed it into a neural network.

Depending on the nature of the given task, the outputs of neural networks can either be in the form of classes (if it is a classification problem) or numeric (if it is a regression problem).
One of the commonly used output functions is the softmax function for classification.

A softmax output is similar to what we get from a multiclass logistic function commonly used to compute the probability of an output belonging to one of the multiple classes. It is given by the following formula:

pi=e^wix′/ ∑c−1t=o e^wt.x′

where c is the number of classes or neurons in the output layer, x′ is the input to the network, and wi’s are the weights associated with the inputs.

An in-depth representation of the cumulative input as the output is given below.

![](assets/20241210_123201_image.png)

In the image above, z is the cumulative input.

z = w1.x1 + w2.x2 + w3.x3 + b

### Activation functions

The activation functions introduce non-linearity in the network, thereby enabling the network to solve highly complex problems. Problems that take the help of neural networks require the ANN to recognise complex patterns and trends in the given data set. If we do not introduce non-linearity, the output will be a linear function of the input vector. This will not help us in understanding more complex patterns present in the data set.

While choosing activation functions, you need to ensure that they are:

1. Non-linear,
2. Continuous, and
3. Monotonically increasing.

**Different types of activation functions:**

![Types of Activation Functions](https://images.upgrad.com/4924e02a-9ba1-4305-b232-7fbc092e857a-Activation_functions.png)

**The features of these activation functions are as follows:**

1. **Sigmoid** : When this type of function is applied, the output from the activation function is bound between 0 and 1 and is not centred around zero. A sigmoid activation function is usually used when we want to regularise the magnitude of the outputs we get from a neural network and ensure that this magnitude does not blow up.
2. **Tanh (Hyperbolic Tangent)** : When this type of function is applied, the output is centred around 0 and bound between -1 and 1, unlike a sigmoid function in which case, it is centred around 0.5 and will give only positive outputs. Hence, the output is centred around zero for tanh.
3. **ReLU (Rectified Linear Unit)** : The output of this activation function is linear in nature when the input is positive and the output is zero when the input is negative. This activation function allows the network to converge very quickly, and hence, its usage is computationally efficient. However, its use in neural networks does not help the network to learn when the values are negative.
4. **Leaky ReLU (Leaky Rectified Linear Unit)** : This activation function is similar to ReLU. However, it enables the neural network to learn even when the values are negative. When the input to the function is negative, it dampens the magnitude, i.e., the input is multiplied with an epsilon factor that is usually a number less than one. On the other hand, when the input is positive, the function is linear and gives the input value as the output. We can control the parameter to allow how much ‘learning emphasis’ should be given to the negative value.

### Parameters and Hyperparameters of a Neural Network

During training, the neural network learning algorithm fits various models to the training data and selects the best prediction model. The learning algorithm is trained with a fixed set of **hyperparameters** associated with the network structure. Some of the important hyperparameters to consider to decide the network structure are given below:

* Number of layers
* Activation function
* Number of neurons in the input, hidden and output layers
* Learning rate (the step size taken each time we update the weights and biases of an ANN)
* Number of epochs (the number of times the entire training data set passes through the neural network)

The purpose of training the learning algorithm is to obtain optimum weights and biases that form the ****parameters**** of the network.

#### Terminologies

1. **W** represents the weight of the matrix
2. **b** stands for bias
3. **x** stands for input
4. **y** stands for the actual label or value that the model is trying to predict
5. **p** represents the probability vector of the predicted output for the classification problem. **h^L** represents the predicted output for the regression problem (where L represents the number of layers)

Detailed Notations

- **Layer-specific superscripts**:

  - Denote which layer the variable belongs to. For example:
    - \( h^n \): Output of the \( n \)-th hidden layer.
    - \( z^n \): Accumulated input to the \( n \)-th layer.
- **Neuron-specific subscripts**:

  - Denote the index of a specific neuron within a layer. For example:
    - \( h nto2 \): Output of the \( n \)-th neuron in the second hidden layer.
    - \( z nto3 \): Accumulated input to the third neuron in the \( n \)-th layer.
- **Weight matrix connections**:

  - \( W^2 \): Weight matrix connecting the first hidden layer to the second hidden layer.
  - \( w 31to2 \): Weight connecting the first neuron of the first hidden layer to the third neuron of the second hidden layer.
- **Bias terms**:

  - \( b 1to3 \): Bias for the third neuron in the first hidden layer.

  ![alt text](image-2.png)

### Assumputions of a Neural Network

1. The neurons in an ANN are arranged in layers, and these layers are arranged sequentially.
2. The neurons within the same layer do not interact with each other.
3. The inputs are fed into the network through the input layer, and the outputs are sent out from the output layer.
4. Neurons in consecutive layers are densely connected, i.e., all neurons in layer l are connected to all neurons in layer l+1.
   Eg: Number of interconnections = Number of neurons in layer **l** x Number of neurons in layer **(l−1)** = 11 * 18 = 198
5. Every neuron in the neural network has a bias value associated with it, and each interconnection has a weight associated with it.
6. All neurons in a particular hidden layer use the same activation function. Different hidden layers can use different activation functions, but in a hidden layer, all neurons use the same activation function.

## Feedforward Neural Network

The information flow in a neural network from the input layer to the output layer to enable the neural network to make a prediction. The information flow in this direction is often called **feedforward**.

![](https://images.upgrad.com/5cdbcd44-0058-4945-b15c-9a98e1da7927-notation.png)![Matrix Representation](https://images.upgrad.com/f3ad356e-5b79-4e73-a31d-5bdc7a358561-matrix_rep1.png)

![](assets/20241210_172853_image.png)

### Algorithm

The pseudocode for a feedforward pass is given below:

1. We initialise the variable h0 as the input: h0=xi
2. We loop through each of the layers computing the corresponding output for each layer, i.e., h^l.
   For l in [1,2,......,L]: h^l=σ(W^l.h^(l−1)+b^l)
3. We compute the prediction p by applying an activation function to the output from the previous layer, i.e., we apply a function to h^L, as shown below.  p=f(h^L)

In both the regression and classification problems, the same algorithm is used till the last step.
In the final step, in the classification problem, p defines the probability vector, which gives the probability of the data point belonging to a particular class among different possible classes or categories. In the regression problem, p represents the predicted output obtained, which we will normally refer to as h^L.

For classification problem, we use **softmax output**, which gives us the probability vector pi:

> Pij = e^(wj.hL) / ∑t=1toc Wt.hL

which is basically  normalising the vector pi.

### Loss function

We want to know how wrong the prediction of the neural network is and want to quantify this error in the prediction. A loss function or cost function will help us quantify such errors.

In the case of regression, the most commonly used loss function is **MSE/RSS**.

> Loss = 1/2 * (actual-predicted)^2
>
> RSS = (actual-predicted)^2
>
> Avg.Loss = MSE = RSS*1/N

In the case of classification, the most commonly used loss function is **Cross Entropy/Log Loss**.

> Individual loss = -∑actual*log(predicted)
> Total Loss = Sum of all individual losses

The task of training neural networks is similar to that of other ML models such as linear regression and logistic regression. The predicted output (output from the last layer) minus the actual output is the cost (or the loss), and we have to tune the parameters **w and b** such that the total cost is minimised.

#### Minimization of total loss

We minimise the average of the total loss and not the total loss. Minimising the average loss implies that the total loss is getting minimised.

This can be done using any optimisation routine such as **gradient descent**.

The gradient vector is in the direction in which the loss value increases most rapidly.

The parameter being optimised is iterated in the direction of reducing cost according to the following rule:

> **W**new = **W**old - α.(∂L/∂W)
> where, W refers to the matrix of all weights and biases

## Back propagation

**Using the chain rule equation**, we can calculate the gradient of loss needed for above function

> ∂L/∂w2 = ∂L/∂h2 * ∂h2/∂z2 * ∂z2/∂w2

![alt text](image-4.png)

> **Loss function**: L = 1/2 * (y−h2)^2

#### Psuedo code

1. Initialise with the input
2. Forward propagation of the input through the network with random initial values for weights and biases
3. Making a prediction and computing the overall loss
4. Updating model parameters using backpropagation i.e., updating the weights and biases in the network, using gradient descent
5. Forward propagation of the input through the network with updated parameters leading to a decrease in the overall loss
6. Repeat the process until the optimum values of weights and biases are obtained such that the model makes acceptable predictions

## TensorFlow

TensorFlow is a deep learning library developed by **Google** . It is used widely in the industry for several different applications. Some of these applications include smart text in Gmail, Google Translate and Google Lens.

### Tensors

A tensor is the fundamental data structure used in TensorFlow. It is a multidimensional array with a uniform data type. The data type for an entire tensor is the same.

**Tensors** are n-dimensional arrays that are quite similar to NumPy arrays. An important difference between these is their performance. NumPy is a highly efficient library that is designed to work on CPUs. On the other hand, TensorFlow can work on CPUs and GPUs. So, if you have a compatible GPU, then it is highly likely that TensorFlow will outperform NumPy.

Types of Tensors:

1. tf.constant - think of const from js
2. tf.variable - think of let/var from js

The differences between these two types of tensors:

- The values of constant tensors cannot be changed once they are declared but those of variable tensors can be.
- Constant tensors need to be initialised with a value while they are being declared, whereas variable tensors can be declared later using operations.
- Differentiation is calculated for variable tensors only, and the gradient operation ignores constants while differentiating.

TensorFlow supports all the basic mathematical operators, and you can call them by simply using the respective operators. To use the operator commands, you need to ensure that both tensors on which the operations are being carried out have the same dimensions

#### Keras flow

The whole process can be summarised as follows:

1. Define a simple sequential model to set the hidden layer(s) and the output layer. The following code allows us to define the simple sequential model:

> model = keras.Sequential(
> [ keras.layers.Dense(2, activation="sigmoid", input_shape=(X.shape[-1],)),
> keras.layers.Dense(1, activation="linear")
> ])

2. Display the properties and dimensions of each layer of the neural network.

> model.summary()

3. Define the type of optimiser to update the weights and biases.

> model.compile(optimizer=keras.optimizers.SGD(), loss="mean_squared_error")

4. Fit all the components defined above into one line of code to train the neural network.

> model.fit(X,Y.values,epochs=10,batch_size=32)

5. Obtain predictions on different input data.

> model.predict(X)[:,0]

---

> model.fit(X_train, y_train, batch_size=64, epochs=5, validation_data=(X_val, y_val))

The number of **epochs** mentioned in the code snippet defines the number of times the learning algorithm will work through the entire data set. One epoch indicates that each training example has had an opportunity to update the internal model parameters, i.e., the weights and biases.

**Batch size** refers to the number of training examples utilised in one iteration. The model decides the number of examples to work with in each iteration before updating the internal model parameters.

#### Dropouts

The main purpose of using dropouts is to reduce overfitting in Neural networks.
The dropout operation is performed by multiplying the weight matrix Wl with an α mask vector. The shape of the vector **α** will be (Weight columns shape,1). Now if the value of **q** (the probability of 1) is 0.66, the **α** vector will have two 1s and one 0.


You can see the differences between the ANN without dropout and the ANN with dropout below. Adding a dropout layer essentially removes the links from the third neuron in the first layer to all the neurons in the next layer. The cross on the interconnections indicates that the interconnection has been removed.


![](https://images.upgrad.com/d9f0dd4d-6330-478d-9a80-67a5ac3b75d3-Dropout1.jpg)

Some important points to note regarding dropouts are:
1. Dropouts can be applied only to some layers of the network (in fact, that is a common practice - you choose some layer arbitrarily to apply dropouts to)
2. The mask α is generated independently for each layer during feedforward, and the same mask is used in backpropagation
3. The mask changes with each minibatch/iteration and is randomly generated in each iteration (sampled from a Bernoulli with some p(1)=q)

> # dropping out 20% neurons in a layer in Keras 
> model.add(Dropout(0.2))

Some important points to note while implementing dropouts are as follows:
1. Here, '0.2' is the probability of zeros and not ones.
2. This is one of the hyperparameters to be experimented with when building a neural network.
3. You do not apply dropout to the output layer.
4. The mask used here is a matrix.
5. Dropout is applied only during training, not while testing.

Dropouts also help in symmetry breaking. There is an extremely high likelihood that communities will be created within neurons, which can restrict the neurons from learning independently. Hence, by setting a random set of the weights to zero in every iteration, this community/symmetry can be broken.

#### Batch Normalisation

Batch normalisation is performed on the output of the layers of each batch, Hl. It is essentially normalising the matrix Hl across all data points in the batch. Each vector in Hl is normalised by the mean vector μ and the standard deviation vector ^σ computed across a batch.

> keras.layers.BatchNormalization()