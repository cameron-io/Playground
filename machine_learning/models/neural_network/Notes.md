## Neural Network

Also known as Artificial Neural Networks (ANN):
- performs well where there are a lot of features:
    - as they automatically do feature engineering.
    - thus, do not require any domain knowledge to restructure the features.

Constructed using Neurons, connected by Synapses through which signals are sent.

## Artificial Neurons

Also known as a node, is modeled after a biological neuron.

It is a simple object that can take input, do some calculations with the input, and produce an output.

For example:
```
    x1
       \
        -> (Neuron) -> y1
       /
    x2
```
Neurons can take any number of inputs and can also produce any number of outputs.

### Neuron Computations

Inside the neuron, to do the computation to produce the output, we first put the inputs into the following equation (just like in logistic regression).

#### Activation Functions

First, the variable inputs must be computed, applying weights & bias.

```
x = (w1 * x1) + (w2 * x2) + b

x1,x2: inputs
w1,w2: weights
b: bias
```

Secondly, there are 3 Activation Functions which may compute 'x'

1. Sigmoid

y = f(x) = 1 / (1 + e^-x)

y ranges from 0 to 1.

2. tanh 
- hyperbolic tan function

y = f(x) = sinh(x)/cosh(x) = (e^x - e^-x) / (e^x + e^-x)

y ranges from -1 to 1

3. ReLU (Rectified Linear Unit)
- identity function for positive numbers 
- sends negative numbers to 0

y = f(x) = { 0 if x <= 0
           { x if x > 0


### Multi-layer Perceptron

A neural network is the combination of neurons where the outputs of some neurons are inputs of other neurons.

Feed-Forward neural networks have the neurons only send signals in one direction.

Multi-Layer Perceptron (MLP) neural networks are most commonly used.

A multi-layer perceptron will always have:
- one input layer, with a neuron (or node) for each input. 
- any number of hidden layers and each hidden layer can have any number of nodes.
- one output layer, with a node for each output.

1. The nodes in the input layer take a single input value and pass it forward.
2. The nodes in the hidden layers as well as the output layer can take multiple inputs but they always produce a single output.


## Training

Accomplished by defining the "loss function":
- measure of how far off our neural network is from being perfect
- the goal is to optimise this loss function

### Cross Entropy

cross entropy = { p   if y = 1
                { 1-p if y = 0

Compute the product of cross entropy values for all datapoints.

For example:
```
Target  M1 Pred M2 Pred
1       0.6     0.5 
1       0.8     0.9
0       0.3     0.1
0       0.4     0.5
```
ce = pred_positives... * (1 - pred_negatives)...
ce_M1 = 0.6 * 0.8 * (1-0.3) * (1-0.4) = 0.2016
ce_M2 = 0.5 * 0.9 * (1-0.1) * (1-0.5) = 0.2025

### Backpropagation

Neural networks have a lot of parameters that we can control. 

There are several coefficients for each node and there can be a lot of nodes.

The process for updating these values to converge on the best possible model is computationally expensive.

The neural network works backwards from the output node iteratively updating the coefficients of the nodes.

The Training Process of Neural Networks summarized:
1. initialize all the Artificial Neurons (nodes) with coefficient values.
2. iteratively change the values so that at every iteration to improve the loss function.
3. Eventually cannot improve the loss function anymore.
4. Thus, arrived at the optimal model.
