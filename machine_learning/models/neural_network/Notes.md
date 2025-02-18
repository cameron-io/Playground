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
