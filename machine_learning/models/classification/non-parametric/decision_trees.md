## Decision Trees

A non-parametric machine learning algorithm.

An example of a decision tree:
```
                        Gender <- Root Internal Node
             -------------|--------------
            /                            \
        Pclass   <- Internal Nodes ->    Age
(1 or 2) /  \ (3)         |     (<= 13) /  \ (> 13)
    Died    Survived       \           /    \
                            --->   Pclass   Died <- Leaf Nodes
                                (1) / \ (2 or 3)    |
                                Died   Survived   <-
```

Every feature is represented by an internal node, which is split into the node's two child nodes.

The final nodes where we make the predictions of survived/didn’t survive are called leaf nodes

When building a decision tree, start by choosing the feature with the most predictive power.

In the case of the titanic dataset, women were often given priority on lifeboats, thus Gender as a feature will be considered first.

## What makes a split valuable

The mathematical term measured is called information gain: 
- a value from 0 to 1 where 0 is the information gain of a useless split and 1 is the information gain of a perfect split.

The goal is to have homogeneity (or purity) on each side.

Looking at the above example:
```
                Gender
              ----|----
    (female) /           \ (male)
    Survived: 233       Survived: 109
    Died:     81        Died:     464
```

The split was successful in distinguishing the feature's target values into each side, thus providing informational gain.

### Gini impurity
- measures how pure a set is.
- represented by a value between 0 and 0.5 
    - 0.5 is completely impure (50% in each class)
    - 0 is completely pure (100% in the same class).
- formula: 2 * p * (1 - p)

In the above example, p = percent of passengers who survived.

Thus, for female side:
```
p = 233/(233+81) = 0.74203822
gini = 2 * p * (1 - p)
gini = 0.382835
```
Male side:
```
p = 109/(109+464) = 0.19022688
gini = 2 * p * (1 - p)
gini = 0.30808122
```

Both values are smaller than eg. the gini values for splitting on Age (<=30: 0.4689, >30: 0.4802), thus splitting on the Gender feature would be a better choice.

## Entropy

Another measure of purity.

A value between 0 and 1 where:
- 1 is completely impure (50% in each class)
- 0 is completely pure (100% the same class)
- formula: - [p log2 p + (1 - p) log 2 (1 - p)]

Using the example above:
```python
# On the left (female):
p = 233/(233+81) = 0.7420
Entropy = -(p * log(p) + (1-p) * log(1-p)) = 0.8237
# On the right (male)
p = 109/(109+464) = 0.1902
Entropy =  -(p * log(p) + (1-p) * log(1-p)) = 0.7019
```

As with the gini calculation, the smaller the entropy values, the better the split.

### Informational gain

Calculated using the numerical value for impurity.

Using the gini values in the example above, for gender:

```python
gini_whole_set = 2 * (survived/total) * (died/total)
gini_whole_set = 2 * 342/887 * 545/887 
gini_whole_set = 0.4738

gini_female = 0.3828
gini_male = 0.3081

info_gain = (gini_whole_set) - (f_survived/total * gini_female) - (m_survived/total * gini_male)
info_gain = 0.4738 - (314/887 * 0.3828) - (573/887 * 0.3081)
info_gain = 0.1393
```

The higher the informational gain, the better.

Another example:
```python
# Sample Input:
#   1 0 1 0 1 0
#   1 1 1
#   0 0 0
S = [int(x) for x in input().split()]
A = [int(x) for x in input().split()]
B = [int(x) for x in input().split()]

# Gini Impurities
gini_s = 2 * S.count(1)/len(S) * (1 - S.count(1)/len(S))
gini_a = 2 * A.count(1)/len(A) * (1 - A.count(1)/len(A))
gini_b = 2 * B.count(1)/len(B) * (1 - B.count(1)/len(B))

# Info gain
info_gain = gini_s - (gini_a*len(A)/len(S)) - (gini_b*len(B)/len(S))
print(round(info_gain, 5))
```