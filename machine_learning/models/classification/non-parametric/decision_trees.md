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
