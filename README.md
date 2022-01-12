# GradientMethods

Gradient Descent methods:
1. Non-convex approach 1 - projecting onto explicit rank constraint: minX∈Rm×n: rank(X)≤r f(x).
2. Non-convex approach 2 - gradient descent over factorized form: minU∈Rm×r,V∈Rn×r f(U).
3. Convex relaxation approach: use conditional gradient for solving minX∈Rm×n: kXk∗≤τ f(x).

Online Gradient Descent methods (finding the best portfolio using online learning):
1. Online gradient descent.
2. Online exponentiated gradient.
3. Online newton step.
