# Grasshoper
A deep learning stayle  optimizer.

# Input:

[value,first-order-gradient,second-order-gradient,cosine_decomposition,sinie_decomposition,...]

# Output:

[update volumn for network weight]

# Idea:

Instead of rule or handcraft based optimizer, we let itself learn the dynamic of the network.

# Assumption:

The dynamic or topology of same-type of data under newtork models has some invariant properties.

We are trying to use a relative small and easy to train network called Grasshoper to learn this dynamic. 


