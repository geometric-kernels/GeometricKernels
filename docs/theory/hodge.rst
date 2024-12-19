#####################
  Hodge Decomposition
#####################

In a simplicial 2-complex, for an edge flow $\mathbf{f}$, we have the following discrete operations to measure its variational properties: 

* The *divergence* operation: $\mathbf{B}_1 \mathbf{f}$, which measures the netflow of edge flows passing through each node. A divergence-free flow has zero netflow at each node, indicating that the in-flow and out-flow are balanced.

* The *curl* operation: $\mathbf{B}_2^\top \mathbf{f}$, which measures the circulation of edge flows around each triangle. A curl-free flow has zero circulation around each triangle, indicating that the flow is irrotational.

For a node function $\mathbf{f}_0$, we have the *gradient* operation: $\mathbf{B}_1^\top \mathbf{f}_0$, which measures the change in the function value along each edge. A gradient-free function has zero change in value along each edge, indicating that the function is constant along each edge.

Given a simplicial 2-complex, the space of edge flows $\mathbb{R}^N_1$ is a direct sum of three subspaces 
$$
\mathbb{R}^{N_1} = \text{im}(\mathbf{B}_1^\top) \oplus \ker(\mathbf{L}) \oplus \text{im}(\mathbf{B}_2),
$$
where $\text{im}(\mathbf{B}_1^\top)$ is the *gradient* space, $\ker(\mathbf{L})$ is the *harmonic* space, and $\text{im}(\mathbf{B}_2)$ is the *curl* space.

It states that any edge flow is composed of three orthogonal components: a gradient component, a harmonic component, and a curl component. 

* The gradient component is the flow that is induced by a node function, which is curl-free. 

* The curl component is the flow that is induced by a triangle function, which is divergence-free.

* The harmonic component is the flow that is divergence-free and curl-free. 
