import lab as B
from beartype.typing import List


def make_product(xs: List[B.Numeric]) -> B.Numeric:
    """
    Embed a list of elements of factor spaces into the product space.
    Assumes that elements are batched along the first dimension.

    :param xs: list of the (batches of) elements, each of the shape [N, ...].
    :return: (a batch of) product space element [N, ...].
    """
    flat_xs = [B.reshape(x, B.shape(x)[0], -1) for x in xs]
    return B.concat(*flat_xs, axis=-1)


def project_product(
    x: B.Numeric, indices: List[int], element_shapes: List[List[int]]
) -> List[B.Numeric]:
    """
    Project an element of the product space into each factor.
    Assumes that elements are batched along the first dimension.

    :param x: (a batch of) product space element [N, ...].
    :param indices: indices of the product space correspoding to each factor.
    :param element_shapes: shapes of the elements in each factor.

    :return: a list of the (batches of) elements in factor spaces, each of the shape [N, *element_shapes[s]]
    """
    N = x.shape[0]
    xs = [
        B.reshape(B.take(x, inds, axis=-1), N, *shape)
        for inds, shape in zip(indices, element_shapes)
    ]
    return xs
