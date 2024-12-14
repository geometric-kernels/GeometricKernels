""" Utilities for dealing with product spaces and product kernels.  """

import lab as B
from beartype.typing import Dict, List

from geometric_kernels.lab_extras import smart_cast


def params_to_params_list(
    number_of_factors: int, params: Dict[str, B.Numeric]
) -> List[Dict[str, B.Numeric]]:
    """
    Takes a dictionary of parameters of a product kernel and returns a list of
    dictionaries of parameters for the factor kernels. The shape of "lengthscale"
    should be the same as the shame of "nu", and the length of both should be
    either 1 or equal to `number_of_factors`.

    :param number_of_factors:
        Number of factors in the product kernel.
    :param params:
        Parameters of the product kernel.
    """
    assert params["lengthscale"].shape == params["nu"].shape
    assert len(params["nu"].shape) == 1

    if params["nu"].shape[0] == 1:
        return [params] * number_of_factors

    assert params["nu"].shape[0] == number_of_factors

    list_of_params: List[Dict[str, B.Numeric]] = []
    for i in range(number_of_factors):
        list_of_params.append(
            {
                "lengthscale": params["lengthscale"][i : i + 1],
                "nu": params["nu"][i : i + 1],
            }
        )

    return list_of_params


def make_product(xs: List[B.Numeric]) -> B.Numeric:
    """
    Embed a list of elements of factor spaces into the product space.
    Assumes that elements are batched along the first dimension.

    :param xs:
        List of the batches of elements, each of the shape [N, <axes(space)>],
        where `<axes(space)>` is the shape of the elements of the respective
        space.

    :return:
        An [N, D]-shaped array, a batch of product space elements, where `D` is
        the sum, over all factor spaces, of `prod(<axes(space)>)`.
    """
    common_dtype = B.promote_dtypes(*[B.dtype(x) for x in xs])

    flat_xs = [B.cast(common_dtype, B.reshape(x, B.shape(x)[0], -1)) for x in xs]
    return B.concat(*flat_xs, axis=-1)


def project_product(
    x: B.Numeric,
    dimension_indices: List[List[int]],
    element_shapes: List[List[int]],
    element_dtypes: List[B.DType],
) -> List[B.Numeric]:
    """
    Project an element of the product space onto each factor.
    Assumes that elements are batched along the first dimension.

    :param x:
        An [N, D]-shaped array, a batch of N product space elements.
    :param dimension_indices:
        Determines how a product space element `x` is to be mapped to inputs
        `xi` of the factor kernels. `xi` are assumed to be equal to
        `x[dimension_indices[i]]`, possibly up to a reshape. Such a reshape
        might be necessary to accommodate the spaces whose elements are matrices
        rather than vectors, as determined by `element_shapes`.
    :param element_shapes:
        Shapes of the elements in each factor. Can be obtained as properties
        `space.element_shape` of any given factor `space`.
    :param element_dtypes:
        Abstract lab data types of the elements in each factor. Can be obtained
        as properties `space.element_dtype` of any given factor `space`.

    :return:
        A list of the batches of elements `xi` in factor spaces, each of the
        shape `[N, *element_shapes[i]]`.
    """
    N = x.shape[0]
    xs = [
        smart_cast(dtype, B.reshape(B.take(x, inds, axis=-1), N, *shape))
        for inds, shape, dtype in zip(dimension_indices, element_shapes, element_dtypes)
    ]
    return xs
