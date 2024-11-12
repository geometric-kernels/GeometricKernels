import inspect

import pytest

import geometric_kernels.spaces

from ..helper import (
    compact_matrix_lie_groups,
    discrete_spectrum_spaces,
    noncompact_symmetric_spaces,
    product_discrete_spectrum_spaces,
    spaces,
)


@pytest.mark.parametrize(
    "fun, cls",
    [
        (compact_matrix_lie_groups, geometric_kernels.spaces.CompactMatrixLieGroup),
        (
            product_discrete_spectrum_spaces,
            geometric_kernels.spaces.ProductDiscreteSpectrumSpace,
        ),
        (discrete_spectrum_spaces, geometric_kernels.spaces.DiscreteSpectrumSpace),
        (
            noncompact_symmetric_spaces,
            geometric_kernels.spaces.NoncompactSymmetricSpace,
        ),
        (spaces, geometric_kernels.spaces.Space),
    ],
)
def test_all_discrete_spectrum_spaces_covered(fun, cls):
    spaces = fun()

    # all classes in the geometric_kernels.spaces module
    classes = [
        (cls_name, cls_obj)
        for cls_name, cls_obj in inspect.getmembers(geometric_kernels.spaces)
        if inspect.isclass(cls_obj)
    ]
    for cls_name, cls_obj in classes:
        if issubclass(cls_obj, cls) and not inspect.isabstract(cls_obj):
            for space in spaces:
                if isinstance(space, cls_obj):
                    break
            else:
                # complain if discrete_spectrum_spaces() does not contain an
                # instance of a non-abstract subclass of DiscreteSpectrumSpace
                # from the geometric_kernels.spaces module.
                assert (
                    False
                ), f"An instance of the class `{cls_name}` is missing from the list returned by the function `{fun.__name__}`"
