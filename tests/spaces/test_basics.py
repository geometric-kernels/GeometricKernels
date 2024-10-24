import inspect

import geometric_kernels.spaces

from ..helper import discrete_spectrum_spaces


def test_all_discrete_spectrum_spaces_covered():
    spaces = discrete_spectrum_spaces()

    # all classes in the geometric_kernels.spaces module
    classes = [
        (cls_name, cls_obj)
        for cls_name, cls_obj in inspect.getmembers(geometric_kernels.spaces)
        if inspect.isclass(cls_obj)
    ]
    for cls_name, cls_obj in classes:
        if issubclass(
            cls_obj, geometric_kernels.spaces.DiscreteSpectrumSpace
        ) and not inspect.isabstract(cls_obj):
            for space in spaces:
                if isinstance(space, cls_obj):
                    break
            else:
                # complain if discrete_spectrum_spaces() does not contain an
                # instance of a non-abstract subclass of DiscreteSpectrumSpace
                # from the geometric_kernels.spaces module.
                assert False, f"Space {cls_name} not covered by tests"
