"""
Standalone script to precompute characters for
:class:`~.spaces.SpecialOrthogonal` and :class:`~.spaces.SpecialUnitary`.

Edit `recalculate`, `storage_file_name`, `order`, and `groups` variables below
in the code and run the script as.

.. code-block:: bash

    python compute_characters.py
"""
import sys
sys.path.append('/data/dragon316/GeometricKernels')

import itertools
import json
import sys
from functools import reduce
import more_itertools
import sympy
from beartype.typing import Union
from sympy.matrices.determinant import _det as sp_det
from jackpy.jack import ZonalPol
from tqdm import tqdm

from geometric_kernels.spaces.so import SOEigenfunctions
from geometric_kernels.spaces.su import SUEigenfunctions  # noqa
from geometric_kernels.spaces.grassmannian import GrassmannianEigenfunctions
from geometric_kernels.utils.utils import (
    get_resource_file_path,
    partition_dominance_cone,
    partition_dominance_or_subpartition_cone,
)

# set to True to recalculate all characters, set to False to add to the already existing without recalculating
recalculate = True

storage_file_name = "../spaces/precomputed_grassmanian_zsf.json"

# the number of representations to be calculated for each group
order = 25

# largest SO(n) to be used
max_n = 10


class CompactJSONEncoder(json.JSONEncoder):
    """A JSON Encoder that puts small containers on single lines.

    Source (probably):
    https://gist.github.com/jannismain/e96666ca4f059c3e5bc28abb711b5c92.
    """

    CONTAINER_TYPES = (list, tuple, dict)
    """Container datatypes include primitives or other containers."""

    MAX_WIDTH = 70
    """Maximum width of a container that might be put on a single line."""

    MAX_ITEMS = 70
    """Maximum number of items in container that might be put on single line."""

    INDENTATION_CHAR = " "

    def __init__(self, *args, **kwargs):
        # using this class without indentation is pointless
        if kwargs.get("indent") is None:
            kwargs.update({"indent": 4})
        super().__init__(*args, **kwargs)
        self.indentation_level = 0

    def encode(self, o):
        """Encode JSON object *o* with respect to single line lists."""
        if isinstance(o, (list, tuple)):
            return "[{}]".format(",".join(self.encode(el) for el in o))
        elif isinstance(o, dict):
            if o:
                if self._put_on_single_line(o):
                    return (
                        "{ "
                        + ", ".join(
                            f"{self.encode(k)}: {self.encode(el)}"
                            for k, el in sorted(o.items())
                        )
                        + " }"
                    )
                else:
                    self.indentation_level += 1
                    output = [
                        self.indent_str + f"{json.dumps(k)}: {self.encode(v)}"
                        for k, v in sorted(o.items())
                    ]
                    self.indentation_level -= 1
                    return "{\n" + ",\n".join(output) + "\n" + self.indent_str + "}"
            else:
                return "{}"
        elif isinstance(
            o, float
        ):  # Use scientific notation for floats, where appropiate
            return format(o, "g")
        elif isinstance(o, str):  # escape newlines
            o = o.replace("\n", "\\n")
            return f'"{o}"'
        else:
            return json.dumps(o, sort_keys=True)

    def iterencode(self, o, **kwargs):
        """Required to also work with `json.dump`."""
        return self.encode(o)

    def _put_on_single_line(self, o):
        return (
            self._primitives_only(o)
            and len(o) <= self.MAX_ITEMS
            and len(str(o)) - 2 <= self.MAX_WIDTH
        )

    def _primitives_only(self, o: Union[list, tuple, dict]):
        if isinstance(o, (list, tuple)):
            return not any(isinstance(el, self.CONTAINER_TYPES) for el in o)
        elif isinstance(o, dict):
            return not any(isinstance(el, self.CONTAINER_TYPES) for el in o.values())

    @property
    def indent_str(self) -> str:
        return self.INDENTATION_CHAR * (self.indentation_level * self.indent)


def compute_grassmanninan_zsf(n, m, signature):
    """
    Refer to the appendix of cite:t:`azangulov2024a`,
    https://arxiv.org/pdf/2208.14960.pdf.
    """
    rank = min(m, n - m)
    
    signature = tuple([x for x in signature if x != 0])
    if len(signature) == 0:
        return [1], [[0] * rank]

    p = ZonalPol(rank, signature) # polynomial with rational coefficients
    coeffs = p.coeffs()
    denom_lcm = reduce(sympy.lcm, [r.q for r in coeffs])
    coeffs = [coef * denom_lcm for coef in coeffs]
    
    coeffs = list(map(int, coeffs))
    monoms = [list(map(int, monom)) for monom in p.monoms()]
    return coeffs, monoms


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                                                                                                                   #
#  Below are the settings and the script for calculating the character parameters and writing them in a JSON file.  #
#                                                                                                                   #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


if __name__ == "__main__":
    characters = {}
    if not recalculate:
        with get_resource_file_path("precomputed_characters.json") as storage_file_name:
            with open(storage_file_name, "r") as file:
                characters = json.load(file)

    for n in range(2, max_n + 1):
        for m in range(2, n-1):
            group_name = "Gr({},{})".format(n, m)
            print(group_name)
            eigenfunctions = GrassmannianEigenfunctions(None, n, m, order, compute_zsf=False)
            if recalculate or (not recalculate and group_name not in characters):
                characters[group_name] = {}
            for signature in tqdm(eigenfunctions._signatures):
                if str(signature) not in characters[group_name]:
                    sys.stdout.write("{}: ".format(str(signature)))
                    coeffs, monoms = compute_grassmanninan_zsf(
                        n, m, signature
                    )
                    print(coeffs, monoms)
                    characters[group_name][str(signature)] = (coeffs, monoms)

        with get_resource_file_path("precomputed_grassmanian_zsf.json") as storage_file_name:
            with open(storage_file_name, "w") as file:
                json.dump(characters, file, cls=CompactJSONEncoder)
