"""
Standalone script to precompute characters for
:class:`~.spaces.SpecialOrthogonal` and :class:`~.spaces.SpecialUnitary`.

Edit `recalculate`, `storage_file_name`, `order`, and `groups` variables below
in the code and run the script as.

.. code-block:: bash

    python compute_characters.py
"""

import itertools
import json
import sys

import more_itertools
import sympy
from beartype.typing import Union
from sympy.matrices.determinant import _det as sp_det
from tqdm import tqdm

from geometric_kernels.spaces.so import SOEigenfunctions
from geometric_kernels.spaces.su import SUEigenfunctions  # noqa
from geometric_kernels.utils.utils import (
    get_resource_file_path,
    partition_dominance_cone,
    partition_dominance_or_subpartition_cone,
)

# set to True to recalculate all characters, set to False to add to the already existing without recalculating
recalculate = False

storage_file_name = "../spaces/precomputed_characters.json"

# the number of representations to be calculated for each group
order = 25

groups = [
    ("SO", 3, SOEigenfunctions),
    ("SO", 4, SOEigenfunctions),
    ("SO", 5, SOEigenfunctions),
    ("SO", 6, SOEigenfunctions),
    ("SO", 7, SOEigenfunctions),
    ("SO", 8, SOEigenfunctions),
    ("SO", 9, SOEigenfunctions),
    ("SO", 10, SOEigenfunctions),
    ("SU", 2, SUEigenfunctions),
    ("SU", 3, SUEigenfunctions),
    ("SU", 4, SUEigenfunctions),
    ("SU", 5, SUEigenfunctions),
    ("SU", 6, SUEigenfunctions),
    ("SU", 7, SUEigenfunctions),
    ("SU", 8, SUEigenfunctions),
    ("SU", 9, SUEigenfunctions),
    ("SU", 10, SUEigenfunctions),
]


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


def compute_character_formula_so(self, signature):
    """
    Refer to the appendix of cite:t:`azangulov2024a`,
    https://arxiv.org/pdf/2208.14960.pdf.
    """
    n = self.n
    rank = self.rank
    gammas = sympy.symbols(" ".join("g{}".format(i + 1) for i in range(rank)))
    gammas = list(more_itertools.always_iterable(gammas))
    gammas_conj = sympy.symbols(" ".join("gc{}".format(i + 1) for i in range(rank)))
    gammas_conj = list(more_itertools.always_iterable(gammas_conj))
    chi_variables = gammas + gammas_conj
    if n % 2:
        gammas_sqrt = sympy.symbols(" ".join("gr{}".format(i + 1) for i in range(rank)))
        gammas_sqrt = list(more_itertools.always_iterable(gammas_sqrt))
        gammas_conj_sqrt = sympy.symbols(
            " ".join("gcr{}".format(i + 1) for i in range(rank))
        )
        gammas_conj_sqrt = list(more_itertools.always_iterable(gammas_conj_sqrt))
        chi_variables = gammas_sqrt + gammas_conj_sqrt

        def xi1(qs):
            mat = sympy.Matrix(
                rank,
                rank,
                lambda i, j: gammas_sqrt[i] ** qs[j] - gammas_conj_sqrt[i] ** qs[j],
            )
            return sympy.Poly(sp_det(mat, method="berkowitz"), chi_variables)

        # qs = [sympy.Integer(2*pk + 2*rank - 2*k - 1) / 2 for k, pk in enumerate(signature)]
        qs = [2 * pk + 2 * rank - 2 * k - 1 for k, pk in enumerate(signature)]
        # denom_pows = [sympy.Integer(2*k - 1) / 2 for k in range(rank, 0, -1)]
        denom_pows = [2 * k - 1 for k in range(rank, 0, -1)]
        numer = xi1(qs)
        denom = xi1(denom_pows)
    else:

        def xi0(qs):
            mat = sympy.Matrix(
                rank,
                rank,
                lambda i, j: gammas[i] ** qs[j] + gammas_conj[i] ** qs[j],
            )
            return sympy.Poly(sp_det(mat, method="berkowitz"), chi_variables)

        def xi1(qs):
            mat = sympy.Matrix(
                rank,
                rank,
                lambda i, j: gammas[i] ** qs[j] - gammas_conj[i] ** qs[j],
            )
            return sympy.Poly(sp_det(mat, method="berkowitz"), chi_variables)

        qs = [
            pk + rank - k - 1 if k != rank - 1 else abs(pk)
            for k, pk in enumerate(signature)
        ]
        pm = signature[-1]
        numer = xi0(qs)
        if pm:
            numer += (1 if pm > 0 else -1) * xi1(qs)
        denom = xi0(list(reversed(range(rank))))
    partition = tuple(map(abs, signature)) + tuple([0] * self.rank)
    monomials_tuples = itertools.chain.from_iterable(
        more_itertools.distinct_permutations(p)
        for p in partition_dominance_or_subpartition_cone(partition)
    )
    monomials_tuples = filter(
        lambda p: all(p[i] == 0 or p[i + rank] == 0 for i in range(rank)),
        monomials_tuples,
    )
    monomials_tuples = list(monomials_tuples)
    monomials = [
        sympy.polys.monomials.Monomial(m, chi_variables).as_expr()
        for m in monomials_tuples
    ]
    chi_coeffs = list(
        more_itertools.always_iterable(
            sympy.symbols(
                " ".join("c{}".format(i) for i in range(1, len(monomials) + 1))
            )
        )
    )
    exponents = [n % 2 + 1] * len(
        monomials
    )  # the correction s.t. chi is the same polynomial for both oddities of n
    chi_poly = sympy.Poly(
        sum(c * m**d for c, m, d in zip(chi_coeffs, monomials, exponents)),
        chi_variables,
    )
    pr = chi_poly * denom - numer
    if n % 2:
        pr = sympy.Poly(
            pr.subs((g * gc, 1) for g, gc in zip(gammas_sqrt, gammas_conj_sqrt)),
            chi_variables,
        )
    else:
        pr = sympy.Poly(
            pr.subs((g * gc, 1) for g, gc in zip(gammas, gammas_conj)), chi_variables
        )
    sol = list(sympy.linsolve(pr.coeffs(), chi_coeffs)).pop()
    if n % 2:
        chi_variables = gammas + gammas_conj
        chi_poly = sympy.Poly(
            chi_poly.subs(
                [gr**2, g]
                for gr, g in zip(gammas_sqrt + gammas_conj_sqrt, chi_variables)
            ),
            chi_variables,
        )
    p = sympy.Poly(
        chi_poly.subs((c, c_val) for c, c_val in zip(chi_coeffs, sol)), chi_variables
    )
    coeffs = list(map(int, p.coeffs()))
    monoms = [list(map(int, monom)) for monom in p.monoms()]
    return coeffs, monoms


def compute_character_formula_su(self, signature):
    """
    Refer to the appendix of cite:t:`azangulov2024a`,
    https://arxiv.org/pdf/2208.14960.pdf.
    """
    n = self.n
    gammas = sympy.symbols(" ".join("g{}".format(i) for i in range(1, n + 1)))
    qs = [pk + n - k - 1 for k, pk in enumerate(signature)]
    numer_mat = sympy.Matrix(n, n, lambda i, j: gammas[i] ** qs[j])
    numer = sympy.Poly(sp_det(numer_mat, method="berkowitz"))
    denom = sympy.Poly(
        sympy.prod(
            gammas[i] - gammas[j] for i, j in itertools.combinations(range(n), r=2)
        )
    )
    monomials_tuples = list(
        itertools.chain.from_iterable(
            more_itertools.distinct_permutations(p)
            for p in partition_dominance_cone(signature)
        )
    )
    monomials = [
        sympy.polys.monomials.Monomial(m, gammas).as_expr() for m in monomials_tuples
    ]
    chi_coeffs = list(
        more_itertools.always_iterable(
            sympy.symbols(
                " ".join("c{}".format(i) for i in range(1, len(monomials) + 1))
            )
        )
    )
    chi_poly = sympy.Poly(sum(c * m for c, m in zip(chi_coeffs, monomials)), gammas)
    pr = chi_poly * denom - numer
    sol = list(sympy.linsolve(pr.coeffs(), chi_coeffs)).pop()
    p = sympy.Poly(sum(c * m for c, m in zip(sol, monomials)), gammas)
    coeffs = list(map(int, p.coeffs()))
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

    for name, n, eigenfunctions_class in groups:
        group_name = "{}({})".format(name, n)
        print(group_name)
        eigenfunctions = eigenfunctions_class(n, order, compute_characters=False)
        if recalculate or (not recalculate and group_name not in characters):
            characters[group_name] = {}
        for signature in tqdm(eigenfunctions._signatures):
            if str(signature) not in characters[group_name]:
                sys.stdout.write("{}: ".format(str(signature)))
                if isinstance(eigenfunctions, SOEigenfunctions):
                    coeffs, monoms = compute_character_formula_so(
                        eigenfunctions, signature
                    )
                elif isinstance(eigenfunctions, SUEigenfunctions):
                    coeffs, monoms = compute_character_formula_su(
                        eigenfunctions, signature
                    )
                print(coeffs, monoms)
                characters[group_name][str(signature)] = (coeffs, monoms)

    with get_resource_file_path("precomputed_characters.json") as storage_file_name:
        with open(storage_file_name, "w") as file:
            json.dump(characters, file, cls=CompactJSONEncoder)
