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
from functools import reduce
import more_itertools
import sympy
from beartype.typing import Union
from sympy.matrices.determinant import _det as sp_det
from tqdm import tqdm
from jackpy.jack import ZonalPol
from sympy import Poly, Rational
from sympy.functions.special.gamma_functions import RisingFactorial
from collections import defaultdict

from geometric_kernels.spaces.so import SOEigenfunctions
from geometric_kernels.spaces.su import SUEigenfunctions  # noqa
from geometric_kernels.spaces.grassmannian import GrassmannianEigenfunctions, Grassmannian
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


def is_sub_partition(sigma, kappa):
    len_sigma = len(sigma)
    len_kappa = len(kappa)
    max_len = max(len_sigma, len_kappa)

    for i in range(max_len):
        s_i = sigma[i] if i < len_sigma else 0
        k_i = kappa[i] if i < len_kappa else 0
        if s_i > k_i:
            return False

    return True


def hypergeometric_coeff(a_parameter, sigma, rank):
    result = sympy.Rational(1, 1)
    sigma_padded = list(sigma) + [0] * (rank - len(sigma))
    for i in range(rank):
        term_for_RisingFactorial = a_parameter - sympy.Rational(1,2)*i
        result *= RisingFactorial(term_for_RisingFactorial, sigma_padded[i])
    return result


def rho(kappa):
    return sum([k*(k-i-1) for i, k in enumerate(kappa)])


def generate_sub_partitions(kappa):
    kappa_len = len(kappa)

    if kappa_len == 0:
        return [tuple()]

    results = []
    current_sigma_parts = [0] * kappa_len

    def _build_recursively(part_idx, prev_part_max_val):
        if part_idx == kappa_len:
            results.append(tuple(current_sigma_parts))
            return

        current_part_upper_bound = min(prev_part_max_val, kappa[part_idx])

        for val in range(current_part_upper_bound, -1, -1):
            current_sigma_parts[part_idx] = val
            _build_recursively(part_idx + 1, val)

    _build_recursively(0, kappa[0])
    results = [tuple(x for x in result if x != 0) for result in results]
    return results


def shift_by_1(poly):
    vars = poly.gens
    subs_dict = {x: x + 1 for x in vars}
    poly_shifted = poly.subs(subs_dict)
    poly_shifted = sympy.Poly(sympy.expand(poly_shifted.as_expr()))
    return poly_shifted


def get_homogeneous_part(poly_expr, x_vars_tuple, degree_to_extract):
    if degree_to_extract < 0:
        return Rational(0, 1)

    if not isinstance(poly_expr, sympy.Expr):
        poly_expr = sympy.sympify(poly_expr)

    if poly_expr.is_zero:
        return Rational(0, 1)

    poly_obj = sympy.Poly(poly_expr, *x_vars_tuple, domain=sympy.QQ)

    homogeneous_part_expr = Rational(0, 1)

    if poly_obj.is_zero:
        if degree_to_extract == 0 and poly_expr.is_constant():
             return poly_expr
        return Rational(0, 1)


    for monom_powers_tuple, coeff in poly_obj.terms():
        current_monom_total_degree = sum(monom_powers_tuple)

        if current_monom_total_degree == degree_to_extract:
            term_expr = Rational(coeff)
            for i, power in enumerate(monom_powers_tuple):
                if power > 0: # Only multiply if power is positive
                    term_expr *= x_vars_tuple[i]**power
            homogeneous_part_expr += term_expr

    return homogeneous_part_expr


class GrassmanianZSF:
    def __init__(self, n, m):
        self.n = n
        self.m = m
        self.rank = min(m, n-m)
        self.vars = tuple(sympy.Symbol(f'x_{i+1}') for i in range(self.rank))
        self.generalized_binomial_coeffs = {}
        self.c_kappa_sigma = {}

    def calculate_C_star(self, kappa):
        if len(kappa) == 0:
          return Poly(sympy.Rational(1,1), self.vars, domain='QQ')

        C_kappa = ZonalPol(self.rank, kappa)
        C_kappa_star = C_kappa/C_kappa.eval([Rational(1, 1)]*self.rank)
        poly_gens = C_kappa.gens
        C_kappa_star = Poly(C_kappa_star, *poly_gens)
        return C_kappa_star

    def calculate_generalized_binomial_coeffs(self, kappa):
        if self.generalized_binomial_coeffs.get(kappa, None) is not None:
            return
        C_kappa_star = self.calculate_C_star(kappa)
        shifted_C_kappa_star = shift_by_1(C_kappa_star)

        LHS = shifted_C_kappa_star
        LHS = LHS.as_expr().expand()
        all_sigmas_ = generate_sub_partitions(kappa)
        all_sigmas = defaultdict(list)
        for sigma in all_sigmas_:
          all_sigmas[sum(sigma)].append(sigma)

        all_sigmas = [(degree, all_sigmas[degree]) for degree
                  in sorted(all_sigmas.keys(), reverse=True)]

        found_coeffs = defaultdict(lambda: sympy.Rational(0, 1))

        for degree, sigmas in all_sigmas:
            lhs_s_part_expr = get_homogeneous_part(LHS, self.vars, degree)
            C_sigmas = [self.calculate_C_star(sigma) for sigma in sigmas]
            if len(sigmas) > 0:

                coeff_symbols = [sympy.Symbol(f"b_{idx}", domain='QQ') for idx in range(len(sigmas))]

                # Equation to make zero: sum(coeff_symbols[i] * C_stars_exprs[i]) - lhs_s_part_expr = 0
                equation_to_solve = sympy.Rational(0, 1)
                for i, sym_b in enumerate(coeff_symbols):
                    equation_to_solve += sym_b * C_sigmas[i]

                equation_to_solve = (equation_to_solve - lhs_s_part_expr).as_expr().expand()

                linear_system_equations = []
                if equation_to_solve != 0:
                    poly_eq = sympy.Poly(equation_to_solve, *self.vars)
                    if not poly_eq.is_zero:
                        for _, coeff_expr_in_b in poly_eq.terms(): # Coeffs are expressions in b_symbols
                            linear_system_equations.append(coeff_expr_in_b)

                solved_b_values_dict = {}
                if linear_system_equations and coeff_symbols:
                    try:
                        solution = sympy.solve(linear_system_equations, *coeff_symbols, dict=True)
                        if solution: # sympy.solve returns list of dicts
                            solved_b_values_dict = solution[0] if isinstance(solution, list) else solution
                    except Exception:
                      pass
                sum_expr_to_subtract = sympy.Rational(0, 1)
                for i, sig in enumerate(sigmas):
                    val = solved_b_values_dict.get(coeff_symbols[i], sympy.Rational(0, 1))
                    found_coeffs[sig] = val
                    sum_expr_to_subtract += val * C_sigmas[i] # Use the full C_sigma*

                if sum_expr_to_subtract != 0:
                    poly_to_subtract = sympy.Poly(sum_expr_to_subtract, *self.vars)
                    LHS = (LHS - poly_to_subtract).as_expr().expand()

        self.generalized_binomial_coeffs[kappa]=found_coeffs

    def calculate_c_kappa_sigma(self, kappa, sigma):
      if kappa not in self.c_kappa_sigma:
        self.c_kappa_sigma[kappa] = {}

      self.calculate_generalized_binomial_coeffs(kappa)
      if self.generalized_binomial_coeffs[kappa][sigma] == 0:
        self.c_kappa_sigma[kappa][sigma] = sympy.Rational(0, 1)
        return

      if kappa == sigma:
        self.c_kappa_sigma[kappa][sigma] = sympy.Rational(1, 1)
        return
      if self.c_kappa_sigma[kappa].get(sigma, None) is not None:
        return
      if not is_sub_partition(sigma, kappa):
        return
      coeff = sympy.Rational(0, 1)
      for i in range(len(sigma)+1):
        sigma_i = list(sigma)
        if len(sigma) == 0:
          sigma_i = (1,)
        elif i == 0:
          sigma_i[0] += 1
        elif i == len(sigma):
          sigma_i.append(1)
        elif sigma_i[i-1] > sigma_i[i]:
          sigma_i[i] += 1
        else:
          continue
        sigma_i = tuple(sigma_i)
        if is_sub_partition(sigma_i, kappa):
          self.calculate_c_kappa_sigma(kappa, sigma_i)

          self.calculate_generalized_binomial_coeffs(sigma_i)

          coeff += (self.c_kappa_sigma[kappa][sigma_i] *
                    self.generalized_binomial_coeffs[kappa][sigma_i] *
                    self.generalized_binomial_coeffs[sigma_i][sigma])

      coeff /= (((sum(kappa) - sum(sigma))* sympy.Rational(self.n, 2) + rho(kappa) - rho(sigma))
      *self.generalized_binomial_coeffs[kappa][sigma])

      self.c_kappa_sigma[kappa][sigma] = coeff

    def compute_P(self, kappa):
      self.calculate_generalized_binomial_coeffs(kappa)
      sigmas = generate_sub_partitions(kappa)
      P_kappa_expr = sympy.Rational(0, 1)

      for sigma in sigmas:
          degree = sum(sigma)
          sign = -1 if (degree % 2 == 1) else 1
          binom_coeff = self.generalized_binomial_coeffs[kappa][sigma]
          a = sympy.Rational(self.rank, 2)
          a_sigma = hypergeometric_coeff(a, sigma, self.rank)
          if binom_coeff == 0 or a_sigma == 0:
              continue

          self.calculate_c_kappa_sigma(kappa, sigma)
          P_kappa_expr += (sign * binom_coeff / a_sigma *
                           self.c_kappa_sigma[kappa][sigma]
                           * self.calculate_C_star(sigma))

      P_kappa_expr = P_kappa_expr/P_kappa_expr.eval(
          [sympy.Rational(1,1)]*self.rank)
      return sympy.Poly(P_kappa_expr.as_expr().expand())


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


def compute_grassmanninan_zsf(n, m, signature, grassmanian_zsf):
    """
    Refer to the appendix of cite:t:`azangulov2024a`,
    https://arxiv.org/pdf/2208.14960.pdf.
    """
    print(signature)
    rank = min(m, n - m)
    signature = tuple([x//2 for x in signature if x != 0])
    if len(signature) == 0:
        return [1], [[0] * rank]

    p = grassmanian_zsf.compute_P(signature) #ZonalPol(rank, signature) # polynomial with rational coefficients
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
        with get_resource_file_path("precomputed_grassmanian_zsf.json") as storage_file_name:
            with open(storage_file_name, "r") as file:
                characters = json.load(file)

    for n in range(2, max_n + 1):
        for m in range(2, n-1):
            group_name = "Gr({},{})".format(n, m)
            print(group_name)
            grassmanian = Grassmannian(n, m)
            eigenfunctions = GrassmannianEigenfunctions(grassmanian, order, compute_zsf=False)
            grassmanina_zsf = GrassmanianZSF(n, m)
            if recalculate or (not recalculate and group_name not in characters):
                characters[group_name] = {}
            for signature in tqdm(eigenfunctions._signatures):
                if str(signature) not in characters[group_name]:
                    sys.stdout.write("{}: ".format(str(signature)))
                    coeffs, monoms = compute_grassmanninan_zsf(
                        n, m, signature, grassmanina_zsf
                    )
                    print(coeffs, monoms)
                    characters[group_name][str(signature)] = (coeffs, monoms)

        with get_resource_file_path("precomputed_grassmanian_zsf.json") as storage_file_name:
            with open(storage_file_name, "w") as file:
                json.dump(characters, file, cls=CompactJSONEncoder)
