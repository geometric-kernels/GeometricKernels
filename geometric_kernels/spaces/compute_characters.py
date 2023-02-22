import json
from typing import Union

import sys

from lie_stationary_kernels.spaces.so import SOEigenfunction


class CompactJSONEncoder(json.JSONEncoder):
    """A JSON Encoder that puts small containers on single lines."""

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
            return '[{}]'.format(','.join(self.encode(el) for el in o))
        elif isinstance(o, dict):
            if o:
                if self._put_on_single_line(o):
                    return "{ " + ", ".join(f"{self.encode(k)}: {self.encode(el)}" for k, el in sorted(o.items())) + " }"
                else:
                    self.indentation_level += 1
                    output = [self.indent_str + f"{json.dumps(k)}: {self.encode(v)}" for k, v in sorted(o.items())]
                    self.indentation_level -= 1
                    return "{\n" + ",\n".join(output) + "\n" + self.indent_str + "}"
            else:
                return "{}"
        elif isinstance(o, float):  # Use scientific notation for floats, where appropiate
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
        return self._primitives_only(o) and len(o) <= self.MAX_ITEMS and len(str(o)) - 2 <= self.MAX_WIDTH

    def _primitives_only(self, o: Union[list, tuple, dict]):
        if isinstance(o, (list, tuple)):
            return not any(isinstance(el, self.CONTAINER_TYPES) for el in o)
        elif isinstance(o, dict):
            return not any(isinstance(el, self.CONTAINER_TYPES) for el in o.values())

    @property
    def indent_str(self) -> str:
        return self.INDENTATION_CHAR*(self.indentation_level*self.indent)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                                                                                                                   #
#  Below are the settings and the script for calculating the character parameters and writing them in a JSON file.  #
#                                                                                                                   #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# set to True to recalculate all characters, set to False to add to the already existing without recalculating
recalculate = False

storage_file_name = 'precomputed_characters.json'

groups = [
    ('SO', 3, SOEigenfunction),
    ('SO', 4, SOEigenfunction),
    ('SO', 5, SOEigenfunction),
    ('SO', 6, SOEigenfunction),
    ('SO', 7, SOEigenfunction),
    ('SO', 8, SOEigenfunction),
    # (SU, 2, SUCharacter),
    # (SU, 3, SUCharacter),
    # (SU, 4, SUCharacter),
    # (SU, 5, SUCharacter),
    # (SU, 6, SUCharacter),
]

# the number of representations to be calculated for each group
order = 20

characters = {}
if not recalculate:
    with open(storage_file_name, 'r') as file:
        characters = json.load(file)

for name, n, eigenfunctions_class in groups:
    group_name = '{}({})'.format(name.__name__, n)
    print(group_name)
    eigenfunctions = eigenfunctions_class(order, n, init_eigenfunctions=False)
    if recalculate or (not recalculate and group_name not in characters):
        characters[group_name] = {}
    for signature in eigenfunctions._signatures:
        if str(signature) not in characters[group_name]:
            sys.stdout.write('{}: '.format(str(signature)))
            coeffs, monoms = eigenfunctions._compute_character_formula(signature)
            print(coeffs, monoms)
            characters[group_name][str(signature)] = (coeffs, monoms)

with open(storage_file_name, 'w') as file:
    json.dump(characters, file, cls=CompactJSONEncoder)
# print(json.dumps(characters, cls=CompactJSONEncoder))