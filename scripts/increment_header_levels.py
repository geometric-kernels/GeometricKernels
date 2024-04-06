"""
This script takes a Jupyter notebook, and turns
# -> ##
## -> ###
### -> ####
...

Usage:
> increment_header_levels.py notebook.ipynb

Created by Viacheslav Borovitskiy in 2024, based on the add_toc.py script
by davide.gerbaudo@gmail.com.
"""

import re
import sys
from collections import namedtuple

import nbformat

Header = namedtuple("Header", ["level", "name"])


def increment_header_levels(nb_name, skip_title):
    RE = re.compile(r"(?:^|\n)(?P<level>#{1,6})(?P<header>(?:\\.|[^\\])*?)#*(?:\n|$)")
    nb = nbformat.read(nb_name, as_version=4)
    title_skipped = not skip_title
    for cell in nb.cells:
        if cell.cell_type == "markdown":
            new_source = cell.source
            offset = 0
            for m in RE.finditer(cell.source):
                print("---", m.group("header"))
                if not title_skipped:
                    print("Skipping title...")
                    title_skipped = True
                    continue
                header_start = m.start("header") + offset
                new_source = new_source[:header_start] + "#" + new_source[header_start:]
                offset += 1
            cell.source = new_source
    nbformat.write(nb, nb_name)


if __name__ == "__main__":
    nb_name = sys.argv[1]
    skip_title = len(sys.argv) >= 3 and sys.argv[2] == "--skip-title"
    increment_header_levels(nb_name, skip_title)
