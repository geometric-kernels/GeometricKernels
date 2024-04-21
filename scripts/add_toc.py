"""
This script takes a Jupyter notebook, and adds a table of contents at the top.

The TOC is added in a new markdown cell.
The items appearing in the TOC are all the markdown headers of the notebook.

Usage:
> add_toc.py notebook.ipynb

Created by davide.gerbaudo@gmail.com in 2020. Modified by Viacheslav
Borovitskiy in 2024.
"""

import os
import re
import sys
from collections import namedtuple

import nbformat
from nbformat.v4.nbbase import new_markdown_cell

TOC_COMMENT = "<!--TABLE OF CONTENTS-->\n"


Header = namedtuple("Header", ["level", "name"])


def is_toc_comment(cell):
    return cell.source.startswith(TOC_COMMENT)


def collect_headers(nb_name):
    headers = []
    RE = re.compile(r"(?:^|\n)(?P<level>#{1,6})(?P<header>(?:\\.|[^\\])*?)#*(?:\n|$)")
    nb = nbformat.read(nb_name, as_version=4)
    for cell in nb.cells:
        if is_toc_comment(cell):
            continue
        elif cell.cell_type == "markdown":
            for m in RE.finditer(cell.source):
                header = m.group("header").strip()
                level = m.group("level").strip().count("#")
                headers.append(Header(level, header))
                print(level * "  ", "-", header)
    return headers


def write_toc(nb_name, headers, top_level):
    nb = nbformat.read(nb_name, as_version=4)
    nb_file = os.path.basename(nb_name)

    def format(header):
        indent = (header.level - 1) * (2 * " ")
        name = header.name
        anchor = "#" + name.replace(" ", "-")
        if header.level <= top_level:
            name = f"**{name}**"
        result = f"{indent}- [{name}]({anchor})"
        return result

    toc = TOC_COMMENT
    toc += "## Contents\n"
    toc += "\n".join([format(h) for h in headers if h.level <= top_level + 1])

    first_cell = nb.cells[0]
    if is_toc_comment(first_cell):
        print("- amending toc for {0}".format(nb_file))
        first_cell.source = toc
    else:
        print("- inserting toc for {0}".format(nb_file))
        nb.cells.insert(0, new_markdown_cell(source=toc))
    nbformat.write(nb, nb_name)


if __name__ == "__main__":
    nb_name = sys.argv[1]
    headers = collect_headers(nb_name)

    title_level = headers[0].level
    if [header.level for header in headers].count(title_level) > 1:
        top_level = title_level
    else:
        top_level = title_level + 1

    if len(sys.argv) >= 3 and sys.argv[2] == "--skip-title":
        del headers[0]  # this is typically the title of the notebook
    write_toc(nb_name, headers, top_level)
