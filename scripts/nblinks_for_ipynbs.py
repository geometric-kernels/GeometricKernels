"""
This script creates .nblink files in the `--nblink-dir` for all the .ipynb
files in the `--ipynb-dir`. It recursively traverses the latter and recreates
the same folder structure in the former.

Created by Viacheslav Borovitskiy in 2024.
"""

import os
from argparse import ArgumentParser
from pathlib import Path

parser = ArgumentParser(
    prog="nblinks_to_ipynbs",
    description="""Create .nblink files for all the .ipynb
                        files, recursively traversing the directory with
                        .ipynb-s and recreating the same folder structure
                        in the destination directory.""",
    epilog="""If the current directory is the root directory of the library,
                the .ipynb files are located in the `./notebooks` directory,
                and you want to put the respective .nblink files into the
                `./docs/examples` directory, run
                `python docs/nblinks_for_ipynbs.py --ipynb-dir ./notebooks --nblink-dir ./docs/examples`
            """,
)
parser.add_argument(
    "--ipynb-dir",
    dest="ipynb_dir",
    help="Directory where the actual .ipynb files are stored.",
    required=True,
)
parser.add_argument(
    "--nblink-dir",
    dest="nblink_dir",
    help="Directory to put the .nblink files.",
    required=True,
)

args = parser.parse_args()

for root, subdirs, filenames in os.walk(args.ipynb_dir):
    # Filter out all the files/folders that start with . (are hidden) and
    # all the files without the .ipynb extension.
    filenames = [f for f in filenames if not f[0] == "." and f.endswith(".ipynb")]
    subdirs[:] = [d for d in subdirs if not d[0] == "."]

    # Modify filenames to include paths relative to the args.ipynb_dir.
    filenames = [
        os.path.relpath(os.path.join(root, f), args.ipynb_dir) for f in filenames
    ]

    for f in filenames:
        nblink_path = Path(os.path.join(args.nblink_dir, f)).with_suffix(".nblink")
        nblink_content = '{"path": "%s"}' % os.path.relpath(
            os.path.join(args.ipynb_dir, f), nblink_path.parent
        )
        print('Writing "%s" to %s.' % (nblink_content, nblink_path))
        os.makedirs(os.path.dirname(nblink_path), exist_ok=True)
        with open(nblink_path, "w") as nblink_file:
            nblink_file.write(nblink_content)
