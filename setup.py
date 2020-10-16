"""Install the aiida-node-shell package."""
import os
import io

try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup

MODULENAME = "aiida_node_shell"

# Get the version number in a dirty way
FOLDER = os.path.split(os.path.abspath(__file__))[0]
FNAME = os.path.join(FOLDER, MODULENAME, "__init__.py")
with open(FNAME) as init:
    # Get lines that match, remove comment part
    # (assuming it's not in the string...)
    VERSIONLINES = [
        l.partition("#")[0] for l in init.readlines() if l.startswith("__version__")
    ]
if len(VERSIONLINES) != 1:
    raise ValueError("Unable to detect the version lines")
VERSIONLINE = VERSIONLINES[0]
VERSION = VERSIONLINE.partition("=")[2].replace('"', "").strip()

setup(
    name=MODULENAME,
    description="A proof of concept of a `verdi node shell` command.",
    url="http://github.com/giovannipizzi/aiida-node-shell",
    license="The MIT license",
    author="Giovanni Pizzi, Bonan Zhu",
    version=VERSION,
    install_requires=["cmd2", "pytz", "ago", "click"],
    python_requires=">=3.6",
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "aiida-node-shell=aiida_node_shell.cmdline:main",
        ],
    }
)
