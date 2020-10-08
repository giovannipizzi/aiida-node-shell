import os
import sys
import argparse

import aiida

from . import AiiDANodeShell


def main():
    # TODO: change this, it's not the recommended way (things written on the command line are default commands)
    parser = argparse.ArgumentParser('aiida-node-shell')
    parser.add_argument('-p', '--profile', required=False, default=None, help="The profile to load")
    parser.add_argument('node_identifier', nargs='?')
    parsed = parser.parse_args()

    aiida.load_profile(parsed.profile)

    try:
        sys.argv = sys.argv[1:]
        shell = AiiDANodeShell(
            node_identifier=parsed.node_identifier,
            startup_script=os.path.expanduser('~/.aiidashellrc'))
    except Exception as exc:
        print("ERROR: {}: {}".format(exc.__class__.__name__, exc))
    else:

        while True:
            try:
                retcode = shell.cmdloop()
                print()
                sys.exit(retcode)
                break
            except KeyboardInterrupt:  # CTRL+C pressed
                # Ignore CTRL+C
                print()
                print()
