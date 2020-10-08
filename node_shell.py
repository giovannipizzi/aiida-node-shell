#!/usr/bin/env runaiida
"""
This is a proof of concept of a `verdi node shell` command to easily browse AiiDA nodes
through an interactive, customisable shell with tab completion.

Expected use-case: quick browsing of node attributes, extras and files in the repository,
with faster interface (node pre-loaded, very fast tab-completion, ...) than `verdi` and
a much more intuitive interface (even if more limited) than the python API.

More details in the README.md file.

Authors: 
  Giovanni Pizzi, EPFL
  Bonan Zhu, UCL

Version: 0.1
"""
import cmd2
from cmd2 import ansi
import functools
from datetime import datetime
import pytz
import io
import sys
import os
import re
import shlex
from traceback import print_exc
from pprint import pformat
from collections import namedtuple
import contextlib
import argparse

from ago import human
import click

from cmd2.utils import basic_complete
from aiida.common.links import LinkType
from aiida.cmdline.commands.cmd_verdi import verdi
from aiida.orm.utils.repository import FileType
from aiida.orm import GroupTypeString
# Examples for autocompletion: https://github.com/python-cmd2/cmd2/blob/master/examples/tab_autocompletion.py

LINK_TYPES_DICT = {
    link_type.value.upper(): link_type
    for link_type in LinkType
}

LINK_TYPES = sorted(LINK_TYPES_DICT.keys())


def with_default_argparse(f):
    """Decorator to use the default argparse parser (e.g. provide -h)."""
    return cmd2.with_argparser(cmd2.Cmd2ArgumentParser())(f)


def now_aware():
    """Get the current datetime object, aware of the timezone (in UTC)."""
    return datetime.utcnow().replace(tzinfo=pytz.utc)


def needs_node(f):
    """Decorator to add to a command if it needs to have a node loaded to work.
    
    It performs validation that the node is there, and prints a suitable error if this is not the case."""
    @functools.wraps(f)
    def wrapper(self, *args, **kwds):  # pylint: disable=inconsistent-return-statements
        """Print an error and return if there is no node loaded."""
        if not self._current_node:  # pylint: disable=protected-access
            self.poutput(
                "ERROR: No node loaded - load a node with `load NODE_IDENTIFIER` first"
            )
            return
        return f(self, *args, **kwds)

    return wrapper


def needs_new_node(always=False):
    """Decorator to mark the command prefers to have the node reloaded.
    The node will be only be reloaded if it is an unsealed ProcessNode.
    """
    def _wrapper(f):
        @needs_node
        @functools.wraps(f)
        def wrapper(self, *args, **kwds):
            """Reload self._current_node"""
            from aiida.orm.utils.loaders import NodeEntityLoader
            from aiida.orm import ProcessNode
            if always:
                self.do_reload('')
            else:
                # Reloading only for unsealed process
                current_node = self._current_node
                if isinstance(current_node, ProcessNode):
                    if not current_node.is_sealed:
                        self.do_reload('')
            return f(self, *args, **kwds)

        return wrapper

    return _wrapper


LinkInfo = namedtuple('LinkInfo', ('direction', 'type', 'label'))
HistInfo = namedtuple('HistInfo', ('node', 'desc', 'linkinfo'))

yellow = functools.partial(ansi.style, fg=ansi.fg.bright_yellow)
blue = functools.partial(ansi.style, fg=ansi.fg.bright_blue)
green = functools.partial(ansi.style, fg=ansi.fg.bright_green)
red = functools.partial(ansi.style, fg=ansi.fg.bright_red)
cyan = functools.partial(ansi.style, fg=ansi.fg.cyan)


class NodeHist:
    """Holds a history of the nodes"""
    def __init__(self):
        """Initialise an entity to keep track of the node history"""
        self.node_history = []
        self.node_history_pointer = -1

    @property
    def current_node(self):
        """Current node"""
        return self.node_history[self.node_history_pointer].node

    def set_current(self, node, desc):
        """Set the current node"""
        if self.node_history and self.node_history[
                self.node_history_pointer].node.pk == node.pk:
            # Loading the current node - do nothing
            return
        # Otherwise we append the node to the history
        if self.node_history_pointer < -1:
            # Drop any 'future'
            self.node_history = self.node_history[:self.node_history_pointer +
                                                  1]
            self.node_history_pointer = -1
        linkinfo = self.get_link_to_previous(node)
        self.node_history.append(HistInfo(node, desc, linkinfo))

    def go_back(self):
        """Go backward in the history"""
        if -self.node_history_pointer > len(self.node_history) - 1:
            # Do nothing if there is no history
            return
        self.node_history_pointer -= 1

    def go_forward(self):
        """Go forward in the history"""
        if self.node_history_pointer < -1:
            self.node_history_pointer += 1

    def go_last(self):
        """Go forward in the history"""
        self.node_history_pointer = -1

    def show_hist(self, cmd=None):
        """Print the history of loaded nodes"""
        n_hist = len(self.node_history)
        current_pos = n_hist + self.node_history_pointer
        data_list = []
        # Define the indentation
        indent = ' | '
        link_indent = '  '
        for i, hist in enumerate(self.node_history):
            here_mark = yellow('<-- We are here') if i == current_pos else ''
            node_line = '{} {} {}'.format(hist.desc, hist.node.label,
                                          here_mark)
            if hist.linkinfo is not None:
                # User unicode symbols for direction
                if hist.linkinfo.direction == '<':
                    link_direction = red('  ▲  ')
                else:
                    link_direction = green('  ▼  ')
                link_line = '---  [{}] {}'.format(hist.linkinfo.label,
                                                  hist.linkinfo.type)
            else:
                link_direction = '  ✖  '
                link_line = ''
            # Only add link if there is a previous node
            if i > 0:
                link_line = link_indent + link_direction + cyan(link_line)
                data_list.append(indent + link_line)

            data_list.append(indent + node_line)
        if cmd is None:
            cmd2.ansi.style_aware_write(sys.stdout,
                                        '\n'.join(data_list) + '\n')
        else:
            cmd.poutput('\n'.join(data_list))

    def get_link_to_previous(self, current_node):
        """Get the link to the previous node"""
        from aiida.orm import QueryBuilder, Node
        from aiida.orm.utils.loaders import NodeEntityLoader

        try:
            previous_node = self.node_history[self.node_history_pointer].node
        except IndexError:
            # No previous node found so there is certain no link to show
            return

        # From previous to current
        direction = None
        q = QueryBuilder()
        q.append(Node, filters={'id': previous_node.pk})
        q.append(Node,
                 filters={'id': current_node.pk},
                 edge_project=['type', 'label'])
        res = q.all()
        # Sort by type, then label
        if res:
            direction = '>'
        else:
            # Now try from current to previous
            q = QueryBuilder()
            q.append(Node, filters={'id': current_node.pk})
            q.append(Node,
                     filters={'id': previous_node.pk},
                     edge_project=['type', 'label'])
            res = q.all()
            if res:
                direction = '<'
        # Stop here if no link is found
        if direction is None:
            return
        # Now we have a link
        res.sort(key=lambda x: (x[0], x[1]))
        ltype, llabel = res[0]
        return LinkInfo(direction, ltype.upper(), llabel)


class AiiDANodeShell(cmd2.Cmd):
    """An interactive shell to browse node and their properties (attributes, extras, repository files, ...)."""
    intro = 'AiiDA node shell. Type help or ? to list commands.\n'
    _current_node = None
    _current_profile = None
    _node_hist = NodeHist()

    def __init__(self, *args, node_identifier=None, **kwargs):
        """Initialise a shell, possibly with a given node preloaded."""
        from aiida.manage.configuration import get_config

        # If I can reach the AiiDA config directory, I set a history
        # In this way, the history is different if different
        # virtualenvs define a different AIIDA_PATH. Note that this is
        # still the same for all profiles in that virtualenv (which is
        # probably OK). Otherwise we would need to insert the profile
        # name in the history file to have a different history per profile,
        # but probably this is not needed.
        aiida_config_dir_path = get_config().dirpath
        if os.path.isdir(aiida_config_dir_path):
            kwargs['persistent_history_file'] = os.path.join(
                aiida_config_dir_path, 'node-shell-history.dat')

        super().__init__(*args, use_ipython=True, **kwargs)
        self.self_in_py = True
        self = cmd2.ansi.STYLE_TERMINAL

        if node_identifier:
            self._set_current_node(node_identifier)

    def _set_current_node(self, identifier):
        """Set a node from an identifier."""
        from aiida.orm.utils.loaders import NodeEntityLoader
        self._current_node = NodeEntityLoader.load_entity(identifier)

    @property
    def _current_profile_object(self):
        """Get the current AiiDA profile object, or None if not set."""
        from aiida.manage.configuration import get_config

        # Caching
        if not self._current_profile:
            self._current_profile = get_config().current_profile

        return self._current_profile

    @property
    def current_profile(self):
        """Return a string with the current profile name (or 'NO_PROFILE' if no profile is loaded)."""
        current_profile_object = self._current_profile_object
        if not current_profile_object:
            return "NO_PROFILE"
        return current_profile_object.name

    def _get_node_string(self, node=None):
        """Return a string representing the current node (to be used in the prompt)."""
        if node is None:
            node = self._current_node
        if node is None:
            return ''
        class_name = node.__class__.__name__
        identifier = node.pk
        return '{}<{}>'.format(class_name, identifier)

    @property
    def prompt(self):
        """Define a custom prompt string."""
        if self._current_node:
            return '({}@{}) '.format(self._get_node_string(),
                                     self.current_profile)
        return '({}) '.format(self.current_profile)

    load_parser = cmd2.Cmd2ArgumentParser()
    load_parser.add_argument('identifier',
                             help='Identifier for loading the node')

    @cmd2.with_argparser(load_parser)
    def do_load(self, arg):
        """Load a node in the shell, making it the 'current node'."""
        # When loading the node I reset the history
        self._set_current_node(arg.identifier)
        self._node_hist.set_current(self._current_node,
                                    self._get_node_string())

    @needs_node
    @with_default_argparse
    def do_reload(self, arg):
        """Reload the node in the shell"""
        self._set_current_node(self._current_node.pk)

    move_parser = cmd2.Cmd2ArgumentParser()
    move_parser.add_argument('--steps',
                             help='the number of steps to move',
                             type=int,
                             default=1)

    @cmd2.with_argparser(move_parser)
    def do_nodehist_backward(self, arg):
        """Back to the previous node"""
        for _ in range(arg.steps):
            self._node_hist.go_back()
        self._set_current_node(self._node_hist.current_node.pk)

    @cmd2.with_argparser(move_parser)
    def do_nodehist_forward(self, arg):
        """Go forward in the node history"""
        for _ in range(arg.steps):
            self._node_hist.go_forward()
        self._set_current_node(self._node_hist.current_node.pk)

    @with_default_argparse
    def do_nodehist_last(self, arg):
        """Go forward in the node history"""
        self._node_hist.go_last()
        self._set_current_node(self._node_hist.current_node.pk)

    @with_default_argparse
    def do_nodehist_show(self, arg):
        """Show node browsering history"""
        self._node_hist.show_hist()

    @needs_node
    @with_default_argparse
    def do_uuid(self, arg):  # pylint: disable=unused-argument
        """Show the UUID of the current node."""
        self.poutput(self._current_node.uuid)

    setter_args = cmd2.Cmd2ArgumentParser()
    setter_args.add_argument('--set', '-s', help='Set the property')

    @needs_new_node(always=True)
    @cmd2.with_argparser(setter_args)
    def do_label(self, arg):  # pylint: disable=unused-argument
        """Show the label of the current node."""
        if arg.set is not None:
            self.poutput("Replacing current label {} with {}".format(
                self._current_node.label, arg.set))
            self._current_node.label = arg.set
        else:
            self.poutput(self._current_node.label)

    @needs_new_node(always=True)
    @cmd2.with_argparser(setter_args)
    def do_description(self, arg):  # pylint: disable=unused-argument
        """Show the description of the current node."""
        if arg.set is not None:
            self.poutput("Replacing current description {} with {}".format(
                self._current_node.description, arg.set))
            self._current_node.description = arg.set
        else:
            self.poutput(self._current_node.description)

    @needs_node
    @with_default_argparse
    def do_ctime(self, arg):  # pylint: disable=unused-argument
        """Show the creation time of the current node."""
        ctime = self._current_node.ctime
        self.poutput("Created {} ({})".format(human(now_aware() - ctime),
                                              ctime))

    @needs_new_node(always=True)
    @with_default_argparse
    def do_mtime(self, arg):  # pylint: disable=unused-argument
        """Show the last modification time of the current node."""
        mtime = self._current_node.mtime
        self.poutput("Last modified {} ({})".format(human(now_aware() - mtime),
                                                    mtime))

    @needs_new_node(always=True)
    @with_default_argparse
    def do_extras(self, arg):  # pylint: disable=unused-argument
        """Show all extras of the current node (keys and values)."""
        extras = self._current_node.extras
        if not extras:
            self.poutput("No extras")
            return
        for key, val in extras.items():
            self.poutput('- {}: {}'.format(key, pformat(val)))

    report_args = cmd2.Cmd2ArgumentParser()
    report_args.add_argument('--levelname', '-l', type=str, default='REPORT')
    report_args.add_argument('--max-depth',
                             '-m',
                             type=int,
                             default=None,
                             help='Limit the number of levels to be printed')
    report_args.add_argument(
        '--indent-size',
        '-i',
        type=int,
        default=2,
        help='Set the number of spaces to indent each level by')

    @needs_new_node()
    @cmd2.with_argparser(report_args)
    def do_report(self, arg):  # pylint: disable=unused-argument
        """Show the report, if the node is a ProcessNode"""
        from aiida.cmdline.utils.common import get_calcjob_report, get_workchain_report, get_process_function_report
        from aiida.orm import CalcJobNode, WorkChainNode, CalcFunctionNode, WorkFunctionNode

        process = self._current_node
        if isinstance(process, CalcJobNode):
            self.poutput(get_calcjob_report(process))
        elif isinstance(process, WorkChainNode):
            self.poutput(
                get_workchain_report(process, arg.levelname, arg.indent_size,
                                     arg.max_depth))
        elif isinstance(process, (CalcFunctionNode, WorkFunctionNode)):
            self.poutput(get_process_function_report(process))
        else:
            self.poutput('Nothing to show for node type {}'.format(
                process.__class__))

    comment_show_parser = cmd2.Cmd2ArgumentParser()
    comment_show_parser.add_argument('--user',
                                     '-u',
                                     help='Filter by user email.')

    @cmd2.with_argparser(comment_show_parser)
    @needs_node
    def do_comment_show(self, arg):
        """Show and comment of a node"""
        from aiida.cmdline.commands.cmd_node import comment_show
        from aiida.orm import User
        from aiida.common.exceptions import NotExistent
        node = self._current_node
        if arg.user is not None:
            try:
                user = User.objects.get(email=arg.user)
            except NotExistent:
                self.poutput(
                    red('Error: ') +
                    'User {} does not exists'.format(arg.user))
                return
        else:
            user = None
        # Use the function directly
        with self.verdi_isolate():
            comment_show.callback(user=user, nodes=[node])

    comment_add_parse = cmd2.Cmd2ArgumentParser()
    comment_add_parse.add_argument('comment', help='Comment to be added')

    @cmd2.with_argparser(comment_add_parse)
    @needs_node
    def do_comment_add(self, arg):
        """Added comment to the current node"""
        content = arg.comment
        if content:
            self._current_node.add_comment(content)
            self.poutput('comment added to {}'.format(
                self._get_node_string(self._current_node)))

    def comment_add_rm_choices_method(self):
        """Method that returns all possible values for the 'extras' command, used for tab-completion."""
        return [comment.pk for comment in self._current_node.get_comments()]

    comment_remove_parser = cmd2.Cmd2ArgumentParser()
    comment_remove_parser.add_argument(
        'id',
        help='ID of the comment to be deleted',
        type=int,
        choices_method=comment_add_rm_choices_method)
    comment_remove_parser.add_argument('--force', '-f', action='store_true')

    @cmd2.with_argparser(comment_remove_parser)
    def do_comment_remove(self, arg):
        """Added comment to the current node"""
        from aiida.orm.comments import Comment
        from aiida.cmdline.utils import echo
        from aiida.common import exceptions
        comment, force = arg.id, arg.force
        with self.verdi_isolate():
            if not force:
                try:
                    click.confirm(
                        'Are you sure you want to remove comment<{}>'.format(
                            comment),
                        abort=True)
                except click.exceptions.Abort:
                    return
            try:
                Comment.objects.delete(comment)
            except exceptions.NotExistent as exception:
                echo.echo_error('failed to remove comment<{}>: {}'.format(
                    comment, exception))
            else:
                echo.echo_success('removed comment<{}>'.format(comment))

    comment_update_parser = cmd2.Cmd2ArgumentParser()
    comment_update_parser.add_argument(
        'id',
        help='ID of the comment to be updated',
        type=int,
        choices_method=comment_add_rm_choices_method)
    comment_update_parser.add_argument('content',
                                       help='Conetnet of the comment')

    @cmd2.with_argparser(comment_update_parser)
    def do_comment_update(self, arg):
        """Update comment for a given comment ID"""
        from aiida.orm.comments import Comment
        from aiida.cmdline.utils import echo
        from aiida.common import exceptions
        with self.verdi_isolate():
            comment_id, content = arg.id, arg.content
            try:
                comment = Comment.objects.get(id=comment_id)
            except (exceptions.NotExistent, exceptions.MultipleObjectsError):
                echo.echo_error('comment<{}> not found'.format(comment_id))

            comment.set_content(content)

            echo.echo_success('comment<{}> updated'.format(comment_id))

    def extras_choices_method(self):
        """Method that returns all possible values for the 'extras' command, used for tab-completion."""
        completions_with_desc = []

        extras_keys = self._current_node.extras_keys()
        for key in extras_keys:
            completions_with_desc.append(
                cmd2.CompletionItem(
                    key, "({})".format(
                        type(self._current_node.get_extra(key)).__name__)))

        # Mark that we already sorted the matches
        # self.matches_sorted = True
        return completions_with_desc

    extrakey_parser = cmd2.Cmd2ArgumentParser()
    extrakey_parser.add_argument('extra_key',
                                 help='The extra key',
                                 choices_method=extras_choices_method)

    @needs_new_node(always=True)
    @cmd2.with_argparser(extrakey_parser)
    def do_extra(self, arg):
        """Show one extra of the current node."""
        extras = self._current_node.extras
        try:
            self.poutput('- {}: {}'.format(arg.extra_key,
                                           pformat(extras[arg.extra_key])))
        except KeyError:
            self.poutput("No extra with key '{}'".format(arg.extra_key))

    @needs_node
    @with_default_argparse
    def do_extrakeys(self, arg):  # pylint: disable=unused-argument
        """Show the keys of all extras of the current node."""
        extras_keys = self._current_node.extras_keys()
        if not extras_keys:
            self.poutput("No extras")
            return
        for key in sorted(extras_keys):
            self.poutput('- {}'.format(key))

    @needs_new_node()
    @with_default_argparse
    def do_attrs(self, arg):  # pylint: disable=unused-argument
        """Show all attributes (keys and values) of the current node."""
        attributes = self._current_node.attributes
        if not attributes:
            self.poutput("No attributes")
            return
        for key, val in attributes.items():
            self.poutput('- {}: {}'.format(key, pformat(val)))

    def attrs_choices_method(self):
        """Method that returns all possible values for the 'attrs' command, used for tab-completion."""
        completions_with_desc = []

        attributes_keys = self._current_node.attributes_keys()
        for key in attributes_keys:
            #if key.startswith(text):
            completions_with_desc.append(
                cmd2.CompletionItem(
                    key, "({})".format(
                        type(self._current_node.get_attribute(key)).__name__)))

        # Mark that we already sorted the matches
        # self.matches_sorted = True
        return completions_with_desc

    attrkey_parser = cmd2.Cmd2ArgumentParser()
    attrkey_parser.add_argument('attribute_key',
                                help='The attribute key',
                                choices_method=attrs_choices_method)

    @needs_new_node()
    @cmd2.with_argparser(attrkey_parser)
    def do_attr(self, arg):
        """Show one attribute of the current node."""
        attributes = self._current_node.attributes
        try:
            self.poutput('- {}: {}'.format(
                arg.attribute_key, pformat(attributes[arg.attribute_key])))
        except KeyError:
            self.poutput("No attribute with key '{}'".format(
                arg.attribute_key))

    @needs_new_node()
    @with_default_argparse
    def do_attrkeys(self, arg):  # pylint: disable=unused-argument
        """Show the keys of all attributes of the current node."""
        attributes_keys = self._current_node.attributes_keys()
        if not attributes_keys:
            self.poutput("No attributes")
            return
        for key in sorted(attributes_keys):
            self.poutput('- {}'.format(key))

    link_parser = cmd2.Cmd2ArgumentParser()
    link_parser.add_argument('-t',
                             '--link-type',
                             help='Filter by link type',
                             choices=LINK_TYPES)

    link_parser.add_argument('-l',
                             '--follow-link-id',
                             help='Follow this link to the next node',
                             type=int)

    @needs_new_node()
    @cmd2.with_argparser(link_parser)
    def do_in(self, arg):
        """List all nodes connected with incoming links to the current node. 
        
        The command also allows to filter by link type."""
        type_filter_string = ''
        if arg.link_type is None:
            incomings = self._current_node.get_incoming().all()
        else:
            incomings = self._current_node.get_incoming(
                link_type=LINK_TYPES_DICT[arg.link_type]).all()
            type_filter_string = ' of type {}'.format(arg.link_type)
        incomings.sort(
            key=lambda x: (x.link_type.value, x.link_label, x.node.pk))
        if not incomings:
            self.poutput("No incoming links{}".format(type_filter_string))
            return
        if arg.follow_link_id is None:
            for ilink, incoming in enumerate(incomings):
                self.poutput(
                    yellow("Link # {} ".format(ilink)) +
                    "- {} ({}) -> {}".format(
                        incoming.link_type.value.upper(), incoming.link_label,
                        self._get_node_string(incoming.node)))
        else:
            if arg.follow_link_id >= len(incomings) or arg.follow_link_id < 0:
                self.poutput("Error: invalid link id {}".format(
                    arg.follow_link_id))
                return

            next_pk = incomings[arg.follow_link_id].node.pk
            self._set_current_node(next_pk)
            self._node_hist.set_current(self._current_node,
                                        self._get_node_string())

    @needs_new_node()
    @cmd2.with_argparser(link_parser)
    def do_out(self, arg):
        """List all nodes connected with outgoing links to the current node. 
        
        The command also allows to filter by link type."""
        type_filter_string = ''
        if arg.link_type is None:
            outgoings = self._current_node.get_outgoing().all()
        else:
            outgoings = self._current_node.get_outgoing(
                link_type=LINK_TYPES_DICT[arg.link_type]).all()
            type_filter_string = ' of type {}'.format(arg.link_type)

        outgoings.sort(
            key=lambda x: (x.link_type.value, x.link_label, x.node.pk))
        if not outgoings:
            self.poutput("No outgoing links{}".format(type_filter_string))
            return
        if arg.follow_link_id is None:
            for ilink, outgoing in enumerate(outgoings):
                self.poutput(
                    yellow("Link # {} ".format(ilink)) +
                    "- {} ({}) -> {}".format(
                        outgoing.link_type.value.upper(), outgoing.link_label,
                        self._get_node_string(outgoing.node)))
        else:
            if arg.follow_link_id >= len(outgoings) or arg.follow_link_id < 0:
                self.poutput("Error: invalid link id {}".format(
                    arg.follow_link_id))
                return
            next_pk = outgoings[arg.follow_link_id].node.pk
            self._set_current_node(next_pk)
            self._node_hist.set_current(self._current_node,
                                        self._get_node_string())

    @needs_new_node(always=True)
    @with_default_argparse
    def do_show(self, arg):  # pylint: disable=unused-argument
        """Show textual information on the current node."""
        from aiida.cmdline.utils.common import get_node_info

        self.poutput(get_node_info(self._current_node))

    def repo_ls_completer_method(self, text, line, begidx, endidx):
        """Method to perform completion for the 'repo_ls' command.
        
        This implements custom logic to make sure it works properly for folders and ignores files."""
        folder, _, start = text.rpartition('/')
        first_level_matching = [
            folder + ('/' if folder else '') + obj.name + '/'
            for obj in self._current_node.list_objects(folder)
            if obj.name.startswith(start) and obj.file_type == FileType.DIRECTORY
        ]
        # I go down one level more to have proper completion of folders
        # without being too expensive for arbitrary-length recursion
        matching = []
        for first_level_folder in first_level_matching:
            matching.append(first_level_folder)
            matching.extend([
                first_level_folder + '/' + obj.name + '/'
                for obj in self._current_node.list_objects(first_level_folder)
                if obj.file_type == FileType.DIRECTORY
            ])

        return self.delimiter_complete(text,
                                       line,
                                       begidx,
                                       endidx,
                                       delimiter='/',
                                       match_against=matching)

    def repo_cat_completer_method(self, text, line, begidx, endidx):
        """Method to perform completion for the 'repo_ls' command.
        
        This implements custom logic to make sure it works properly both for folders and for files."""
        folder, _, start = text.rpartition('/')
        first_level_matching = [
            folder + ('/' if folder else '') + obj.name +
            ('/' if obj.file_type == FileType.DIRECTORY else '')
            for obj in self._current_node.list_objects(folder)
            if obj.name.startswith(start)
        ]
        matching = []
        for first_level_object in first_level_matching:
            matching.append(first_level_object)
            if first_level_object.endswith('/'):
                matching.extend([
                    first_level_object + '/' + obj.name +
                    ('/' if obj.file_type == FileType.DIRECTORY else '')
                    for obj in self._current_node.list_objects(
                        first_level_object)
                ])

        return self.delimiter_complete(text,
                                       line,
                                       begidx,
                                       endidx,
                                       delimiter='/',
                                       match_against=matching)

    ls_parser = cmd2.Cmd2ArgumentParser()
    ls_parser.add_argument(
        '-l',
        '--long',
        action='store_true',
        help=
        "Show additional information on each object, e.g. if it's a file or a folder"
    )
    ls_parser.add_argument('-s',
                           '--no-trailing-slashes',
                           action='store_true',
                           help="Do not show trailing slashes for folders")
    ls_parser.add_argument('PATH',
                           nargs='?',
                           default='.',
                           help="The path to list",
                           completer_method=repo_ls_completer_method)

    @needs_node
    @cmd2.with_argparser(ls_parser)
    def do_repo_ls(self, arg):
        """List all files and folders in the repository of the current node."""
        with self.verdi_isolate():
            for obj in self._current_node.list_objects(arg.PATH):
                if arg.long:
                    click.secho('[{}] '.format(obj.file_type.name[0].lower()),
                                fg='blue',
                                bold=True,
                                nl=False)
                click.secho(obj.name,
                            nl=False,
                            bold=(obj.file_type == FileType.DIRECTORY))
                if arg.no_trailing_slashes:
                    click.secho("")
                else:
                    if obj.file_type == FileType.DIRECTORY:
                        click.secho("/")
                    else:
                        click.secho("")

    cat_parser = cmd2.Cmd2ArgumentParser()
    cat_parser.add_argument('PATH',
                            nargs='?',
                            default='.',
                            help="The path to the file to output",
                            completer_method=repo_cat_completer_method)
    cat_parser.add_argument('--byte',
                            '-b',
                            help='Open in byte mode',
                            action='store_true')

    @needs_node
    @cmd2.with_argparser(cat_parser)
    def do_repo_cat(self, arg):
        """Echo on screen the content of a file in the repository of the current node."""
        mode = 'rb' if arg.byte else 'r'
        try:
            content = self._current_node.get_object_content(arg.PATH,
                                                            mode=mode)
        except IsADirectoryError:
            self.perror("Error: '{}' is a directory".format(arg.PATH))
        except FileNotFoundError:
            self.perror("Error: '{}' not found if node repository".format(
                arg.PATH))
        else:
            if mode == 'rb':
                self.stdout.buffer.write(content)
            else:
                self.stdout.write(content)

    @with_default_argparse
    @needs_node
    def do_unload(self, arg):  # pylint: disable=unused-argument
        """Unload the node from the repository, making no node to be selected as the 'current node'."""
        self._current_node = None

    @with_default_argparse
    def do_exit_with_error(self, arg):  # pylint: disable=unused-argument
        """Exit the shell with a non-zero error code.
        
        This function is mostly for internal use."""
        self.exit_code = 1
        return True

    @with_default_argparse
    def do_exit(self, arg):  # pylint: disable=unused-argument
        """Exit the shell."""
        return True

    #def precmd(self, line):
    #    'To be implemented in case we want to manipulate the line'
    #    #line = line.lower()
    #    return line

    def get_verdi_completion(self, arg_pieces):
        """Call the click verdi completion command.

        This requires preparing the environment as reqired by click when going
        through bash completion.

        Note if you are looking into the code: don't look only in click, but also
        in the code mokey-patched by click-completion!
        https://github.com/click-contrib/click-completion/blob/master/click_completion/patch.py

        For this reason, for now I go via the BASH environment variables, that seems
        less prone to code changes in the immediate future.
        """
        # Additional environment variables
        added_env = {}
        # Use a TAB as a separator
        added_env['IFS'] = r"$'\t'"
        # Variable needed by click to perform tab completion
        added_env['_VERDI_COMPLETE'] = "complete-bash"

        # Prepare the pieces
        comp_words = "verdi " + " ".join(
            shlex.quote(arg_piece) for arg_piece in arg_pieces)
        pos = len(arg_pieces)
        added_env['COMP_WORDS'] = comp_words
        added_env['COMP_CWORD'] = str(pos)

        try:
            my_stdout = io.StringIO()
            with self.verdi_isolate(added_env, my_stdout):
                verdi.main(args=[], prog_name='verdi')
        except SystemExit:
            # SystemExit means the command-line tool finished as intented
            my_stdout.seek(0)
            pieces = my_stdout.read().split('\t')
            # Manual attempt to unescape... Not perfect, and might break in the future :-(
            # But for now I'm reverting what is done here:
            # https://github.com/click-contrib/click-completion/blob/6e08a5fa43149c822152d40c07e00be5ec2c5c7e/click_completion/core.py#L170
            # namely re.sub(r"""([\s\\"'()])""", r'\\\1', opt)
            # i.e. it's prepending a backslash to the following characters:
            # - a space-like character
            # - a backslash
            # - a double quote
            # - a single quote
            # open and closed brackets: ( )
            pieces = [
                re.sub(r"""\\([\s\\"'()])""", r'\1', piece) for piece in pieces
            ]
            return pieces
        except Exception:
            ## IGNORE EXCEPTIONS DURING COMPLETION
            # In any case, it'd be good if quickly fix this issue first:
            # https://github.com/aiidateam/aiida-core/issues/3815
            # (but I would still keep this logic here to ignore possible exceptions)
            return []
        return []

    def verdi_args_completer_method(self, text, line, begidx, endidx,
                                    arg_tokens):  # pylint: disable=too-many-arguments
        """Method to perform completion for the 'verdi' command.

        This pipes through the Bash completion via click."""
        match_against = self.get_verdi_completion(arg_tokens['args'])
        complete_vals = basic_complete(text, line, begidx, endidx,
                                       match_against)
        # This apparently happens if there is no completion
        # Actually in bash this sometimes triggers file completion; to check
        # if we want to investigate using self.path_complete here (but it does not
        # make sense in many cases, see e.g. 'verdi group list')
        if complete_vals == ['']:
            complete_vals = []
        return complete_vals

    verdi_complete_parser = cmd2.Cmd2ArgumentParser()
    verdi_complete_parser.add_argument(
        'args',
        nargs=argparse.REMAINDER,
        suppress_tab_hint=True,
        completer_method=verdi_args_completer_method)

    @cmd2.with_argparser(verdi_complete_parser)
    def do_verdi(self, arg):
        """Run verdi commands using the current profile.
        The argument will be passed to ``verdi`` as it is, except that {} will be 
        expanded to the currently loaded node's pk.
        You may reference other nodes in the load history using offsets in {}.
        For example, {-1} will be subsituted with the last loaded nodes' pk.
        """

        # Here I force verdi to use the current profile, otherwise the
        # command won't work for the node shell launched with non-default profile
        verdi_args = ['-p', self.current_profile]
        try:
            passed_args = [
                expand_node_subsitute(the_arg, self._node_hist)
                for the_arg in arg.args
            ]
        except RuntimeError as exc:
            self.poutput(str(exc))
            return

        if passed_args:
            # Print help for this command (not verdi)
            if passed_args[0] in ('-h', '--help', '-help'):
                self.poutput(AiiDANodeShell.do_verdi.__doc__)
            # Passing -p is not allowed, we can only use the current profile
            if passed_args[0] in ('-p', '--profile'):
                self.poutput(
                    red('Error: ') +
                    'Manual profile selection is not allowed in the node shell.'
                )
                self.poutput(
                    'To switch profile, relaunch the node shell with verdi -p <profile} run.'
                )
                self.poutput('Your current profile is: {}'.format(
                    green(self.current_profile)))
                return

        verdi_args.extend(passed_args)
        try:
            with self.verdi_isolate():
                verdi.main(args=verdi_args, prog_name='verdi')
        except SystemExit as exception:
            # SystemExit means the command-line tool finished as intented
            # No action needed
            pass
        except Exception as exception:
            # Catch all python exceptions raised during the verdi command execution
            self.poutput(
                'Verdi Command raised an exception {}'.format(exception))
            print_exc(file=self.stdout)

    group_belong_parser = cmd2.Cmd2ArgumentParser()
    group_belong_parser.add_argument('--with-description',
                                     '-d',
                                     help='Show also the decription',
                                     action='store_true')
    group_belong_parser.add_argument(
        '--with-count',
        '-c',
        help='Show also the number of nodes in each group',
        action='store_true')

    @needs_node
    @cmd2.with_argparser(group_belong_parser)
    def do_group_belong(self, args):
        """Command for list groups"""
        from aiida.orm import QueryBuilder, Node, Group
        from tabulate import tabulate

        q = QueryBuilder()
        q.append(Node, filters={'id': self._current_node.pk})
        q.append(Group, with_node=Node, project=['*'])

        projection_lambdas = {
            'pk': lambda group: str(group.pk),
            'label': lambda group: group.label,
            'type_string': lambda group: group.type_string,
            'count': lambda group: group.count(),
            'user': lambda group: group.user.email.strip(),
            'description': lambda group: group.description
        }

        table = []
        projection_header = ['PK', 'Label', 'Type string', 'User']
        projection_fields = ['pk', 'label', 'type_string', 'user']

        if args.with_count:
            projection_header.append('Node Count')
            projection_fields.append('count')

        if args.with_description:
            projection_header.append('Description')
            projection_fields.append('description')

        for (group, ) in q.all():
            table.append([
                projection_lambdas[field](group) for field in projection_fields
            ])

        self.poutput(tabulate(table, headers=projection_header))

    def do_cd(self, args):
        """Change directory"""
        path = args
        if not path:
            path = os.path.expanduser("~")
        os.chdir(path)

    complete_cd = cmd2.Cmd.path_complete

    @with_default_argparse
    def do_pwd(self, arg):
        """Return the current workding directory"""
        self.poutput(os.getcwd())

    @with_default_argparse
    def do_verdi_shell(self, _):
        """Enter an ipython shell simimlar to that launched by `verdi shell`
        """
        from cmd2.py_bridge import PyBridge

        def load_ipy(cmd2_app, py_bridge):
            """Embed an IPython shell in an environment that is restricted to only the variables in this function
            :param cmd2_app: instance of the cmd2 app
            :param py_bridge: a PyBridge
            """
            from aiida.cmdline.utils.shell import get_start_namespace
            from IPython import embed

            # Create a variable pointing to py_bridge
            exec("{} = py_bridge".format(cmd2_app.py_bridge_name))

            # Add node_shell variable pointing to this app
            exec("node_shell = cmd2_app")
            # Add current_node varible pointing to the current loaded node
            exec("current_node = cmd2_app._current_node")

            # Initialse the namespace - this is what used by `verdi shell`
            locals().update(get_start_namespace())
            _cnode_string = cmd2_app._get_node_string()

            # Delete these names from the environment so IPython can't use them
            del cmd2_app
            del py_bridge

            # Start ipy shell
            banner = (
                'Entering an embedded verdi shell. Type quit or <Ctrl>-d to exit.\n'
                'Run Python code from external files with: run filename.py\n')

            if _cnode_string:
                banner += 'The loaded node \'{}\' can be accessed with \'current_node\'\n'.format(
                    _cnode_string)

            embed(
                banner1=banner,
                exit_msg='Leaving verdi shell, back to the node shell',
                colors='Neutral',
            )

        if self.in_pyscript():
            self.perror(
                "Recursively entering interactive verdi shells is not allowed")
            return

        try:
            self._in_py = True
            new_py_bridge = PyBridge(self)
            load_ipy(self, new_py_bridge)
            return new_py_bridge.stop
        finally:
            self._in_py = False

    @contextlib.contextmanager
    def verdi_isolate(self, new_env_vars=None, custom_stdout=None):
        """A context manager that sets up the isolation for invoking of a
        command line tool. The sys.stdin and sys.sydout are temporarily redirected
        to self.stdout, self.stderr. This is useful for calling verdi commends that
        writes directly to stdout and stderr
        """
        old_stdout = sys.stdout
        old_env = os.environ.copy()

        if new_env_vars:
            for key, val in new_env_vars.items():
                os.environ[key] = val

        if custom_stdout:
            sys.stdout = custom_stdout
        else:
            sys.stdout = self.stdout

        try:
            yield
        finally:
            sys.stdout = old_stdout
            os.environ = old_env


def expand_node_subsitute(arg, hist_obj):
    """Expand the subsitution for node PK in the argument.
    {} expands to node_history[current_pos].pk
    {n} expands to node_history[current_pos + n].pk
    """
    regex = r"{([-\d]*)}"
    for occ in re.finditer(regex, arg):
        whole_str = occ.group(0)
        if occ.group(1):
            idx = int(occ.group(1))
        else:
            idx = 0
        pointer = hist_obj.node_history_pointer + idx
        try:
            node = hist_obj.node_history[pointer].node
        except IndexError:
            if hist_obj.node_history:
                how_many = ("are {} nodes".format(len(hist_obj.node_history))
                            if len(hist_obj.node_history) != 1 else
                            "is 1 node")
                raise RuntimeError(
                    'Invalid offset in argument "{}" for node history '
                    '(there {} in the history)'.format(arg, how_many))
            raise RuntimeError(
                'No node in node history, you need to load a node before using it in argument "{}"'
                .format(arg))

        # Replace the line
        else:
            arg = arg.replace(whole_str, str(node.pk))

    return arg


if __name__ == '__main__':
    import os

    # TODO: change this, it's not the recommended way (things written on the command line are default commands)
    try:
        node_identifier = sys.argv[1]
        sys.argv = sys.argv[1:]
    except IndexError:
        node_identifier = None

    try:
        shell = AiiDANodeShell(
            node_identifier=node_identifier,
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
