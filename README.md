# aiida-node-shell

## Goal

A proof of concept of a possible future `verdi node shell` command.

This command helps users to easily browse AiiDA nodes
through an interactive, customisable shell with tab completion.

The main expected use-case is that of AiiDA users who want to quickly browse nodes,
and their attributes, extras and files in the repository.

This is facilitated by a very fast interface. Indeed:

- nodes are pre-loaded and multiple commands act by default on the "current node"
- tab completion is *very* fast (ms vs a few hundreds of ms in `verdi`, since the database environment
  and connections are setup only once when starting the shell.
- tab completion is more customisable, e.g. showing the type of incoming/outgoing links together with the
  value to complete when inspecting links (commands `in` and `out`), or can properly browse directory and files
  in the node repository
- a much more intuitive interface (even if more limited) than the python API, that needs to be
  complete but therefore necessarily more verbose

The goal is not to replace neither the python API nor the `verdi commands`, but to provide a faster browsing 
interface for a number of common tasks.

## Usage
First, install the required dependencies in you virtual environment: `pip install -r requirements.txt`.

Then just execute the script (`./node_shell.py`) if you want to use the default profile in your environment.
If you want to use a different profile, you can run with:
  ```
  verdi -p <MY_PROFILE> run node_shell.py
  ```

You will get a prompt. Use the `help` command to see all commands available, or the `-h` option of every command.

The guide below is a quick (but by no means complete) description of some useful commands.

If you want to preload a node, add its identifier as the first command line parameter.

## Quick start
Here we use some example data. Actual PKs and output will of course
differ in your DB.

The shell by default has no preloaded node, and the shell shows just the current profile name: `(default) `
(in this example, the profile is called `default`).

- Load a node, making it the current node: `load 1283` will load node with identifier 1283. 
  Identifiers use the same logic as verdi,
  [described here](https://aiida.readthedocs.io/projects/aiida-core/en/v1.0.1/verdi/verdi_user_guide.html#cli-identifiers)
  to specify PKs, UUIDs and/or labels.

  The prompt now also indicates the type and PK of the current node: `(CalcJobNode<1283>@default)` - Note that you can unload the node, if needed, with the `unload` command.

- Let's check the last modification time with the command `mtime`: we get

    ```text
    Last modified 3 hours, 6 minutes ago (2020-02-19 10:46:49.307814+00:00)
    ```

- Similar commands allow to inspect other similar node properties: `ctime`, `label`, `description`, `uuid`. You can also the same output that you would get with `verdi node show` (in bash) with the `show` command.

- Inspect attributes of the current node with:
  
  - `attrkeys` to get a list of the keys of all attributes of the current node
  - `attr <attr_key>` to get the value of the attribute with key `<attr_key>`. You can use tab completion that will also show the type of the possible matching attributes! E.g. typing `attr re<TAB>` you get:
    ```text
    ATTRIBUTE_KEY             Description
    remote_workdir            (str)                                                                               
    resources                 (dict)                                                                              
    retrieve_list             (list)                                                                              
    retrieve_singlefile_list  (list)                                                                              
    retrieve_temporary_list   (list)  
    ```

  - `attrs` to print all keys and values of all attributes

- Use similar commands for extras: `extrakeys`, `extra <extra_key>`, `extras`.

- Inspect all nodes that are incoming into the current node with the `in` command, showing the link type, the link label and the PK of the node. **Reminder**: you can jump to a node using the `load <PK>` command, making it easy to browse the graph. The output looks like:
    ```
    # Link 0 - INPUT_CALC (settings) -> 1282
    # Link 1 - INPUT_CALC (structure) -> 1211
    # Link 2 - INPUT_CALC (pseudo_Al) -> 1527
    # Link 3 - INPUT_CALC (code) -> 631
    # Link 4 - INPUT_CALC (parameters) -> 1281
    # Link 5 - INPUT_CALC (pseudo_Os) -> 1335
    # Link 6 - INPUT_CALC (kpoints) -> 1197
    ```

- Similarly, you can inspect outgoing nodes with `out`. Both for `in` and `out`, you can add a `-t` option to limit results to only one link type.
- Note that the links are labelled by numbers. If you pass `-l <link id>` argument to the `in` command, it will bring you to the node connected by the link.
- Browse the folders and files in the repository. You can check list files using `repo_ls`: 

    ```
    .aiida/
    _aiidasubmit.sh
    aiida.in
    out/
    pseudo/
    ```

  By default, folders are in bold and end with a slash. Options allow to change the behaviour or to explicitly show if it is a folder or a file.

- You can show the content of a subfolder simply adding the name (tab-completion available! Note that with `repo_ls`, only folders are completed, not files). For instance, `repo_ls .aiida/` gives:

    ```
    calcinfo.json
    job_tmpl.json
    ```

- Get the content of a file with `repo_cat <PATH>`; also in this case tab-completion works!

- The interesting feature is that you can then pipe the output into any bash command. E.g. you can do `repo_cat aiida.in | less` to paginate the output, or `repo_cat aiida.in | grep UPF` to filter lines, giving e.g.:
    ```
    Al     26.981538 Al.pbe-n-kjpaw_psl.1.0.0.UPF
    Os     190.23 Os.pbe-spfn-rrkjus_psl.1.0.0.UPF
    ```   
  You can even redirect to file on your computer using `repo_cat aiida.in > test.txt` (and you can then show the file with bash commands by prepending with `!`, see also below): `!ls -l test.txt`, or `!cat test.txt`.

- The report command prints the *report* for a process. It can be used for quick inspection of complex workchains.
- The `nodehist_show` command shows a list the visited nodes with theiry relationships,
  and you can quickly go back and forth using the `nodehist_backward` and `nodehist_forward` commands.

  ```text
  (ArrayData<30029>@demo) nodehist_show
  CalcJobNode<30036> MyStructure RELAX 
    🢁  ---  [structure] INPUT_CALC
  StructureData<30030>  
    🢁  ---  [structure] CREATE
  CalcJobNode<30025> MyStructure RELAX 
    🢁  ---  [CALL] CALL_CALC
  WorkChainNode<30023> MyStructure RELAX 
    🢃  ---  [energies] RETURN
  ArrayData<30029>  <-- We are here
  ```

- There are also a set of useful things that are enabled by the `cmd2` library:
  - prepend a command with `!` to run in bash, for instance `!ls` to list the files in the current directory
  - you can run python commands with `run_pyscript`, or prepended by `@` (you can check all shortcuts with the `shortcuts` command)
  - exit the shell using `CTRL+D` or typing `exit` or `quit`
  - you can create command aliases, e.g. with `alias create rls repo_ls` to have a shorter alias for the `repo_ls` command (check also the `macro` command to also define aliases with argument placeholders)
  - edit a file using the `edit` command
  - use arrow keys to get previous commands; moreover there is a `history` command, and `CTRL+R` to quickly find commands executed recently (like in bash)
  - Define a startup script to execute commands at every shell invocation. The current file that is read (if it exists) is `~/.aiidashellrc`. For instance, you could create a file containing
    
    ```
    alias create rls repo_ls
    alias create rcat repo_cat
    ```

  to have the `rls` and `rcat` commands always available.
  An example `.aiidashellrc` file is provided in the repository.

## Konwn issues

- The use of `verdi status` in the node shell without the RabbitMQ server running will cause CPU usage surge. This is probably due a spawned thread getting stuck in a loop.

## Current status of the code and feedback
I believe that the current code is already very useful, but it must be considered a first draft, to allow people to test
the functionality and provide feedback.
In particular, commands might be added, renamed or removed at any time, and their interface might change.

Any feedback (additional commands, changes to their name or interface, changes to their parameters, 
improvements on the tab-completion, ...) are highly welcome. Please use GitHub issues for this (or make PRs).

Once we get the feeling that enough people are interested and we converge on a preliminary interface,
we will stabilise it releasing a 1.0 version and not do backward-incompatible changes without changing also
the major version number. We might also possibly evaluate its inclusion in aiida-core 
(e.g. as a `verdi node shell` command).
