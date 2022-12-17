# Installation

There is a [quick way](#quick-setup-not-recommended) to get started, but we advise 
setting up a virtual environment and guide through the process in the section
[Proper Python setup with virtual environment
](#proper-python-setup-with-virtual-environment--recommended)

## Quick setup (**not recommended**)

If you just want to use the software, the easiest way is to run from your
system's command line in the root folder of the project

```shell
pip install --user .
```

This will install the current version into your local folder of third-party libraries. 
Note that pytorch_gum_uncertainty_propagation runs with **Python 
version 3.10**. Usage in any Python environment on your computer is then possible by

```python
import pytorch_gum_uncertainty_propagation
```

or, for example, for the custom activation function QuadLU:

```python
from pytorch_gum_uncertainty_propagation.modules import QuadLU
```

### Updating to the newest version

Updates can be installed via the same command

```shell
pip install --user .
```

## Proper Python setup with virtual environment  (**recommended**)

The setup described above allows the quick and easy use of
pytorch_gum_uncertainty_propagation, but it also has its downsides. 
When working with Python we should rather always work in so-called virtual 
environments, in which our project specific dependencies are satisfied without 
polluting or breaking other projects' dependencies and to avoid breaking all your 
dependencies in case of an update of our Python distribution.

### Set up a virtual environment

If you are not familiar with [Python virtual environments
](https://docs.python.org/3/glossary.html#term-virtual-environment) you can get the
motivation and an insight into the mechanism in the
[official docs](https://docs.python.org/3/tutorial/venv.html).

You have the option to set up pytorch_gum_uncertainty_propagation 
using the Python built-in tool 
`venv`. The commands differ slightly between [Windows
](#create-a-venv-python-environment-on-windows) and [Mac/Linux
](#create-a-venv-python-environment-on-mac--linux).

#### Create a `venv` Python environment on Windows

In your Windows PowerShell execute the following to set up a virtual environment in
a folder of your choice.

```shell
PS C:> cd C:\LOCAL\PATH\TO\ENVS
PS C:\LOCAL\PATH\TO\ENVS> py -3 -m venv gum_compliant_nn_unc_prop_venv
PS C:\LOCAL\PATH\TO\ENVS> gum_compliant_nn_unc_prop_venv\Scripts\activate
```

Proceed to [the next step
](#install-pytorch_gum_uncertainty_propagation-via-pip).

#### Create a `venv` Python environment on Mac & Linux

In your terminal execute the following to set up a virtual environment in a folder
of your choice.

```shell
$ cd /LOCAL/PATH/TO/ENVS
$ python3 -m venv gum_compliant_nn_unc_prop_venv
$ source gum_compliant_nn_unc_prop_venv/bin/activate
```

Proceed to [the next step
](#install-pytorch_gum_uncertainty_propagation-via-pip).

### Install pytorch_gum_uncertainty_propagation via `pip`

Once you activated your virtual environment, you can install
pytorch_gum_uncertainty_propagation via:

```shell
pip install .
```

```shell
Collecting pytorch_gum_uncertainty_propagation
[...]
Successfully installed pytorch_gum_uncertainty_propagation-[...] [...]
```

That's it!

### Optional Jupyter Notebook dependencies

If you are familiar with Jupyter Notebooks, you find some examples in the _src/examples_
subfolder of the source code repository. To execute these you need additional 
dependencies which you get by appending `[examples]` to
pytorch_gum_uncertainty_propagation in the above installation command, 
e.g.

```shell
(gum_compliant_nn_unc_prop_venv) $ pip install .[examples]
```

### Install known to work dependencies' versions

In case errors arise within pytorch_gum_uncertainty_propagation, 
the first thing you can try is installing the known to work configuration of 
dependencies against which we run our test suite. This you can easily achieve with 
our requirements file. This is done with the following sequence of commands after 
activating:

```shell
(gum_compliant_nn_unc_prop_venv) $ pip install --upgrade pip-tools
Collecting pip-tools
[...]
Successfully installed pip-tools-6.11.0
(gum_compliant_nn_unc_prop_venv) $ python --version
Python 3.10.7
(gum_compliant_nn_unc_prop_venv) $ python -m piptools sync requirements.txt 
requirements.txt
Collecting [...]
[...]
Successfully installed [...]
(gum_compliant_nn_unc_prop_venv) $
```
