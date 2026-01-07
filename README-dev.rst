Setup for development
=====================

.. contents:: Table of Contents
   :local:
   :depth: 2

.. _Poetry: https://python-poetry.org/docs/

After you have cloned the Git repository, you will need to:
    1. set up `Poetry`_
    2. create a virtual environment for development, where to install the dependencies
       of  the project
    3. execute the tests
    4. setup Git LFS if needed
    5. configure the pre-commit hooks for static code analysis and auto-formatting
    6. configure the Python IDE (PyCharm)

Installing Poetry
^^^^^^^^^^^^^^^^^
`Poetry`_ makes it easy to install the dependencies and start a virtual environment.

Poetry can be installed using a Python installation on the system, or one from a conda environment.

- To install Poetry from PowerShell with Python in your system path:

..  code-block:: ps1

    (Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | py -

- Or to install from an activated conda environment:

..  code-block:: bash

    curl -sSL https://install.python-poetry.org | python -

Either way, the installer creates a Poetry wrapper in a well-known, platform-specific directory
(``%APPDATA%\Python\Scripts`` on Windows). If this directory is not present in your PATH,
you can add it in order to invoke Poetry as ``poetry``.

You can confirm that Poetry is now found from your PATH by running ``poetry --version``.

For the next step, you have two choices:
    - tell Poetry to use a Conda environment (more convenient to
    - let Poetry create a virtual environment for the project.

Prefer the Conda environment if you want to test your project in conjunction with another application,
as Conda provides a solution for end users, while Poetry primarily manages the development environment.

..  note::
    You can have several environments for different versions of Python,
    (e.g. one for Python 3.10 and another one for Python 3.11) and work with single project
    location on disk in all these environments.


Option 1: an environment created by Conda
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
You can use Conda to create an environment that will then be used by Poetry to install
the project dependencies.

..  warning::
    Poetry ties a unique Python environment to each combination of Python version X.Y
    and project folder: you cannot have two environments for the same Python version and
    the same project location on disk.

    As a consequence, once Poetry has created a virtual environment or recorded a Conda
    environment for a given project location, you will not be able to use a different
    Conda environment with the same Python version for that project location.

First, create a new conda environment for the desired version of Python (3.10 in this example):

.. code-block:: bash

    conda create -n my-env python=3.10

Then, tell Poetry to use this environment for the project.
Simply **active the conda environment** before running ``poetry install``:

.. code-block:: bash

    conda activate my-env
    poetry install

Poetry automatically detects the conda environment and uses it for the project.

Finally, confirm that Poetry is using the correct Python executable by running the following command:

.. code-block:: bash

    poetry run where python
    #> C:\...\envs\my-env\python.exe

And also confirm that the package is installed in the conda environment:

.. code-block:: bash

    conda list my-app
    #> packages in environment at C:\...\envs\my-env:
    #> my-app                    0.1.0                    pypi_0    pypi


.. note::
    To install without development dependencies, use the following command instead

    .. code-block:: bash

        poetry install --without=dev


Option 2: a virtual environment created by Poetry
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
If you prefer not to use Conda, Poetry will create its own virtual environment for the project.

Simply run the following command from the project folder:

.. code-block:: bash

    poetry install

This will use the Python executable that was used to install Poetry.
If the version of that Python executable is compatible with the project,
skip to the next section, else read on.

**To specify a different version of Python to Poetry**, you need that version
of Python to be installed on the system. The easiest way to do that is
probably to use Conda, and create a new environment with the desired version of Python.

Here is an example that creates a conda environment for Python 3.10,
and prints out the path to the corresponding Python executable:

.. code-block:: bash

    conda create -n py310 python=3.10
    conda run -n py310 where python
    #> C:\...\envs\py310\python.exe

Then, tell Poetry to create the project environment using a Python executable
of the desired version (replace path from example with the actual one):

.. code-block:: bash

    poetry env use C:\...\envs\py310\python.exe

Specify and install the project dependencies
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Update `pyproject.toml`_ with the desired packages.
Dependencies for development and testing should be added to the section ``[tool.poetry.group.dev.dependencies]``.

Install the dependencies by executing the following command from the project folder:

.. code-block:: bash

    poetry install

If you update a dependency in ``pyproject.toml``, or wish Poetry to resolve again dependencies
to the latest compatible versions, tell Poetry to update its .lock file and install:

.. code-block:: bash

    poetry lock
    poetry install

To execute a module from you can either use the ``poetry run`` command:

.. code-block:: bash

    poetry run python -m my-app

or you can activate the virtual environment and run the module directly:

.. code-block:: bash

    poetry shell
    python -m my-app


..  note::
    Add dependencies with Poetry from command line.

    The following command, at once, inserts a dependency into ``pyproject.yaml``,
    updates the ``poetry.lock`` file, and installs the new dependency in the environment:

    .. code-block:: bash

        poetry add some-dependency

    With a specific version:

    .. code-block:: bash

        poetry add some-other-dependency==1.2.3

    For a development dependency:

    .. code-block:: bash

        poetry add -G dev some-dev-dependency


Run the tests
^^^^^^^^^^^^^^^^^
Test files are placed under the ``tests`` folder. Inside this folder and sub-folders,
Python test files are to be named with ``_test.py`` as a suffix.


To execute the tests, run the following command:

.. code-block:: bash

    poetry run pytest


Code coverage with Pytest
^^^^^^^^^^^^^^^^^^^^^^^^^
.. _pytest-cov: https://pypi.org/project/pytest-cov/

When installing the environment with ``poetry``, `pytest-cov`_ gets installed
as specified in ``pyproject.toml``.
It allows you to visualize the code coverage of your tests.
You can run the tests from the console with coverage:

.. code-block:: bash

    poetry run pytest --cov --cov-report html

Or if the Poetry environment is activated, simply:

.. code-block:: bash

    pytest --cov --cov-report html

The html report is generated in the folder ``htmlcov`` at the root of the project.
You can then explore the report by opening ``index.html`` in a browser.

In ``pyproject.toml``, the section ``[tool.coverage.report]`` defines the common options
for the coverage reports. The minimum accepted percentage of code coverage is specified
by the option ``fail_under``.

The section ``[tool.coverage.html]`` defines the options specific to the HTML report.


Git LFS
^^^^^^^
In the case your package requires large files, `git-lfs`_ can be used to store those files.
Copy it from the `git-lfs`_ website, and install it.

Then, in the project folder, run the following command to install git-lfs:

.. code-block:: bash

    git lfs install


It will update the file ``.gitattributes`` with the list of files to track.

Then, add the files and the ``.gitattributes`` to the git repository, and commit.

.. _git-lfs: https://git-lfs.com/

Then, add the files to track with git-lfs:

.. code-block:: bash

    git lfs track "*.desire_extension"


Configure the pre-commit hooks
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`pre-commit`_ is used to automatically run static code analysis upon commit.
The list of tools to execute upon commit is configured in the file `.pre-commit-config.yaml`_.

pre-commit can be installed using a Python installation on the system, or one from a conda environment.

- To install pre-commit using Python (and pip) in your system path:

..  code-block:: bash

    pip install --user pre-commit

- Or to install from an activated conda environment:

..  code-block:: bash

    conda install -c conda-forge pre-commit

Then, in either way, install the pre-commit hooks as follow (**current directory is the project folder**):

..  code-block:: bash

    pre-commit install

To run pre-commit manually, use the following command:

..  code-block:: bash

    pre-commit run --all-files

If any error occurs, it might be caused by an obsolete versions of the tools that pre-commit is trying to execute.
Try the following command to update them:

..  code-block:: bash

    pre-commit autoupdate

Upon every commit, all the pre-commit checks run automatically for you, and reformat files when required. Enjoy...

If you prefer to run pre-commit upon push, and not upon every commit, use the following commands:

..  code-block:: bash

    pre-commit uninstall -t pre-commit
    pre-commit install -t pre-push

.. _pre-commit: https://pre-commit.com/


IDE : PyCharm
^^^^^^^^^^^^^
`PyCharm`_, by JetBrains, is a very good IDE for developing with Python.

Configure the Python interpreter in PyCharm
-------------------------------------------
For PyCharm to offer code completion, and to run tests from the IDE,
make sure to specify the Python interpreter.

..  note:: If Poetry is in the ``PATH``, PyCharm will offer automatically to configure the environment with Poetry
    when a ``pyproject.toml`` file is present at the root of the project.

In PyCharm settings, open ``File > Settings``, go to ``Python Interpreter``,
and add click add interpreter (at the top left):

    ..  image:: docs/images/readme/pycharm-add_Python_interpreter.png
        :alt: PyCharm: Python interpreter settings
        :align: center
        :width: 80%

For an environment created by Poetry
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Select ``Poetry Environment``, ``Existing environment``,
navigate to the Poetry installation folder, and select the ``python.exe`` file:

    ..  image:: docs/images/readme/pycharm-set_Poetry_Python_as_interpreter.png
        :alt: PyCharm: Set Python from Poetry environment as interpreter
        :align: center
        :width: 80%

On Windows, Poetry typically creates the virtual environment for the project under
``%LOCALAPPDATA%\pypoetry\Cache\virtualenvs\[the-project-name-with-some-suffix]\Script``).

You can also find this location by running from the command line:

.. code-block:: bash

    poetry env info

For a conda environment
~~~~~~~~~~~~~~~~~~~~~~~
Select ``Conda Environment``, ``Use existing environment``,
and select the desired environment from the list:

    ..  image:: docs/images/readme/pycharm-set_conda_env_as_interpreter.png
        :alt: PyCharm: Set conda environment as interpreter
        :align: center
        :width: 80%

Then you can check the list of installed packages in the ``Packages`` table. You should see
**my-app** and its dependencies. Make sure to turn off the ``Use Conda Package Manager``
option to see also the packages installed with Poetry through pip:

    ..  image:: docs/images/readme/pycharm-list_all_conda_packages.png
        :alt: PyCharm: Conda environment packages
        :align: center
        :width: 80%


Mark the sources and tests folder in PyCharm
--------------------------------------------
First, right click on the ``my_app`` folder and select ``Mark Directory as > Sources Root``:

Then, right click on the ``tests`` folder and select ``Mark Directory as > Test Sources Root``:

    ..  image:: docs/images/readme/pycharm-mark_directory_as_tests.png
        :alt: PyCharm: Add Python interpreter
        :align: center
        :width: 40%


Run the tests from PyCharm
--------------------------
After you have marked the ``tests`` folder as the test root, you can start tests with a right click on
the ``tests`` folder and select ``Run 'pytest in tests'``, or select the folder and just hit ``Ctrl+Shift+F10``.

PyCharm will nicely present the test results and logs:

    ..  image:: docs/images/readme/pycharm-test_results.png
        :alt: PyCharm: Run tests
        :align: center
        :width: 80%


Execute tests with coverage from PyCharm
----------------------------------------
You can run the tests with a nice report of the code coverage, thanks to the pytest-cov plugin
(already installed in the virtual environment as development dependency as per `pyproject.toml`_).


To set up this option in PyCharm, right click on the ``tests`` folder and ``Modify Run Configuration...``,
then add the following option in the ``Additional Arguments`` field:

    ..  image:: docs/images/readme/pycharm-menu_modify_test_run_config.png
        :alt: PyCharm tests contextual menu: modify run configuration
        :width: 30%

    ..  image:: docs/images/readme/pycharm-dialog_edit_test_run_config.png
        :alt: PyCharm dialog: edit tests run configuration
        :width: 60%

Select ``pytest in tests``, and add the following option in the ``Additional Arguments`` field::

    --cov --cov-report html

Then, run the tests as usual, and you will get a nice report of the code coverage.

.. note::
    Running tests with coverage disables the debugger, so breakpoints will be ignored.

Some useful plugins for PyCharm
-------------------------------
Here is a suggestion for some plugins you can install in PyCharm.

- `Toml`_, to edit and validate ``pyproject.toml`` file.
- `IdeaVim`_, for Vim lovers.
- `GitHub Copilot`_, for AI assisted coding.

.. _PyCharm: https://www.jetbrains.com/pycharm/

.. _Toml: https://plugins.jetbrains.com/plugin/8195-toml/
.. _IdeaVim: https://plugins.jetbrains.com/plugin/164-ideavim/
.. _GitHub Copilot: https://plugins.jetbrains.com/plugin/17718-github-copilot

.. _pyproject.toml: pyproject.toml
.. _.pre-commit-config.yaml: .pre-commit-config.yaml

Build the docs
--------------

To build the api docs using autodocs

.. code-block:: bash

  sphinx-apidoc -o source/ ../geoh5py -t docs/templates


Copyright
^^^^^^^^^
Copyright (c) 2020-2026 Mira Geoscience Ltd.
