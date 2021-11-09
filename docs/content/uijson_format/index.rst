UI.JSON Format
==============

About
^^^^^

The **ui.json** format provides a User Interface (UI) between geoh5py and `Geoscience ANALYST Pro
<http://www.mirageoscience.com/our-products/software-product/geoscience-analyst>`_. The file structure is built on json objects, each representing a parameter that is used in a python script. In general only a label and value is required, however as outline below there are many types of input and dependencies that can be drawn on thoughout the file. On output from Geoscience ANALYST, the value and whether the parameter is enabled will be updated or added to each json. Extra entries in the json are allowed and are ignored, but written out by Geoscience ANALYST. In general, entries will be put in order that they are set in the json. The exception is data parameters that depend on object parameters. Placing those parameters in the same group will ensure that they are close in the UI.


Entries
^^^^^^^
Within the file, the following objects could be present:

run_command ``str``
    Name of python script (i.e., "run_me" for run_me.py) required for Geoscience ANALYST Pro to run on save or auto-load.
conda_environment ``str``
    Optional name of conda environment to activate when running the python script in *run_command*
title ``str``
    Optional title of user interface window

Each json object with the following members become a parameter in the user interface. The members ``label`` and ``value`` are required. The members that can be given to all parameter objects are:

label ``str``
    Required string describing parameter. A colon will automatically be added within Geoscience ANALYST, so this should be omitted.
value ``str``, ``int``, ``bool`` , or ``float``
    Required empty or a unique identifier of a workspace object. The value is updated when written from Geoscience ANALYST.
main ``bool``
    If set to true, the parameter is shown in the first tab and will throw an error if not given and not optional. Optional parameters may be set to main. When main is not given or is false, the parameter will be under the *Optional Parameters* tab.
tooltip ``str``
   String describing the parameter in detail that appears when the mouse hovers over it.
visible ``bool``
    *true* or *false* on whether the parameter is visible. Default is true (visible).
optional ``bool``
    *true* or *false* on whether the parameter is optional. On output, check if *enabled* is set to true.
enabled ``bool``
    *true* or *false* if the parameter is enabled. Default is true.
group ``str``
    Name of the group to which the parameter belongs.
groupOptional ``bool``
    If true, adds a checkbox the entire group name. The group parameters will be disabled if not checked.
dependency ``str``
    The name of the object of which this object is dependent upon. The dependency parameter should be optional or boolean parameter (i.e., has a checkbox).
dependencyType ``str``
    What happens when the dependency entry is checked. Options are ``enabled`` or ``disabled``
groupDependency ``str``
    The name of the group of which this object is dependent upon. The dependency group should be optional (ie ``groupOptional``). The group should not be the parameter's group.
groupDependencyType ``str``
    What happens when the dependency group is checked. Options are ``enabled`` or ``disabled``


Parameters
^^^^^^^^^^
There are other entries that may be required based on the parameter type. The following sections define different parameter types that can be found in the **ui.json** format.

 .. toctree::
   :maxdepth: 1

   json_objects.rst


Exporting from Geoscience ANALYST
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
When a **ui.json** is saved with Geoscience ANALYST Pro, members of entries are updated. This allows a Python program to both create and read the same **ui.json** with ease. The following object members are updated or written:

- The **value** member of each entry is updated with the appropriate type
- The **enabled** member ``bool`` for whether the parameter is enabled
- In the case of the :ref:`Data parameter <data_parameter>`, **isValue** and **property** will be updated if given. Use the **isValue** ``bool`` member to figure out if you should use the **value** (true) or **property** (false) members when dissecting the output json.

The following json objects will be written (and overwritten if given) upon export from Geoscience ANALYST Pro:

- monitoring_directory ``str`` the absolute path of a monitoring directory. Workspace files written to this folder will be automatically processed by Geoscience ANALYST.
- workspace_geoh5 ``str`` the absolute path to the current workspace (if previously saved) being used
- geoh5 ``str`` the absolute path to the geoh5 written containing all the objects of the workspace within the parameters of the **ui.json**. One only needs to use this workspace along with the json file to access the objects with geoh5py.


Tips on creating UIs
^^^^^^^^^^^^^^^^^^^^
Here are a few tips on creating good looking UIs:

- Keep labels short and concise. Be consistent with capitalization and do not include the colons. Geoscience ANALYST will add colons and align them.
- Tooltips are great
- Utilize **optional** object members and dependencies. Objects without a valid value will invoke an error when exporting or running from Geoscience ANALYST. Truly optional objects, should be optional. If a single workspace object input is optional, use the :ref:`Object parameter <object_parameter>` rather than two parameters with a dependency.
- Group related objects, but do not use a group if there are fewer than 3 objects.
- The **main** member is for general, required parameters. Do not include this member with every object, unless there are only a handful of objects. "Non-main" members are designated to a second page under *Optional parameters*.


External Links
^^^^^^^^^^^^^^

-  `C++ JSON Library <https://github.com/nlohmann/json>`_
-  `C++ JSON value types <https://nlohmann.github.io/json/doxygen/classnlohmann_1_1basic__json_ac68cb65a7f3517f0c5b1d3a4967406ad.html#ac68cb65a7f3517f0c5b1d3a4967406ad>`_
