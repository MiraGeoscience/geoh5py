UI.JSON Format
==============

About
^^^^^

The **ui.json** format provides a User Interface (UI) between geoh5py and `Geoscience ANALYST Pro
<http://www.mirageoscience.com/our-products/software-product/geoscience-analyst>`_. The file structure is built on an array of `JSON objects <https://JSON-schema.org/draft/2020-12/JSON-schema-core.html>`_, each representing a parameter that is used in a python script. An object contains members that control the style and behaviour of the UI. In general only a **label** and **value** member is required in each object, however as outlined below, there are many types of input and dependencies that can be drawn on throughout the file. On output from Geoscience ANALYST, the value and whether the parameter is enabled will be updated or added to each JSON. Extra objects in the JSON are allowed and are ignored, but written out by Geoscience ANALYST. In general, objects will be put in order that they are set in the JSON. The exception is data parameters that depend on object parameters. Placing those parameters in the same group will ensure that they are close in the UI.


Input Objects
^^^^^^^^^^^^^
Within the **ui.json** file, each JSON object with **value** and **label** members will be considered a parameter to the UI. The following JSON objects could also be present:

run_command ``str``
    Name of python script excluding the .py extension (i.e., "run_me" for run_me.py) required for Geoscience ANALYST Pro to run on save or auto-load.
conda_environment ``str``
    Optional name of conda environment to activate when running the python script in *run_command*
title ``str``
    Optional title of user interface window

Object Members
^^^^^^^^^^^^^^
Each JSON object with the following members become a parameter in the user interface. Each object must have the members ``label`` and ``value``. Each member will contribute to the appearence and behaviour within Geoscience ANALYST>. The possible members that can be given to all parameter objects are:

label ``str``
    Required string describing parameter. A colon will automatically be added within Geoscience ANALYST, so this should be omitted.
value ``str``, ``int``, ``bool`` , or ``float``
    This require member takes a different form, including empty, depending on the :ref:`parameter type <json_param_examples>`. The value is updated when written from Geoscience ANALYST.
main ``bool``
    If set to true, the parameter is shown in the first tab and will throw an error if not given and not optional. Optional parameters may be set to main. When main is not given or is false, the parameter will be under the *Optional Parameters* tab.
tooltip ``str``
   String describing the parameter in detail that appears when the mouse hovers over it.
optional ``bool``
    *true* or *false* on whether the parameter is optional. On output, check if *enabled* is set to true.
enabled ``bool``
    *true* or *false* if the parameter is enabled. The default is true. If a parameter is optional and not enabled, it will start as disabled (grey and inactive in the UI).
group ``str``
    Name of the group to which the parameter belongs. Adds a box and name around the parameters with the same case-sensitive group name.
groupOptional ``bool``
    If true, adds a checkbox in the top of the group box next to the name. The group parameters will be disabled if not checked. The initial statedpends on the **groupDependency** and **groupDependencyType** members and the **enabled** member of the group's parameters.
dependency ``str``
    The name of the object of which this object is dependent upon. The dependency parameter should be optional or boolean parameter (i.e., has a checkbox).
dependencyType ``str``
    What happens when the dependency member is checked. Options are ``enabled`` or ``disabled``
groupDependency ``str``
    The name of the object of which the group of the parameter is dependent upon. This member will also require the **groupOptional** member to be present and set to ``true``. Be sure that the object is not within the group.
groupDependencyType ``str``
    What happens when the group's dependency parameter is checked. Options are ``enabled`` or ``disabled``.


.. _json_param_examples:

Parameter Types
^^^^^^^^^^^^^^^
There are other JSON members that may be available or required based on the parameter type. The following sections define different parameter types that can be found in the **ui.json** format.

 .. toctree::
   :maxdepth: 1

   json_objects.rst


Exporting from Geoscience ANALYST
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
When a **ui.json** is saved with Geoscience ANALYST Pro, the following object members are updated or added:

- The **value** member with the appropriate type
- The **enabled** member ``bool`` for whether the parameter is enabled
- The :ref:`Data parameter <data_parameter>` will also have updated **isValue** and **property** members. The **isValue** ``bool`` member is *true* if the **value** member was selected and *false* if the **property** member was selected.

The following JSON objects will be written (and overwritten if given) upon export from Geoscience ANALYST Pro:

- monitoring_directory ``str`` the absolute path of a monitoring directory. Workspace files written to this folder will be automatically processed by Geoscience ANALYST.
- workspace_geoh5 ``str`` the absolute path to the current workspace (if previously saved) being used
- geoh5 ``str`` the absolute path to the geoh5 written containing all the objects of the workspace within the parameters of the **ui.json**. One only needs to use this workspace along with the JSON file to access the objects with geoh5py.


Tips on creating UIs
^^^^^^^^^^^^^^^^^^^^
Here are a few tips on creating good looking UIs:

- Keep labels short and concise. Be consistent with capitalization and do not include the colons. Geoscience ANALYST will add colons and align them.
- Write detailed tooltips.
- Group related objects, but do not use a group if there are fewer than 3 objects.
- The **main** member is for general, required parameters. Do not include this member with every object, unless there are only a handful of objects. Objects that are in the required parameters without a valid value will invoke an error when exporting or running from Geoscience ANALYST. "Non-main" members are designated to a second page under *Optional parameters*.
- Utilize **optional** object members and dependencies. If a single workspace object input is optional, use the :ref:`Object parameter <object_parameter>` rather than two parameters with a dependency.


External Links
^^^^^^^^^^^^^^

- `JSON Terminology <https://JSON-schema.org/draft-04/JSON-schema-core.html>`_
- `Universally Unique IDentifier (UUID) <https://en.wikipedia.org/wiki/Universally_unique_identifier>`_
- `C++ JSON Library <https://github.com/nlohmann/JSON>`_
