Members
=======

The following members are available to all input forms in the ui.json schema. They are used to define the parameters that will be displayed in the UI or
to define the behaviour of the parameters in relation to others (association, visibility, etc.).

.. _common_members:

label
^^^^^
``str``
(Required) Describes the parameter. Geoscience ANALYST will add a colon when rendering the interface, so this should
be omitted.

value
^^^^^
``str``, ``int``, ``float``, or ``bool``
(Required) The value of the parameter.  Value is updated with data selected within Analyst.  In addition to the
types listed here, the value may be an empty string '' depending on the :ref:`parameter type <parameter_types>`.

main
^^^^^
``bool``
If set to true, the parameter is shown in the first tab of the UI and will throw an error if no value is set (and not
optional). Optional parameters may be set to main. When main is not given or is false, the parameter will be under
the *Optional Parameters* tab.

tooltip
^^^^^^^
``str``
   String describing the parameter in detail that appears when the mouse hovers over it.

optional
^^^^^^^^
``bool``
If set *true* the parameter will be rendered with a checkbox that will control the enabled state on writing from
ANALYST.

enabled
^^^^^^^
``bool``
Controls the enabled/disabled state of the parameter.  Disabled parameters will be rendered in a grey font and will
not be editable.  The default is true. If a parameter is optional and not enabled, it will start as disabled.

group
^^^^^
``str``
Name of the group to which the parameter belongs. Adds a box and name around the parameters with the same group name.

groupOptional
^^^^^^^^^^^^^
``bool``
If true, adds a checkbox in the top of the group box next to the name. The group parameters will be disabled if not
checked. The initial state depends on the **groupDependency** and **groupDependencyType** members and the **enabled**
member of the group's parameters.

dependency
^^^^^^^^^^
``str``
The name of the parameter which the current parameter is dependent upon. The dependency parameter should be optional
or boolean parameter (i.e., has a checkbox).

dependencyType
^^^^^^^^^^^^^^
``str``
Provides the behaviour when the dependency is enabled.  If ``enabled``, the current parameter's enabled state will
match the dependency, if ``disabled`` it will be the opposite of its dependency.  Alternatively, a ``show`` or
``hide`` string may be used to control the visibility of the dependent parameter

groupDependency
^^^^^^^^^^^^^^^
``str``
The name of the parameter on which the group depends. Requires the **groupOptional** member to be ``true``. The
parameter cannot be in the current group.

groupDependencyType
^^^^^^^^^^^^^^^^^^^
``str``
Provides the behaviour when the dependency is enabled.  If ``enabled``, all the parameters within the group will
match the dependency's enabled state, if ``disabled`` they will be the opposite of the dependency.
