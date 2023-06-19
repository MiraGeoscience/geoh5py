Parameters available for all ui.json objects
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The following members are available to all input parameters in the ui.json schema.

label ``str``
    Required string describing parameter. A colon will automatically be added within Geoscience ANALYST, so this should be omitted.
value ``str``, ``int``, ``bool`` , or ``float``
    This required member takes a different form, including the empty string '', depending on the :ref:`parameter type <json_param_examples>`. The value is updated when written from Geoscience ANALYST.
main ``bool``
    If set to true, the parameter is shown in the first tab of the UI and will throw an error if not present (and not optional). Optional parameters may be set to main. When main is not given or is false, the parameter will be under the *Optional Parameters* tab.
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
    The name of the parameter which this parameter is dependent upon. The dependency parameter should be optional or boolean parameter (i.e., has a checkbox).
dependencyType ``str``
    What happens when the dependency member is checked. Options are ``enabled`` or ``disabled``
groupDependency ``str``
    The name of the object of which the group of the parameter is dependent upon. This member will also require the **groupOptional** member to be present and set to ``true``. Be sure that the object is not within the group.
groupDependencyType ``str``
    What happens when the group's dependency parameter is checked. Options are ``enabled`` or ``disabled``.
