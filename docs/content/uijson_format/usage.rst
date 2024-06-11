Usage with Geoscience ANALYST Pro
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
A ui.json file contains the parameters that reference a Python script to run, as well as the conda environment to activate when running the script if provided.

run_command ``str``
    Name of python script excluding the .py extension (i.e., "run_me" for run_me.py) required for Geoscience ANALYST Pro to run on save or auto-load.
conda_environment ``str``
    [Optional] Name of conda environment to activate when running the python script in *run_command*
title ``str``
    Title of user interface window

To complete the example above, add the **run_command**, and **title** parameter to the ui.json file.

.. code-block:: json

    {
    "grid_object": {
    "meshType": ["{B020A277-90E2-4CD7-84D6-612EE3F25051}"],
    "main": true,
    "label": "Select Block Model",
    "value": ""
    },
    "title":"My first UI",
    "run_command": "run_me"
    }

.. figure:: ./images/block_model_uijson.png

Within the accompanying python script, the parameters from the ui.json is passed to the python script as a dictionary of arguments, and can be accessed using the InputFile module of geoh5py as shown below:

.. code-block:: python

    import sys
    from geoh5py.ui_json import InputFile

    ui_json = sys.argv[1]
    ifile = InputFile.read_ui_json(ui_json)
    params = ifile.data

    # Get the block model grid object
    bm = params["grid_object"]
    print(f"The selected object name is {bm.name}")


.. figure:: ./images/block_model_output.png

When a **ui.json** is run within Geoscience ANALYST Pro, the following parameters are updated or added:

- The **value** member is updated with the UUID of the object selected in the UI
- The **enabled** member ``bool`` is set for whether the parameter is enabled
- The :ref:`Data parameter <data_parameter>` will also have updated **isValue** and **property** members. The **isValue** ``bool`` member is *true* if the **value** member was selected and *false* if the **property** member was selected.

The following JSON objects will be written (and overwritten if given) upon running a ui.json from Geoscience ANALYST Pro:

- **monitoring_directory** ``str`` the absolute path of a monitoring directory. Workspace files written to this folder will be automatically processed by Geoscience ANALYST.
- **workspace_geoh5** ``str`` (Optional) Path to the source geoh5 file (for reference only)
- **geoh5** ``str`` the absolute path to the geoh5 written containing all the objects of the workspace within the parameters of the **ui.json**. One only needs to use this workspace along with the JSON file to access the objects with geoh5py.
