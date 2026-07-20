UI.JSON Format
--------------

The **ui.json** format provides a schema to create a simple User Interface (UI) connecting `Geoscience ANALYST Pro <http://www.mirageoscience.com/our-products/software-product/geoscience-analyst>`_ and Python. The format uses the `JSON format <https://json-schema.org/specification.html>`_ to store the state of the UI and to pass those parameters to an accompanying Python script.

.. figure:: ./images/python_scripts.png
        :align: center
        :width: 800

Usage
=====

A ui.json file contains `forms <Forms>`_ that drive an accompanying Python script to be executed from Geoscience ANALYST.

To be valid, the **ui.json** file must contain at least the following fields:

- **title** ``str``
    Title of user interface window
- **run_command** ``str``
    Name of Python script excluding the .py extension (i.e., "run_me" for run_me.py) required for Geoscience ANALYST Pro
    to run on save or auto-load.
- **conda_environment** ``str``
    [Optional] Name of conda environment to activate when running the Python script in *run_command*

For example, a simple ui.json below describes a single ``grid_object`` parameter, which is used to select a block model within a geoh5 file. See the `Object form <Object form>`_ section for more details on this type of selection form.

.. code-block:: json

    {
        "title":"My first UI",
        "run_command": "run_me",
        "conda_environment": "mirageo",
        "grid_object": {
            "meshType": ["{B020A277-90E2-4CD7-84D6-612EE3F25051}"],
            "main": true,
            "label": "Select Block Model",
            "value": ""
        }
    }

.. figure:: ./images/block_model_uijson.png


Execution
^^^^^^^^^

When a **ui.json** is run within Geoscience ANALYST Pro (either ``OK`` or ``Apply``), the following steps occur:

- The **value** and **enabled** fields of every forms are updated to reflect the current state of the UI. For example, the **value** field of the ``grid_object`` is set to the selected object ``UUID``. If no object is selected, the **value** field is set to an empty string.
- A ``ui.json`` file is written to disk at the selected location, along with a geoh5 file containing all the objects within the parameters of the **ui.json**.
- ANALYST Pro executes the Python script specified in the **run_command** field of the **ui.json** file. The script is executed in the conda environment specified in the **conda_environment** field of the **ui.json** file (if provided). For example, the ``run_me.py`` script is executed in the ``mirageo`` conda environment:



Note that the ``mirageo`` conda environment is the default environment for Geoscience ANALYST Pro. Users can create their own conda environments and specify them in the **conda_environment** field of the **ui.json** file.

Within the accompanying Python script, the parameters from the ui.json may be accessed using the UIJson module of
geoh5py as shown below:

.. code-block:: python

    import sys
    from geoh5py.ui_json import UIJson

    ui_json = sys.argv[1]
    ifile = UIJson.read(ui_json)
    selector = ifile.grid_object


Rendering
^^^^^^^^^

The user interface defined by the ``ui.json`` file can be opened in ANALYST Pro in two ways:

a. **Drag and Drop:**
   Simply drag the ``ui.json`` file into the viewport. The corresponding dialog will open immediately.

   .. figure:: ./images/drag_drop.gif
      :align: center
      :width: 800

b. **Add to Python Script Menu:**

   1. From the ANALYST menubar, open the *Python* menu and select *Script Directory* to launch the file explorer.

   2. Copy the ``ui.json`` file into the displayed folder.

   3. Close the workspace or restart ANALYST Pro. The new application will appear under the Python script menu.

   .. figure:: ./images/dropdown.gif
      :align: center
      :width: 800


General Tips
^^^^^^^^^^^^
- Keep labels concise
- Write detailed tooltips
- Group related attributes
- Don't include the **main** member with every parameter. "Non-main" members are designated to a second page under *Optional parameters*
- Utilize **optional** object members and dependencies.
