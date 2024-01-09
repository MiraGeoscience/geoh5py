About
^^^^^

The **ui.json** format provides a schema to create a simple User Interface (UI) between geoh5py and `Geoscience ANALYST Pro
<http://www.mirageoscience.com/our-products/software-product/geoscience-analyst>`_. The format uses `JSON objects <https://json-schema.org/specification.html>`_ to represent `script parameters <./json_objects.rst>`_ used in the UI, and pass those parameters to an accompanying python script.

Each ui.json object requires at least a **label** and **value** member, however additional members can be used to define different types of input and additional dependencies between parameters.

For example, a simple ui.json below describes a single parameter called 'grid_object', which is used to select a block model within a geoh5 file.

.. code-block:: json

    {
    "grid_object": {
    "meshType": ["{B020A277-90E2-4CD7-84D6-612EE3F25051}"],
    "main": true,
    "label": "Select Block Model",
    "value": ""
    }
    }

Note: The **meshType** used to select the grid object is defined by a list of UUID. A complete list of UUID's for geoh5 object types are available in the `geoh5 objects documentation <../content/geoh5_format/analyst/objects.rst>`_.
