.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/psf/black

GeoH5io
=======
API to read and write GeoH5 files from Python 3.

The API is described in terms data types and service interfaces in a
[Thrift](https://thrift.apache.org/) definition file.
You are free to generate a client from this definition file in one of the many
languages supported by Thrift. You can then use the client in your custom code
to communicate with the GeoH5 server.