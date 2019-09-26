import uuid

import h5py
import numpy as np

KNOWN_TYPES = {
    "{dd99b610-be92-48c0-873c-5b5946ea2840}": "Unknown type",
    "{61fbb4e8-a480-11e3-8d5a-2776bdf4f982}": "Container",
    "{825424fb-c2c6-4fea-9f2b-6cd00023d393}": "DrillHole container",
    "{202c5db1-a56d-4004-9cad-baafd8899406}": "Points",
    "{6a057fdc-b355-11e3-95be-fd84a7ffcb88}": "Curve",
    "{f26feba3-aded-494b-b9e9-b2bbcbe298e1}": "Surface",
    "{b020a277-90e2-4cd7-84d6-612ee3f25051}": "3D mesh (block model)",
    "{7caebf0e-d16e-11e3-bc69-e4632694aa37}": "Drillhole",
    "{77ac043c-fe8d-4d14-8167-75e300fb835a}": "GeoImage",
    "{e79f449d-74e3-4598-9c9c-351a28b8b69e}": "Label",
}


class H5Reader:
    """
        H5 file Reader
    """

    @classmethod
    def get_project_attributes(cls, h5file: str, base: str) -> dict:

        project = h5py.File(h5file, "r+")
        project_attrs = {}

        for key in list(project[base].attrs.keys()):

            attr = key.lower().replace(" ", "_")

            project_attrs[attr] = project[base].attrs[key]

        project.close()

        return project_attrs

    # def get_objects(self) -> dict:
    #
    #     project = h5py.File(h5file, "r+")
    #
    #     # Build a dictionary of objects
    #     for obj in project[self.workspace.base]["Objects"].keys():
    #
    #         h5_pointer = project[self.workspace.base]["Objects"][obj]
    #         name = h5_pointer.attrs["Name"]
    #         # otype = self._types['Object'][name]
    #
    #         if h5_pointer["Type"].attrs["ID"] in list(KNOWN_TYPES.keys()):
    #
    #             self._objects[name] = KNOWN_TYPES[h5_pointer["Type"].attrs["ID"]]
    #
    #         else:
    #             self._objects[name] = Object(name, uid=uuid.UUID(obj))
    #
    #         self._objects[name].load(h5_pointer)
    #         self._uuid[obj] = self._objects[name]
    #
    #     project.close()
    #     return object_list

    @staticmethod
    def bool_value(value: np.uint8) -> bool:
        return bool(value)

    @staticmethod
    def uuid_value(value: str) -> uuid.UUID:
        return uuid.UUID(value)
