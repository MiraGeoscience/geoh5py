# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#  Copyright (c) 2026 Mira Geoscience Ltd.                                     '
#                                                                              '
#  This file is part of geoh5py.                                               '
#                                                                              '
#  geoh5py is free software: you can redistribute it and/or modify             '
#  it under the terms of the GNU Lesser General Public License as published by '
#  the Free Software Foundation, either version 3 of the License, or           '
#  (at your option) any later version.                                         '
#                                                                              '
#  geoh5py is distributed in the hope that it will be useful,                  '
#  but WITHOUT ANY WARRANTY; without even the implied warranty of              '
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the               '
#  GNU Lesser General Public License for more details.                         '
#                                                                              '
#  You should have received a copy of the GNU Lesser General Public License    '
#  along with geoh5py.  If not, see <https://www.gnu.org/licenses/>.           '
# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

from pathlib import Path

from geoh5py import Workspace
from geoh5py.groups import FusionTableGroup
from geoh5py.shared.utils import stringify
from geoh5py.ui_json import UIJson
from geoh5py.ui_json.utils import monitored_directory_copy


def main(ui_file):
    print("Creating Exploratory Data Analysis group...")

    # Read the file as pydantic class
    ifile = UIJson.read(ui_file)

    with Workspace(ifile.geoh5) as workspace:
        # Convert to a dict with geoh5py entities
        my_inputs = ifile.to_params(workspace=workspace)

        # Create top group
        eda_group = FusionTableGroup.create(
            workspace,
            mesh=my_inputs["data_mesh"],
            referenced_data=my_inputs["target_channel"],
            name="EDA Group",
            parent=my_inputs["out_group"],
        )

        # Create feature list group
        source = Path(__name__).resolve().parent / "feature_list.ui.json"
        features = UIJson.read(source)
        features.set_values(
            data_channel=my_inputs["data_channel"], data_mesh=my_inputs["data_mesh"]
        )

        features_group = features.to_ui_json_group(workspace, name="Feature List")

        prop_group = my_inputs["data_mesh"].create_property_group(
            name="Feature List", properties=my_inputs["data_channel"]
        )
        # Update Features outputs
        options = features_group.options
        for key in options["children"]:
            options["children"][key]["value"] = prop_group.uid

        features_group.options = stringify(options)

        # Create weight of evidence group and link to the EDA group
        source = Path(__name__).resolve().parent / "woe.ui.json"
        woe = UIJson.read(source)
        woe.set_values(data_channel=key, data_mesh=my_inputs["data_mesh"])
        woe.to_ui_json_group(workspace, name="Weight of Evidence")

        # Update EDA outputs
        options = my_inputs["out_group"].options

        for key, value in zip(options["children"], [eda_group, features_group]):
            options["children"][key]["value"] = value.uid

        my_inputs["out_group"].options = stringify(options)

        # Send the result back to ANALYST
        if ifile.monitoring_directory:
            monitored_directory_copy(ifile.monitoring_directory, my_inputs["out_group"])

        print("Done")


if __name__ == "__main__":
    # ui_file = sys.argv[1]
    ui_file = r"C:\Users\dominiquef\Documents\tests\prototype_workflows\eda.ui.json"
    main(ui_file)
