import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import ipywidgets as widgets
from ipywidgets.widgets import Label, Dropdown, Layout, VBox, HBox, Text


from geoh5io.workspace import Workspace
from geoh5io.objects import Curve, BlockModel
import json
from .plotting import plot_profile_data_selection, plot_plan_data_selection
from ..utils import find_value, rotate_xy


def pf_inversion_widget(h5file, plot=False, resolution=50, inducing_field="50000, 90, 0"):
    workspace = Workspace(h5file)

    units = {"Gravity": "g/cc", "Magnetics": "SI"}
    names = list(workspace.list_objects_name.values())

    def update_data_list(_):
        obj = workspace.get_entity(objects.value)[0]

        data.options = [name for name in obj.get_data_list() if "visual" not in name.lower()]
        if data.options:
            data.value = data.options[0]

        sensor_value.options = [name for name in obj.get_data_list() if "visual" not in name.lower()] + ["Vertices"]
        sensor_value.value = "Vertices"

    def update_data_options(_):
        obj = workspace.get_entity(objects.value)[0]

        if obj.get_data(data.value):
            data_obj = obj.get_data(data.value)[0]
            uncertainties.value = f"0, {np.percentile(np.abs(data_obj.values), 10)}"

        if data.value is not None:
            data_type.value = find_value(data_type.options, [data.value.lower()])

    def update_topo_list(_):
        obj = workspace.get_entity(topo_objects.value)[0]
        topo_value.options = [name for name in obj.get_data_list() if "visual" not in name.lower()] + ['Vertices']

    objects = Dropdown(
        options=names,
        description='Object:',
    )

    objects.observe(update_data_list, names="value")

    data = Dropdown(
        description='Channel: ',
    )

    uncertainties = Text(
        description='Uncertainty (%, floor): ',
        style={'description_width': 'initial'}
    )

    data.observe(update_data_options, names="value")

    ###################### Data selection ######################
    # Fetch vertices in the project
    lim_x = [1e+8, 0]
    lim_y = [1e+8, 0]
    for name in names:
        obj = workspace.get_entity(name)[0]
        if obj.vertices is not None:
            lim_x[0], lim_x[1] = np.min([lim_x[0], obj.vertices[:, 0].min()]), np.max(
                [lim_x[1], obj.vertices[:, 0].max()])
            lim_y[0], lim_y[1] = np.min([lim_y[0], obj.vertices[:, 1].min()]), np.max(
                [lim_y[1], obj.vertices[:, 1].max()])
        elif hasattr(obj, "centroids"):
            lim_x[0], lim_x[1] = np.min([lim_x[0], obj.centroids[:, 0].min()]), np.max(
                [lim_x[1], obj.centroids[:, 0].max()])
            lim_y[0], lim_y[1] = np.min([lim_y[0], obj.centroids[:, 1].min()]), np.max(
                [lim_y[1], obj.centroids[:, 1].max()])

    center_x = widgets.FloatSlider(
        min=lim_x[0], max=lim_x[1], value=np.mean(lim_x),
        steps=10, description="Easting", continuous_update=False
    )
    center_y = widgets.FloatSlider(
        min=lim_y[0], max=lim_y[1], value=np.mean(lim_y),
        steps=10, description="Northing", continuous_update=False,
        orientation='vertical',
    )
    azimuth = widgets.FloatSlider(
        min=-90, max=90, value=0, steps=5, description="Orientation", continuous_update=False
    )
    width_x = widgets.FloatSlider(
        max=lim_x[1] - lim_x[0],
        min=100,
        value=lim_x[1] - lim_x[0],
        steps=10, description="Width", continuous_update=False
    )
    width_y = widgets.FloatSlider(
        max=lim_y[1] - lim_y[0],
        min=100,
        value=lim_y[1] - lim_y[0],
        steps=10, description="Height", continuous_update=False,
        orientation='vertical'
    )
    resolution = widgets.FloatText(value=resolution, description="Resolution (m)",
                                   style={'description_width': 'initial'})

    def plot_selection(
            entity_name, data_name, uncertainties, resolution,
            center_x, center_y,
            width_x, width_y, azimuth
    ):
        obj = workspace.get_entity(entity_name)[0]

        if obj.get_data(data_name):
            fig = plt.figure(figsize=(10, 10))
            ax1 = plt.subplot()

            corners = np.r_[np.c_[-1., -1.], np.c_[-1., 1.], np.c_[1., 1.], np.c_[1., -1.], np.c_[-1., -1.]]
            corners[:, 0] *= width_x/2
            corners[:, 1] *= width_y/2
            corners = rotate_xy(corners, [0,0], -azimuth)
            ax1.plot(corners[:, 0] + center_x, corners[:, 1] + center_y, 'k')

            data_obj = obj.get_data(data_name)[0]
            floor = float(uncertainties.split(",")[1])
            plot_plan_data_selection(
                obj, data_obj,
                **{
                    "ax": ax1,
                    "downsampling": resolution,
                    "window": {
                        "center": [center_x, center_y],
                        "size": [width_x, width_y],
                        "azimuth": azimuth
                    },
                    "contours": [-floor, floor]
                }
            )


    plot_window = widgets.interactive_output(
        plot_selection, {
            "entity_name": objects,
            "data_name": data,
            "uncertainties": uncertainties,
            "resolution": resolution,
            "center_x": center_x,
            "center_y": center_y,
            "width_x": width_x,
            "width_y": width_y,
            "azimuth": azimuth,
        }
    )

    selection_panel = VBox([
        Label("Window & Downsample"),
        VBox([
            center_x,
            HBox([
                center_y,
                plot_window,
                width_y
            ], layout=Layout(align_items='center')),
            VBox([width_x, azimuth, resolution], layout=Layout(align_items='center'))
        ], layout=Layout(align_items='center'))
    ])

    def update_survey_type(_):
        if survey_type.value == "Magnetics":
            data_type.options = ["tmi", 'bxx', "bxy", "bxz", 'byy', "byz", "bzz"]

            survey_type_panel.children = [Label("Data"), survey_type, objects, data, data_type, uncertainties, inducing_field]
        else:
            data_type.options = ["gz", 'gxx', "gxy", "gxz", 'gyy', "gyz", "gzz"]
            survey_type_panel.children = [Label("Data"), survey_type, objects, data, data_type, uncertainties]

        if data.value is not None:
            data_type.value = find_value(data_type.options, [data.value.lower()])
        if ref_mod.children[1].children[1].children:
            ref_mod.children[1].children[1].children[0].decription = units[survey_type.value]

    survey_type = Dropdown(
        options=["Magnetics", "Gravity"],
        description="Survey Type:",
    )
    inducing_field = widgets.Text(
        value=inducing_field,
        description='Inducing Field [Amp, Inc, Dec]',
        style={'description_width': 'initial'}
    )
    survey_type.observe(update_survey_type)

    data_type = Dropdown(
        description="Data Type:",
    )

    survey_type_panel = VBox([survey_type, objects, data, data_type, uncertainties])

    ###################### Spatial parameters ######################
    ########## TOPO #########
    topo_objects = Dropdown(
        options=names,
        value=find_value(names, ['topo', 'dem']),
        description='Object:',
    )
    topo_objects.observe(update_topo_list, names="value")

    topo_value = Dropdown(
        description='Channel: ',
    )
    update_topo_list("")

    topo_panel = VBox([topo_objects, topo_value])

    topo_offset = widgets.FloatText(
        value=-30,
        description="Vertical offset (m)",
        style={'description_width': 'initial'}
    )
    topo_options = {
        "Object": topo_panel,
        "Drape Height": topo_offset
    }

    topo_options_button = widgets.RadioButtons(
        options=['Object', 'Drape Height'],
        description='Define by:',
    )

    def update_topo_options(_):
        topo_options_panel.children = [topo_options_button, topo_options[topo_options_button.value]]

    topo_options_button.observe(update_topo_options)

    topo_options_panel = VBox([topo_options_button, topo_options[topo_options_button.value]])

    ########## RECEIVER #########
    sensor_value = Dropdown(
        options=["Vertices"],
        value="Vertices",
        description='Channel: ',
    )

    sensor_offset = widgets.FloatText(
        value=30,
        description="Vertical offset (m)",
        style={'description_width': 'initial'}
    )

    sensor_options = {
        "Channel": sensor_value,
        "Drape Height (Topo required)": sensor_offset
    }

    sensor_options_button = widgets.RadioButtons(
        options=['Channel', 'Drape Height (Topo required)'],
        description='Define by:',
    )

    def update_sensor_options(_):
        if topo_value.value is None:
            sensor_options_button.value = "Channel"

        sensor_options_panel.children = [sensor_options_button, sensor_options[sensor_options_button.value]]

    sensor_options_button.observe(update_sensor_options)

    sensor_options_panel = VBox([sensor_options_button, sensor_options[sensor_options_button.value]])

    ###############################
    spatial_options = {
        "Topography": topo_options_panel,
        "Sensor Height": sensor_options_panel
    }

    spatial_choices = widgets.Dropdown(
        options=list(spatial_options.keys()),
        value=list(spatial_options.keys())[0],
        disabled=False
    )

    spatial_panel = VBox(
        [Label("Spatial Information"), VBox([spatial_choices, spatial_options[spatial_choices.value]])])

    def spatial_option_change(_):
        spatial_panel.children[1].children = [spatial_choices, spatial_options[spatial_choices.value]]

    spatial_choices.observe(spatial_option_change)

    update_data_list("")

    ###################### Inversion options ######################
    def write_unclick(_):
        if write.value:
            input_dict = {}
            input_dict['out_group'] = out_group.value
            input_dict['workspace'] = h5file
            input_dict['save_to_geoh5'] = h5file
            if survey_type.value == 'Gravity':
                input_dict["inversion_type"] = 'grav'
            elif survey_type.value == "Magnetics":
                input_dict["inversion_type"] = 'mvis'
                input_dict["inducing_field_aid"] = np.asarray(inducing_field.value.split(",")).astype(float).tolist()

            input_dict["data_type"] = {
                "GA_object": {
                    "name": objects.value,
                    "data": data.value,
                    "components": [data_type.value],
                    "uncertainties": np.asarray(uncertainties.value.split(",")).astype(float).tolist()
                }
            }

            if sensor_options_button.value == 'Channel':
                input_dict["data_type"]["GA_object"]['z_channel'] = sensor_value.value
            else:
                input_dict['drape_data'] = sensor_offset.value

            if topo_options_button.value == "Object":
                input_dict["topography"] = {
                    "GA_object": {
                        "name": topo_objects.value,
                        "data": topo_value.value,
                    }
                }
            else:
                input_dict["topography"] = {"drapped": topo_offset}

            input_dict['resolution'] = resolution.value
            input_dict["window"] = {
                        "center": [center_x.value, center_y.value],
                        "size": [width_x.value, width_y.value],
                        "azimuth": azimuth.value
            }

            if ref_type.value == "None":
                input_dict["alphas"] = [0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1]
            else:
                input_dict["model_reference"] = ref_mod.children[1].children[1].children[0].value

            input_dict["core_cell_size"] = [resolution.value/2, resolution.value/2, resolution.value/2]
            input_dict["octree_levels_topo"] = [0, 3, 3]
            input_dict["octree_levels_obs"] = [5, 5, 5]
            input_dict["padding_distance"] = [
                [width_x.value/2, width_x.value/2],
                [width_y.value / 2, width_y.value / 2],
                [np.min([width_x.value / 2, width_y.value / 2]), 0]
            ]
            input_dict['depth_core'] = {'auto': 0.2}

            with open(f"{out_group.value}.json", 'w') as f:
                json.dump(input_dict, f)

        write.value = False

    def invert_unclick(_):
        if invert.value:
            prompt = os.system(
                "start cmd.exe @cmd /k " + f"\"python functions/pf_inversion.py {out_group.value}.json\"")
            invert.value = False

    invert = widgets.ToggleButton(
        value=False,
        description='Invert',
        button_style='danger',
        tooltip='Run simpegEM1D',
        icon='check'
    )

    invert.observe(invert_unclick)

    out_group = widgets.Text(
        value='Inversion_',
        description='Save to:',
        disabled=False
    )

    write = widgets.ToggleButton(
        value=False,
        description='Write input',
        button_style='',
        tooltip='Write json input file',
        icon='check'
    )

    write.observe(write_unclick)

    chi_factor = widgets.FloatText(
        value=1,
        description='Target misfit',
        disabled=False
    )

    ref_type = widgets.RadioButtons(
        options=["None", 'Value', 'Model'],
        value='None',
        disabled=False
    )

    def update_ref(_):

        if ref_mod.children[1].children[0].value == 'Model':

            model_list = []

            for obj in workspace.all_objects():
                if isinstance(obj, BlockModel):

                    for data in obj.children:

                        if getattr(data, "values", None) is not None:
                            model_list += [data.name]

            ref_mod.children[1].children[1].children = [widgets.Dropdown(
                description='3D Model',
                options=model_list,
            )]

        elif ref_mod.children[1].children[0].value == 'Value':

            ref_mod.children[1].children[1].children = [widgets.FloatText(
                description=units[survey_type.value],
                value=0.,
            )]

        else:
            ref_mod.children[1].children[1].children = []

    ref_type.observe(update_ref)
    ref_mod = widgets.VBox([Label('Reference model'), widgets.VBox([ref_type, widgets.VBox([])])])

    inversion_options = {
        "output name": out_group,
        "target misfit": chi_factor,
        "reference model": ref_mod,
    }

    option_choices = widgets.Dropdown(
        options=list(inversion_options.keys()),
        value=list(inversion_options.keys())[0],
        disabled=False
    )

    def inv_option_change(_):
        inversion_panel.children[1].children = [option_choices, inversion_options[option_choices.value]]

    option_choices.observe(inv_option_change)

    inversion_panel = widgets.VBox([
        widgets.HBox([widgets.Label("Inversion Options")]),
        widgets.HBox([option_choices, inversion_options[option_choices.value]], )
        # layout=widgets.Layout(height='500px')
    ], layout=Layout(width="100%"))

    survey_type.value = "Gravity"

    return VBox([HBox([survey_type_panel, spatial_panel]), selection_panel, inversion_panel, write, invert])


def em1d_inversion_widget(h5file, plot_profile=True, start_channel=None, object_name=None):
    """
    Setup and invert time or frequency domain data
    """
    workspace = Workspace(h5file)

    curves = [entity.parent.name + "." + entity.name for entity in workspace.all_objects() if isinstance(entity, Curve)]
    names = [name for name in sorted(curves)]

    # Load all known em systems
    dir_path = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(dir_path, "../AEM_systems.json"), 'r') as aem_systems:
        em_system_specs = json.load(aem_systems)

    def get_parental_child(parental_name):

        parent, child = parental_name.split(".")

        parent_entity = workspace.get_entity(parent)[0]

        children = [entity for entity in parent_entity.children if entity.name==child]
        return children

    def find_value(labels, strings):
        value = None
        for name in labels:
            for string in strings:
                if string.lower() in name.lower():
                    value = name
        return value

    def get_comp_list(entity):
        component_list = []

        for pg in entity.property_groups:
            component_list.append(pg.name)

        return component_list

    objects = Dropdown(
        options=names,
        description='Object:',
    )
    def object_observer(_):

        entity = get_parental_child(objects.value)[0]
        data_list = entity.get_data_list()

        # Update topo field
        topo.options = data_list
        topo.value = find_value(data_list, ['dem', "topo"])
        line_field.options = data_list
        line_field.value = find_value(data_list, ['line'])

        if get_comp_list(entity):
            components.options = get_comp_list(entity)
            components.value = get_comp_list(entity)[0]

        for aem_system, specs in em_system_specs.items():
            if any([specs["flag"] in channel for channel in data_list]):
                system.value = aem_system

        system_observer("")
        line_field_observer("")

    objects.observe(object_observer, names='value')

    systems = list(em_system_specs.keys())
    system = Dropdown(
        options=systems,
        description='System: ',
    )

    scale = Dropdown(
        options=['linear', 'symlog'],
        value='symlog',
        description='Scaling',
    )

    def system_observer(_, start_channel=start_channel):
        entity = get_parental_child(objects.value)[0]
        rx_offsets = em_system_specs[system.value]["rx_offsets"]
        uncertainties = em_system_specs[system.value]["uncertainty"]

        if start_channel is None:
            start_channel = em_system_specs[system.value]["channel_start_index"]
        em_type = em_system_specs[system.value]["type"]

        if em_system_specs[system.value]["type"] == 'time':
            label = "Time (s)"
            scale.value = "symlog"
        else:
            label = "Frequency (Hz)"
            scale.value = "linear"

        data_channel_options = {}
        for ind, (key, time) in enumerate(
            em_system_specs[system.value]["channels"].items()
        ):
            if ind+1 < start_channel:
                continue

            if len(rx_offsets) > 1:
                offsets = rx_offsets[ind]
            else:
                offsets = rx_offsets[0]

            data_list = entity.get_data_list()
            if components.value is not None:
                p_g = entity.get_property_group(components.value)
                if p_g is not None:
                    data_list = [workspace.get_entity(data)[0].name for data in p_g.properties]

            data_channel_options[key] = VBox([
                        widgets.Checkbox(
                            value=ind+1 >= start_channel,
                            indent=True,
                            description="Active"
                        ),
                        widgets.Text(
                            value=f"{time:.5e}",
                            description=label,
                            style={'description_width': 'initial'}
                        ),
                        Dropdown(
                            options=data_list,
                            value=find_value(data_list, [key]),
                            description="Channel",
                            style={'description_width': 'initial'}
                        ),
                        widgets.Text(
                            value=', '.join([
                                str(uncert) for uncert in uncertainties[ind][:2]
                            ]),
                            description="Error (%, floor)",
                            style={'description_width': 'initial'}
                        ),
                        widgets.Text(
                            value=', '.join([
                                str(offset) for offset in offsets
                            ]),
                            description='Offset',
                            style={'description_width': 'initial'}
                        )
                ])

        data_channel_choices.options = list(data_channel_options.keys())
        data_channel_choices.value = list(data_channel_options.keys())[0]
        system.data_channel_options = data_channel_options
        data_channel_panel.children = [
            data_channel_choices,
            data_channel_options[data_channel_choices.value]
        ]

    system.observe(system_observer, names='value')

    components = Dropdown(
        description='Data Group',
    )

    def data_channel_choices_observer(_):
        if (
            hasattr(system, "data_channel_options") and
            data_channel_choices.value in (system.data_channel_options.keys())
        ):
            data_channel_panel.children = [data_channel_choices, system.data_channel_options[data_channel_choices.value]]

    data_channel_choices = widgets.Dropdown(
            description="Data field:",
            style={'description_width': 'initial'}
        )

    data_channel_choices.observe(data_channel_choices_observer, names='value')
    data_channel_panel = widgets.VBox([data_channel_choices])  #layout=widgets.Layout(height='500px')

    def auto_pick_channels(_):
        entity = get_parental_child(objects.value)[0]

        if components.value is not None:

            p_g = entity.get_property_group(components.value)

            if hasattr(system, "data_channel_options"):
                for key, data_widget in system.data_channel_options.items():
                    value = []
                    if p_g is not None:
                        data_list = [workspace.get_entity(data)[0].name for data in p_g.properties]
                        value = find_value(data_list, [p_g.name+key])
                    else:
                        data_list = entity.get_data_list()
                    if not value:
                        value = find_value(data_list, [key])
        #               if data_widget.children[0].children[0].value:
                    data_widget.children[2].options = data_list
                    data_widget.children[2].value = value

    components.observe(auto_pick_channels, names='value')

    topo = Dropdown(
        description='DEM:',
    )

    line_field = Dropdown(
        description='Lines field',
    )

    def line_field_observer(_):
        if (
            objects.value is not None and
            line_field.value is not None and
            'line' in line_field.value.lower()
        ):
            entity = get_parental_child(objects.value)[0]
            if entity.get_data(line_field.value):
                lines.options = np.unique(
                    entity.get_data(line_field.value)[0].values
                )

                if lines.options[1]:
                    lines.value = [lines.options[1]]
                if lines.options[0]:
                    lines.value = [lines.options[0]]

    line_field.observe(line_field_observer, names='value')
    lines = widgets.SelectMultiple(
        description=f"Select data:",
    )
    downsampling = widgets.FloatText(
        value=0,
        description='Downsample (m)',
        style={'description_width': 'initial'}
    )

    object_fields_options = {
        "Data Channels": data_channel_panel,
        "Topo": topo,
        "Line ID": line_field,
    }

    object_fields_dropdown = widgets.Dropdown(
            options=list(object_fields_options.keys()),
            value=list(object_fields_options.keys())[0],
    )

    object_fields_panel = widgets.VBox([
        widgets.VBox([object_fields_dropdown], layout=widgets.Layout(height='75px')),
        widgets.VBox([object_fields_options[object_fields_dropdown.value]])  #layout=widgets.Layout(height='500px')
    ], layout=widgets.Layout(height='150px'))

    def object_fields_panel_change(_):
        object_fields_panel.children[1].children = [object_fields_options[object_fields_dropdown.value]]

    object_fields_dropdown.observe(object_fields_panel_change, names='value')

    def get_fields_list(field_dict):
        plot_field = []
        for field_widget in field_dict.values():
            if field_widget.children[0].value:
                plot_field.append(field_widget.children[2].value)

        return plot_field

    def fetch_uncertainties():
        uncerts = []
        if hasattr(system, "data_channel_options"):
            for key, data_widget in system.data_channel_options.items():
                if data_widget.children[0].value:
                    uncerts.append(np.asarray(data_widget.children[3].value.split(",")).astype(float))

        return uncerts

    def show_selection(
        line_ids, downsampling, plot_uncert, scale
    ):
        workspace = Workspace(h5file)
        entity = get_parental_child(objects.value)[0]

        if plot_uncert:
            uncerts = fetch_uncertainties()
        else:
            uncerts = None

        locations = entity.vertices
        parser = np.ones(locations.shape[0], dtype='bool')

        fig = plt.figure(figsize=(12, 8))
        ax1 = plt.subplot(2, 1, 1)
        ax2 = plt.subplot(2, 1, 2)
        if hasattr(system, "data_channel_options"):
            plot_field = get_fields_list(system.data_channel_options)

            if entity.get_data(plot_field[0]):
                data = entity.get_data(plot_field[0])[0]
                plot_plan_data_selection(
                        entity, data, **{
                            "highlight_selection": {line_field.value: line_ids},
                            "downsampling": downsampling,
                            "ax": ax1,
                            "color_norm": colors.SymLogNorm(linthresh=np.percentile(np.abs(data.values), 5))
                        }
                )

                if plot_profile:
                    ax2, threshold = plot_profile_data_selection(
                        entity, plot_field,
                        selection={line_field.value: line_ids},
                        downsampling=downsampling,
                        uncertainties=uncerts,
                        ax=ax2
                    )
                    plt.yscale(scale, linthreshy=threshold)

    uncert_mode = widgets.RadioButtons(
        options=['Estimated (%|data| + background)', 'User input (\%|data| + floor)'],
        value='Estimated (%|data| + background)',
        disabled=False
    )

    # uncert_mode.observe(uncert_values_active)

    uncert_panel = widgets.VBox(
        [Label("Apply to:"), uncert_mode],
        layout=widgets.Layout(width='50%')
    )

    plot_uncert = widgets.Checkbox(
        value=False,
        description="Plot uncertainties"
        # indent=False
    )

    data_selection_panel = VBox([
        lines, downsampling, plot_uncert, scale
    ], layout=Layout(width="50%"))

    interactive_plot = widgets.interactive_output(
                show_selection, {
                    "line_ids": lines,
                    "downsampling": downsampling,
                    "plot_uncert": plot_uncert,
                    "scale": scale,
                }
    )
    data_panel = widgets.HBox([
            data_selection_panel,
            HBox([interactive_plot], layout=Layout(width="50%"))]
    )

    ############# Inversion panel ###########
    input_file = "simpegEM1D_inputs.json"

    def write_unclick(_):
        if write.value:
            workspace = Workspace(h5file)
            entity = get_parental_child(objects.value)[0]
            input_dict = {}
            input_dict["system"] = system.value
            input_dict["topo"] = topo.value
            input_dict['workspace'] = h5file
            input_dict['entity'] = entity.name
            input_dict['lines'] = {line_field.value: [str(line) for line in lines.value]}
            input_dict['downsampling'] = str(downsampling.value)
            input_dict['chi_factor'] = [chi_factor.value]
            input_dict['out_group'] = out_group.value

            if ref_mod.children[1].children[1].children:
                input_dict['reference'] = ref_mod.children[1].children[1].children[0].value
            else:
                input_dict['reference'] = []

            input_dict["data"] = {}

            input_dict["uncert"] = {"mode": uncert_mode.value}
            input_dict["uncert"]['channels'] = {}

            if em_system_specs[system.value]['type'] == 'time' and hasattr(system, "data_channel_options"):
                data_widget = list(system.data_channel_options.values())[0]
                input_dict['rx_offsets'] = np.asarray(data_widget.children[4].value.split(",")).astype(float).tolist()
            else:
                input_dict['rx_offsets'] = {}

            if hasattr(system, "data_channel_options"):
                for key, data_widget in system.data_channel_options.items():
                    if data_widget.children[0].value:
                        input_dict["data"][key] = data_widget.children[2].value
                        input_dict["uncert"]['channels'][key] = np.asarray(data_widget.children[3].value.split(",")).astype(float).tolist()

                        if em_system_specs[system.value]['type'] == 'frequency':
                            input_dict['rx_offsets'][key] = np.asarray(data_widget.children[4].value.split(",")).astype(float).tolist()

            with open(input_file, 'w') as f:
                json.dump(input_dict, f)

            write.value = False

    def invert_unclick(_):
        if invert.value:

            if em_system_specs[system.value]['type'] == 'time':
                prompt = os.system("start cmd.exe @cmd /k " + f"\"python functions/tem1d_inversion.py {input_file}\"")
            else:
                prompt = os.system("start cmd.exe @cmd /k " + f"\"python functions/fem1d_inversion.py {input_file}\"")

            invert.value = False

    def update_ref(_):

        if ref_mod.children[1].children[0].value == 'Best-fitting halfspace':

            ref_mod.children[1].children[1].children = []

        elif ref_mod.children[1].children[0].value == 'Model':

            model_list = []

            for obj in workspace.all_objects():
                if isinstance(obj, BlockModel):

                    for data in obj.children:

                        if getattr(data, "values", None) is not None:

                            model_list += [data.name]

            ref_mod.children[1].children[1].children = [widgets.Dropdown(
                description='3D Model',
                options=model_list,
            )]

        else:

            ref_mod.children[1].children[1].children = [widgets.FloatText(
                description='S/m',
                value=1e-3,
            )]

    invert = widgets.ToggleButton(
        value=False,
        description='Invert',
        button_style='danger',
        tooltip='Run simpegEM1D',
        icon='check'
    )

    invert.observe(invert_unclick)

    out_group = widgets.Text(
        value='Inversion_',
        description='Save to:',
        disabled=False
    )

    write = widgets.ToggleButton(
        value=False,
        description='Write input',
        button_style='',
        tooltip='Write json input file',
        icon='check'
    )

    write.observe(write_unclick)

    chi_factor = widgets.FloatText(
        value=1,
        description='Target misfit',
        disabled=False
    )

    ref_type = widgets.RadioButtons(
        options=['Best-fitting halfspace', 'Model', 'Value'],
        value='Best-fitting halfspace',
        disabled=False
    )

    ref_type.observe(update_ref)
    ref_mod = widgets.VBox([Label('Reference conductivity'), widgets.VBox([ref_type, widgets.VBox([])])])

    inversion_options = {
        "output name": out_group,
        "target misfit": chi_factor,
        "reference model": ref_mod,
        "uncertainties": uncert_panel
    }

    option_choices = widgets.Dropdown(
            options=list(inversion_options.keys()),
            value=list(inversion_options.keys())[0],
            disabled=False
    )

    def inv_option_change(_):
        inversion_panel.children[1].children = [option_choices, inversion_options[option_choices.value]]

    option_choices.observe(inv_option_change)
    inversion_panel = widgets.VBox([
        widgets.HBox([widgets.Label("Inversion Options")]),
        widgets.HBox([option_choices, inversion_options[option_choices.value]], )  #layout=widgets.Layout(height='500px')
    ], layout=Layout(width="100%"))

    # Trigger all observers
    if object_name is not None and object_name in names:
        objects.value = object_name
    else:
        objects.value = names[0]

    object_observer("")

    return widgets.VBox([
        HBox([
            VBox([Label("EM survey"), objects, system, components]),
            VBox([Label("Parameters"), object_fields_panel])
        ]),
        data_panel, inversion_panel, write, invert
    ])
