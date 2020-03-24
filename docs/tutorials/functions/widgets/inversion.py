import os
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from ipywidgets.widgets import Label, Dropdown, Layout, VBox, HBox
from scipy.spatial import cKDTree

from geoh5io.workspace import Workspace
from geoh5io.objects import Curve, BlockModel
import json
from ..utils import plot_profile_data_selection, plot_plan_data_selection, find_value


def pf_inversion_widget(h5file, plot=False):
    """

    """
    workspace = Workspace(h5file)

    names = list(workspace.list_objects_name.values())

    def updateList(_):
        obj = workspace.get_entity(objects.value)[0]
        data.options = obj.get_data_list()

    def create_widgets(_):
        if data.value:
            options = ['TMI', 'gx']
            field_type = []
            for val in data.value:
                dtype = widgets.Dropdown(
                    options=options,
                    value='TMI',
                    description=val.value + ' Type:',

                    )

                uncert = widgets.Text(
                    value="0, 0"
                    )

                field_type.append(VBox([dtype, uncert]))

            field_param.children = field_type

    objects = widgets.Dropdown(
        options=names,
        value=names[0],
        description='Object:',
    )

    topo = widgets.Dropdown(
        options=names,
        value=find_value(names, ["topo"]),
        description='Topo:',
    )

    entity = workspace.get_entity(objects.value)[0]
    data = widgets.widgets.SelectMultiple(
            options=entity.get_data_list(),
            description=f"Data fields:",

        )

    data.observe(create_widgets)

    objects.observe(updateList)

    field_param = HBox([])

    stack = VBox([objects, topo, data, field_param])

    return stack


def em1d_inversion_widget(h5file, plot_profile=True, start_channel=None, object_name=None):

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
    ], layout=widgets.Layout(height='225px'))

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

        parser = None
        if downsampling > 0:
            locations = entity.vertices
            xx = np.arange(locations[:, 0].min()-downsampling, locations[:, 0].max()+downsampling, downsampling)
            yy = np.arange(locations[:, 1].min()-downsampling, locations[:, 1].max()+downsampling, downsampling)

            X, Y = np.meshgrid(xx, yy)

            tree = cKDTree(locations[:, :2])
            rad, ind = tree.query(np.c_[X.ravel(), Y.ravel()])
            keep = np.unique(ind[rad < downsampling])

            parser = np.zeros(locations.shape[0], dtype='bool')
            parser[keep] = True

        fig = plt.figure(figsize=(12, 8))
        ax1 = plt.subplot(1, 2, 1)
        ax2 = plt.subplot(1, 2, 2)
        if hasattr(system, "data_channel_options"):
            plot_field = get_fields_list(system.data_channel_options)
        else:
            plot_field = []

        plot_plan_data_selection(
                entity, plot_field,
                selection={line_field.value: line_ids},
                downsampling=parser,
                ax=ax1
        )
        if plot_profile:
            ax2, threshold = plot_profile_data_selection(
                entity, plot_field,
                selection={line_field.value: line_ids},
                downsampling=parser,
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

    # Trigger all observers
    if object_name is not None and object_name in names:
        objects.value = object_name
    else:
        objects.value = names[0]

    object_observer("")

    data_panel = widgets.HBox([
            data_selection_panel,
            widgets.interactive_output(
                show_selection, {
                    "line_ids": lines,
                    "downsampling": downsampling,
                    "plot_uncert": plot_uncert,
                    "scale": scale,
                }
            )]
    )

    ############# Inversion panel ###########
    input_file = "simpegEM1D_inputs.json"

    def write_unclick(_):
        if write.value:
            workspace = Workspace(h5file)
            entity = get_parental_child(objects.value)[0]
            input_params = {}
            input_params["system"] = system.value
            input_params["topo"] = topo.value
            input_params['workspace'] = h5file
            input_params['entity'] = entity.name
            input_params['lines'] = {line_field.value: [str(line) for line in lines.value]}
            input_params['downsampling'] = str(downsampling.value)
            input_params['chi_factor'] = [chi_factor.value]
            input_params['out_group'] = out_group.value

            if ref_mod.children[1].children[1].children:
                input_params['reference'] = ref_mod.children[1].children[1].children[0].value
            else:
                input_params['reference'] = []

            input_params["data"] = {}

            input_params["uncert"] = {"mode": uncert_mode.value}
            input_params["uncert"]['channels'] = {}

            if em_system_specs[system.value]['type'] == 'time' and hasattr(system, "data_channel_options"):
                data_widget = list(system.data_channel_options.values())[0]
                input_params['rx_offsets'] = np.asarray(data_widget.children[4].value.split(",")).astype(float).tolist()
            else:
                input_params['rx_offsets'] = {}

            if hasattr(system, "data_channel_options"):
                for key, data_widget in system.data_channel_options.items():
                    if data_widget.children[0].value:
                        input_params["data"][key] = data_widget.children[2].value
                        input_params["uncert"]['channels'][key] = np.asarray(data_widget.children[3].value.split(",")).astype(float).tolist()

                        if em_system_specs[system.value]['type'] == 'frequency':
                            input_params['rx_offsets'][key] = np.asarray(data_widget.children[4].value.split(",")).astype(float).tolist()

            with open(input_file, 'w') as f:
                json.dump(input_params, f)

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
    return widgets.VBox([
        HBox([
            VBox([Label("EM survey"), objects, system, components]),
            VBox([Label("Parameters"), object_fields_panel])
        ]),
        data_panel, inversion_panel, write, invert
    ])
