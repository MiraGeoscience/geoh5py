import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from ipywidgets.widgets import Dropdown, VBox, HBox

from geoh5io.workspace import Workspace
from geoh5io.objects import Curve
from ..utils import plot_profile_data_selection


def plot_em_data_widget(h5file):
    workspace = Workspace(h5file)

    curves = [entity.parent.name + "." + entity.name for entity in workspace.all_objects() if isinstance(entity, Curve)]
    names = [name for name in sorted(curves)]

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

    def plot_profiles(
        entity_name, groups, line_field, lines, scale
    ):

        fig = plt.figure(figsize=(12, 12))
        entity = get_parental_child(entity_name)[0]

        ax = plt.subplot()
        colors = [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ]

        threshold = 1e-16
        for group, color in zip(groups, colors):

            prop_group = entity.get_property_group(group)

            if prop_group is not None:
                fields = [entity.workspace.get_entity(uid)[0].name for uid in prop_group.properties]

                ax, threshold = plot_profile_data_selection(
                    prop_group.parent, fields,
                    selection={line_field: lines},
                    ax=ax,
                    color=color
                )

        ax.grid(True)
        plt.yscale(scale, linthreshy=threshold)

    def updateList(_):
        entity = get_parental_child(objects.value)[0]
        data_list = entity.get_data_list()
        obj = get_parental_child(objects.value)[0]

        options = [pg.name for pg in obj.property_groups]
        options = [option for option in sorted(options)]
        groups.options = options
        groups.value = [groups.options[0]]
        line_field.options = data_list
        line_field.value = find_value(data_list, ['line'])

        if line_field.value is None:
            line_ids = []
            value = []
        else:
            line_ids = np.unique(entity.get_data(line_field.value)[0].values)
            value = [line_ids[0]]

        lines.options = line_ids
        lines.value = value

    objects = widgets.Dropdown(
        options=names,
        value=names[0],
        description='Object:',
    )

    obj = get_parental_child(objects.value)[0]

    order = np.sort(obj.vertices[:, 0])

    entity = get_parental_child(objects.value)[0]

    data_list = entity.get_data_list()
    line_field = Dropdown(
        options=data_list,
        value=find_value(data_list, ["line"]),
        description='Lines field',
    )

    options = [pg.name for pg in obj.property_groups]
    options = [option for option in sorted(options)]
    groups = widgets.SelectMultiple(
        options=options,
        value=[options[0]],
        description='Data: ',
    )

    if line_field.value is None:
        line_list = []
        value = []
    else:

        line_list = np.unique(entity.get_data(line_field.value)[0].values)
        value = [line_list[0]]

    lines = widgets.SelectMultiple(
        options=line_list,
        value=value,
        description='Data: '
    )

    objects.observe(updateList, names='value')

    scale = Dropdown(
        options=['linear', 'symlog'],
        value='symlog',
        description='Scaling',
    )

    apps = VBox([objects, line_field, lines, groups, scale])
    layout = HBox([
        apps,
        widgets.interactive_output(
                    plot_profiles, {
                        "entity_name": objects,
                        "groups": groups,
                        "line_field": line_field,
                        "lines": lines,
                        "scale": scale,
                        }
                    )
    ])
    return layout

