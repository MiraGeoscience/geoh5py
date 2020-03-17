import sys
import os
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.colors as colors

from skimage.feature import canny
from skimage.transform import probabilistic_hough_line

import ipywidgets as widgets
from ipywidgets.widgets import Label, Dropdown, Layout, VBox, HBox

from scipy.spatial import cKDTree

from geoh5io.workspace import Workspace
from geoh5io.objects import Grid2D, Curve, Points, Surface, BlockModel
from geoh5io.data import Data
from geoh5io.groups import ContainerGroup
import json


def object_data_selection(h5file, plot=False):
    """

    """
    workspace = Workspace(h5file)

    def listObjects(obj_name, data_name):
        obj = workspace.get_entity(obj_name)[0]

        data = obj.get_data(data_name)[0]

        if plot:
            fig = plt.figure(figsize=(10, 10))
            axs = plt.subplot()

            if data.entity_type.color_map is not None:
                new_cmap = data.entity_type.color_map.values
                values = new_cmap['Value']
                values -= values.min()
                values /= values.max()

                cdict = {
                    'red': np.c_[values, new_cmap['Red']/255, new_cmap['Red']/255].tolist(),
                    'green': np.c_[values, new_cmap['Green']/255, new_cmap['Green']/255].tolist(),
                    'blue': np.c_[values, new_cmap['Blue']/255, new_cmap['Blue']/255].tolist(),
                }
                cmap = colors.LinearSegmentedColormap(
                    'custom_map', segmentdata=cdict, N=len(values)
                )

            else:
                cmap = None

            values = None
            if isinstance(getattr(data, "values", None), np.ndarray):
                if not isinstance(data.values[0], str):
                    values = data.values

            if isinstance(obj, Grid2D) and values is not None:
                X = obj.centroids[:, 0].reshape(obj.shape, order="F")
                Y = obj.centroids[:, 1].reshape(obj.shape, order="F")
                obj_data = values.reshape(obj.shape, order="F")
                out = plt.pcolormesh(X, Y, obj_data, cmap=cmap)
            elif isinstance(obj, Points) or isinstance(obj, Curve):
                X, Y = obj.vertices[:, 0], obj.vertices[:, 1]
                out = plt.scatter(
                    X, Y, 5, values, cmap=cmap)
            else:
                print("Sorry, 'plot=True' option only implemented for Grid2D, Points and Curve objects")
            format_labels(X, Y, axs)
            plt.colorbar(out)

        return obj, data

    names = list(workspace.list_objects_name.values())

    def updateList(_):
        obj = workspace.get_entity(objects.value)[0]
        data.options = obj.get_data_list()

    objects = widgets.Dropdown(
        options=names,
        value=names[0],
        description='Object:',
    )

    obj = workspace.get_entity(objects.value)[0]
    data = widgets.Dropdown(
        options=obj.get_data_list(),
        value=obj.get_data_list()[0],
        description='Data: ',
    )

    objects.observe(updateList)

    out = widgets.interactive(
        listObjects, obj_name=objects, data_name=data
    )

    return out


def coordinate_transformation(
    h5file, plot=False, epsg_in=None, epsg_out=None, object_names=[]
):
    """

    """

    from pyproj import Proj, transform

    workspace = Workspace(h5file)

    def listObjects(obj_names, epsg_in, epsg_out, export, plot_it):

        out_list = []
        if epsg_in != 0 and epsg_out != 0:
            try:
                inProj = Proj(f'epsg:{int(epsg_in)}')
                labels_in = None
                if str(inProj.definition).find('longlat') != -1:
                    labels_in = ['Lon', "Lat"]
            except Exception:
                print(sys.exc_info()[1])

            try:
                outProj = Proj(f'epsg:{int(epsg_out)}')
                labels_out = None
                if str(outProj.definition).find('longlat') != -1:
                    labels_out = ['Lon', "Lat"]
            except Exception:
                print(sys.exc_info()[1])

            if plot_it:
                fig = plt.figure(figsize=(10, 10))
                ax1 = plt.subplot(1, 2, 1)
                ax2 = plt.subplot(1, 2, 2)
                X1, Y1, X2, Y2 = [], [], [], []

            if export:
                group = ContainerGroup.create(
                    workspace,
                    name=f'Projection epsg:{int(epsg_out)}'
                )
            for name in obj_names:
                obj = workspace.get_entity(name)[0]

                if not hasattr(obj, 'vertices'):
                    print(f"Skipping {name}. Entity dies not have vertices")
                    continue

                x, y = obj.vertices[:, 0], obj.vertices[:, 1]
                if str(inProj.definition).find('longlat') != -1:
                    x, y = y, x

                x2, y2 = transform(inProj, outProj, x, y)
                if str(outProj.definition).find('longlat') != -1:
                    x2, y2 = y2, x2

                if export:
                    # Save the result to geoh5
                    out_list.append(
                        type(obj).create(
                            obj.workspace,
                            name=name,
                            parent=group,
                            vertices=np.c_[x2, y2, obj.vertices[:, 2]],
                            cells=getattr(obj, 'cells', None)
                        )
                    )

                if plot_it:
                    ax1.scatter(x, y, 5)
                    ax2.scatter(x2, y2, 5)
                    X1.append(x), Y1.append(y), X2.append(x2), Y2.append(y2)

            if plot_it:
                format_labels(np.hstack(X1), np.hstack(Y1), ax1, labels=labels_in)
                format_labels(np.hstack(X2), np.hstack(Y2), ax2, labels=labels_out)

        return out_list

    names = []
    for obj in workspace._objects.values():
        if isinstance(obj.__call__(), (Curve, Points, Surface)):
            names.append(obj.__call__().name)

    def saveIt(_):
        if export.value:
            export.value = False
            print('Export completed!')

    objects = widgets.SelectMultiple(
        options=names,
        value=object_names,
        description='Object:',
    )

    export = widgets.ToggleButton(
        value=False,
        description='Export to GA',
        button_style='',
        tooltip='Description',
        icon='check'
        )

    export.observe(saveIt)
    plot_it = widgets.ToggleButton(
        value=True,
        description='Plot',
        button_style='',
        tooltip='Description',
        icon='check'
        )

    # export.observe(saveIt)
    epsg_in = widgets.FloatText(
        value=epsg_in,
        description='EPSG # in:',
        disabled=False
    )

    epsg_out = widgets.FloatText(
        value=epsg_out,
        description='EPSG # out:',
        disabled=False
    )

    out = widgets.interactive(
        listObjects, obj_names=objects, epsg_in=epsg_in, epsg_out=epsg_out, export=export, plot_it=plot_it
    )

    return out


def edge_detection(
    grid: Grid2D, data: Data,
    sigma=1.0, threshold=3, line_length=4.0, line_gap=2
):
    """
    Widget for Grid2D objects for the automated detection of line features.
    The application relies on the Canny and Hough tranforms from the
    Scikit-Image library.

    :param grid: Grid2D object
    :param data: Children data object for the provided grid

    Optional
    --------

    :param sigma [Canny]: standard deviation of the Gaussian filter
    :param threshold [Hough]: Value threshold
    :param line_length [Hough]: Minimum accepted pixel length of detected lines
    :param line_gap [Hough]: Maximum gap between pixels to still form a line.
    """

    assert isinstance(grid, Grid2D), "This application is only designed for Grid2D objects"

    X = grid.centroids[:, 0].reshape(grid.shape, order="F")
    Y = grid.centroids[:, 1].reshape(grid.shape, order="F")
    grid_data = data.values.reshape(grid.shape, order="F")

    def compute_plot(sigma, threshold, line_length, line_gap, export_as, export):

        # Parameters controling the edge detection
        edges = canny(grid_data.T, sigma=sigma)

        # Parameters controling the line detection
        lines = probabilistic_hough_line(edges, line_length=line_length, threshold=threshold, line_gap=line_gap, seed=0)

        if data.entity_type.color_map is not None:
                new_cmap = data.entity_type.color_map.values
                values = new_cmap['Value']
                values -= values.min()
                values /= values.max()

                cdict = {
                    'red': np.c_[values, new_cmap['Red']/255, new_cmap['Red']/255].tolist(),
                    'green': np.c_[values, new_cmap['Green']/255, new_cmap['Green']/255].tolist(),
                    'blue': np.c_[values, new_cmap['Blue']/255, new_cmap['Blue']/255].tolist(),
                }
                cmap = colors.LinearSegmentedColormap(
                    'custom_map', segmentdata=cdict, N=len(values)
                )

        else:
            cmap = None

        fig = plt.figure(figsize=(10, 10))
        axs = plt.subplot()
        plt.pcolormesh(X, Y, grid_data, cmap=cmap)
        format_labels(X, Y, axs)

        xy = []
        cells = []
        count = 0

        for line in lines:
            p0, p1 = line

            points = np.r_[
                np.c_[X[p0[0], 0], Y[0, p0[1]], 0],
                np.c_[X[p1[0], 0], Y[0, p1[1]], 0]
                ]
            xy.append(points)

            cells.append(np.c_[count, count+1].astype("uint32"))

            count += 2

            plt.plot(points[:, 0], points[:, 1], 'k--', linewidth=2)

        if export:
            # Save the result to geoh5
            curve = Curve.create(
                grid.workspace,
                name=export_as,
                vertices=np.vstack(xy),
                cells=np.vstack(cells)
            )

        return lines

    def saveIt(_):
        if export.value:
            export.value = False
            print(f'Lines {export_as.value} exported to: {grid.workspace.h5file}')

    def saveItAs(_):
        export_as.value = (
            f"S:{sigma.value}" + f" T:{threshold.value}" + f" LL:{line_length.value}" + f" LG:{line_gap.value}"
        )

    export = widgets.ToggleButton(
        value=False,
        description='Export to GA',
        button_style='',
        tooltip='Description',
        icon='check'
        )

    export.observe(saveIt)

    sigma = widgets.FloatSlider(
        min=0., max=10, step=0.1, value=sigma, continuous_update=False
    )

    sigma.observe(saveItAs)

    line_length = widgets.IntSlider(
        min=1., max=10., step=1., value=line_length, continuous_update=False
    )
    line_length.observe(saveItAs)

    line_gap = widgets.IntSlider(
        min=1., max=10., step=1., value=line_gap, continuous_update=False
    )
    line_gap.observe(saveItAs)

    threshold = widgets.IntSlider(
        min=1., max=10., step=1., value=threshold, continuous_update=False
    )
    threshold.observe(saveItAs)

    export_as = widgets.Text(
        value=(f"S:{sigma.value}" + f" T:{threshold.value}" + f" LL={line_length.value}" + f" LG={line_gap.value}"),
        description="Save as:",
        disabled=False
    )

    out = widgets.interactive(
        compute_plot,
        sigma=sigma,
        threshold=threshold,
        line_length=line_length,
        line_gap=line_gap,
        export_as=export_as,
        export=export,
    )

    return out


def format_labels(x, y, axs, labels=None, aspect='equal'):
    if labels is None:
        axs.set_ylabel("Northing (m)")
        axs.set_xlabel("Easting (m)")
    else:
        axs.set_xlabel(labels[0])
        axs.set_ylabel(labels[1])
    xticks = np.linspace(x.min(), x.max(), 5, dtype=int)
    yticks = np.linspace(y.min(), y.max(), 5, dtype=int)
    axs.set_yticks(yticks)
    axs.set_yticklabels(yticks, rotation=90, va='center')
    axs.set_xticks(xticks)
    axs.set_xticklabels(xticks, va='center')
    axs.set_aspect(aspect)


def plot_profile_data_selection(
    entity, field_list,
    uncertainties=None, selection={}, downsampling=None,
    plot_legend=False, ax=None, color=[0, 0, 0]
):

    locations = entity.vertices

    if downsampling is None:
        downsampling = np.ones(locations.shape[0], dtype='bool')

    if ax is None:
        fig = plt.figure(figsize=(8, 8))
        ax = plt.subplot()

    pos = ax.get_position()
    xx, yy = [], []
    threshold = 1e-14
    for key, values in selection.items():

        for line in values:

            if entity.get_data(key):
                ind = np.where((entity.get_data(key)[0].values == line) * downsampling)[0]
            else:
                continue
            if len(ind) == 0:
                continue

            xyLocs = locations[ind, :]

            if np.std(xyLocs[:, 0]) > np.std(xyLocs[:, 1]):
                dist = xyLocs[:, 0].copy()
            else:
                dist = xyLocs[:, 1].copy()

            dist -= dist.min()
            order = np.argsort(dist)
            legend = []

            c_increment = [(1-c)/(len(field_list)+1) for c in color]

            for ii, field in enumerate(field_list):
                if entity.get_data(field) and entity.get_data(field)[0].values is not None:
                    values = entity.get_data(field)[0].values[ind][order]

                    xx.append(dist[order][~np.isnan(values)])
                    yy.append(values[~np.isnan(values)])

                    if uncertainties is not None:
                        ax.errorbar(
                            xx[-1], yy[-1],
                            yerr=uncertainties[ii][0]*np.abs(yy[-1]) + uncertainties[ii][1],
                            color=[c+ii*i for c, i in zip(color, c_increment)]
                        )
                    else:
                        ax.plot(xx[-1], yy[-1], color=[c+ii*i for c, i in zip(color, c_increment)])
                    legend.append(field)

                    threshold = np.max([
                        threshold,
                        np.percentile(
                            np.abs(yy[-1]), 2)
                    ])

            if plot_legend:
                ax.legend(legend, loc=3, bbox_to_anchor=(0, -0.25), ncol=3)

            if xx and yy:
                format_labels(
                    np.hstack(xx),
                    np.hstack(yy),
                    ax,
                    labels=["Distance (m)", "Fields"],
                    aspect='auto'
                )

    # ax.set_position([pos.x0, pos.y0, pos.width*2., pos.height])
    # ax.set_aspect(1)
    return ax, threshold


def plot_plan_data_selection(
    entity, field_list,
    selection={}, downsampling=None,
    ax=None
):

    locations = entity.vertices

    if downsampling is None:
        downsampling = np.ones(locations.shape[0], dtype='bool')

    if ax is None:
        plt.figure(figsize=(8, 8))
        ax = plt.subplot()

    if entity.get_data(field_list[0]):
        data = entity.get_data(field_list[0])[0].values[downsampling]
        ax.scatter(
            locations[downsampling, 0],
            locations[downsampling, 1], 1,
            data, marker=',',
            norm=colors.SymLogNorm(linthresh=np.percentile(np.abs(data), 5))
        )
        xx, yy = [], []
        for key, values in selection.items():

            for line in values:

                ind = np.where((entity.get_data(key)[0].values == line) * downsampling)[0]

                xyLocs = locations[ind, :]

                ax.scatter(xyLocs[:, 0], xyLocs[:, 1], 3, 'r')

        ax.set_aspect("equal")
        format_labels(locations[:, 0], locations[:, 1], ax)

    return ax


def find_value(labels, strings):
    value = None
    for name in labels:
        for string in strings:
            if string.lower() in name.lower():
                value = name
    return value


def pf_survey_setup(h5file, plot=False):
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


def em1d_inversion_setup(h5file, plot_profile=True, start_channel=None):

    workspace = Workspace(h5file)

    curves = [entity.parent.name + "." + entity.name for entity in workspace.all_objects() if isinstance(entity, Curve)]
    names = [name for name in sorted(curves)]

    # Load all known em systems
    with open("functions/AEM_systems.json", 'r') as aem_systems:
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
            component_list.append(pg.group_name)

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
                        value = find_value(data_list, [p_g.group_name+key])
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


def plot_em_data(h5file):
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

        options = [pg.group_name for pg in obj.property_groups]
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

    options = [pg.group_name for pg in obj.property_groups]
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

