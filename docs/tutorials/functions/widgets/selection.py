import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import ipywidgets as widgets
from geoh5io.workspace import Workspace
from geoh5io.objects import Grid2D, Curve, Points
from ..utils import format_labels


def object_data_selection_widget(h5file, plot=False):
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
        workspace = Workspace(h5file)
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

    objects.observe(updateList, names='value')

    out = widgets.interactive(
        listObjects, obj_name=objects, data_name=data
    )

    return objects, data