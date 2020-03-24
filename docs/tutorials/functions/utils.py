import sys
import os
import numpy as np
import re
import matplotlib.pyplot as plt
import matplotlib.colors as colors

from skimage.feature import canny
from skimage.transform import probabilistic_hough_line

import ipywidgets as widgets
from ipywidgets.widgets import Label, Dropdown, Layout, VBox, HBox

from scipy.spatial import cKDTree
from scipy.interpolate import LinearNDInterpolator

from geoh5io.workspace import Workspace
from geoh5io.objects import Grid2D, Curve, Points, Surface, BlockModel
from geoh5io.groups import ContainerGroup
import json


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


def export_curve_2_shapefile(
        curve, attribute=None, epsg=None, file_name=None
):
    from shapely.geometry import mapping, LineString
    import fiona
    from fiona.crs import from_epsg
    import urllib

    if epsg is not None and epsg.isdigit():
        crs = from_epsg(int(epsg))

        wkt = urllib.request.urlopen(
            "http://spatialreference.org/ref/epsg/{0}/prettywkt/".format(
                str(int(epsg))
            )
        )
        # remove spaces between charachters
        remove_spaces = wkt.read().replace(b" ", b"")
        # create the .prj file
        prj = open(file_name + ".prj", "w")

        epsg = remove_spaces.replace(b"\n", b"")
        prj.write(epsg.decode("utf-8"))
        prj.close()
    else:
        crs = None

    if attribute is not None:
        if curve.get_data(attribute):
            attribute = curve.get_data(attribute)[0]

    polylines, values = [], []
    for lid in curve.unique_lines:

        ind_line = np.where(curve.line_id == lid)[0]
        ind_vert = np.r_[
            curve.cells[ind_line, 0], curve.cells[ind_line[-1], 1]
        ]

        polylines += [curve.vertices[ind_vert, :2]]

        if attribute is not None:
            values += [attribute.values[ind_vert]]

    # Define a polygon feature geometry with one attribute
    schema = {'geometry': 'LineString'}

    if attribute:
        attr_name = attribute.name.replace(":", "_")
        schema['properties'] = {attr_name: "float"}
    else:
        schema['properties'] = {"id": "int"}

    with fiona.open(
            file_name + '.shp', 'w', driver='ESRI Shapefile', schema=schema, crs=crs
    ) as c:

        # If there are multiple geometries, put the "for" loop here
        for ii, poly in enumerate(polylines):

            if len(poly) > 1:
                pline = LineString(list(tuple(map(tuple, poly))))

                res = {}
                res['properties'] = {}

                if attribute and values:
                    res['properties'][attr_name] = np.mean(values[ii])
                else:
                    res['properties']["id"] = ii

                # geometry of of the original polygon shapefile
                res['geometry'] = mapping(pline)
                c.write(res)
