import numpy as np
from scipy.spatial import cKDTree


def find_value(labels, strings):
    value = None
    for name in labels:
        for string in strings:
            if (string.lower() in name.lower()) or (name.lower() in string.lower()):
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


def filter_xy(x, y, data, distance, return_indices=False, window=None):
    """
    Downsample xy data based on minimum distance
    """

    filter_xy = np.zeros_like(x, dtype='bool')
    if x.ndim == 1:
        if distance > 0:
            xx = np.arange(x.min() - distance, x.max() + distance, distance)
            yy = np.arange(y.min() - distance, y.max() + distance, distance)

            X, Y = np.meshgrid(xx, yy)

            tree = cKDTree(np.c_[x, y])
            rad, ind = tree.query(np.c_[X.ravel(), Y.ravel()])
            takeout = np.unique(ind[rad < 2**0.5*distance])

            filter_xy[takeout] = True

        else:
            filter_xy = np.ones_like(x, dtype='bool')
    elif distance > 0:

        dwn_x = int(np.ceil(distance / np.min(x[1:] - x[:-1])))
        dwn_y = int(np.ceil(distance / np.min(x[1:] - x[:-1])))
        filter_xy[::dwn_x, ::dwn_y] = True


    mask = np.ones_like(x, dtype='bool')
    if window is not None:
        x_lim = [
            window['center'][0] - window['size'][0] / 2,
            window['center'][0] + window['size'][0] / 2
        ]
        y_lim = [
            window['center'][1] - window['size'][1] / 2,
            window['center'][1] + window['size'][1] / 2
        ]

        xy_rot = rotate_xy(
            np.c_[x.ravel(), y.ravel()], window['center'], window['azimuth']
        )

        mask = (
                (xy_rot[:, 0] > x_lim[0]) *
                (xy_rot[:, 0] < x_lim[1]) *
                (xy_rot[:, 1] > y_lim[0]) *
                (xy_rot[:, 1] < y_lim[1])
            ).reshape(x.shape)

    if data is not None:
        data = data.copy()
        data[(filter_xy * mask)==False] = np.nan

    if x.ndim == 1:
        x, y = x[filter_xy], y[filter_xy]
        if data is not None:
            data = data[filter_xy]
    else:
        x, y = x[::dwn_x, ::dwn_y], y[::dwn_x, ::dwn_y]
        if data is not None:
            data = data[::dwn_x, ::dwn_y]

    if return_indices:
        return x, y, data, filter_xy*mask
    else:
        return x, y, data


def rotate_xy(xyz, center, angle):
    R = np.r_[
        np.c_[np.cos(np.pi * angle / 180), -np.sin(np.pi * angle / 180)],
        np.c_[np.sin(np.pi * angle / 180), np.cos(np.pi * angle / 180)]
    ]

    locs = xyz.copy()
    locs[:, 0] -= center[0]
    locs[:, 1] -= center[1]

    xy_rot = np.dot(R, locs[:, :2].T).T

    return np.c_[xy_rot[:, 0] + center[0], xy_rot[:, 1] + center[1], locs[:, 2:]]