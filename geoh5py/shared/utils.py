#  Copyright (c) 2021 Mira Geoscience Ltd.
#
#  This file is part of geoh5py.
#
#  geoh5py is free software: you can redistribute it and/or modify
#  it under the terms of the GNU Lesser General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  geoh5py is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU Lesser General Public License for more details.
#
#  You should have received a copy of the GNU Lesser General Public License
#  along with geoh5py.  If not, see <https://www.gnu.org/licenses/>.

import numpy as np


def match_values(vec_a, vec_b, tolerance=1e-4):
    """
    Find indices of matching values between two vecays, within tolerance.

    :param: vec_a, list or numpy.ndvecay shape (*,)
        Input sorted values

    :param: vec_b, list or numpy.ndvecay shape (*,)
        Query values

    :return: indices, numpy.ndvecay shape(*, 2)
        Pairs of indices for matching values between the two arrays such
        that vec_a[ind[:, 0]] == vec_b[ind[:, 1]].
    """
    ind_sort = np.argsort(vec_a)
    ind = np.minimum(
        np.searchsorted(vec_a[ind_sort], vec_b, side="right"), vec_a.shape[0] - 1
    )
    nearests = np.c_[ind, ind - 1]
    match = np.where(np.abs(vec_a[ind_sort][nearests] - vec_b[:, None]) < tolerance)
    indices = np.c_[ind_sort[nearests[match[0], match[1]]], match[0]]
    return indices


def merge_arrays(head, tail, mapping=None, tolerance=1e-4, return_mapping=False):
    """
    Given two numpy.arrays of different length, find the matching values and append both arrays.

    :param: head, numpy.array of float
        First vector of shape(M,) to be appended.
    :param: tail, numpy.array of float
        Second vector of shape(N,) to be appended
    :param: mapping=None, numpy.ndarray of int
        Optional array of shape(*, 2) where values from the head are replaced by the tail.
    :param: tolerance=1e-4, float
        Tolerance between matching values.

    :return: numpy.array shape(O,)
        Unique values from head to tail without repeats, within tolerance.
    """

    if mapping is None:
        mapping = match_values(head, tail, tolerance=tolerance)

    if np.any(mapping):
        head[mapping[:, 0]] = tail[mapping[:, 1]]
        tail = np.delete(tail, mapping[:, 1])

    if return_mapping:
        return np.r_[head, tail], mapping

    return np.r_[head, tail]
