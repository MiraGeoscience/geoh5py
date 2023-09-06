#  Copyright (c) 2023 Mira Geoscience Ltd.
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

import pytest

from geoh5py.shared.exceptions import AggregateValidationError, TypeValidationError
from geoh5py.ui_json.enforcers import TypeEnforcer, ValueEnforcer
from geoh5py.ui_json.validation import Validations


def test_validate_raises_single_error():
    validations = Validations("my_param", [TypeEnforcer(str)])
    validations.validate("1")
    msg = "Type 'int' provided for 'my_param' is invalid. " "Must be: 'str'."
    with pytest.raises(TypeValidationError, match=msg):
        validations.validate(1)


def test_validate_raises_aggregate_error():
    validations = Validations(
        "my_param", [TypeEnforcer(str), ValueEnforcer(["onlythis"])]
    )
    validations.validate("onlythis")
    msg = (
        "Validation of 'my_param' collected 2 errors:\n\t"
        "0. Type 'int' provided for 'my_param' is invalid"
    )
    with pytest.raises(AggregateValidationError, match=msg):
        validations.validate(1)
