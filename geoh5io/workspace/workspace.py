from __future__ import annotations

import uuid
import weakref
from contextlib import contextmanager
from typing import TYPE_CHECKING, Callable, Dict, Optional, Type, Union, cast

from geoh5io.shared import EntityType

if TYPE_CHECKING:
    from geoh5io.groups import group
    from geoh5io.objects import object
    from geoh5io.shared import type


class Workspace:
    _active_ref: Union[
        weakref.ReferenceType, Callable[[], Optional[Workspace]]
    ] = lambda: None

    def __init__(self):
        self.version = None
        self._distance_unit = None
        self._contributors = []
        self._groups: Dict[uuid.UUID, group.Group] = {}
        self._objects: Dict[uuid.UUID, object.Object] = {}
        self._types: Dict[uuid.UUID, type.EntityType] = {}

        # TODO: must always have a root group (cannot be None)
        # self._root: group.Group

    def activate(self):
        """ Makes this workspace the active one.

            In case te workspace gets deleted, Workspace.active() safely returns None.
        """
        if Workspace._active_ref() is not self:
            Workspace._active_ref = weakref.ref(self)

    def deactivate(self):
        """ Deactivate this workspace if it was the active one, else does nothing.
        """
        if Workspace._active_ref() is self:
            Workspace._active_ref = lambda: None

    @staticmethod
    def active() -> Workspace:
        """ Get the active workspace. """
        active_one = Workspace._active_ref()
        if active_one is None:
            raise RuntimeError("No active workspace.")

        # so that type check does not complain of possible returned None
        return cast(Workspace, active_one)

    def find_type(
        self, type_uid: uuid.UUID, type_class: Type["type.EntityType"]
    ) -> Optional[EntityType]:
        found_type = self._types.get(type_uid, None)
        if found_type is not None and isinstance(found_type, type_class):
            return found_type

        return None


@contextmanager
def active_workspace(workspace: Workspace):
    previous_active_ref = Workspace._active_ref
    workspace.activate()
    yield workspace

    workspace.deactivate()
    # restore previous active workspace when leaving the context
    previous_active = previous_active_ref()
    if previous_active is not None:
        previous_active.activate()
