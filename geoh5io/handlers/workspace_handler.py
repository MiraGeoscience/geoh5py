from typing import List

from geoh5io import interfaces


class WorkspaceHandler:
    @staticmethod
    def get_api_version() -> interfaces.shared.VersionString:
        version = interfaces.shared.VersionString()
        version.value = interfaces.api.API_VERSION
        return version

    def create_geoh5(self, file_path: str) -> interfaces.workspace.Workspace:
        # TODO
        pass

    # pylint: disable=unused-argument
    @staticmethod
    def open_geoh5(file_path: str) -> interfaces.workspace.Workspace:
        # TODO
        return interfaces.workspace.Workspace()

    def save(
        self, file_path: str, overwrite_file: bool
    ) -> interfaces.workspace.Workspace:
        # TODO
        pass

    def save_copy(
        self, file_path: str, overwrite_file: bool
    ) -> interfaces.workspace.Workspace:
        # TODO
        pass

    def export_objects(
        self,
        objects_or_groups: List[interfaces.shared.Uuid],
        file_path: str,
        overwrite_file: bool,
    ) -> interfaces.workspace.Workspace:
        # TODO
        pass

    def close(self,) -> None:
        # TODO
        pass

    def get_contributors(self,) -> List[str]:
        # TODO
        pass
