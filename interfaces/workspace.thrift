include "shared.thrift"

namespace * workspace

/**
 * Describes an workspace.
 */
struct Workspace {
    1: string file_path = ""; // empty if this workspace has not been saved yet.
    2: shared.VersionNumber version;
    3: shared.DistanceUnit distance_unit;
    4: shared.DateTime date_created;
    5: shared.DateTime date_modified;
}

/**
 * Exception thrown upon a read/write error on file.
 */
exception FileIOException {
    1: optional string message = "";
}

/**
 * Exception thrown upon a parse error on file.
 */
exception FileFormatException {
    1: optional string message = "";
}

/**
 * Defines operation on the current open workspace.
 *
 * The workspace may be created from scratch, or loaded from a geoh5 file.
 */
service WorkspaceService {
    /**
     * Returns the API version of the server,
     * which may differ from the one of the client (see API_VERSION).
     */
    shared.VersionString get_api_version();

    /**
     * Creates a new empty Workspace and saves at the given file location on server.
     *
     * In case of success, the new workspace becomes current.
     * Closes the current workspace beforehand.
     */
    Workspace create_geoh5(1: required string file_path) throws (1:FileIOException io_ex);

    /**
     * Opens the geoh5 file as read-only at the given path location on server,
     * and loads its content as the current Workspace.
     *
     * Closes the current workspace beforehand.
     */
    Workspace open_geoh5(1: required string file_path)
        throws (1:FileIOException io_ex, 2: FileFormatException fmt_ex);

    /**
     * Saves the current workspace at the given file location on server.
     *
     * In case of success, the current workspace now refers to this new file location.
     * Closes the previously opened workspace file, if any.
     *
     * If file_path is the same the current workspace file, and overwrite_file is true,
     * then the original file is updated.
     */
    Workspace save(1: required string file_path, 2: optional bool overwrite_file = false)
        throws (1:FileIOException io_ex);

    /**
     * Saves the current workspace at the given file location on server.
     *
     * The current open workspace does not change and still refers to the initially
     * opened geoh5 file, if any.
     *
     * Even it overwrite_file is true, it will refuse to overwrite the file of the
     * current open workspace.
     */
    Workspace save_copy(1: required string file_path, 2: optional bool overwrite_file = false)
        throws (1:FileIOException io_ex);

    /**
     * Saves the given list of Objects/Groups as a new geoh5 file at the given file location on server.
     *
     * The current open workspace does not change and still refers to the initially
     * opened geoh5 file, if any.
     *
     * Even it overwrite_file is true, it will refuse to overwrite the file of the
     * current open workspace.
     */
    Workspace export_objects(
        1: required list<shared.Uuid> objects_or_groups, 2: required string file_path,
        3: optional bool overwrite_file = false
    ) throws (1:FileIOException io_ex, 2:shared.InvalidUid uuid_ex, 3:shared.BadEntityType entity_type_ex);

    /**
     * Closes the current workspace and the associated geoh5 file if any.
     */
    void close();

    /**
     * Gets list of contributors for the current open geoh5 workspace.
     *
     * Note: the contributor list of a workspace is update with current
     * user every time the workspace is saved
     */
    list<string> get_contributors();
}
