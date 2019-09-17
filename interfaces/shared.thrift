namespace * shared

/**
 * A version string. Refers to `Semantic Versioning <https://semver.org/>` for a proper formatting
 * (major.minor.patch[-preReleaseID][+buildID]).
 */
struct VersionString {
    1: optional string value = "";
}

/**
 * A version represented as floating point number (major.minor),
 * as in the geoh5 format version.
 */
struct VersionNumber {
    1: optional double value = 0.;
}

/**
 * A UUID. Can be null.
 */
struct Uuid {
   1: optional string id = "";  // 'optional' allows for null Uuid.
}

/**
 * UTC date and time, represented by text according to the ISO 8601 specifications,
 * taking the form YYYY-MM-DDTHH:mm:ss[Z|[+|-]HH:mm].
 */
struct DateTime {
    1: optional string value = ""; // optional allows for null DateTime
}

/**
 * A distance unit ("meter" or "feet").
 */
struct DistanceUnit {
    1: optional string unit = "";
}

/**
 * The unit for data values.
 */
struct DataUnit {
    1: optional string unit = "";
}

struct Coord3D {
    1: optional double x = 0.;
    2: optional double y = 0.;
    3: optional double z = 0.;
}

/**
 * Base structure that holds attributes common to Object, Data and Group entities.
 */
struct Entity {
   1: required Uuid uid;
   2: required Uuid type;
   3: required string name;
   4: optional bool visible = false;
   5: optional bool allow_delete = true;
   6: optional bool allow_rename = true;
   7: optional bool is_public = true;
}

/**
 * Exception thrown when a given Uuid is invalid or refers to an unexpected entity type.
 */
exception InvalidUid {
    1: optional string message = "";
}

/**
 * Exception thrown when a given Entity reference (usually through a Uuid)
 * is not pointing to an Entity of the expected type.
 */
exception BadEntityType {
    1: optional string message = "";
}

/**
 * Exception thrown when trying to rename an entity with an unauthorized name.
 */
exception BadEntityName {
    1: optional string message = "";
}

/**
 * Defines operations common to all Entity types (Objects, Groups, Data).
 */
service EntityService {

    /**
     * Sets/unsets the public flag on each Entity of the given list.
     */
    void set_public(1: required list<Uuid> entities , 2: required bool is_public)
        throws (1:InvalidUid uuid_ex);

    /**
     * Sets/unsets the visible flag on each Entity of the given list.
     */
    void set_visible(1: required list<Uuid> entities, 2: required bool visible)
        throws (1:InvalidUid uuid_ex);

    /**
     * Sets/unsets the delete permission on each Entity of the given list.
     */
    void set_allow_delete(1: required list<Uuid> entities, 2: required bool allow)
        throws (1:InvalidUid uuid_ex);

    /**
     * Sets/unsets the rename permission on each Entity of the given list.
     */
    void set_allow_rename(1: required list<Uuid> entities, 2: required bool allow)
        throws (1:InvalidUid uuid_ex);

    /**
     * Renames the given Entity.
     */
    void rename(1: required Uuid entities, 2: required string new_name)
        throws (1:InvalidUid uuid_ex, 2:BadEntityName name_ex);
}
