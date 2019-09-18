include "shared.thrift"

namespace * objects

/**
 * Contains the base Entity attributes and more attributes specific to Object.
 */
struct Object {
   1: required shared.Entity base_;
   2: optional bool allow_move = true;
   // geoh5 also has: list<shared.Uuid> clipping_uids; -- Do not expose in the API, until it allows for manipulation of clipping planes
}

struct Points {
    1: required Object base_;
    // TODO more attributes
}

struct Curve {
    1: required Object base_;
    // TODO more attributes
}

struct Surface {
    1: required Object base_;
    // TODO more attributes
}

struct Grid2D {
    1: required Object base_;
    // TODO more attributes
}

struct Drillhole {
    1: required Object base_;
    // TODO more attributes
}

struct BlockModel {
    1: required Object base_;
    // TODO more attributes
}

struct Octree {
    1: required Object base_;
    // TODO more attributes
}

struct GeoImage {
    1: required Object base_;
    // TODO more attributes
}

struct Label {
    1: required Object base_;
    // TODO more attributes
}

/**
 * Query parameters to look for Objects.
 */
struct ObjectQuery {
   1: optional string name = "";
   2: optional shared.Uuid type;
   3: optional shared.Uuid in_group;
   4: optional bool recursive = false;
}

/**
 * Describes the known type of Objects.
 *
 * Use ObjectService to obtain a ObjectClass value from an Object type Uuid.
 */
enum ObjectClass {
   UNKNOWN = 0
   POINTS,
   CURVE,
   SURFACE,
   GRID2D,
   DRILLHOLE,
   BLOCKMODEL,
   OCTREE,
   GEOIMAGE,
   LABEL,
}

struct GeometryTransformation {
   1: optional shared.Coord3D translation;
   2: optional double rotation_deg = 0.;
}

/**
 * Exception thrown when failing to apply an operation that modifies an Object.
 */
exception InvalidObjectOperation {
    1: optional string message = "";
}

/**
 * Defines operations on Objects.
 */
service ObjectsService extends shared.EntityService {
    /**
     * Returns the Uuid for the Object type that corresponds to the given ObjectClass.
     *
     * Returns an empty Uuid for ObjectClass.UNKNOWN.
     */
    shared.Uuid get_type(1: required ObjectClass object_class);

    /**
     * Returns the ObjectClass for the given Object type.
     *
     * May return ObjectClass.UNKNOWN if the given type does not correspond to any known type.
     */
    ObjectClass get_class(1: required shared.Uuid type)
        throws (1:shared.InvalidUid uuid_ex);

    list<Object> get_all();
    list<Object> find(1: required ObjectQuery query)
        throws (1:shared.InvalidUid uuid_ex, 2:shared.BadEntityType entity_type_ex);

    /**
     * Sets/unsets the move permission on each Object of the given list.
     */
    void set_allow_move(1: required list<shared.Uuid> objects, 2: required bool allow)
        throws (1:shared.InvalidUid uuid_ex, 2:shared.BadEntityType entity_type_ex);

    /**
     * Moves each Object of the given list under the destination group.
     */
    void move_to_group(1: required list<shared.Uuid> objects, 2: required shared.Uuid destination_group)
        throws (1:shared.InvalidUid uuid_ex, 2:shared.BadEntityType entity_type_ex, 3:InvalidObjectOperation op_ex);

    Object get(1: required shared.Uuid uid)
        throws (1:shared.InvalidUid uuid_ex, 2:shared.BadEntityType entity_type_ex);

    Points narrowPoints(1: required shared.Uuid uid)
        throws (1:shared.InvalidUid uuid_ex, 2:shared.BadEntityType entity_type_ex);

    Curve narrowCurve(1: required shared.Uuid uid)
        throws (1:shared.InvalidUid uuid_ex, 2:shared.BadEntityType entity_type_ex);

    Surface narrowSurface(1: required shared.Uuid uid)
        throws (1:shared.InvalidUid uuid_ex, 2:shared.BadEntityType entity_type_ex);

    Grid2D narrowGrid2D(1: required shared.Uuid uid)
        throws (1:shared.InvalidUid uuid_ex, 2:shared.BadEntityType entity_type_ex);

    Drillhole narrowDrillhole(1: required shared.Uuid uid)
        throws (1:shared.InvalidUid uuid_ex, 2:shared.BadEntityType entity_type_ex);

    BlockModel narrowBlockModel(1: required shared.Uuid uid)
        throws (1:shared.InvalidUid uuid_ex, 2:shared.BadEntityType entity_type_ex);

    Octree narrowOctree(1: required shared.Uuid uid)
        throws (1:shared.InvalidUid uuid_ex, 2:shared.BadEntityType entity_type_ex);

    GeoImage narrowGeoImage(1: required shared.Uuid uid)
        throws (1:shared.InvalidUid uuid_ex, 2:shared.BadEntityType entity_type_ex);

    Label narrowLabel(1: required shared.Uuid uid)
        throws (1:shared.InvalidUid uuid_ex, 2:shared.BadEntityType entity_type_ex);

    // Below, methods for creating new Objects.
    // When creating object, group is optional. The object gets created under the root container
    // if group is not specified.

    Object createAnyObject(
        1: required shared.Uuid type, 2: required string name,
        3: optional shared.Uuid parent_group, 4: map<string, string> attributes
     ) throws (1:shared.BadEntityName name_ex, 2:shared.InvalidUid uuid_ex, 3:shared.BadEntityType entity_type_ex);

    // TODO: for each object type, have a specific set of parameters
    // Points objectService.createPoints(1: required string name, 2: optional shared.Uuid parent_group /*specific parameters... */) throws (1:shared.BadEntityName name_ex, 2:shared.InvalidUid uuid_ex, 3:shared.BadEntityType entity_type_ex);
    // Curve objectService.createCurve(1: required string name, 2: optional shared.Uuid parent_group, /*specific parameters... */) throws (1:shared.BadEntityName name_ex, 2:shared.InvalidUid uuid_ex, 3:shared.BadEntityType entity_type_ex);
    // ...

    void transform(1: required list<shared.Uuid> objects, 2: required GeometryTransformation transformation)
        throws (1:shared.InvalidUid uuid_ex, 2:shared.BadEntityType entity_type_ex, 3:InvalidObjectOperation op_ex);

    // TODO: get the object coordinates, cells, survey location, ...

    // TODO: allow creation of new custom ObjectTypes?
}

// TODO if specific operations per class, other than create: introduce CurveService, SurfaceService, ...
