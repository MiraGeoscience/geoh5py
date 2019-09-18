include "shared.thrift"

namespace * groups

/**
 * Contains the base Entity attributes and more attributes specific to Group.
 */
struct Group {
   1: required shared.Entity base_;
   2: optional bool allowMove = true;
   // geoh5 also has: list<shared.Uuid> clipping_uids; -- Do not expose in the API, until it allows for manipulation of clipping planes
}

struct GroupQuery {
   1: optional string name;
   2: optional shared.Uuid type;
   3: optional shared.Uuid in_group;
   4: optional bool recursive = false;
}

/**
 * Describes the known type of Groups.
 *
 * Use GroupService to obtain a GroupClass value from an Group type Uuid.
 */
enum GroupClass {
   UNKNOWN = 0,
   CONTAINER,
   DRILLHOLE,
   // TODO: more undocumented groups
}

/**
 * Exception thrown when failing to apply an operation that modifies Group.
 */
exception InvalidGroupOperation {
    1: optional string message = "";
}

service GroupsService extends shared.EntityService {
    /** Returns the root container of the current open Workspace. */
    Group get_root();

    /**
     * Returns the Uuid for the Group type that corresponds to the given GroupClass.
     *
     * Returns an empty Uuid for GroupClass.UNKNOWN.
     */
    shared.Uuid get_type(1: required GroupClass group_class);

    /**
     * Returns the GroupClass for the given Group type.
     *
     * May return GroupClass.UNKNOWN if the given type does not correspond to any known type.
     */
    GroupClass get_class(1: required shared.Uuid type)
        throws (1:shared.InvalidUid uuid_ex);

    list<Group> get_all();
    list<Group> find(1: required GroupQuery query)
        throws (1:shared.InvalidUid uuid_ex, 2:shared.BadEntityType entity_type_ex);

    /**
     * Sets/unsets the move permission on each Group of the given list.
     */
    void set_allow_move(1: required list<shared.Uuid> groups, 2: required bool allow)
        throws (1:shared.InvalidUid uuid_ex, 2:shared.BadEntityType entity_type_ex);

    /**
     * Moves each Group of the given list under the given destination group.
     */
    void move_to_group(1: required list<shared.Uuid> groups, 2: required shared.Uuid destination_group)
        throws (1:shared.InvalidUid uuid_ex, 2:shared.BadEntityType entity_type_ex, 3:InvalidGroupOperation op_ex);

    // type defaults to Uuid for GroupClass.CONTAINER if not specified
    Group create(1: optional shared.Uuid type/*, TODO...*/)
        throws (1:shared.InvalidUid uuid_ex, 2:shared.BadEntityType entity_type_ex);

    // TODO: allow creation of new custom GroupTypes?
}
