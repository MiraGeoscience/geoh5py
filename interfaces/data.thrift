include "shared.thrift"

namespace * data

/**
 * Describes how the Data is tied to its parent Entity: vertex, face, cell, or the object/group itself.
 */
enum DataAssociation {
   UNKNOWN = 0,
   OBJECT,
   CELL,
   FACE,
   VERTEX,
}

enum PrimitiveType {
    UNKNOWN = 0,
    INTEGER,
    FLOAT,
    REFERENCED,
    TEXT,
    FILENAME,
    DATETIME,
    BLOB,
}

/**
 * Contains the base Entity attributes and more attributes specific to Data.
 */
struct Data {
   1: required shared.Entity entity_;
   2: required DataAssociation association;
}

/**
 * The unit for data values.
 */
struct DataUnit {
    1: optional string unit = "";
}

struct DataType {
   1: required shared.Uuid uid;
   2: required string name;
   3: optional string description = "";
   4: optional DataUnit units,
   5: PrimitiveType primitive_type,
}

struct DataSlab {
   1: optional i64 start = 0;
   2: optional i64 stride = 1;
   3: optional i64 count = 0;
   4: optional i64 block = 1;
}

struct ReferencedDataEntry {
    1: required i32 key; // TODO: actually unsigned int
    2: optional string value;
}

struct ReferencedValues {
    1: list<i32> indices; // TODO: deal with unsigned integer in file, but not support in Thrift
    2: list<ReferencedDataEntry> entries;
}

/**
 * Query parameters to look for Data.
 */
struct DataQuery {
   1: optional string name;
   2: optional shared.Uuid object_or_group; // Data can be attached to an Object or a Group
   3: optional shared.Uuid data_type;
   4: optional PrimitiveType primitive_type;
   5: optional DataAssociation association;
}

/**
 * Query parameters to look for DataType.
 */
struct DataTypeQuery {
   1: optional string name;
   2: optional PrimitiveType primitive_type;
   3: optional DataUnit units;
}

/**
 * Exception thrown when failing to apply an operation that modifies Data.
 */
exception InvalidDataOperation {
    1: optional string message = "";
}

/**
 * Exception thrown when trying to access Data values with the wrong PrimitiveType.
 */
exception BadPrimitiveType {
    1: optional string message = "";
}

service DataService extends shared.EntityService {
    list<Data> get_all();
    list<Data> find(1: required DataQuery query)
        throws (1:shared.InvalidUid uuid_ex, 2:shared.BadEntityType entity_type_ex);

    Data get(1: required shared.Uuid uid)
        throws (1:shared.InvalidUid uuid_ex, 2:shared.BadEntityType entity_type_ex);

    /**
     * Returns values for Data of primitive type FLOAT.
     */
    list<double> get_float_values(1: required shared.Uuid data, 2: optional DataSlab slab)
        throws (1:shared.InvalidUid uuid_ex, 2:shared.BadEntityType entity_type_ex, 3:BadPrimitiveType primitive_type_ex);

    /**
     * Returns values for Data of primitive type INTEGER.
     */
    list<i32> get_integer_values(1: required shared.Uuid data, 2: optional DataSlab slab)
        throws (1:shared.InvalidUid uuid_ex, 2:shared.BadEntityType entity_type_ex, 3:BadPrimitiveType primitive_type_ex);

    /**
     * Returns values for Data of primitive type TEXT.
     */
    list<string> get_text_values(1: required shared.Uuid data, 2: optional DataSlab slab)
        throws (1:shared.InvalidUid uuid_ex, 2:shared.BadEntityType entity_type_ex, 3:BadPrimitiveType primitive_type_ex);

    /**
     * Returns values for Data of primitive type REFERENCED.
     */
    ReferencedValues get_referenced_values(1: required shared.Uuid data, 2: optional DataSlab slab)
        throws (1:shared.InvalidUid uuid_ex, 2:shared.BadEntityType entity_type_ex, 3:BadPrimitiveType primitive_type_ex);

    /**
     * Returns values for Data of primitive type DATETIME.
     */
    list<string> get_datetime_values(1: required shared.Uuid data, 2: optional DataSlab slab)
        throws (1:shared.InvalidUid uuid_ex, 2:shared.BadEntityType entity_type_ex, 3:BadPrimitiveType primitive_type_ex);

    /**
     * Returns values for Data of primitive type FILENAME.
     *
     * For each filename within Data, an opaque dataset named after the filename is expected under the Data instance,
     * containing a complete binary dump of the file. See get_file_data().
     */
    list<string> get_filename_values(1: required shared.Uuid data, 2: optional DataSlab slab)
        throws (1:shared.InvalidUid uuid_ex, 2:shared.BadEntityType entity_type_ex, 3:BadPrimitiveType primitive_type_ex);

    /** For Data of primitive type FILENAME, gets the binary dump for the given file name. */
    binary get_file_content(1: required shared.Uuid data, 2: required string file_name)
        throws (1:shared.InvalidUid uuid_ex, 2:shared.BadEntityType entity_type_ex, 3:BadPrimitiveType primitive_type_ex);

    /**
     * Returns values for Data of primitive type BLOB.
     *
     * 1D array of bytes, where each byte represents boolean values (8-bit char type, value "0" or "1").
     * For each index set to 1, an opaque dataset named after the index (e.g. "1", "2", etc) is expected under the Data instance,
     * containing the binary data tied to that index. See get_blob_element().
     */
    list<byte> get_blob_values(1: required shared.Uuid data, 2: optional DataSlab slab)
        throws (1:shared.InvalidUid uuid_ex, 2:shared.BadEntityType entity_type_ex, 3:BadPrimitiveType primitive_type_ex);

    /** For Data of primitive type BLOB, gets the binary data for the given index. */
    binary get_blob_element(1: required shared.Uuid data, 2: required i64 index);

    /** Returns all defined DataTypes. */
    list<DataType> get_all_types();

    /** Returns all DataTypes that match the query. */
    list<DataType> find_types(1: required DataTypeQuery query)

    /** Get the DataType from the given DataType Uuid. */
    DataType get_type(1: required shared.Uuid uid)
        throws (1:shared.InvalidUid uuid_ex, 2:shared.BadEntityType entity_type_ex);

    // TODO: DataType create_type(...) throws ...;

    // TODO: get colorMap from DataType -- optional records colors assigned to value ranges
    // TODO: get valueMap from DataType -- for REFERENCED DataType only

    // TODO: Data create(objectUuid, association, dataTypeID, ...) throws ...

    // TODO: write Data, for each primitive type...
}

// TODO: how are managed data groups? Should a group be part of DataQuery
