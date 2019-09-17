// keep in mind: these same services will also be implement by Geoscience Analyst
// to operate on the current open workspace.
// The same geoh5io client can connect indifferently to a geoh5 server or to Geoscience Analyst.

// TODO: the server flag any modified entity so that it knows what needs to be to saved
// (thus, flag any new object as modified). Also update last modified date?

// TODO: define a scenario for a demo, so that implementation is prioritized accordingly

include "shared.thrift"
include "workspace.thrift"
include "objects.thrift"
include "groups.thrift"
include "data.thrift"

namespace * api

/**
 * The version of this API. This is different form the version of the geoh5 file format.
 *
 * Follows Semantic Versioning.
 */
const string API_VERSION = "0.1.0";

// other services?
// TODO: read/write and assign clipping planes?
// TODO: manage viewports, and visibility per viewport
