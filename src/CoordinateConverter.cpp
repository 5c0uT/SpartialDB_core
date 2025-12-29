#include "CoordinateConverter.hpp"

#ifdef USE_PROJ
#include <proj.h>
#else
// Stub implementation when PROJ is not available
struct PJ_CONTEXT {};
struct PJ {};
#endif

#include <stdexcept>
#include <string>

CoordinateConverter::CoordinateConverter(const std::string& sourceCRS, const std::string& targetCRS)
    : mContext(nullptr), mConversion(nullptr) {
#ifdef USE_PROJ
    mContext = proj_context_create();
    if (!mContext) {
        throw std::runtime_error("Failed to create PROJ context");
    }

    mConversion = proj_create_crs_to_crs(mContext, sourceCRS.c_str(), targetCRS.c_str(), nullptr);
    if (!mConversion) {
        std::string error = proj_errno_string(proj_context_errno(mContext));
        proj_context_destroy(mContext);
        throw std::runtime_error(std::string("Failed to create PROJ transformation: ") + error);
    }
#else
    throw std::runtime_error("CoordinateConverter requires PROJ library which is not available");
#endif
}

CoordinateConverter::~CoordinateConverter() {
#ifdef USE_PROJ
    if (mConversion) {
        proj_destroy(mConversion);
    }
    if (mContext) {
        proj_context_destroy(mContext);
    }
#endif
}

void CoordinateConverter::convert(double& x, double& y, double& z) const {
#ifdef USE_PROJ
    if (!mConversion) {
        throw std::runtime_error("Conversion not initialized");
    }

    // Create input coordinate
    PJ_COORD coord_in = proj_coord(x, y, z, 0.0);

    // Transform coordinate
    PJ_COORD coord_out = proj_trans(mConversion, PJ_FWD, coord_in);

    // Check for errors
    if (proj_errno(mConversion) != 0) {
        throw std::runtime_error(
            std::string("PROJ conversion failed: ") + 
            proj_errno_string(proj_errno(mConversion))
        );
    }

    // Update output values
    x = coord_out.xy.x;
    y = coord_out.xy.y;
    z = coord_out.xyz.z;
#else
    throw std::runtime_error("PROJ library not available");
#endif
}
