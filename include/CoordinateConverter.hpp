#pragma once

#include <string>
#include <memory>

// Forward declarations для PROJ (если не доступны)
struct PJ_CONTEXT;
struct PJ;

class CoordinateConverter {
public:
    CoordinateConverter(const std::string& sourceCRS, const std::string& targetCRS);
    ~CoordinateConverter();

    void convert(double& x, double& y, double& z) const;

private:
    PJ_CONTEXT* mContext;
    PJ* mConversion;
};
