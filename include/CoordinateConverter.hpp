#pragma once

#include <proj.h>
#include <string>
#include <memory>

class CoordinateConverter {
public:
    CoordinateConverter(const std::string& sourceCRS, const std::string& targetCRS);
    ~CoordinateConverter();

    void convert(double& x, double& y, double& z) const;

private:
    PJ_CONTEXT* mContext;
    PJ* mConversion;
};