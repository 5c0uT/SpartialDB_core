#pragma once

#include <string>
#include <memory>

#ifdef USE_PROJ
#include <proj.h>
#else
// Forward declarations для stub-конфигурации
struct PJ_CONTEXT;
struct PJ;
#endif

class CoordinateConverter {
public:
    CoordinateConverter(const std::string& sourceCRS, const std::string& targetCRS);
    ~CoordinateConverter();

    void convert(double& x, double& y, double& z) const;

private:
    PJ_CONTEXT* mContext;
    PJ* mConversion;
};
