#include "CoordinateConverter.hpp"
#include <stdexcept>

CoordinateConverter::CoordinateConverter(const std::string& sourceCRS, const std::string& targetCRS)
    : mContext(nullptr),
      mConversion(nullptr) {

    // Создание контекста PROJ
    mContext = proj_context_create();
    if (!mContext) {
        throw std::runtime_error("Failed to create PROJ context");
    };

    // Создание преобразования
    mConversion = proj_create_crs_to_crs(
        mContext,
        sourceCRS.c_str(),
        targetCRS.c_str(),
        nullptr
    );

    if (!mConversion) {
        // Получение сообщения об ошибке
        const char* errorMsg = proj_errno_string(proj_context_errno(mContext));
        std::string fullError = "Failed to create coordinate conversion: ";
        if (errorMsg) fullError += errorMsg;

        // Освобождение ресурсов
        proj_context_destroy(mContext);
        mContext = nullptr;

        throw std::runtime_error(fullError);
    };
};

CoordinateConverter::~CoordinateConverter() {
    if (mConversion) {
        proj_destroy(mConversion);
        mConversion = nullptr;
    };
    if (mContext) {
        proj_context_destroy(mContext);
        mContext = nullptr;
    };
};

void CoordinateConverter::convert(double& x, double& y, double& z) const {
    if (!mConversion) {
        throw std::runtime_error("Coordinate conversion not initialized");
    };

    // Создание координаты
    PJ_COORD coord = proj_coord(x, y, z, 0);

    // Выполнение преобразования
    PJ_COORD result = proj_trans(mConversion, PJ_FWD, coord);

    // Проверка ошибок
    if (proj_errno(mConversion)) {
        const char* errorMsg = proj_errno_string(proj_errno(mConversion));
        std::string fullError = "Coordinate conversion failed: ";
        if (errorMsg) fullError += errorMsg;

        throw std::runtime_error(fullError);
    };

    // Обновление значений
    x = result.xyz.x;
    y = result.xyz.y;
    z = result.xyz.z;
};