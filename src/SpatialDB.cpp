#include "SpatialDB.hpp"
#include "BVHManager.hpp"
#include <string>

SpatialDB::SpatialDB() {
    mBVHManager = new BVHManager();
}

SpatialDB::~SpatialDB() {
    delete mBVHManager;
}

void SpatialDB::loadLAS(const std::string& path, const char* crs) {
    // Реализация загрузки LAS
}

void SpatialDB::loadMesh(const std::string& path) {
    // Реализация загрузки меша
}

RayHit SpatialDB::queryRay(const float origin[3], const float direction[3], float maxDist) {
    RayHit hit;
    // Реализация raycast
    return hit;
}

std::vector<RayHit> SpatialDB::batchQueryRay(
    const std::vector<float>& origins,
    const std::vector<float>& directions,
    const std::vector<float>& maxDists)
{
    std::vector<RayHit> hits;
    // Реализация пакетного raycast
    return hits;
}

std::vector<uint32_t> SpatialDB::querySphere(const float center[3], float radius) {
    std::vector<uint32_t> results;
    // Реализация сферического запроса
    return results;
}

void SpatialDB::buildBVH() {
    if (mBVHManager) {
        mBVHManager->buildBVH();
    }
}

void SpatialDB::clearScene() {
    if (mBVHManager) {
        mBVHManager->clearScene();
    }
}
