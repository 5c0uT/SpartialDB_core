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
    hit.position = {origin[0], origin[1], origin[2]};
    hit.normal = {direction[0], direction[1], direction[2]};
    hit.distance = -1.0f;
    hit.objectID = 0;
    return hit;
}

std::vector<RayHit> SpatialDB::batchQueryRay(
    const std::vector<float>& origins,
    const std::vector<float>& directions,
    const std::vector<float>& maxDists)
{
    std::vector<RayHit> hits;
    const size_t rayCount = maxDists.size();
    hits.reserve(rayCount);

    for (size_t i = 0; i < rayCount; ++i) {
        RayHit hit;
        const size_t offset = i * 3;
        if (offset + 2 < origins.size()) {
            hit.position = {origins[offset], origins[offset + 1], origins[offset + 2]};
        }
        if (offset + 2 < directions.size()) {
            hit.normal = {directions[offset], directions[offset + 1], directions[offset + 2]};
        }
        hits.push_back(hit);
    }

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
