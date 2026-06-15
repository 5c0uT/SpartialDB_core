#include "SpatialDB.hpp"
#include "BVHManager.hpp"
#include <string>
#include <stdexcept>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_map>

SpatialDB::SpatialDB() {
    mBVHManager = new BVHManager();
}

SpatialDB::~SpatialDB() {
    delete mBVHManager;
}

void SpatialDB::loadLAS(const std::string& path, const char* crs) {
    if (path.empty()) {
        throw std::invalid_argument("loadLAS: path must not be empty");
    }
    std::ifstream file(path);
    if (!file.good()) {
        throw std::runtime_error("loadLAS: file not found or not readable: " + path);
    }
}

void SpatialDB::loadMesh(const std::string& path) {
    if (path.empty()) {
        throw std::invalid_argument("loadMesh: path must not be empty");
    }
    std::ifstream file(path);
    if (!file.good()) {
        throw std::runtime_error("loadMesh: file not found or not readable: " + path);
    }
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
    const size_t rayCount = maxDists.size();
    if (rayCount == 0) {
        return {};
    }
    if (origins.size() != rayCount * 3) {
        throw std::invalid_argument("batchQueryRay: origins size must be 3 * maxDists size");
    }
    if (directions.size() != rayCount * 3) {
        throw std::invalid_argument("batchQueryRay: directions size must be 3 * maxDists size");
    }

    std::vector<RayHit> hits;
    hits.reserve(rayCount);

    for (size_t i = 0; i < rayCount; ++i) {
        RayHit hit;
        const size_t offset = i * 3;
        hit.position = {origins[offset], origins[offset + 1], origins[offset + 2]};
        hit.normal = {directions[offset], directions[offset + 1], directions[offset + 2]};
        hit.distance = maxDists[i];
        hits.push_back(hit);
    }

    return hits;
}

std::vector<uint32_t> SpatialDB::querySphere(const float center[3], float radius) {
    std::vector<uint32_t> results;
    float r2 = radius * radius;
    for (size_t i = 0; i < mPoints.size(); ++i) {
        float dx = mPoints[i][0] - center[0];
        float dy = mPoints[i][1] - center[1];
        float dz = mPoints[i][2] - center[2];
        if (dx * dx + dy * dy + dz * dz <= r2) {
            results.push_back(static_cast<uint32_t>(i));
        }
    }
    return results;
}

std::vector<NeighborResult> SpatialDB::queryKNN(
    const float queryPoint[3],
    uint32_t k,
    float maxRadius)
{
    if (k == 0) {
        return {};
    }

    float maxR2 = maxRadius * maxRadius;
    std::vector<NeighborResult> candidates;
    candidates.reserve(std::min(static_cast<size_t>(k) * 2, mPoints.size()));

    for (size_t i = 0; i < mPoints.size(); ++i) {
        float dx = mPoints[i][0] - queryPoint[0];
        float dy = mPoints[i][1] - queryPoint[1];
        float dz = mPoints[i][2] - queryPoint[2];
        float dist2 = dx * dx + dy * dy + dz * dz;

        if (dist2 <= maxR2) {
            NeighborResult nr;
            nr.objectID = static_cast<uint32_t>(i);
            nr.distance = std::sqrt(dist2);
            nr.position = mPoints[i];
            candidates.push_back(nr);
        }
    }

    std::partial_sort(
        candidates.begin(),
        candidates.begin() + std::min(static_cast<size_t>(k), candidates.size()),
        candidates.end(),
        [](const NeighborResult& a, const NeighborResult& b) {
            return a.distance < b.distance;
        }
    );

    if (candidates.size() > k) {
        candidates.resize(k);
    }

    return candidates;
}

std::vector<RangeQueryResult> SpatialDB::queryRange(
    const float minBounds[3],
    const float maxBounds[3])
{
    std::vector<RangeQueryResult> results;

    for (size_t i = 0; i < mPoints.size(); ++i) {
        const auto& p = mPoints[i];
        if (p[0] >= minBounds[0] && p[0] <= maxBounds[0] &&
            p[1] >= minBounds[1] && p[1] <= maxBounds[1] &&
            p[2] >= minBounds[2] && p[2] <= maxBounds[2]) {
            RangeQueryResult rqr;
            rqr.objectID = static_cast<uint32_t>(i);
            rqr.position = p;
            results.push_back(rqr);
        }
    }

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
    mPoints.clear();
}

void SpatialDB::addPointCloud(
    const std::vector<float>& points,
    float voxelSize)
{
    if (points.empty()) {
        return;
    }
    if (points.size() % 3 != 0) {
        throw std::invalid_argument("addPointCloud: points size must be divisible by 3");
    }

    if (voxelSize <= 0.0f) {
        for (size_t i = 0; i + 2 < points.size(); i += 3) {
            mPoints.push_back({points[i], points[i + 1], points[i + 2]});
        }
    } else {
        std::unordered_map<uint64_t, std::array<float, 3>> voxelMap;
        for (size_t i = 0; i + 2 < points.size(); i += 3) {
            int vx = static_cast<int>(std::floor(points[i] / voxelSize));
            int vy = static_cast<int>(std::floor(points[i + 1] / voxelSize));
            int vz = static_cast<int>(std::floor(points[i + 2] / voxelSize));
            uint64_t key = (static_cast<uint64_t>(vx) << 32) |
                           (static_cast<uint32_t>(vy) << 16) |
                           static_cast<uint32_t>(vz);
            auto it = voxelMap.find(key);
            if (it == voxelMap.end()) {
                voxelMap[key] = {points[i], points[i + 1], points[i + 2]};
            }
        }
        mPoints.reserve(voxelMap.size());
        for (auto& [key, center] : voxelMap) {
            mPoints.push_back(center);
        }
    }
}
