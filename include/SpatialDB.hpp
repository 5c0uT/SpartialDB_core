#pragma once
#include <vector>
#include <array>
#include <string>
#include <cstdint>

class BVHManager;

struct RayHit {
    std::array<float, 3> position {0.0f, 0.0f, 0.0f};
    std::array<float, 3> normal {0.0f, 0.0f, 0.0f};
    float distance = -1.0f;
    uint32_t objectID = 0;
};

struct NeighborResult {
    uint32_t objectID = 0;
    float distance = 0.0f;
    std::array<float, 3> position {0.0f, 0.0f, 0.0f};
};

struct RangeQueryResult {
    uint32_t objectID = 0;
    std::array<float, 3> position {0.0f, 0.0f, 0.0f};
};

class SpatialDB {
public:
    SpatialDB();
    ~SpatialDB();

    void loadLAS(const std::string& path, const char* crs = "EPSG:4326");
    void loadMesh(const std::string& path);

    RayHit queryRay(const float origin[3], const float direction[3], float maxDist);
    std::vector<RayHit> batchQueryRay(
        const std::vector<float>& origins,
        const std::vector<float>& directions,
        const std::vector<float>& maxDists
    );
    std::vector<uint32_t> querySphere(const float center[3], float radius);

    std::vector<NeighborResult> queryKNN(
        const float queryPoint[3],
        uint32_t k,
        float maxRadius = 1e6f
    );
    std::vector<RangeQueryResult> queryRange(
        const float minBounds[3],
        const float maxBounds[3]
    );

    void buildBVH();
    void clearScene();

    void addPointCloud(
        const std::vector<float>& points,
        float voxelSize = 0.1f
    );

private:
    BVHManager* mBVHManager;
    std::vector<std::array<float, 3>> mPoints;
};
