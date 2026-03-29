#pragma once
#include <vector>
#include <array>
#include <string>

// Предварительное объявление
class BVHManager;

struct RayHit {
    std::array<float, 3> position {0.0f, 0.0f, 0.0f};
    std::array<float, 3> normal {0.0f, 0.0f, 0.0f};
    float distance = -1.0f;
    uint32_t objectID = 0;
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

    void buildBVH();
    void clearScene();

private:
    BVHManager* mBVHManager;
};
