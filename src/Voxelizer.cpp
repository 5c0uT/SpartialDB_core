#include "Voxelizer.hpp"
#include <unordered_map>
#include <cmath>
#include <PxPhysicsAPI.h>

struct VoxelKey {
    int x, y, z;
    bool operator==(const VoxelKey& other) const {
        return x == other.x && y == other.y && z == other.z;
    };
};

namespace std {
    template<> struct hash<VoxelKey> {
        size_t operator()(const VoxelKey& k) const {
            return hash<int>()(k.x) ^ hash<int>()(k.y) ^ hash<int>()(k.z);
        };
    };
};

void Voxelizer::voxelize(
    const std::vector<physx::PxVec3>& points,
    std::vector<physx::PxVec3>& outVertices,
    std::vector<uint32_t>& outIndices,
    float voxelSize
) {
    std::unordered_map<VoxelKey, physx::PxVec3> voxelCenters;
    std::unordered_map<VoxelKey, int> voxelCounts;

    // Вычисление центров вокселей
    for (const auto& p : points) {
        VoxelKey key{
            static_cast<int>(std::floor(p.x / voxelSize)),
            static_cast<int>(std::floor(p.y / voxelSize)),
            static_cast<int>(std::floor(p.z / voxelSize))
        };

        voxelCenters[key] += p;
        voxelCounts[key]++;
    };

    // Нормализация центров
    for (auto& [key, center] : voxelCenters) {
        int count = voxelCounts[key];
        center.x /= count;
        center.y /= count;
        center.z /= count;
    };

    // Генерация кубов для вокселей
    const float hs = voxelSize * 0.5f;
    const physx::PxVec3 offsets[8] = {
        {-hs, -hs, -hs}, { hs, -hs, -hs}, { hs,  hs, -hs}, {-hs,  hs, -hs},
        {-hs, -hs,  hs}, { hs, -hs,  hs}, { hs,  hs,  hs}, {-hs,  hs,  hs}
    };

    const uint32_t indices[36] = {
        0,1,2, 0,2,3, // bottom
        4,5,6, 4,6,7, // top
        0,4,7, 0,7,3, // left
        1,5,6, 1,6,2, // right
        0,1,5, 0,5,4, // front
        3,2,6, 3,6,7  // back
    };

    uint32_t vertexOffset = 0;
    for (const auto& [key, center] : voxelCenters) {
        // Вершины куба
        for (int i = 0; i < 8; i++) {
            outVertices.push_back(center + offsets[i]);
        };

        // Индексы куба
        for (int i = 0; i < 36; i++) {
            outIndices.push_back(vertexOffset + indices[i]);
        };

        vertexOffset += 8;
    };
};