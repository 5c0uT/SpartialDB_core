#pragma once
#include <vector>
#include <foundation/PxVec3.h>

class Voxelizer {
public:
    static void voxelize(
        const std::vector<physx::PxVec3>& points,
        std::vector<physx::PxVec3>& outVertices,
        std::vector<uint32_t>& outIndices,
        float voxelSize
    );
};