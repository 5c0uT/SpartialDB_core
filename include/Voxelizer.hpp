#pragma once

#include <vector>
#include <PxPhysicsAPI.h>

class Voxelizer {
public:
    static void voxelize(
        const std::vector<physx::PxVec3>& points,
        std::vector<float>& outVertices,
        std::vector<uint32_t>& outIndices,
        float voxelSize
    );
};
