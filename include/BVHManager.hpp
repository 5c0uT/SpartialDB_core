#pragma once

#include <vector>
#include <memory>
#include <PxPhysicsAPI.h>

class BVHManager {
public:
    BVHManager();
    ~BVHManager();

    void addObject(const std::vector<float>& vertices, const std::vector<uint32_t>& indices);
    void buildBVH();
    void clearScene();

private:
    struct Impl;
    std::unique_ptr<Impl> mImpl;
};
