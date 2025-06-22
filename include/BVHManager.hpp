#pragma once
#include <vector>
#include <array>

class BVHManager {
public:
    BVHManager();
    ~BVHManager(); // Объявляем деструктор

    void addObject(const std::vector<float>& vertices, const std::vector<uint32_t>& indices);
    void buildBVH();
    void clearScene();

private:
    struct Impl;
    Impl* mImpl;
};