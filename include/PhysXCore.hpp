#pragma once
#include <PxPhysicsAPI.h>
#include <memory>
#include <mutex>

class PhysXCore {
public:
    static PhysXCore& instance();

    void init(int cudaDevice = 0);
    physx::PxPhysics* physics();

    ~PhysXCore();

private:
    PhysXCore();

    struct Impl;
    std::unique_ptr<Impl> mImpl;

    // Объявляем как inline static (C++17)
    inline static PhysXCore* sInstance = nullptr;
    inline static std::mutex sMutex;
};