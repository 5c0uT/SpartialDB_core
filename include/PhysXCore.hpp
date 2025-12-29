#pragma once

#include <PxPhysicsAPI.h>
#include <memory>
#include <mutex>

class PhysXCore {
public:
    static PhysXCore& instance();

    // Initialization and cleanup
    void init(int cudaDevice = 0);
    void cleanup();

    // Physics access
    physx::PxPhysics* physics();
    physx::PxScene* createScene();
    void stepSimulation(float timestep = 0.016f);

    // Getters for common objects
    physx::PxCpuDispatcher* getDispatcher() const;
    physx::PxMaterial* getDefaultMaterial() const;

    ~PhysXCore();

private:
    PhysXCore();

    struct Impl;
    std::unique_ptr<Impl> mImpl;

    // Объявляем как inline static (C++17)
    inline static PhysXCore* sInstance = nullptr;
    inline static std::mutex sMutex;
};
