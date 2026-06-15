#pragma once

#include <PxPhysicsAPI.h>
#include <memory>
#include <mutex>
#include <cstdint>

struct GPUMemoryStats {
    uint64_t allocated_bytes = 0;
    uint64_t peak_bytes = 0;
    uint32_t active_allocations = 0;
    int cuda_device = -1;
    bool gpu_available = false;
};

class PhysXCore {
public:
    static PhysXCore& instance();

    void init(int cudaDevice = 0);
    void cleanup();

    physx::PxPhysics* physics();
    physx::PxScene* createScene();
    void stepSimulation(float timestep = 0.016f);

    physx::PxCpuDispatcher* getDispatcher() const;
    physx::PxMaterial* getDefaultMaterial() const;

    GPUMemoryStats getMemoryStats() const;
    int getCudaDevice() const;
    int getDeviceCount() const;

    ~PhysXCore();

private:
    PhysXCore();

    struct Impl;
    std::unique_ptr<Impl> mImpl;

    inline static PhysXCore* sInstance = nullptr;
    inline static std::mutex sMutex;
};
