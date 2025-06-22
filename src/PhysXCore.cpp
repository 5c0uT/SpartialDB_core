#include "PhysXCore.hpp"
#include <foundation/PxFoundation.h>
#include <extensions/PxDefaultCpuDispatcher.h>
#include <extensions/PxDefaultErrorCallback.h>
#include <iostream>

using namespace physx;

struct PhysXCore::Impl {
    PxDefaultAllocator mAllocator;
    PxDefaultErrorCallback mErrorCallback;
    PxFoundation* mFoundation = nullptr;
    PxPhysics* mPhysics = nullptr;
    PxPvd* mPvd = nullptr;

    ~Impl() {
        if (mPhysics) mPhysics->release();
        if (mPvd) mPvd->release();
        if (mFoundation) mFoundation->release();
    }
};

PhysXCore::~PhysXCore() = default;

PhysXCore::PhysXCore() : mImpl(std::make_unique<Impl>()) {}

PhysXCore& PhysXCore::instance() {
    if (!sInstance) {
        std::lock_guard<std::mutex> lock(sMutex);
        if (!sInstance) sInstance = new PhysXCore();
    }
    return *sInstance;
}

void PhysXCore::init(int cudaDevice) {
    Impl& p = *mImpl;

    p.mFoundation = PxCreateFoundation(PX_PHYSICS_VERSION, p.mAllocator, p.mErrorCallback);
    if (!p.mFoundation) throw std::runtime_error("PxCreateFoundation failed!");

    // Полностью отключаем PVD
    p.mPvd = nullptr;

    PxTolerancesScale scale;
    p.mPhysics = PxCreatePhysics(PX_PHYSICS_VERSION, *p.mFoundation, scale, true, p.mPvd);
    if (!p.mPhysics) throw std::runtime_error("PxCreatePhysics failed!");
}
physx::PxPhysics* PhysXCore::physics() {
    return mImpl->mPhysics;
}