#include "PhysXCore.hpp"

#include <iostream>
#include <stdexcept>

using namespace physx;

struct PhysXCore::Impl {
    PxDefaultAllocator mAllocator;
    PxDefaultErrorCallback mErrorCallback;
    PxFoundation* mFoundation = nullptr;
    PxPhysics* mPhysics = nullptr;
    PxPvd* mPvd = nullptr;
    PxScene* mScene = nullptr;
    PxCpuDispatcher* mDispatcher = nullptr;
    PxMaterial* mDefaultMaterial = nullptr;

    ~Impl() {
        cleanup();
    }

    void cleanup() {
        if (mScene) {
            mScene->release();
            mScene = nullptr;
        }
        if (mDispatcher) {
            // PxCpuDispatcher doesn't have release() - it's released with scene
            mDispatcher = nullptr;
        }
        if (mDefaultMaterial) {
            mDefaultMaterial->release();
            mDefaultMaterial = nullptr;
        }
        if (mPhysics) {
            mPhysics->release();
            mPhysics = nullptr;
        }
        if (mPvd) {
            mPvd->release();
            mPvd = nullptr;
        }
        if (mFoundation) {
            mFoundation->release();
            mFoundation = nullptr;
        }
    }
};

PhysXCore::PhysXCore() : mImpl(std::make_unique<Impl>()) {}

PhysXCore::~PhysXCore() = default;

PhysXCore& PhysXCore::instance() {
    if (!sInstance) {
        std::lock_guard<std::mutex> lock(sMutex);
        if (!sInstance) {
            sInstance = new PhysXCore();
        }
    }
    return *sInstance;
}

void PhysXCore::init(int cudaDevice) {
    Impl& p = *mImpl;

    p.mFoundation = PxCreateFoundation(PX_PHYSICS_VERSION, p.mAllocator, p.mErrorCallback);
    if (!p.mFoundation) {
        throw std::runtime_error("PxCreateFoundation failed!");
    }

    // Disable PVD completely
    p.mPvd = nullptr;

    PxTolerancesScale scale;
    p.mPhysics = PxCreatePhysics(PX_PHYSICS_VERSION, *p.mFoundation, scale, true, p.mPvd);
    if (!p.mPhysics) {
        throw std::runtime_error("PxCreatePhysics failed!");
    }

    // Create default material
    p.mDefaultMaterial = p.mPhysics->createMaterial(0.5f, 0.5f, 0.6f);
    if (!p.mDefaultMaterial) {
        throw std::runtime_error("Failed to create default material!");
    }

    // Create CPU dispatcher
    p.mDispatcher = PxDefaultCpuDispatcherCreate(4);
    if (!p.mDispatcher) {
        throw std::runtime_error("Failed to create CPU dispatcher!");
    }

    std::cout << "PhysXCore initialized successfully" << std::endl;
}

void PhysXCore::cleanup() {
    if (mImpl) {
        mImpl->cleanup();
    }
}

PxPhysics* PhysXCore::physics() {
    return mImpl->mPhysics;
}

PxScene* PhysXCore::createScene() {
    if (!mImpl->mPhysics || !mImpl->mDispatcher) {
        throw std::runtime_error("PhysXCore not initialized!");
    }

    PxSceneDesc sceneDesc(mImpl->mPhysics->getTolerancesScale());
    sceneDesc.gravity = PxVec3(0.0f, -9.81f, 0.0f);
    sceneDesc.cpuDispatcher = mImpl->mDispatcher;
    sceneDesc.filterShader = PxDefaultSimulationFilterShader;
    sceneDesc.flags |= PxSceneFlag::eENABLE_CCD;

    PxScene* scene = mImpl->mPhysics->createScene(sceneDesc);
    if (!scene) {
        throw std::runtime_error("Failed to create PhysX scene!");
    }

    // Store reference if this is the main scene
    if (!mImpl->mScene) {
        mImpl->mScene = scene;
    }

    return scene;
}

void PhysXCore::stepSimulation(float timestep) {
    if (!mImpl->mScene) {
        throw std::runtime_error("Scene not created!");
    }
    mImpl->mScene->simulate(timestep);
    mImpl->mScene->fetchResults(true);
}

PxCpuDispatcher* PhysXCore::getDispatcher() const {
    return mImpl->mDispatcher;
}

PxMaterial* PhysXCore::getDefaultMaterial() const {
    return mImpl->mDefaultMaterial;
}
