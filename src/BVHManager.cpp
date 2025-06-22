#include "BVHManager.hpp"
#include "PhysXCore.hpp"
#include <PxPhysics.h>
#include <extensions/PxDefaultCpuDispatcher.h>
#include <extensions/PxDefaultSimulationFilterShader.h>
#include <geometry/PxBoxGeometry.h>
#include <iostream>
#include <vector>
#include <limits>
#include <algorithm>

using namespace physx;

struct BVHManager::Impl {
    PxScene* mScene = nullptr;
    PxMaterial* mMaterial = nullptr;
    std::vector<PxRigidStatic*> mActors;

    ~Impl() {
        clearScene();
    }

    void clearScene() {
        for (auto actor : mActors) {
            if (mScene) mScene->removeActor(*actor);
            actor->release();
        }
        mActors.clear();

        if (mMaterial) {
            mMaterial->release();
            mMaterial = nullptr;
        }

        if (mScene) {
            mScene->release();
            mScene = nullptr;
        }
    }
};

BVHManager::BVHManager() : mImpl(new Impl()) {
    PhysXCore& core = PhysXCore::instance();
    PxPhysics* physics = core.physics();

    PxSceneDesc sceneDesc(physics->getTolerancesScale());
    sceneDesc.gravity = PxVec3(0.0f, 0.0f, 0.0f);
    sceneDesc.cpuDispatcher = PxDefaultCpuDispatcherCreate(1);
    sceneDesc.filterShader = PxDefaultSimulationFilterShader;
    sceneDesc.flags |= PxSceneFlag::eENABLE_CCD;

    mImpl->mScene = physics->createScene(sceneDesc);
    if (!mImpl->mScene) {
        throw std::runtime_error("Failed to create scene");
    }

    mImpl->mMaterial = physics->createMaterial(0.5f, 0.5f, 0.1f);
    if (!mImpl->mMaterial) {
        throw std::runtime_error("Failed to create material");
    }
}

// Определяем деструктор после определения Impl
BVHManager::~BVHManager() {
    delete mImpl;
}

void BVHManager::addObject(const std::vector<float>& vertices, const std::vector<uint32_t>& indices) {
    // Рассчитываем ограничивающий объем (AABB)
    PxVec3 minBounds(FLT_MAX, FLT_MAX, FLT_MAX);
    PxVec3 maxBounds(-FLT_MAX, -FLT_MAX, -FLT_MAX);

    for (size_t i = 0; i < vertices.size(); i += 3) {
        float x = vertices[i];
        float y = vertices[i+1];
        float z = vertices[i+2];

        minBounds.x = std::min(minBounds.x, x);
        minBounds.y = std::min(minBounds.y, y);
        minBounds.z = std::min(minBounds.z, z);

        maxBounds.x = std::max(maxBounds.x, x);
        maxBounds.y = std::max(maxBounds.y, y);
        maxBounds.z = std::max(maxBounds.z, z);
    }

    // Создаем бокс на основе AABB
    PxVec3 center = (minBounds + maxBounds) * 0.5f;
    PxVec3 halfExtents = (maxBounds - minBounds) * 0.5f;
    PxBoxGeometry geometry(halfExtents);

    PhysXCore& core = PhysXCore::instance();
    PxPhysics* physics = core.physics();

    PxRigidStatic* actor = physics->createRigidStatic(PxTransform(center));
    if (!actor) {
        throw std::runtime_error("Failed to create rigid actor");
    }

    PxShape* shape = physics->createShape(geometry, *mImpl->mMaterial, true);
    if (!shape) {
        actor->release();
        throw std::runtime_error("Failed to create shape");
    }

    actor->attachShape(*shape);
    shape->release();
    mImpl->mScene->addActor(*actor);
    mImpl->mActors.push_back(actor);
}

void BVHManager::buildBVH() {
    std::cout << "Building BVH..." << std::endl;
    if (mImpl->mScene) {
        PxU32 nbActors = mImpl->mScene->getNbActors(PxActorTypeFlag::eRIGID_STATIC);
        std::cout << "Scene contains " << nbActors << " actors" << std::endl;
    }
}

void BVHManager::clearScene() {
    mImpl->clearScene();
}