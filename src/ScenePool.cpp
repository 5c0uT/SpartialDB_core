#include "ScenePool.hpp"
#include "PhysXCore.hpp"
#include <PxPhysicsAPI.h>
#include <extensions/PxDefaultCpuDispatcher.h>
#include <extensions/PxDefaultSimulationFilterShader.h>
#include <atomic>

using namespace physx;

struct ScenePool::Impl {
    std::vector<PxScene*> scenes;
    std::atomic<size_t> currentIndex = 0;

    ~Impl() {
        cleanup();
    }

    void initialize(int numScenes) {
        PhysXCore& core = PhysXCore::instance();
        PxPhysics* physics = core.physics();
        PxTolerancesScale scale = physics->getTolerancesScale();

        for (int i = 0; i < numScenes; ++i) {
            PxSceneDesc sceneDesc(scale);
            sceneDesc.gravity = PxVec3(0.0f, 0.0f, 0.0f);
            sceneDesc.cpuDispatcher = PxDefaultCpuDispatcherCreate(1);
            sceneDesc.filterShader = PxDefaultSimulationFilterShader;
            sceneDesc.flags |= PxSceneFlag::eENABLE_CCD;

            PxScene* scene = physics->createScene(sceneDesc);
            if (scene) {
                scenes.push_back(scene);
            }
        }
    }

    void cleanup() {
        for (auto scene : scenes) {
            if (scene) scene->release();
        }
        scenes.clear();
        currentIndex = 0;
    }

    PxScene* getScene() {
        if (scenes.empty()) return nullptr;
        size_t index = currentIndex.fetch_add(1) % scenes.size();
        return scenes[index];
    }
};

ScenePool::ScenePool() : mImpl(new Impl()) {}
ScenePool::~ScenePool() { delete mImpl; }

void ScenePool::initialize(int numScenes) {
    mImpl->initialize(numScenes);
}

void ScenePool::cleanup() {
    mImpl->cleanup();
}

physx::PxScene* ScenePool::getScene() {
    return mImpl->getScene();
}