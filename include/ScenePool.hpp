#pragma once
#include <vector>

namespace physx {
    class PxScene;
}

class ScenePool {
public:
    ScenePool();
    ~ScenePool();

    void initialize(int numScenes = 4);
    void cleanup();
    physx::PxScene* getScene();

private:
    struct Impl;
    Impl* mImpl;
};