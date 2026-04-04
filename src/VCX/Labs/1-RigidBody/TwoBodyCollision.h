#pragma once
#include <fcl/narrowphase/collision.h>
#include "Labs/1-RigidBody/RigidBody.h"
#include <iostream>

namespace VCX::Labs::RigidBody {
    void CollisionDetection(Box &b0, Box &b1, fcl::CollisionResult<float> &collisionResult);

    void CollisionResponse(Box &b0, Box &b1, fcl::CollisionResult<float> &collisionResult, float _restitution);

} 