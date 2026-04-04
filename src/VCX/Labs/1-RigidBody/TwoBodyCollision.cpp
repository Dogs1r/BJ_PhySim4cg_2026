#include <fcl/narrowphase/collision.h>
#include "Labs/1-RigidBody/CaseTwoBody.h"
#include "Labs/1-RigidBody/RigidBody.h"
#include "Labs/Common/ImGuiHelper.h"
#include "Engine/app.h"
#include "TwoBodyCollision.h"
#include <iostream>

namespace VCX::Labs::RigidBody {
    void CollisionDetection(Box &b0, Box &b1, fcl::CollisionResult<float> &collisionResult) {
        collisionResult.clear();
        // Eigen::Vector3f RigidBody::dim - size of a box
        using CollisionGeometryPtr_t = std::shared_ptr<fcl::CollisionGeometry<float>>;
        CollisionGeometryPtr_t box_geometry_A(new fcl::Box<float>(b0.dim[0], b0.dim[1],
        b0.dim[2]));
        CollisionGeometryPtr_t box_geometry_B(new fcl::Box<float>(b1.dim[0], b1.dim[1],
        b1.dim[2]));
        // Eigen::Vector3f RigidBody::x - position of a box, Eigen::Quaternionf
        fcl::CollisionObject<float> box_A(box_geometry_A,
        fcl::Transform3f(Eigen::Translation3f(b0.center)*b0.orientation));

        fcl::CollisionObject<float> box_B(box_geometry_B,
        fcl::Transform3f(Eigen::Translation3f(b1.center)*b1.orientation));
        // Compute collision - at most 8 contacts and return contact information.
        fcl::CollisionRequest<float> collisionRequest(1, true);
        fcl::collide(&box_A, &box_B, collisionRequest, collisionResult);
    }

    void CollisionResponse(Box &b0, Box &b1, fcl::CollisionResult<float> &collisionResult, float _restitution) {
        if(!collisionResult.isCollision()) {
            collisionResult.clear();
            return;
        }

        std::vector<fcl::Contact<float>> contacts;
        collisionResult.getContacts(contacts);
        Eigen::Vector3f contact_pos = contacts[0].pos;
        Eigen::Vector3f contact_normal = -contacts[0].normal;

        Eigen::Vector3f v0 = b0.velocity + b0.angularvelocity.cross(contact_pos - b0.center);
        Eigen::Vector3f v1 = b1.velocity + b1.angularvelocity.cross(contact_pos - b1.center);
        float relative_velocity = (v0 - v1).dot(contact_normal);

        if(relative_velocity > 0) {
            return;
        }


        float e = _restitution; // coefficient of restitution
        float numerator = -(1 + e) * relative_velocity;
        float denominator = 1 / b0.mass + 1 / b1.mass + contact_normal.dot((b0.GetInertiaMatrix().inverse() * (contact_pos - b0.center).cross(contact_normal)).cross(contact_pos - b0.center)) + contact_normal.dot((b1.GetInertiaMatrix().inverse() * (contact_pos - b1.center).cross(contact_normal)).cross(contact_pos - b1.center));
        float impulse = numerator / denominator;

        b0.velocity += impulse / b0.mass * contact_normal;
        b1.velocity -= impulse / b1.mass * contact_normal;
        b0.angularvelocity += impulse * b0.GetInertiaMatrix().inverse() * (contact_pos - b0.center).cross(contact_normal);
        b1.angularvelocity -= impulse * b1.GetInertiaMatrix().inverse() * (contact_pos - b1.center).cross(contact_normal);
        collisionResult.clear();
    }

} 