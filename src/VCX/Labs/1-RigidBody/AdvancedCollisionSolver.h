#pragma once

#include <algorithm>
#include <vector>

#include <fcl/narrowphase/collision.h>

#include "Labs/1-RigidBody/RigidBody.h"

namespace VCX::Labs::RigidBody {

    // Iterative impulse + positional correction for more stable resting contacts.
    inline void SolveContactAdvanced(
        Box & boxA,
        Box & boxB,
        fcl::CollisionResult<float> & collisionResult,
        float restitution,
        float friction,
        float restitutionVelocityThreshold,
        float correctionBeta,
        float penetrationSlop,
        bool boxBStatic = false) {

        if (!collisionResult.isCollision()) {
            return;
        }

        std::vector<fcl::Contact<float>> contacts;
        collisionResult.getContacts(contacts);
        if (contacts.empty()) {
            return;
        }

        Eigen::Matrix3f invIA = boxA.GetInertiaMatrix().inverse();
        Eigen::Matrix3f invIB = Eigen::Matrix3f::Zero();
        if (!boxBStatic) {
            invIB = boxB.GetInertiaMatrix().inverse();
        }

        float invMassA = boxA.mass > 1e-6f ? 1.0f / boxA.mass : 0.0f;
        float invMassB = (boxBStatic || boxB.mass <= 1e-6f) ? 0.0f : 1.0f / boxB.mass;

        for (fcl::Contact<float> const & contact: contacts) {
            Eigen::Vector3f normal = -contact.normal;
            Eigen::Vector3f point = contact.pos;

            Eigen::Vector3f ra = point - boxA.center;
            Eigen::Vector3f rb = point - boxB.center;

            Eigen::Vector3f va = boxA.velocity + boxA.angularvelocity.cross(ra);
            Eigen::Vector3f vb = boxB.velocity + boxB.angularvelocity.cross(rb);
            float relativeNormalVelocity = (va - vb).dot(normal);
            float normalImpulse = 0.0f;

            if (relativeNormalVelocity < 0.0f) {
                float denom = invMassA + invMassB;
                denom += normal.dot((invIA * ra.cross(normal)).cross(ra));
                denom += normal.dot((invIB * rb.cross(normal)).cross(rb));

                if (denom > 1e-6f) {
                    float effectiveRestitution = relativeNormalVelocity < -restitutionVelocityThreshold ? restitution : 0.0f;
                    float impulse = -(1.0f + effectiveRestitution) * relativeNormalVelocity / denom;
                    normalImpulse = impulse;

                    boxA.velocity += impulse * invMassA * normal;
                    boxA.angularvelocity += impulse * invIA * ra.cross(normal);

                    if (!boxBStatic) {
                        boxB.velocity -= impulse * invMassB * normal;
                        boxB.angularvelocity -= impulse * invIB * rb.cross(normal);
                    }
                }
            }

            if (normalImpulse > 0.0f) {
                va = boxA.velocity + boxA.angularvelocity.cross(ra);
                vb = boxB.velocity + boxB.angularvelocity.cross(rb);
                Eigen::Vector3f tangentVelocity = (va - vb) - (va - vb).dot(normal) * normal;
                float tangentSpeed = tangentVelocity.norm();

                if (tangentSpeed > 1e-6f) {
                    Eigen::Vector3f tangent = tangentVelocity / tangentSpeed;
                    float tangentDenom = invMassA + invMassB;
                    tangentDenom += tangent.dot((invIA * ra.cross(tangent)).cross(ra));
                    tangentDenom += tangent.dot((invIB * rb.cross(tangent)).cross(rb));

                    if (tangentDenom > 1e-6f) {
                        float tangentImpulse = -tangentSpeed / tangentDenom;
                        float maxFrictionImpulse = std::max(0.0f, friction) * normalImpulse;
                        tangentImpulse = std::clamp(tangentImpulse, -maxFrictionImpulse, maxFrictionImpulse);

                        boxA.velocity += tangentImpulse * invMassA * tangent;
                        boxA.angularvelocity += tangentImpulse * invIA * ra.cross(tangent);

                        if (!boxBStatic) {
                            boxB.velocity -= tangentImpulse * invMassB * tangent;
                            boxB.angularvelocity -= tangentImpulse * invIB * rb.cross(tangent);
                        }
                    }
                }
            }

            float penetration = std::max(0.0f, contact.penetration_depth - penetrationSlop);
            if (penetration > 0.0f) {
                float totalInvMass = invMassA + invMassB;
                if (totalInvMass > 1e-6f) {
                    Eigen::Vector3f correction = correctionBeta * penetration / totalInvMass * normal;
                    boxA.center += invMassA * correction;
                    if (!boxBStatic) {
                        boxB.center -= invMassB * correction;
                    }
                }
            }
        }
    }

} // namespace VCX::Labs::RigidBody
