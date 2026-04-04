//to be completed
#pragma once

#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "Engine/GL/Frame.hpp"
#include "Engine/GL/Program.h"
#include "Engine/GL/RenderItem.h"
#include "Labs/1-RigidBody/RigidBody.h"
#include"Labs/1-RigidBody/TwoBodyCollision.h"
#include "Labs/Common/ForceManager.h"
#include "Labs/Common/ICase.h"
#include "Labs/Common/ImageRGB.h"
#include "Labs/Common/OrbitCameraManager.h"
#include <fcl/narrowphase/collision.h>

namespace VCX::Labs::RigidBody {

    class CaseMultiBody : public Common::ICase {
    public:
        CaseMultiBody();

        virtual std::string_view const GetName() override { return "MultiBody scene"; }

        virtual void                     OnSetupPropsUI() override;
        virtual Common::CaseRenderResult OnRender(std::pair<std::uint32_t, std::uint32_t> const desiredSize) override;
        virtual void                     OnProcessInput(ImVec2 const & pos) override;

        void AddBox();
        void OnProcessMouseControl();
        void OnUpdate(float deltaTime);
        void Reset();
        void BuildScene();


        void CollisionDetection();
        void CollisionResponse();

    private:
        std::vector<glm::vec3> BuildBoxVertices(Box const & box) const;

        struct ContactEntry {
            std::size_t                bodyA = 0;
            std::size_t                bodyB = 0;
            bool                       withGround = false;
            fcl::CollisionResult<float> result;
        };

    private:
        Engine::GL::UniqueProgram                       _program;
        Engine::GL::UniqueRenderFrame                   _frame;
        Engine::Camera                                  _camera { .Eye = glm::vec3(-6, 6, 6) };
        Common::OrbitCameraManager                      _cameraManager;
        Common::ForceManager                            _forceManager;
        Engine::GL::UniqueIndexedRenderItem             _boxItem;  // render faces
        Engine::GL::UniqueIndexedRenderItem             _lineItem; // render wireframe

        std::vector<Box>                                _initialBoxes;
        std::vector<Box>                                _boxes;
        Box                                             _ground;
        std::vector<ContactEntry>                       _contacts;

        int                                             _boxCount = 3;
        bool                                            _recompute = true;
        float                                           _restitution = 0.35f;
        float                                           _gravity = 9.8f;
        float                                           _groundHeight = -1.5f;
        int                                             _solverIterations = 10;
        float                                           _correctionBeta = 0.25f;
        float                                           _penetrationSlop = 0.003f;
        float                                           _friction = 0.65f;
        float                                           _restitutionVelocityThreshold = 0.5f;
        float                                           _fixedTimeStep = 1.0f / 120.0f;
        int                                             _maxSubSteps = 8;
        float                                           _timeAccumulator = 0.0f;
        
        float                                           _linearDamping = 0.08f;
        float                                           _angularDamping = 0.12f;
        float                                           _sleepLinearThreshold = 0.06f;
        float                                           _sleepAngularThreshold = 0.08f;
        int                                             _sleepFramesThreshold = 20;
        float                                           _spawnPlaneHeight = 2.0f;
        std::vector<int>                                _sleepCounters;

        bool                                           _isCharging=false;
        float                                          _chargeTime=0.f;
        float                                          _maxChargeTime=2.f;
        float                                          _minChargeTime=0.05f;
        float                                         _phro=2.f;
        Box                                           _PreviewBox;
        ImVec2                                        _mousePos { 0.f, 0.f };
        std::pair<std::uint32_t, std::uint32_t>      _renderSize { 1u, 1u };
    };
} // namespace VCX::Labs::RigidBody
