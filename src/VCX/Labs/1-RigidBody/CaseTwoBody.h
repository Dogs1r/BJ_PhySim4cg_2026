//to be completed
#pragma once

#include<Eigen/Core>
#include<Eigen/Geometry>

#include "Engine/GL/Frame.hpp"
#include "Engine/GL/Program.h"
#include "Engine/GL/RenderItem.h"
#include "Labs/Common/ICase.h"
#include "Labs/Common/ImageRGB.h"
#include "Labs/Common/OrbitCameraManager.h"
#include "Labs/Common/ForceManager.h"
#include "Labs/1-RigidBody/RigidBody.h"
#include <fcl/narrowphase/collision.h>

namespace VCX::Labs::RigidBody {

    class CaseTwoBody : public Common::ICase {
    public:
        CaseTwoBody();

        virtual std::string_view const GetName() override { return "Box Collide"; }

        virtual void                     OnSetupPropsUI() override;
        virtual Common::CaseRenderResult OnRender(std::pair<std::uint32_t, std::uint32_t> const desiredSize) override;
        virtual void                     OnProcessInput(ImVec2 const & pos) override;

        void OnProcessMouseControl();
        void OnUpdate(float deltaTime);
        void CollisionDetection();
        void CollisionResponse();
        void Reset();

    private:
        Engine::GL::UniqueProgram           _program;
        Engine::GL::UniqueRenderFrame       _frame;
        Engine::Camera                      _camera { .Eye = glm::vec3(-3, 3, 3) };
        Common::OrbitCameraManager          _cameraManager;
        Common::ForceManager                _forceManager;
        Engine::GL::UniqueIndexedRenderItem _boxItem;  // render the box
        Engine::GL::UniqueIndexedRenderItem _lineItem; // render line on box
        fcl::CollisionResult<float>                  _collisionResult;
        Box                                _initialbox[2];
        Box                                 _box[2];
        int                                _caseId = 0;
        bool                               _recompute = true;
        float                             _restitution = 0.5f;
    };
} // namespace VCX::Labs::RigidBody
