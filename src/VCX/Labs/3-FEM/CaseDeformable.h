#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>


#include "Engine/GL/Frame.hpp"
#include "Engine/GL/Program.h"
#include "Engine/GL/RenderItem.h"
#include "Labs/3-FEM/Femsys.h"
#include "Labs/Common/ICase.h"
#include "Labs/Common/ImageRGB.h"
#include "Labs/Common/OrbitCameraManager.h"
#include "Labs/Common/ForceManager.h"

namespace VCX::Labs::FEM{
    class CaseDeformable : public Common::ICase{
        public:
            CaseDeformable();
            virtual std::string_view const      GetName() override { return "Deformable Object"; }
            virtual void                        OnSetupPropsUI() override;
            virtual Common::CaseRenderResult    OnRender(std::pair<std::uint32_t, std::uint32_t> const desiredSize) override;
            virtual void                        OnProcessInput(ImVec2 const & pos) override;
            void                                OnProcessMouseControl(std::pair<glm::vec3, int> force);
            void                                ResetSystem();
        private:
            Engine::GL::UniqueProgram           _program;
        Engine::GL::UniqueRenderFrame           _frame;
        Engine::Camera                          _camera { .Eye = glm::vec3(-3, 3, 3) };
        Common::OrbitCameraManager              _cameraManager;
        Engine::GL::UniqueIndexedRenderItem     _tetItem;  
        FEM::Simulator                          _femSystem;
        Common::ForceManager                    _forceManager;

    };
}