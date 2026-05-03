#pragma once

#include "Engine/GL/Frame.hpp"
#include "Engine/GL/Program.h"
#include "Engine/GL/UniformBlock.hpp"
#include "Engine/Sphere.h"
// #include "Labs/0-GettingStarted/FluidSimulator.h"
#include "Labs/2-FluidSimulation/FluidSimulator.h"
#include "Labs/Common/ICase.h"
#include "Labs/Common/ImageRGB.h"
#include "Labs/Common/OrbitCameraManager.h"
#include "Labs/Scene/Content.h"
#include "Labs/Scene/SceneObject.h"

namespace VCX::Labs::FluidSimulation {

    class CaseFluid : public Common::ICase {
    public:
        CaseFluid(std::initializer_list<Assets::ExampleScene> && scenes);

        virtual std::string_view const GetName() override { return "Fluid Simulation"; }

        virtual void                     OnSetupPropsUI() override;
        virtual Common::CaseRenderResult OnRender(std::pair<std::uint32_t, std::uint32_t> const desiredSize) override;
        virtual void                     OnProcessInput(ImVec2 const & pos) override;

    private:
        std::vector<Assets::ExampleScene> const _scenes;

        Engine::GL::UniqueProgram         _program;
        Engine::GL::UniqueProgram         _lineprogram;
        Engine::GL::UniqueRenderFrame     _frame;
        VCX::Labs::Rendering::SceneObject _sceneObject;
        std::size_t                       _sceneIdx { 0 };
        bool                              _recompute { true };
        bool                              _uniformDirty { true };
        int                               _msaa { 2 };
        int                               _useBlinn { 0 };
        float                             _shininess { 32 };
        float                             _ambientScale { 1 };
        bool                              _useGammaCorrection { true };
        int                               _attenuationOrder { 2 };
        int                               _bumpMappingPercent { 20 };

        Engine::GL::UniqueIndexedRenderItem _BoundaryItem;
        Common::OrbitCameraManager          _cameraManager;
        float                               _BndWidth { 2.0 };
        bool                                _stopped { false };
        Engine::Model                       _sphere;
        int                                 _res {24 };
        float                               _r;
        int                                 numofSpheres;
        Fluid::Simulator                    _simulation;

        int   _numSubSteps { 1 };
        int   _numParticleIters { 5 };
        int   _numPressureIters { 20 };
        bool  _separateParticles { true };
        float _overRelaxation { 0.5f };
        bool  _compensateDrift { true };
        bool  _enableObstacleDrag { true };
        bool  _draggingObstacle { false };
        glm::vec3 _dragPlanePoint { 0.0f };
        glm::vec3 _dragPlaneNormal { 0.0f, 0.0f, 1.0f };
        glm::vec3 _dragAnchorOffset { 0.0f };

        glm::vec3 _obstaclePos { 0.0f };
        glm::vec3 _obstacleVel { 0.0f };
        float     _obstacleRadius { 0.12f };

        char const *          GetSceneName(std::size_t const i) const { return VCX::Labs::Rendering::Content::SceneNames[std::size_t(_scenes[i])].c_str(); }
        Engine::Scene const & GetScene(std::size_t const i) const { return VCX::Labs::Rendering::Content::Scenes[std::size_t(_scenes[i])]; }
        void                  ResetSystem();
    };
} // namespace VCX::Labs::GettingStarted
