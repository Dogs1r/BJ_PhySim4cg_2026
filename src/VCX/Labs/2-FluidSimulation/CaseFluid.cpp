#include <spdlog/spdlog.h>
#include "Engine/app.h"
#include "Labs/2-FluidSimulation/CaseFluid.h"
#include "Labs/Common/ImGuiHelper.h"
#include <glm/ext.hpp>
#include <iostream>
#include <algorithm>

namespace VCX::Labs::FluidSimulation {
    namespace {
        struct MouseRay {
            glm::vec3 Origin;
            glm::vec3 Dir;
        };

        MouseRay MakeMouseRay(glm::vec2 const mousePos, std::pair<std::uint32_t, std::uint32_t> const size, VCX::Engine::Camera const & camera) {
            float const width  = std::max(1.0f, static_cast<float>(size.first));
            float const height = std::max(1.0f, static_cast<float>(size.second));
            float const x      = 2.0f * mousePos.x / width - 1.0f;
            float const y      = 1.0f - 2.0f * mousePos.y / height;

            glm::mat4 const inv = glm::inverse(camera.GetProjectionMatrix(width / height) * camera.GetViewMatrix());
            glm::vec4 const nearClip(x, y, -1.0f, 1.0f);
            glm::vec4 const farClip(x, y, 1.0f, 1.0f);
            glm::vec4 const nearWorld4 = inv * nearClip;
            glm::vec4 const farWorld4  = inv * farClip;
            glm::vec3 const nearWorld  = glm::vec3(nearWorld4) / nearWorld4.w;
            glm::vec3 const farWorld   = glm::vec3(farWorld4) / farWorld4.w;
            return { nearWorld, glm::normalize(farWorld - nearWorld) };
        }

        bool IntersectRaySphere(MouseRay const & ray, glm::vec3 const & center, float radius, float & tHit) {
            glm::vec3 const oc = ray.Origin - center;
            float const a      = glm::dot(ray.Dir, ray.Dir);
            float const b      = 2.0f * glm::dot(oc, ray.Dir);
            float const c      = glm::dot(oc, oc) - radius * radius;
            float const disc   = b * b - 4.0f * a * c;
            if (disc < 0.0f)
                return false;

            float const sqrtDisc = std::sqrt(disc);
            float const t0       = (-b - sqrtDisc) / (2.0f * a);
            float const t1       = (-b + sqrtDisc) / (2.0f * a);
            tHit                 = t0 > 0.0f ? t0 : t1;
            return tHit > 0.0f;
        }

        bool IntersectRayPlane(MouseRay const & ray, glm::vec3 const & planePoint, glm::vec3 const & planeNormal, glm::vec3 & hitPoint) {
            float const denom = glm::dot(ray.Dir, planeNormal);
            if (std::abs(denom) < 1e-6f)
                return false;
            float const t = glm::dot(planePoint - ray.Origin, planeNormal) / denom;
            if (t < 0.0f)
                return false;
            hitPoint = ray.Origin + t * ray.Dir;
            return true;
        }

        glm::vec3 ClampObstacleToTank(glm::vec3 const & pos, float radius) {
            float const minC = -0.5f + radius;
            float const maxC = 0.5f - radius;
            return {
                std::clamp(pos.x, minC, maxC),
                std::clamp(pos.y, minC, maxC),
                std::clamp(pos.z, minC, maxC),
            };
        }
    } // namespace

    const std::vector<glm::vec3> vertex_pos = {
            glm::vec3(-0.5f, -0.5f, -0.5f),
            glm::vec3(0.5f, -0.5f, -0.5f),  
            glm::vec3(0.5f, 0.5f, -0.5f),  
            glm::vec3(-0.5f, 0.5f, -0.5f), 
            glm::vec3(-0.5f, -0.5f, 0.5f),  
            glm::vec3(0.5f, -0.5f, 0.5f),   
            glm::vec3(0.5f, 0.5f, 0.5f),   
            glm::vec3(-0.5f, 0.5f, 0.5f)
    };
    const std::vector<std::uint32_t> line_index = { 0, 1, 1, 2, 2, 3, 3, 0, 4, 5, 5, 6, 6, 7, 7, 4, 0, 4, 1, 5, 2, 6, 3, 7 }; // line index

    CaseFluid::CaseFluid(std::initializer_list<Assets::ExampleScene> && scenes) :
        _scenes(scenes),
        _program(
            Engine::GL::UniqueProgram({
                Engine::GL::SharedShader("assets/shaders/fluid.vert"),
                Engine::GL::SharedShader("assets/shaders/fluid.frag") })),
        _lineprogram(
            Engine::GL::UniqueProgram({
                Engine::GL::SharedShader("assets/shaders/flat.vert"),
                Engine::GL::SharedShader("assets/shaders/flat.frag") })),
        _sceneObject(1),
        _BoundaryItem(Engine::GL::VertexLayout()
            .Add<glm::vec3>("position", Engine::GL::DrawFrequency::Stream , 0), Engine::GL::PrimitiveType::Lines){ 
        _cameraManager.AutoRotate = false;
        _program.BindUniformBlock("PassConstants", 1);
        _program.GetUniforms().SetByName("u_DiffuseMap" , 0);
        _program.GetUniforms().SetByName("u_SpecularMap", 1);
        _program.GetUniforms().SetByName("u_HeightMap"  , 2);

        _lineprogram.GetUniforms().SetByName("u_Color",  glm::vec3(1.0f));
        _BoundaryItem.UpdateElementBuffer(line_index);
        ResetSystem();
        _sphere = Engine::Model { Engine::Sphere(6, _r), 0 };
    }

    void CaseFluid::OnSetupPropsUI() {
        if(ImGui::Button("Reset System")) 
            ResetSystem();
        ImGui::SameLine();
        if(ImGui::Button(_stopped ? "Start Simulation":"Stop Simulation"))
            _stopped = ! _stopped;
        ImGui::Spacing();
        ImGui::SliderInt("numSubSteps", &_numSubSteps, 1, 30);
        ImGui::SliderFloat("Flip Ratio", &_simulation.m_fRatio, 0.0f, 1.0f);
        ImGui::Checkbox("Drag obstacle with mouse", &_enableObstacleDrag);
        ImGui::SliderFloat("obstacleRadius", &_obstacleRadius, 0.02f, 0.5f);

    }


    Common::CaseRenderResult CaseFluid::OnRender(std::pair<std::uint32_t, std::uint32_t> const desiredSize) {
        if (_recompute) {
            _recompute = false;
            _sceneObject.ReplaceScene(GetScene(_sceneIdx));
            _cameraManager.Save(_sceneObject.Camera);
        }
        if (! _stopped) {
            float dt = Engine::GetDeltaTime();
            int numSubSteps = std::max(1, _numSubSteps);
            float sdt = dt / static_cast<float>(numSubSteps);

            for (int step = 0; step < numSubSteps; ++step) {
                _simulation.integrateParticles(sdt);
                _simulation.handleParticleCollisions(_obstaclePos, _obstacleRadius, _obstacleVel, sdt);
                if (_separateParticles)
                    _simulation.pushParticlesApart(_numParticleIters, sdt);
                _simulation.handleParticleCollisions(_obstaclePos, _obstacleRadius, _obstacleVel, sdt);
                _simulation.transferVelocities(true, _simulation.m_fRatio);
                _simulation.updateParticleDensity();
                _simulation.solveIncompressibility(_numPressureIters, sdt, _overRelaxation, _compensateDrift);
                _simulation.transferVelocities(false, _simulation.m_fRatio);
            }
            _simulation.updateParticleColors();
        }
        
        _BoundaryItem.UpdateVertexBuffer("position", Engine::make_span_bytes<glm::vec3>(vertex_pos));
        _frame.Resize(desiredSize);

        _cameraManager.Update(_sceneObject.Camera);
        _sceneObject.PassConstantsBlock.Update(&VCX::Labs::Rendering::SceneObject::PassConstants::Projection, _sceneObject.Camera.GetProjectionMatrix((float(desiredSize.first) / desiredSize.second)));
        _sceneObject.PassConstantsBlock.Update(&VCX::Labs::Rendering::SceneObject::PassConstants::View, _sceneObject.Camera.GetViewMatrix());
        _sceneObject.PassConstantsBlock.Update(&VCX::Labs::Rendering::SceneObject::PassConstants::ViewPosition, _sceneObject.Camera.Eye);
        _lineprogram.GetUniforms().SetByName("u_Projection", _sceneObject.Camera.GetProjectionMatrix((float(desiredSize.first) / desiredSize.second)));
        _lineprogram.GetUniforms().SetByName("u_View"      , _sceneObject.Camera.GetViewMatrix());
        
        if (_uniformDirty) {
            _uniformDirty = false;
            _program.GetUniforms().SetByName("u_AmbientScale"      , _ambientScale);
            _program.GetUniforms().SetByName("u_UseBlinn"          , _useBlinn);
            _program.GetUniforms().SetByName("u_Shininess"         , _shininess);
            _program.GetUniforms().SetByName("u_UseGammaCorrection", int(_useGammaCorrection));
            _program.GetUniforms().SetByName("u_AttenuationOrder"  , _attenuationOrder);            
            _program.GetUniforms().SetByName("u_BumpMappingBlend"  , _bumpMappingPercent * .01f);            

        }
        
        gl_using(_frame);

        glEnable(GL_DEPTH_TEST);
        glLineWidth(_BndWidth);
        _BoundaryItem.Draw({ _lineprogram.Use() });
        glLineWidth(1.f);

        // Rendering::ModelObject m = Rendering::ModelObject(_sphere,_simulation.Positions);
        Rendering::ModelObject m = Rendering::ModelObject(_sphere,_simulation.m_particlePos,_simulation.m_particleColor);
        auto const & material    = _sceneObject.Materials[0];
        m.Mesh.Draw({ material.Albedo.Use(),  material.MetaSpec.Use(), material.Height.Use(),_program.Use() },
            _sphere.Mesh.Indices.size(), 0, _simulation.m_iNumSpheres);
        
        if (_obstacleRadius > 0.0f) {
            Engine::Model obstacle_sphere { Engine::Sphere(6, _obstacleRadius), 0 };
            glm::vec3 obstacleColor = _draggingObstacle ? glm::vec3(1.0f, 0.55f, 0.15f) : glm::vec3(0.0f, 0.0f, 1.0f);
            Rendering::ModelObject obstacle = Rendering::ModelObject(obstacle_sphere, { _obstaclePos }, { obstacleColor });
            obstacle.Mesh.Draw({ material.Albedo.Use(),  material.MetaSpec.Use(), material.Height.Use(),_program.Use() },
                obstacle_sphere.Mesh.Indices.size(), 0, 1);
        }

        glDepthFunc(GL_LEQUAL);
        glDepthFunc(GL_LESS);
        glDisable(GL_DEPTH_TEST);

        return Common::CaseRenderResult {
            .Fixed     = false,
            .Flipped   = true,
            .Image     = _frame.GetColorAttachment(),
            .ImageSize = desiredSize,
        };
    }

    void CaseFluid::OnProcessInput(ImVec2 const& pos) {
        auto window = ImGui::GetCurrentWindow();
        bool hovered = false;
        bool anyHeld = false;
        ImGui::ButtonBehavior(window->Rect(), window->GetID("##io"), &hovered, &anyHeld);
        (void)anyHeld;

        if (_enableObstacleDrag && hovered && ImGui::IsMouseClicked(ImGuiMouseButton_Left) && _obstacleRadius > 0.0f) {
            MouseRay const ray = MakeMouseRay(glm::vec2(pos.x, pos.y), { static_cast<std::uint32_t>(window->Rect().GetWidth()), static_cast<std::uint32_t>(window->Rect().GetHeight()) }, _sceneObject.Camera);
            float tHit = 0.0f;
            if (IntersectRaySphere(ray, _obstaclePos, _obstacleRadius, tHit)) {
                _draggingObstacle = true;
                glm::vec3 hitPoint {};
                _dragPlanePoint  = _obstaclePos;
                _dragPlaneNormal = glm::normalize(_sceneObject.Camera.Target - _sceneObject.Camera.Eye);
                if (! IntersectRayPlane(ray, _dragPlanePoint, _dragPlaneNormal, hitPoint))
                    hitPoint = _obstaclePos;
                _dragAnchorOffset = _obstaclePos - hitPoint;
                _obstacleVel      = glm::vec3(0.0f);
            }
        }

        if (_draggingObstacle && ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
            MouseRay const ray = MakeMouseRay(glm::vec2(pos.x, pos.y), { static_cast<std::uint32_t>(window->Rect().GetWidth()), static_cast<std::uint32_t>(window->Rect().GetHeight()) }, _sceneObject.Camera);
            glm::vec3 hitPoint {};
            if (IntersectRayPlane(ray, _dragPlanePoint, _dragPlaneNormal, hitPoint)) {
                glm::vec3 const prevPos = _obstaclePos;
                _obstaclePos            = ClampObstacleToTank(hitPoint + _dragAnchorOffset, _obstacleRadius);
                float const dt          = std::max(Engine::GetDeltaTime(), 1e-6f);
                _obstacleVel            = (_obstaclePos - prevPos) / dt;
            }
            return;
        }

        if (_draggingObstacle && ImGui::IsMouseReleased(ImGuiMouseButton_Left)) {
            _obstacleVel     = glm::vec3(0.0f);
            _dragPlanePoint  = _obstaclePos;
            _draggingObstacle = false;
            return;
        }

        _cameraManager.ProcessInput(_sceneObject.Camera, pos);
    }

    void CaseFluid::ResetSystem(){
        _simulation.setupScene(_res);
        numofSpheres = _simulation.m_iNumSpheres;
        _r = _simulation.m_particleRadius; //cell size
        _obstaclePos = glm::vec3(0.0f);
        _obstacleVel = glm::vec3(0.0f);
        _draggingObstacle = false;
        _dragPlanePoint = glm::vec3(0.0f);
        _dragPlaneNormal = glm::vec3(0.0f, 0.0f, 1.0f);
        _dragAnchorOffset = glm::vec3(0.0f);
    }
}
