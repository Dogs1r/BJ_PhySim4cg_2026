#include "Labs/3-FEM/CaseDeformable.h"
#include "Labs/3-FEM/FemSys.h"
#include "Labs/Common/ImGuiHelper.h"
#include "Engine/app.h"
#include <iostream>
#include "Labs/Common/ForceManager.h"



namespace VCX::Labs::FEM {

    CaseDeformable::CaseDeformable():
        _program(
            Engine::GL::UniqueProgram({ Engine::GL::SharedShader("assets/shaders/flat.vert"),
                                        Engine::GL::SharedShader("assets/shaders/flat.frag") })),
        _tetItem(Engine::GL::VertexLayout().Add<glm::vec3>("position", Engine::GL::DrawFrequency::Stream, 0), Engine::GL::PrimitiveType::Triangles)
         {

        _femSystem.setupScene(8*2, 2*2, 2*2, 0.5f);
        // _femSystem.setupSceneSimple();

        const std::vector<std::uint32_t> tri_index_tet = { 0, 1, 2, 0, 1, 3, 0, 2, 3, 1, 2, 3};
        std::vector<std::uint32_t> tri_index;

        for(int i=0; i < _femSystem.tetrahedra.size(); i++)
        {
            glm::ivec4 curtet = _femSystem.tetrahedra[i];
            for(int j=0; j<tri_index_tet.size(); j++)
            {
                tri_index.push_back(curtet[tri_index_tet[j]]);
            }
        }
        _tetItem.UpdateElementBuffer(tri_index);
        _cameraManager.AutoRotate = false;
        _cameraManager.Save(_camera);

    }

    void CaseDeformable::ResetSystem() {
        _femSystem.particlePos = _femSystem.particlePosRest;
        _femSystem.particleVel = _femSystem.particleVelRest;
    }

    void CaseDeformable::OnSetupPropsUI() {
        if(ImGui::Button("Reset System")) 
            ResetSystem();

        ImGui::SliderFloat("Gravity", &_femSystem.g, 0.0f, 1.0f);
        ImGui::SliderFloat("Young", &_femSystem.young, 1000.0f, 100000.0f);
        ImGui::SliderFloat("Poison", &_femSystem.poison, -1.0f, 0.5f);
        ImGui::SliderFloat("Friction", &_femSystem.friction, 0.0f, 100.0f);
        ImGui::Spacing();
    }

    Common::CaseRenderResult CaseDeformable::OnRender(std::pair<std::uint32_t, std::uint32_t> const desiredSize) {
        // apply mouse control first
        std::pair<glm::vec3,int> force =  _forceManager.getForce(_femSystem.particlePos);

        OnProcessMouseControl(force);

        // Advance(Engine::GetDeltaTime());
        _femSystem.Simulate(Engine::GetDeltaTime(), 1);

        // rendering
        _frame.Resize(desiredSize);

        _cameraManager.Update(_camera);
        _program.GetUniforms().SetByName("u_Projection", _camera.GetProjectionMatrix((float(desiredSize.first) / desiredSize.second)));
        _program.GetUniforms().SetByName("u_View", _camera.GetViewMatrix());

        gl_using(_frame);
        glEnable(GL_LINE_SMOOTH);
        glLineWidth(.5f);

        auto span_bytes = Engine::make_span_bytes<glm::vec3>(_femSystem.particlePos);

        _program.GetUniforms().SetByName("u_Color", glm::vec3{ 121.0f / 255, 207.0f / 255, 171.0f / 255 });
        _tetItem.UpdateVertexBuffer("position", span_bytes);
        _tetItem.Draw({ _program.Use() });

        glLineWidth(1.f);
        glPointSize(1.f);
        glDisable(GL_LINE_SMOOTH);

        return Common::CaseRenderResult {
            .Fixed     = false,
            .Flipped   = true,
            .Image     = _frame.GetColorAttachment(),
            .ImageSize = desiredSize,
        };
    }

    void CaseDeformable::OnProcessInput(ImVec2 const & pos) {
        _cameraManager.ProcessInput(_camera, pos);
        _forceManager.ProcessInput(_camera, pos);
    }

    void CaseDeformable::OnProcessMouseControl(std::pair<glm::vec3,int> force) {
        glm::vec3 forceVec = force.first;
        int pointId = force.second;

        glm::vec3 pointPos = _femSystem.particlePos[pointId];
        
        float forceScale = 200.0f; 
        float forceRange = 2.0f;
        const float maxForcePerMass = 10000.0f; 

        for(int i=0; i<_femSystem.particlePos.size(); i++)
        {
            glm::vec3 pos = _femSystem.particlePos[i];
            glm::vec3 diff = pos - pointPos;
            float dist = glm::length(diff);
            if(dist < forceRange)
            {
                glm::vec3 applied = forceScale * forceVec * (1.0f - dist/forceRange);
                float maxF = maxForcePerMass * _femSystem.particle_weight;
                float mag = glm::length(applied);
                if(mag > maxF && mag > 0.0f)
                    applied = applied / mag * maxF;
                _femSystem.particleForce[i] += applied;
            }
        }


    }

} // namespace VCX::Labs::FEM
