#include "Labs/1-RigidBody/CaseSingleBody.h"
#include "Labs/Common/ImGuiHelper.h"
#include "Engine/GL/Shader.h"
#include "Engine/app.h"
#include <iostream>

// static std::vector<glm::vec3> eigen2glm(Eigen::VectorXf const & eigen_v) {
//     return std::vector<glm::vec3>(
//         reinterpret_cast<glm::vec3 const *>(eigen_v.data()),
//         reinterpret_cast<glm::vec3 const *>(eigen_v.data() + eigen_v.size())
//     );
// }

static glm::vec3 eigen2glm(Eigen::Vector3f const & v) {
    return glm::vec3(v.x(), v.y(), v.z());
}

// static Eigen::VectorXf glm2eigen(std::vector<glm::vec3> const & glm_v) {
//     Eigen::VectorXf v = Eigen::Map<Eigen::VectorXf const, Eigen::Aligned>
//     (reinterpret_cast<float const *>(glm_v.data()), static_cast<int>(glm_v.size() * 3));
//     return v;
// }

static Eigen::Vector3f glm2eigen(glm::vec3 const & v) {
    return Eigen::Vector3f(v.x, v.y, v.z);
}

namespace VCX::Labs::RigidBody {

    CaseSingleBody::CaseSingleBody():
        _program(
            Engine::GL::UniqueProgram({ Engine::GL::SharedShader("assets/shaders/flat.vert"),
                                        Engine::GL::SharedShader("assets/shaders/flat.frag") })),
        _boxItem(Engine::GL::VertexLayout().Add<glm::vec3>("position", Engine::GL::DrawFrequency::Stream, 0), Engine::GL::PrimitiveType::Triangles),
        _lineItem(Engine::GL::VertexLayout().Add<glm::vec3>("position", Engine::GL::DrawFrequency::Stream, 0), Engine::GL::PrimitiveType::Lines) {
        //     3-----2
        //    /|    /|
        //   0 --- 1 |
        //   | 7 - | 6
        //   |/    |/
        //   4 --- 5
        const std::vector<std::uint32_t> line_index = { 0, 1, 1, 2, 2, 3, 3, 0, 4, 5, 5, 6, 6, 7, 7, 4, 0, 4, 1, 5, 2, 6, 3, 7 }; // line index
        _lineItem.UpdateElementBuffer(line_index);

        // const std::vector<std::uint32_t> tri_index = { 0, 1, 2, 0, 2, 3, 1, 4, 0, 1, 4, 5, 1, 6, 5, 1, 2, 6, 2, 3, 7, 2, 6, 7, 0, 3, 7, 0, 4, 7, 4, 5, 6, 4, 6, 7 };
        const std::vector<std::uint32_t> tri_index = { 0, 1, 2, 0, 2, 3, 1, 0, 4, 1, 4, 5, 1, 5, 6, 1, 6, 2, 2, 7, 3, 2, 6, 7, 0, 3, 7, 0, 7, 4, 4, 6, 5, 4, 7, 6 };
        _boxItem.UpdateElementBuffer(tri_index);
        _cameraManager.AutoRotate = false;
        _cameraManager.Save(_camera);

        _box.velocity=  Eigen::Vector3f(0.f,0.f,0.f);
        _box.angularvelocity= Eigen::Vector3f(1.f,0.f,0.f);
    }

    void CaseSingleBody::OnSetupPropsUI() {
        if (ImGui::CollapsingHeader("Appearance", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::ColorEdit3("Box Color", glm::value_ptr(_box.color));
            ImGui::SliderFloat("x", &_box.dim[0], 0.5, 4);
            ImGui::SliderFloat("y", &_box.dim[1], 0.5, 4);
            ImGui::SliderFloat("z", &_box.dim[2], 0.5, 4);

            ImGui::InputFloat("pos_x", &_box.center[0]);
            ImGui::InputFloat("pos_y", &_box.center[1]);
            ImGui::InputFloat("pos_z", &_box.center[2]);

            ImGui::InputFloat("vel_x", &_box.velocity[0]);
            ImGui::InputFloat("vel_y", &_box.velocity[1]);
            ImGui::InputFloat("vel_z", &_box.velocity[2]);
        }
        ImGui::Spacing();
    }

    Common::CaseRenderResult CaseSingleBody::OnRender(std::pair<std::uint32_t, std::uint32_t> const desiredSize) {
        // apply mouse control first
        std::pair<glm::vec3,glm::vec3> force= _forceManager.getForce(eigen2glm(_box.center));
        OnProcessMouseControl(force);

        OnUpdate(VCX::Engine::GetDeltaTime());
        // rendering
        _frame.Resize(desiredSize);

        _cameraManager.Update(_camera);
        _program.GetUniforms().SetByName("u_Projection", _camera.GetProjectionMatrix((float(desiredSize.first) / desiredSize.second)));
        _program.GetUniforms().SetByName("u_View", _camera.GetViewMatrix());

        gl_using(_frame);
        glEnable(GL_DEPTH_TEST);
        glEnable(GL_LINE_SMOOTH);
        glLineWidth(.5f);

        std::vector<glm::vec3> VertsPosition;
        Eigen::Matrix3f R=_box.orientation.toRotationMatrix();
        Eigen::Vector3f              new_x = _box.dim[0] / 2 * R * Eigen::Vector3f(1.f, 0.f, 0.f);
        Eigen::Vector3f              new_y = _box.dim[1] / 2 * R * Eigen::Vector3f(0.f, 1.f, 0.f);
        Eigen::Vector3f              new_z = _box.dim[2] / 2 * R * Eigen::Vector3f(0.f, 0.f, 1.f);
        VertsPosition.resize(8);
        VertsPosition[0] = eigen2glm( _box.center - new_x + new_y + new_z);
        VertsPosition[1] = eigen2glm( _box.center + new_x + new_y + new_z);
        VertsPosition[2] = eigen2glm( _box.center + new_x + new_y - new_z);
        VertsPosition[3] = eigen2glm( _box.center - new_x + new_y - new_z);
        VertsPosition[4] = eigen2glm( _box.center - new_x - new_y + new_z);
        VertsPosition[5] = eigen2glm( _box.center + new_x - new_y + new_z);
        VertsPosition[6] = eigen2glm( _box.center + new_x - new_y - new_z);
        VertsPosition[7] = eigen2glm( _box.center - new_x - new_y - new_z);

        auto span_bytes = Engine::make_span_bytes<glm::vec3>(VertsPosition);

        _program.GetUniforms().SetByName("u_Color", _box.color);
        _boxItem.UpdateVertexBuffer("position", span_bytes);
        _boxItem.Draw({ _program.Use() });

        _program.GetUniforms().SetByName("u_Color", glm::vec3(1.f, 1.f, 1.f));
        _lineItem.UpdateVertexBuffer("position", span_bytes);
        _lineItem.Draw({ _program.Use() });

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

    void CaseSingleBody::OnProcessInput(ImVec2 const & pos) {
        _cameraManager.ProcessInput(_camera, pos);
        _forceManager.ProcessInput(_camera, pos);
    }

    void CaseSingleBody::OnProcessMouseControl(std::pair<glm::vec3,glm::vec3> force) {
        glm::vec3 forceVec=force.first;
        glm::vec3 forceCenter=force.second;
        Eigen::Vector3f torque=(glm2eigen(forceCenter)-_box.center).cross(glm2eigen(forceVec));
        
        float movingScale=1.f;
        _box.velocity+=glm2eigen(forceVec)*movingScale/_box.mass;
        _box.angularvelocity+=(_box.GetInertiaMatrix().inverse())*torque;

    }

    void CaseSingleBody::OnUpdate(float deltaTime){
        _box.center+=_box.velocity*deltaTime;
        Eigen::Quaternionf _angularVelocityQ(0.f,_box.angularvelocity.x(),_box.angularvelocity.y(),_box.angularvelocity.z());
        _box.orientation.coeffs()+=0.5f*deltaTime*(_angularVelocityQ*_box.orientation).coeffs();
        _box.orientation.normalize();
    }
} // namespace VCX::Labs::GettingStarted
