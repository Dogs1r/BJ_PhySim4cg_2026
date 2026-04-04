#include "Labs/1-RigidBody/CaseTwoBody.h"
#include "Labs/Common/ImGuiHelper.h"
#include "Engine/GL/Shader.h"
#include "Engine/app.h"
#include "Labs/1-RigidBody/TwoBodyCollision.h"

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

    static constexpr auto c_Cases = std::array<char const *, 3> {
        "Face-Vertex Collision",
        "Line-Line Collision",
        "Face-Face Collision"
    };

    static const std::array<std::pair<Eigen::Vector3f, Eigen::Vector3f>, 3> c_BoxDims = {
        std::pair<Eigen::Vector3f, Eigen::Vector3f>{Eigen::Vector3f{1, 1, 1}, Eigen::Vector3f{1, 1, 1}},
        std::pair<Eigen::Vector3f, Eigen::Vector3f>{Eigen::Vector3f{0.5, 0.5, 1.5}, Eigen::Vector3f{0.5, 1.5, 0.5}},
        std::pair<Eigen::Vector3f, Eigen::Vector3f>{Eigen::Vector3f{1, 1, 1}, Eigen::Vector3f{1, 1, 1}}
    };

    static const std::array<std::pair<Eigen::Vector3f, Eigen::Vector3f>, 3> c_BoxPositions = {
        std::pair<Eigen::Vector3f, Eigen::Vector3f>{Eigen::Vector3f{0, 0, 0}, Eigen::Vector3f{3, 0, 0}},
        std::pair<Eigen::Vector3f, Eigen::Vector3f>{Eigen::Vector3f{0, 0, 0}, Eigen::Vector3f{3, -0.5, 0}},
        std::pair<Eigen::Vector3f, Eigen::Vector3f>{Eigen::Vector3f{0, 0, 0}, Eigen::Vector3f{3, 0, 0}}
    };

    static const std::array<std::pair<Eigen::Quaternionf, Eigen::Quaternionf>, 3> c_BoxOrientations = {
        std::pair<Eigen::Quaternionf, Eigen::Quaternionf>{Eigen::Quaternionf{1, 0, 0, 0}, Eigen::Quaternionf{0.3, 0.9, 0.3, 0.1}},
        std::pair<Eigen::Quaternionf, Eigen::Quaternionf>{Eigen::Quaternionf{1, 0, 0, 0}, Eigen::Quaternionf{0.3, 0.3, 0.9, 0.1}},
        std::pair<Eigen::Quaternionf, Eigen::Quaternionf>{Eigen::Quaternionf{1, 0, 0, 0}, Eigen::Quaternionf{1, 0, 0, 0}}
    };

    CaseTwoBody::CaseTwoBody():
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

        _box[0].velocity=  Eigen::Vector3f(2.f,0.f,0.f);
        _box[0].angularvelocity= Eigen::Vector3f(0.f,0.f,0.f);
        _box[1].velocity=  Eigen::Vector3f(-2.f,0.f,0.f);
        _box[1].angularvelocity= Eigen::Vector3f(0.f,0.f,0.f);
        _initialbox[0]=_box[0];
        _initialbox[1]=_box[1];
    }

    void CaseTwoBody::OnSetupPropsUI() {
        _recompute |= ImGui::Combo("Case", &_caseId, c_Cases.data(), c_Cases.size());
        if (ImGui::CollapsingHeader("Appearance", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::ColorEdit3("Box1 Color", glm::value_ptr(_box[0].color));
            ImGui::SliderFloat("Box1 x", &_initialbox[0].dim[0], 0.5, 4);
            ImGui::SliderFloat("Box1 y", &_initialbox[0].dim[1], 0.5, 4);
            ImGui::SliderFloat("Box1 z", &_initialbox[0].dim[2], 0.5, 4);

            ImGui::ColorEdit3("Box2 Color", glm::value_ptr(_box[1].color));
            ImGui::SliderFloat("Box2 x", &_initialbox[1].dim[0], 0.5, 4);
            ImGui::SliderFloat("Box2 y", &_initialbox[1].dim[1], 0.5, 4);
            ImGui::SliderFloat("Box2 z", &_initialbox[1].dim[2], 0.5, 4);
        }
        if (ImGui::CollapsingHeader("Position", ImGuiTreeNodeFlags_DefaultOpen)){
            ImGui::InputFloat("Box1 pos_x", &_initialbox[0].center[0]);
            ImGui::InputFloat("Box1 pos_y", &_initialbox[0].center[1]);
            ImGui::InputFloat("Box1 pos_z", &_initialbox[0].center[2]);

            ImGui::InputFloat("Box2 pos_x", &_initialbox[1].center[0]);
            ImGui::InputFloat("Box2 pos_y", &_initialbox[1].center[1]);
            ImGui::InputFloat("Box2 pos_z", &_initialbox[1].center[2]);
        }
        if (ImGui::CollapsingHeader("Velocity", ImGuiTreeNodeFlags_DefaultOpen)){

            ImGui::InputFloat("Box1 vel_x", &_initialbox[0].velocity[0]);
            ImGui::InputFloat("Box1 vel_y", &_initialbox[0].velocity[1]);
            ImGui::InputFloat("Box1 vel_z", &_initialbox[0].velocity[2]);

            ImGui::InputFloat("Box2 vel_x", &_initialbox[1].velocity[0]);
            ImGui::InputFloat("Box2 vel_y", &_initialbox[1].velocity[1]);
            ImGui::InputFloat("Box2 vel_z", &_initialbox[1].velocity[2]);
        }
        ImGui::Spacing();
    }

    Common::CaseRenderResult CaseTwoBody::OnRender(std::pair<std::uint32_t, std::uint32_t> const desiredSize) {
        // apply mouse control first
        // std::pair<glm::vec3,glm::vec3> force= _forceManager.getForce(eigen2glm(_box.center));
        OnProcessMouseControl();

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
        std::array<std::vector<glm::vec3>, 2> boxVerts;
        boxVerts[0].resize(8);
        boxVerts[1].resize(8);
        for (int i=0;i<2;i++){
            Eigen::Matrix3f R=_box[i].orientation.toRotationMatrix();
            Eigen::Vector3f              new_x = _box[i].dim[0] / 2 * R * Eigen::Vector3f(1.f, 0.f, 0.f);
            Eigen::Vector3f              new_y = _box[i].dim[1] / 2 * R * Eigen::Vector3f(0.f, 1.f, 0.f);
            Eigen::Vector3f              new_z = _box[i].dim[2] / 2 * R * Eigen::Vector3f(0.f, 0.f, 1.f);

            boxVerts[i][0] = eigen2glm( _box[i].center - new_x + new_y + new_z);
            boxVerts[i][1] = eigen2glm( _box[i].center + new_x + new_y + new_z);
            boxVerts[i][2] = eigen2glm( _box[i].center + new_x + new_y - new_z);
            boxVerts[i][3] = eigen2glm( _box[i].center - new_x + new_y - new_z);
            boxVerts[i][4] = eigen2glm( _box[i].center - new_x - new_y + new_z);
            boxVerts[i][5] = eigen2glm( _box[i].center + new_x - new_y + new_z);
            boxVerts[i][6] = eigen2glm( _box[i].center + new_x - new_y - new_z);
            boxVerts[i][7] = eigen2glm( _box[i].center - new_x - new_y - new_z);
        }

        for (int i = 0; i < 2; ++i) {
            auto span_bytes = Engine::make_span_bytes<glm::vec3>(boxVerts[i]);

            _program.GetUniforms().SetByName("u_Color", _box[i].color);
            _boxItem.UpdateVertexBuffer("position", span_bytes);
            _boxItem.Draw({ _program.Use() });

            _program.GetUniforms().SetByName("u_Color", glm::vec3(1.f, 1.f, 1.f));
            _lineItem.UpdateVertexBuffer("position", span_bytes);
            _lineItem.Draw({ _program.Use() });
        }

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

    void CaseTwoBody::OnProcessMouseControl(){
        if(ImGui::IsKeyDown(ImGuiKey_LeftCtrl)) {
            _forceManager.ProcessInput(_camera, ImGui::GetMousePos());
        }
        if(ImGui::IsKeyDown(ImGuiKey_LeftAlt)&&ImGui::IsKeyDown(ImGuiKey_F)) {
            Reset();
            return;
            
        }
        if(_recompute){
            _recompute=false;
            _initialbox[0].center = c_BoxPositions[_caseId].first;
            _initialbox[1].center = c_BoxPositions[_caseId].second;

            _initialbox[0].orientation = c_BoxOrientations[_caseId].first;
            _initialbox[1].orientation = c_BoxOrientations[_caseId].second;

            _initialbox[0].dim = c_BoxDims[_caseId].first;
            _initialbox[1].dim = c_BoxDims[_caseId].second;
            _box[0] = _initialbox[0];
            _box[1] = _initialbox[1];           
        }
    }

    void CaseTwoBody::Reset(){
        _box[0]=_initialbox[0];
        _box[1]=_initialbox[1];
    }

    void CaseTwoBody::OnUpdate(float deltaTime){
        for(int i=0;i<2;i++){
            _box[i].center+=_box[i].velocity*deltaTime;
            Eigen::Quaternionf _angularVelocityQ(0.f,_box[i].angularvelocity.x(),_box[i].angularvelocity.y(),_box[i].angularvelocity.z());
            _box[i].orientation.coeffs()+=0.5f*deltaTime*(_angularVelocityQ*_box[i].orientation).coeffs();
            _box[i].orientation.normalize();
        }
        CollisionDetection();
        CollisionResponse();
    }

    void CaseTwoBody::OnProcessInput(ImVec2 const & pos) {
        _cameraManager.ProcessInput(_camera, pos);
    }

    void CaseTwoBody::CollisionDetection(){
        VCX::Labs::RigidBody::CollisionDetection(_box[0],_box[1],_collisionResult);
    }

    void CaseTwoBody::CollisionResponse(){
        VCX::Labs::RigidBody::CollisionResponse(_box[0],_box[1],_collisionResult, _restitution);
    }
}