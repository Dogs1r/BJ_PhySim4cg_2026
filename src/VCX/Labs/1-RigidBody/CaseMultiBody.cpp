#include "Labs/1-RigidBody/CaseMultiBody.h"

#include <algorithm>
#include <random>

#include <glm/glm.hpp>

#include "Labs/1-RigidBody/AdvancedCollisionSolver.h"
#include "Engine/GL/Shader.h"
#include "Engine/app.h"
#include "Labs/1-RigidBody/TwoBodyCollision.h"
#include "Labs/Common/ImGuiHelper.h"

static glm::vec3 eigen2glm(Eigen::Vector3f const & v) {
    return glm::vec3(v.x(), v.y(), v.z());
}

static Eigen::Vector3f glm2eigen(glm::vec3 const & v) {
    return Eigen::Vector3f(v.x, v.y, v.z);
}


static Eigen::Quaternionf randomQuaternion(std::mt19937 & gen) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    Eigen::Quaternionf                    q(dist(gen), dist(gen), dist(gen), dist(gen));
    q.normalize();
    return q;
}

static Eigen::Vector3f randomVector(Eigen::Vector3f const & lowerBound, Eigen::Vector3f const & upperBound, std::mt19937 & gen) {
    std::uniform_real_distribution<float> distx(lowerBound.x(), upperBound.x());
    std::uniform_real_distribution<float> disty(lowerBound.y(), upperBound.y());
    std::uniform_real_distribution<float> distz(lowerBound.z(), upperBound.z());
    return Eigen::Vector3f(distx(gen), disty(gen), distz(gen));
}

namespace VCX::Labs::RigidBody {

    CaseMultiBody::CaseMultiBody():
        _program(
            Engine::GL::UniqueProgram({ Engine::GL::SharedShader("assets/shaders/flat.vert"),
                                        Engine::GL::SharedShader("assets/shaders/flat.frag") })),
        _boxItem(Engine::GL::VertexLayout().Add<glm::vec3>("position", Engine::GL::DrawFrequency::Stream, 0), Engine::GL::PrimitiveType::Triangles),
        _lineItem(Engine::GL::VertexLayout().Add<glm::vec3>("position", Engine::GL::DrawFrequency::Stream, 0), Engine::GL::PrimitiveType::Lines) {
        const std::vector<std::uint32_t> lineIndex = { 0, 1, 1, 2, 2, 3, 3, 0, 4, 5, 5, 6, 6, 7, 7, 4, 0, 4, 1, 5, 2, 6, 3, 7 };
        const std::vector<std::uint32_t> triIndex  = { 0, 1, 2, 0, 2, 3, 1, 0, 4, 1, 4, 5, 1, 5, 6, 1, 6, 2, 2, 7, 3, 2, 6, 7, 0, 3, 7, 0, 7, 4, 4, 6, 5, 4, 7, 6 };

        _lineItem.UpdateElementBuffer(lineIndex);
        _boxItem.UpdateElementBuffer(triIndex);

        _cameraManager.AutoRotate = false;
        _cameraManager.Save(_camera);

        _ground = Box(Eigen::Vector3f(15.f, 1.f, 15.f), Eigen::Vector3f(0.f, _groundHeight - 0.5f, 0.f), Eigen::Quaternionf::Identity(), 100000.f);
        _ground.color = glm::vec3(0.35f, 0.35f, 0.4f);
        _ground.velocity = Eigen::Vector3f::Zero();
        _ground.angularvelocity = Eigen::Vector3f::Zero();

        _PreviewBox.color = glm::vec3(0.95f, 0.85f, 0.2f);
        _PreviewBox.orientation = Eigen::Quaternionf::Identity();
        _PreviewBox.velocity = Eigen::Vector3f::Zero();
        _PreviewBox.angularvelocity = Eigen::Vector3f::Zero();
        _PreviewBox.center = Eigen::Vector3f(0.f, _groundHeight + 2.f, 0.f);
        _PreviewBox.dim = Eigen::Vector3f::Constant(0.6f);

        BuildScene();
    }

    void CaseMultiBody::BuildScene() {
        std::random_device rd;
        std::mt19937       gen(rd());

        _initialBoxes.clear();
        _boxes.clear();
        _initialBoxes.reserve(static_cast<std::size_t>(_boxCount));

        for (int i = 0; i < _boxCount; ++i) {
            Box box;
            box.dim = randomVector(Eigen::Vector3f(0.55f, 0.55f, 0.55f), Eigen::Vector3f(1.15f, 1.15f, 1.15f), gen);
            box.center = randomVector(Eigen::Vector3f(-2.2f, _groundHeight + 2.5f, -2.2f), Eigen::Vector3f(2.2f, _groundHeight + 5.0f, 2.2f), gen);
            box.orientation = randomQuaternion(gen);
            box.velocity = Eigen::Vector3f::Zero();
            box.angularvelocity = Eigen::Vector3f::Zero();

            float volume = box.dim.x() * box.dim.y() * box.dim.z();
            box.mass = std::max(0.1f, _phro * volume);

            _initialBoxes.push_back(box);
        }

        _boxes = _initialBoxes;
        _sleepCounters.assign(_boxes.size(), 0);
        _recompute = false;
    }

    void CaseMultiBody::AddBox() {
        Box box = _PreviewBox;

        float sx = std::max(0.5f, box.dim.x());
        float sy = std::max(0.5f, box.dim.y());
        float sz = std::max(0.5f, box.dim.z());
        box.dim = Eigen::Vector3f(sx, sy, sz);

        float volume = sx * sy * sz;
        box.mass = std::max(0.1f, _phro * volume);

        box.velocity = Eigen::Vector3f::Zero();
        box.angularvelocity = Eigen::Vector3f::Zero();
        box.orientation = Eigen::Quaternionf::Identity();

        _boxes.push_back(box);
        _sleepCounters.push_back(0);
        _recompute = true;
    }

    std::vector<glm::vec3> CaseMultiBody::BuildBoxVertices(Box const & box) const {
        Eigen::Matrix3f R = box.orientation.toRotationMatrix();
        Eigen::Vector3f newX = box.dim[0] * 0.5f * R * Eigen::Vector3f(1.f, 0.f, 0.f);
        Eigen::Vector3f newY = box.dim[1] * 0.5f * R * Eigen::Vector3f(0.f, 1.f, 0.f);
        Eigen::Vector3f newZ = box.dim[2] * 0.5f * R * Eigen::Vector3f(0.f, 0.f, 1.f);

        std::vector<glm::vec3> verts(8);
        verts[0] = eigen2glm(box.center - newX + newY + newZ);
        verts[1] = eigen2glm(box.center + newX + newY + newZ);
        verts[2] = eigen2glm(box.center + newX + newY - newZ);
        verts[3] = eigen2glm(box.center - newX + newY - newZ);
        verts[4] = eigen2glm(box.center - newX - newY + newZ);
        verts[5] = eigen2glm(box.center + newX - newY + newZ);
        verts[6] = eigen2glm(box.center + newX - newY - newZ);
        verts[7] = eigen2glm(box.center - newX - newY - newZ);
        return verts;
    }

    void CaseMultiBody::OnSetupPropsUI() {
        if (ImGui::CollapsingHeader("Spawner", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::Text("Hold Shift + Left Mouse to charge and release a box.");
            ImGui::SliderFloat("Spawn Plane Height", &_spawnPlaneHeight, _groundHeight + 0.2f, 8.0f, "%.2f");
            if (ImGui::Button("Add Box Now")) {
                AddBox();
            }
        }

        if (ImGui::CollapsingHeader("Physics", ImGuiTreeNodeFlags_DefaultOpen)) {
            if (ImGui::Button("Reset Scene")) {
                Reset();
            }
        }
        ImGui::Spacing();
    }

    Common::CaseRenderResult CaseMultiBody::OnRender(std::pair<std::uint32_t, std::uint32_t> const desiredSize) {
        _renderSize = desiredSize;
        OnProcessMouseControl();
        OnUpdate(VCX::Engine::GetDeltaTime());

        _frame.Resize(desiredSize);
        _cameraManager.Update(_camera);
        _program.GetUniforms().SetByName("u_Projection", _camera.GetProjectionMatrix(float(desiredSize.first) / desiredSize.second));
        _program.GetUniforms().SetByName("u_View", _camera.GetViewMatrix());

        gl_using(_frame);
        glEnable(GL_DEPTH_TEST);
        glEnable(GL_LINE_SMOOTH);
        glLineWidth(0.5f);

        {
            auto groundVerts = BuildBoxVertices(_ground);
            auto span        = Engine::make_span_bytes<glm::vec3>(groundVerts);

            _program.GetUniforms().SetByName("u_Color", _ground.color);
            _boxItem.UpdateVertexBuffer("position", span);
            _boxItem.Draw({ _program.Use() });

            _program.GetUniforms().SetByName("u_Color", glm::vec3(0.85f, 0.85f, 0.9f));
            _lineItem.UpdateVertexBuffer("position", span);
            _lineItem.Draw({ _program.Use() });
        }

        for (Box const & box: _boxes) {
            auto verts = BuildBoxVertices(box);
            auto span  = Engine::make_span_bytes<glm::vec3>(verts);

            _program.GetUniforms().SetByName("u_Color", box.color);
            _boxItem.UpdateVertexBuffer("position", span);
            _boxItem.Draw({ _program.Use() });

            _program.GetUniforms().SetByName("u_Color", glm::vec3(1.f, 1.f, 1.f));
            _lineItem.UpdateVertexBuffer("position", span);
            _lineItem.Draw({ _program.Use() });
        }

        if (_isCharging) {
            auto previewVerts = BuildBoxVertices(_PreviewBox);
            auto span         = Engine::make_span_bytes<glm::vec3>(previewVerts);

            _program.GetUniforms().SetByName("u_Color", _PreviewBox.color);
            _boxItem.UpdateVertexBuffer("position", span);
            _boxItem.Draw({ _program.Use() });

            _program.GetUniforms().SetByName("u_Color", glm::vec3(0.2f, 0.2f, 0.2f));
            _lineItem.UpdateVertexBuffer("position", span);
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

    void CaseMultiBody::OnProcessInput(ImVec2 const & pos) {
        _mousePos = pos;
        _cameraManager.ProcessInput(_camera, pos);
        _forceManager.ProcessInput(_camera, pos);
    }

    void CaseMultiBody::OnProcessMouseControl() {
        if (ImGui::IsKeyDown(ImGuiKey_LeftAlt) && ImGui::IsKeyDown(ImGuiKey_F)) {
            Reset();
            return;
        }

        bool addMode = ImGui::IsKeyDown(ImGuiKey_LeftShift);
        if (addMode && !_isCharging && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
            _isCharging = true;
            _chargeTime = 0.f;
        }

        if (_isCharging && ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
            _chargeTime += VCX::Engine::GetDeltaTime();

            float t = std::clamp(_chargeTime / std::max(0.01f, _maxChargeTime), 0.f, 1.f);
            float s = 0.35f + 1.4f * t;

            _PreviewBox.dim = Eigen::Vector3f::Constant(s);

            float width = std::max(1u, _renderSize.first);
            float height = std::max(1u, _renderSize.second);
            float ndcX = 2.0f * (_mousePos.x / width) - 1.0f;
            float ndcY = 1.0f - 2.0f * (_mousePos.y / height);

            glm::mat4 projection = _camera.GetProjectionMatrix(width / height);
            glm::mat4 view = _camera.GetViewMatrix();
            glm::mat4 invVP = glm::inverse(projection * view);

            glm::vec4 nearH = invVP * glm::vec4(ndcX, ndcY, -1.0f, 1.0f);
            glm::vec4 farH = invVP * glm::vec4(ndcX, ndcY, 1.0f, 1.0f);
            if (nearH.w != 0.f && farH.w != 0.f) {
                glm::vec3 nearP = glm::vec3(nearH) / nearH.w;
                glm::vec3 farP = glm::vec3(farH) / farH.w;
                glm::vec3 rayDir = glm::normalize(farP - nearP);

                float targetY = _spawnPlaneHeight;
                if (std::abs(rayDir.y) > 1e-5f) {
                    float hitT = (targetY - _camera.Eye.y) / rayDir.y;
                    if (hitT > 0.f) {
                        glm::vec3 hit = _camera.Eye + hitT * rayDir;
                        _PreviewBox.center = Eigen::Vector3f(hit.x, targetY + 0.5f * s, hit.z);
                    }
                }
            }
        }

        if (_isCharging && ImGui::IsMouseReleased(ImGuiMouseButton_Left)) {
            if (_chargeTime >= _minChargeTime) {
                AddBox();
            }
            _isCharging = false;
            _chargeTime = 0.f;
        }
    }

    void CaseMultiBody::Reset() {
        _boxes = _initialBoxes;
        _sleepCounters.assign(_boxes.size(), 0);
        _isCharging = false;
        _chargeTime = 0.f;
        _recompute = false;
        _timeAccumulator = 0.f;
    }

    void CaseMultiBody::CollisionDetection() {
        _contacts.clear();
        _contacts.reserve(_boxes.size() * (_boxes.size() + 1) / 2);

        for (std::size_t i = 0; i < _boxes.size(); ++i) {
            for (std::size_t j = i + 1; j < _boxes.size(); ++j) {
                ContactEntry contact;
                contact.bodyA = i;
                contact.bodyB = j;
                contact.withGround = false;
                VCX::Labs::RigidBody::CollisionDetection(_boxes[i], _boxes[j], contact.result);
                if (contact.result.isCollision()) {
                    _contacts.push_back(std::move(contact));
                }
            }

            ContactEntry groundContact;
            groundContact.bodyA = i;
            groundContact.withGround = true;

            Box groundProxy = _ground;
            groundProxy.velocity = Eigen::Vector3f::Zero();
            groundProxy.angularvelocity = Eigen::Vector3f::Zero();
            VCX::Labs::RigidBody::CollisionDetection(_boxes[i], groundProxy, groundContact.result);
            if (groundContact.result.isCollision()) {
                _contacts.push_back(std::move(groundContact));
            }
        }
    }

    void CaseMultiBody::CollisionResponse() {
        int iterations = std::max(1, _solverIterations);
        for (int iter = 0; iter < iterations; ++iter) {
            for (ContactEntry & contact: _contacts) {
                if (contact.withGround) {
                    SolveContactAdvanced(_boxes[contact.bodyA], _ground, contact.result, _restitution, _friction, _restitutionVelocityThreshold, _correctionBeta, _penetrationSlop, true);
                } else {
                    SolveContactAdvanced(_boxes[contact.bodyA], _boxes[contact.bodyB], contact.result, _restitution, _friction, _restitutionVelocityThreshold, _correctionBeta, _penetrationSlop, false);
                }
            }
        }
    }

    void CaseMultiBody::OnUpdate(float deltaTime) {
        if (deltaTime <= 0.f) {
            return;
        }

        float fixedDt = std::max(1e-5f, _fixedTimeStep);
        int maxSteps = std::max(1, _maxSubSteps);
        _timeAccumulator += std::min(deltaTime, 0.1f);

        int steps = 0;
        while (_timeAccumulator >= fixedDt && steps < maxSteps) {
            float linearDecay = std::max(0.0f, 1.0f - _linearDamping * fixedDt);
            float angularDecay = std::max(0.0f, 1.0f - _angularDamping * fixedDt);
            for (Box & box: _boxes) {
                box.velocity += Eigen::Vector3f(0.f, -_gravity, 0.f) * fixedDt;
                box.velocity *= linearDecay;
                box.angularvelocity *= angularDecay;
                box.center += box.velocity * fixedDt;

                Eigen::Quaternionf angularVelocityQ(0.f, box.angularvelocity.x(), box.angularvelocity.y(), box.angularvelocity.z());
                box.orientation.coeffs() += 0.5f * fixedDt * (angularVelocityQ * box.orientation).coeffs();
                box.orientation.normalize();
            }

            CollisionDetection();
            CollisionResponse();

            for (std::size_t i = 0; i < _boxes.size(); ++i) {
                bool nearlyStopped = _boxes[i].velocity.norm() < _sleepLinearThreshold && _boxes[i].angularvelocity.norm() < _sleepAngularThreshold;
                if (nearlyStopped) {
                    _sleepCounters[i] = std::min(_sleepCounters[i] + 1, _sleepFramesThreshold + 1);
                    if (_sleepCounters[i] > _sleepFramesThreshold) {
                        _boxes[i].velocity = Eigen::Vector3f::Zero();
                        _boxes[i].angularvelocity = Eigen::Vector3f::Zero();
                    }
                } else {
                    _sleepCounters[i] = 0;
                }
            }

            _timeAccumulator -= fixedDt;
            ++steps;
        }

        if (steps == maxSteps && _timeAccumulator > fixedDt) {
            _timeAccumulator = fixedDt;
        }
    }

} // namespace VCX::Labs::RigidBody