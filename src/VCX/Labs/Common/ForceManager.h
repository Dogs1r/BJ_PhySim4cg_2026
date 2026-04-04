#pragma once

#include "Engine/Camera.hpp"
#include "Engine/math.hpp"
#include "Labs/Common/ImGuiHelper.h"

namespace VCX::Labs::Common {
    class ForceManager {
    public:
        ForceManager() = default;

        // Set to true to enable force application
        bool EnableForce = true;
        float ForceScale = 1.0f; // Adjust this to scale the force applied

        void ProcessInput(Engine::Camera& camera, ImVec2 const& mousePos);
        glm::vec3 getForce();
        std::pair<glm::vec3,glm::vec3> getForce(glm::vec3 cubeCenter);  // force with applied point, return force + point
        std::pair<glm::vec3,int> getForce(std::vector<glm::vec3> candidatePoints);

    private:
        glm::vec3 _forceDelta = glm::vec3(0.f);
        Engine::Camera _camera;
        glm::vec3 _rayDirection = glm::vec3(0.f);
    };
} // namespace VCX::Labs::Common
