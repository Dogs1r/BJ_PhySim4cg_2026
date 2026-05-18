#pragma once

#include <algorithm>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <glm/glm.hpp>
#include <iostream>
#include <utility>
#include <vector>
#include <cmath>

namespace VCX::Labs::FEM{
    struct Simulator{
        std::vector<glm::vec3>          particlePos;
        std::vector<glm::vec3>          particleVel;
        std::vector<glm::ivec4>         tetrahedra;
        std::vector<glm::vec3>          particleForce;
        std::vector<glm::vec3>          particleVelRest;
        std::vector<glm::vec3>          particlePosRest;

        int                             wx;
        int                             wy;
        int                             wz;
        float                           delta;
        float                           poison=.3f;
        float                           young=1000.0f;
        float                           density=10.f;
        float                           friction=0.4f;
        float                           g=1.0f;   
        float                           particle_weight;



        inline float trace(glm::mat3 const & m) {
            return m[0][0] + m[1][1] + m[2][2];
        }

        inline int GetID(std::size_t i, std::size_t j, std::size_t k) {
            return static_cast<int>(i * (wy+1) *(wz+1) + j * (wz+1) + k);
        }

        inline glm::ivec3 GetCoord(int const id){
            int i = id / ((wy+1) * (wz+1));
            int j = (id % ((wy+1) * (wz+1))) / (wz+1);
            int k = id % (wz+1);
            return glm::ivec3(i, j, k);
        }

        inline bool is_fixed(int const id){
            glm::ivec3 coord = GetCoord(id);
            return coord.x == 0 ;
        }

        std::vector<glm::vec3> computeForceTet(const int tetId);

        void SimulateSubstep(float const dt);

        void Simulate(float const dt, int const numIters);

        void AddParticle(glm::vec3 const & pos, glm::vec3 const & vel);

        void AddTet(int const i0, int const i1, int const i2, int const i3);

        void setupScene(int  _wx, int _wy, int _wz,float _delta);

    };
}