#include "Femsys.h"

namespace VCX::Labs::FEM{
    std::vector<glm::vec3> Simulator::computeForceTet(const int tetId){
        glm::vec3 x0 = particlePos[tetrahedra[tetId][0]];
        glm::vec3 x1 = particlePos[tetrahedra[tetId][1]];
        glm::vec3 x2 = particlePos[tetrahedra[tetId][2]];
        glm::vec3 x3 = particlePos[tetrahedra[tetId][3]];

        glm::mat3 Ds = glm::mat3(x1 - x0, x2 - x0, x3 - x0);

        glm::mat3 Ds_rest = glm::mat3(particlePosRest[tetrahedra[tetId][1]] - particlePosRest[tetrahedra[tetId][0]],
                        particlePosRest[tetrahedra[tetId][2]] - particlePosRest[tetrahedra[tetId][0]],
                        particlePosRest[tetrahedra[tetId][3]] - particlePosRest[tetrahedra[tetId][0]]);
        glm::mat3 F = Ds * glm::inverse(Ds_rest);
        float lambda = young * poison / ((1 + poison) * (1 - 2 * poison));
        float mu = young / (2 * (1 + poison));
        glm::mat3 G = 0.5f*(glm::transpose(F) * F - glm::mat3(1.0f));   // Green-Lagrange Strain
        glm::mat3 S = 2 * mu * G + lambda * trace(G) * glm::mat3(1.0f); // Cauchy Stress

        float V_rest = abs(glm::determinant(Ds_rest)) / 6.0f;

        glm::mat3 force = -V_rest * F * S * glm::transpose(glm::inverse(Ds_rest)) ;
        glm::vec3 f1 = force[0];
        glm::vec3 f2 = force[1];
        glm::vec3 f3 = force[2];
        glm::vec3 f0 = -f1 - f2 - f3;

        return {f0, f1, f2, f3};
    }
    void Simulator::SimulateSubstep(float const dt){
        glm::vec3 gravity { 0, -g, 0 };

        for(int i=0; i<particlePos.size(); i++)
        {
            particleForce[i] += gravity * particle_weight;
            particleForce[i] -= friction * particleVel[i];
        }

        const float maxForcePerMass = 1e5f; // multiplier, tuned conservatively
        for(int i=0; i<particleForce.size(); i++){
            float maxF = maxForcePerMass * particle_weight;
            float mag = glm::length(particleForce[i]);
            if(mag > maxF && mag > 0.0f){
                particleForce[i] = particleForce[i] / mag * maxF;
            }
        }

        for(int i=0; i<tetrahedra.size(); i++)
        {
            std::vector<glm::vec3> force = computeForceTet(i);
            for(int j=0; j<4; j++)
            {
                particleForce[tetrahedra[i][j]] += force[j];
            }
        }


        for(int i=0; i<particleVel.size(); i++)
        {
            if(!is_fixed(i))
                particleVel[i] += particleForce[i] / particle_weight * dt;
        }

        for(int i=0; i<particlePos.size(); i++)
        {
            particlePos[i] += particleVel[i] * dt;
        }

        for(int i=0; i<particlePos.size(); i++){
            if(particlePos[i].x < 0.0f){
                if(!is_fixed(i)){
                    particlePos[i].x = 0.0f;
                    particleVel[i].x = 0.0f;
                    particleVel[i].y *= (1.0f - std::min(friction, 0.99f));
                    particleVel[i].z *= (1.0f - std::min(friction, 0.99f));
                } else {
                    particlePos[i].x = 0.0f;
                    particleVel[i] = glm::vec3(0.0f);
                }
            }
        }

        // reset particle force
        for(int i=0; i<particleForce.size(); i++)
        {
            particleForce[i] = {0, 0, 0};
        }


    }

    void Simulator::Simulate(float const dt, int const numIters){
        for(int i=0; i<numIters; i++)
        {
            SimulateSubstep(dt/numIters);
        }
    }

    void Simulator::AddParticle(glm::vec3 const & pos, glm::vec3 const & vel){
        particlePos.push_back(pos);
        particleVel.push_back(vel);
        particleForce.push_back({0, 0, 0});
    }

    void Simulator::AddTet(int const i0, int const i1, int const i2, int const i3){
        tetrahedra.push_back(glm::ivec4(i0, i1, i2, i3));
    }

    void Simulator::setupScene(int  _wx, int _wy, int _wz,float _delta){
        wx = _wx;
        wy = _wy;
        wz = _wz;
        delta = _delta;
        particle_weight = density * delta * delta * delta;

        for(std::size_t i=0; i<=wx; i++)
        {
            for(std::size_t j=0; j<=wy; j++)
            {
                for(std::size_t k=0; k<=wz; k++)
                {
                    glm::vec3 pos = glm::vec3(i*delta, j*delta, k*delta);
                    AddParticle(pos, {0, 0, 0});
                }
            }
        }

        for(std::size_t i=0;i<wx;i++){
            for(std::size_t j=0;j<wy;j++){
                for(std::size_t k=0;k<wz;k++){
                        AddTet(GetID(i, j, k), GetID(i, j, k + 1), GetID(i, j + 1, k + 1), GetID(i + 1, j + 1, k + 1));
                        AddTet(GetID(i, j, k), GetID(i, j + 1, k), GetID(i, j + 1, k + 1), GetID(i + 1, j + 1, k + 1));
                        AddTet(GetID(i, j, k), GetID(i, j, k + 1), GetID(i + 1, j, k + 1), GetID(i + 1, j + 1, k + 1));
                        AddTet(GetID(i, j, k), GetID(i + 1, j, k), GetID(i + 1, j, k + 1), GetID(i + 1, j + 1, k + 1));
                        AddTet(GetID(i, j, k), GetID(i, j + 1, k), GetID(i + 1, j + 1, k), GetID(i + 1, j + 1, k + 1));
                        AddTet(GetID(i, j, k), GetID(i + 1, j, k), GetID(i + 1, j + 1, k), GetID(i + 1, j + 1, k + 1));
                }
            }
        }
        particlePosRest = particlePos;
        particleVelRest = particleVel;
    }
}