#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace VCX::Labs::RigidBody{

    struct Box{
    public:

        float                               mass{1.f};
        Eigen::Vector3f                     dim{1.f,2.f,3.f};
        Eigen::Vector3f                     center{0.f,0.f,0.f};
        Eigen::Vector3f                     velocity{1.f,0.f,0.f};
        Eigen::Vector3f                     angularvelocity{0.f,1.f,0.f};
        Eigen::Quaternionf                  orientation{1.f,0.f,0.f,0.f};
        glm::vec3                           color{252.0f/255,157.0f/255,154.0f/255};//猛男粉！
        //glm::vec3 color{131.0f/255,175.0f/255,155.0f/255};//青绿灰
        //glm::vec3 color{175.0f/255,215.0f/255,237.0f/255};//灰蓝

        Box(Eigen::Vector3f dim={1.f,2.f,3.f},Eigen::Vector3f center={0.f,0.f,0.f},Eigen::Quaternionf orientation={1.f,0.f,0.f,0.f},float mass=1.f){
            this->dim=dim;
            this->center=center;
            this->orientation=orientation;
            this->mass=mass;
        }

        Eigen::Matrix3f GetInertiaMatrix(){
            Eigen::Matrix3f Iref;
            float                           m=mass;
            float                           w=dim.x();
            float                           h=dim.y();
            float                           d=dim.z();

            Iref<<1.f/12.f*m*(h*h+d*d),0,0,
                  0,1.f/12.f*m*(d*d+w*w),0,
                  0,0,1.f/12.f*m*(w*w+h*h);

            Eigen::Matrix3f R=orientation.toRotationMatrix();
            Eigen::Matrix3f inertiaMatrix=R*Iref*R.transpose();

            return inertiaMatrix;
        }
    };
}//namespace VCX::Labs::RigidBody