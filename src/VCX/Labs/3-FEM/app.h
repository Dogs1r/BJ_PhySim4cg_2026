#pragma once

#include <vector>

#include "Engine/app.h"
#include "Labs/3-FEM/CaseDeformable.h"
#include "Labs/Common/UI.h"

namespace VCX::Labs::FEM {
    class App : public Engine::IApp {
    private:
        Common::UI _ui;

        CaseDeformable   _casedeform;

        std::size_t _caseId = 0;

        std::vector<std::reference_wrapper<Common::ICase>> _cases = {_casedeform};

    public:
        App();

        void OnFrame() override;
    };
}
