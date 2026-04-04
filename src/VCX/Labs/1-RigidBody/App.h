#pragma once

#include <vector>

#include "Engine/app.h"
#include "Labs/1-RigidBody/CaseSingleBody.h"
#include "Labs/1-RigidBody/CaseTwoBody.h"
#include "Labs/1-RigidBody/CaseMultiBody.h"
#include "Labs/Common/UI.h"

namespace VCX::Labs::RigidBody {
    class App : public Engine::IApp {
    private:
        Common::UI _ui;

        CaseSingleBody      _caseSingleBody;
        CaseTwoBody         _caseTwoBody;
        CaseMultiBody       _caseMultiBody;
        std::size_t _caseId = 0;

        std::vector<std::reference_wrapper<Common::ICase>> _cases = { _caseSingleBody, _caseTwoBody ,_caseMultiBody};

    public:
        App();

        void OnFrame() override;
    };
}
