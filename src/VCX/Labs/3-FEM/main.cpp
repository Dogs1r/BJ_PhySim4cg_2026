#include "Assets/bundled.h"
#include "Engine/app.h"
#include "Labs/3-FEM/App.h"

int main() {
    return VCX::Engine::RunApp<VCX::Labs::FEM::App>({
        .Title = "Deformable Object",
        .WindowSize = { 1280, 720 },
        .FontSize = 18,
        .IconFileNames = VCX::Assets::DefaultIcons,
        .FontFileNames = VCX::Assets::DefaultFonts,
    });
}
