#include "Assets/bundled.h"
#include "Engine/app.h"
#include "Labs/2-FluidSimulation/App.h"
// Your implementation of Fluid Simulation.

int main() {
    //namespace FluidSimulation = VCX::Labs::FluidSimulation;
    //namespace Engine           = VCX::Engine;
    //namespace Assets           = VCX::Assets;
    using namespace VCX;
    return Engine::RunApp<Labs::FluidSimulation::App>(Engine::AppContextOptions {
        .Title         = "VCX-sim Labs 2: Fluid Simulation",
        .WindowSize    = {1024, 768},
        .FontSize      = 16,
        .IconFileNames = Assets::DefaultIcons,
        .FontFileNames = Assets::DefaultFonts,
    });
}
