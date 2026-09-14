#include "universe.h"

using namespace BNM;

extern "C" void V240RunPostV240FeasibilityProbe();

namespace {
void RunV240PostV240ProbeAfterLoad() {
    V240RunPostV240FeasibilityProbe();
}

struct V240PostV240ProbeRegistrar {
    V240PostV240ProbeRegistrar() {
        Loading::AddOnLoadedEvent(RunV240PostV240ProbeAfterLoad);
    }
};

V240PostV240ProbeRegistrar g_v240PostV240ProbeRegistrar;
} // namespace
