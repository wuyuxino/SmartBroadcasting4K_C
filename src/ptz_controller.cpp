#include "ptz_controller.h"
#include <iostream>

void PTZStub::sendPanTilt(double pan_deg, double tilt_deg, double zoom) {
    std::cout << "[PTZ STUB] pan=" << pan_deg << " tilt=" << tilt_deg << " zoom=" << zoom << std::endl;
}