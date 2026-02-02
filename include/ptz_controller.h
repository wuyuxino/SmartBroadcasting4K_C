#pragma once

class IPTZController {
public:
    virtual ~IPTZController() = default;
    // pan, tilt in degrees, zoom arbitrary scale
    virtual void sendPanTilt(double pan_deg, double tilt_deg, double zoom) = 0;
};

class PTZStub : public IPTZController {
public:
    void sendPanTilt(double pan_deg, double tilt_deg, double zoom) override;
};