package com.driftless.fusion

/**
 * TFLite wrapper for the learned speed/vibration filter trained offline
 * on IO-VNBD (see training/). Takes a windowed IMU segment, returns a
 * denoised forward-velocity estimate — this is what stands in for the
 * OBD-II speed feed we don't have.
 */
class VelocityModel {
    // TODO: load assets/models/velocity_model.tflite, Interpreter.run(window) -> Float
}
