#pragma once

#include "driftless/imu_derived.h"
#include "driftless/types.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <optional>

namespace driftless {

// Downsamples a raw ~200Hz FOG stream to the ~10Hz cadence the shared model
// was trained at. Emits the LATEST raw sample once at least target_period_s
// has elapsed since the last emission -- nearest-sample decimation, not
// block-averaging, so the model sees the same kind of instantaneous reading
// a native 10Hz device would have reported (averaging would low-pass motion
// content the model may rely on).
class Decimator {
public:
    explicit Decimator(double target_period_s = 0.1) : target_period_s_(target_period_s) {}

    // Feed a raw sample. Returns it if it should be emitted now, else
    // nullopt.
    std::optional<ImuSample> push(const ImuSample& raw) {
        const auto period_nanos = static_cast<std::int64_t>(target_period_s_ * 1e9);
        if (!last_emit_nanos_.has_value()) {
            last_emit_nanos_ = raw.timestamp_nanos;
            return raw;  // first sample seeds the cadence
        }
        if (raw.timestamp_nanos - *last_emit_nanos_ >= period_nanos) {
            last_emit_nanos_ = raw.timestamp_nanos;
            return raw;
        }
        return std::nullopt;
    }

private:
    double target_period_s_;
    std::optional<std::int64_t> last_emit_nanos_;
};

// Builds the exact [1,14,80] channels-first tensor the shared model expects
// (models/MODEL_CONTRACT.md) from a stream of DECIMATED (~10Hz) samples.
//
// Channel order matches FEATURE_CHANNELS = IMU_CHANNELS + DERIVED_CHANNELS
// in training/driftless_train/preprocess.py:
//   0 acc_x   1 acc_y   2 acc_z
//   3 gyro.x ("gyro_yaw" slot)  4 gyro.Z ("gyro_pitch" slot)  5 gyro.Y ("gyro_roll" slot)
//     ^ note 4 and 5: the slot names follow the phone dataset's raw column
//       order, which is NOT physical x/y/z. Physical z (vertical rate) belongs
//       in slot 4.
//   6 grav_x  7 grav_y  8 grav_z   (running gravity-lowpass estimate, not raw accel)
//   9 acc_norm  10 acc_vert  11 acc_horiz  12 gyro_vert  13 gyro_horiz
//
// The gyro-slot naming is inherited from the phone dataset's column names,
// not a physical claim -- see the caveat in imu_derived.h.
class ModelWindow {
public:
    static constexpr int kChannels = 14;
    static constexpr int kTimesteps = 80;
    using Tensor = std::array<float, kChannels * kTimesteps>;  // channels-first, flat

    // Feed one decimated (~10Hz) sample. Returns the full tensor once at
    // least kTimesteps samples have been seen, else nullopt. Internally
    // runs the gravity low-pass and derived-channel computation, so do not
    // call computeImuDerived() separately for the same stream.
    std::optional<Tensor> push(const ImuSample& sample) {
        double dt_s = 0.1;
        if (last_timestamp_nanos_.has_value()) {
            const double dt = static_cast<double>(sample.timestamp_nanos - *last_timestamp_nanos_) * 1e-9;
            if (dt > 0.0) dt_s = dt;
        }
        last_timestamp_nanos_ = sample.timestamp_nanos;

        const Vec3 g = gravity_.push(sample.accel, dt_s);
        const auto derived = computeImuDerived(sample.accel, sample.gyro, g);

        std::array<float, kChannels> row{};
        row[0] = static_cast<float>(sample.accel.x());
        row[1] = static_cast<float>(sample.accel.y());
        row[2] = static_cast<float>(sample.accel.z());
        // Physical x/y/z map to feature slots 3/5/4, NOT 3/4/5. The slot
        // names are inherited from the phone dataset's raw column order, in
        // which the column headed "gyro_pitch" (slot 4) is the vertical-axis
        // rate -- see training/driftless_train/augment.py, GYRO_XYZ_IDX ==
        // [3, 5, 4], and trap 1 in training/README.md. Pinned by
        // test_model_window_golden.
        row[3] = static_cast<float>(sample.gyro.x());
        row[5] = static_cast<float>(sample.gyro.y());
        row[4] = static_cast<float>(sample.gyro.z());
        row[6] = static_cast<float>(g.x());
        row[7] = static_cast<float>(g.y());
        row[8] = static_cast<float>(g.z());
        row[9] = static_cast<float>(derived.acc_norm);
        row[10] = static_cast<float>(derived.acc_vert);
        row[11] = static_cast<float>(derived.acc_horiz);
        row[12] = static_cast<float>(derived.gyro_vert);
        row[13] = static_cast<float>(derived.gyro_horiz);

        history_.push_back(row);
        if (history_.size() > static_cast<std::size_t>(kTimesteps)) history_.pop_front();
        if (history_.size() < static_cast<std::size_t>(kTimesteps)) return std::nullopt;

        Tensor tensor{};
        for (int t = 0; t < kTimesteps; ++t) {
            for (int c = 0; c < kChannels; ++c) {
                tensor[c * kTimesteps + t] = history_[static_cast<std::size_t>(t)][static_cast<std::size_t>(c)];
            }
        }
        return tensor;
    }

private:
    GravityLowpass gravity_{10.0};
    std::deque<std::array<float, kChannels>> history_;
    std::optional<std::int64_t> last_timestamp_nanos_;
};

}  // namespace driftless