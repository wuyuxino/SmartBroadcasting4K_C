#include "prediction_manager.h"

#if defined(__has_include)
  #if __has_include("kalman_predictor.h")
    #include "kalman_predictor.h"
  #else
    // Stub KalmanPredictor when the real header / dependencies aren't available
    class KalmanPredictor {
    public:
        KalmanPredictor(const std::string&, const std::string&) {}
        bool isInitialized() const { return false; }
        std::vector<FrameCoord> predict(const std::vector<FrameCoord>&) { return {}; }
    };
  #endif
#else
  #include "kalman_predictor.h"
#endif

#include <algorithm>
#include <iostream>
#include <optional>

PredictionManager::PredictionManager(DetectionResultQueue& q,
                                   IPTZController* ptz,
                                   const std::string& kalman_json,
                                   const std::string& norm_json,
                                   double sum_move_thresh,
                                   int prediction_horizon)
    : queue_(q), ptz_(ptz), sum_move_thresh_(sum_move_thresh), prediction_horizon_(prediction_horizon) {
    // 延迟构造 KalmanPredictor（若依赖缺失则使用 stub）
    predictor_ = std::make_unique<KalmanPredictor>(kalman_json, norm_json);
}

PredictionManager::~PredictionManager() {
    stop();
}

void PredictionManager::start() {
    if (running_) return;
    running_ = true;
    last_fire_ = std::chrono::steady_clock::now() - std::chrono::milliseconds(COOLDOWN_MS);
    thread_ = std::thread(&PredictionManager::loop, this);
}

void PredictionManager::stop() {
    if (!running_) return;
    running_ = false;
    if (thread_.joinable()) thread_.join();
}

void PredictionManager::loop() {
    while (running_) {
        auto start = std::chrono::steady_clock::now();
        auto all = queue_.peek_all();

        // Debug: 打印队列大小
        // std::cout << "[PredictionManager] peek_all size=" << all.size() << std::endl;

        if (!all.empty()) {
            bool changed = hasSignificantChange(all);
            // std::cout << "[PredictionManager] hasSignificantChange=" << (changed ? "YES" : "NO") << std::endl;
            if (changed) {
                auto now = std::chrono::steady_clock::now();
                if (std::chrono::duration_cast<std::chrono::milliseconds>(now - last_fire_).count() >= COOLDOWN_MS) {
                    // build history of FrameCoord from last WINDOW frames
                    std::vector<FrameCoord> history;
                    int taken = std::min((int)all.size(), WINDOW);
                    // we assume each vector corresponds to a frame in chronological order left->right (peek_all returns queue copy order)
                    int start_idx = (int)all.size() - taken;
                    for (int i = start_idx; i < (int)all.size(); ++i) {
                        const auto &frame_boxes = all[i];
                        // pick highest-confidence "ball" box
                        float best_conf = 0.0f;
                        int best_frame_id = i; // fallback
                        DetectionBox best_box;
                        bool found = false;
                        for (const auto &b : frame_boxes) {
                            if (!isBall(b.class_name)) continue;
                            if (b.confidence > best_conf) {
                                best_conf = b.confidence;
                                best_box = b;
                                found = true;
                            }
                        }
                        if (found) {
                            auto c = centroid(best_box);
                            // We need frame_id; DetectionBox should now carry it
                            history.emplace_back((int)best_box.frame_id, c.x, c.y);
                        }
                    }

                    if (history.size() >= 2) {
                        // std::cout << "[PredictionManager] history size=" << history.size() << " -> calling predictor" << std::endl;

                        // Debug: print input history (frame_id, x, y)
                        // std::cout << "[PredictionManager] predictor input history: ";
                        // for (const auto &hc : history) {
                        //     std::cout << "[fid=" << hc.frame_id << ", x=" << hc.x1 << ", y=" << hc.y1 << "] ";
                        // }
                        // std::cout << std::endl;

                        try {
                            auto pred_start = std::chrono::steady_clock::now();
                            auto futures = predictor_->predict(history);
                            auto pred_end = std::chrono::steady_clock::now();
                            auto pred_us = std::chrono::duration_cast<std::chrono::microseconds>(pred_end - pred_start).count();
                            total_prediction_time_us_.fetch_add((uint64_t)pred_us);
                            prediction_count_.fetch_add(1);

                            // std::cout << "[PredictionManager] predict_time_us=" << pred_us << " (avg=" << get_avg_prediction_time_ms() << " ms)" << std::endl;

                            // Debug: print all predicted frames
                            if (!futures.empty()) {
                                // std::cout << "[PredictionManager] predictor returned " << futures.size() << " frames: ";
                                // for (const auto &f : futures) {
                                //     std::cout << "[fid=" << f.frame_id << ", x=" << f.x1 << ", y=" << f.y1 << "] ";
                                // }
                                // std::cout << std::endl;

                                // choose index based on prediction_horizon_, clamp to available frames
                                int idx = std::min((int)futures.size() - 1, std::max(0, prediction_horizon_));
                                const auto &sel = futures[idx];
                                double px = sel.x1;
                                double py = sel.y1;
                                // std::cout << "[PredictionManager] using prediction index="<<idx<<" (fid="<<sel.frame_id<<")" << std::endl;
                                // std::cout << "[PredictionManager] predicted px=" << px << " py=" << py << std::endl;

                                // Map pixel center -> absolute pan/tilt using trained linear fit (from your Python)
                                // Pan = 0.033440 * center_x - 19.79
                                // Tilt = -0.005492 * center_y - 8.35
                                double center_x = px;
                                double center_y = py;
                                double pan = 0.033440 * center_x - 19.79;  // 坐标系转换
                                double tilt = -0.005492 * center_y - 8.35; // 坐标系转换
                                double zoom = 1.0;
                                // std::cout << "[PredictionManager] mapping pixel->Pan/Tilt: Pan="<<pan<<" Tilt="<<tilt<<"\n";
                                ptz_->sendPanTilt(pan, tilt, zoom);
                                last_fire_ = std::chrono::steady_clock::now();
                            } else {
                                std::cout << "[PredictionManager] predictor returned empty" << std::endl;
                            }
                        } catch (const std::exception &e) {
                            std::cerr << "Prediction error: " << e.what() << std::endl;
                        }
                    } else {
                        std::cout << "[PredictionManager] not enough history for predictor" << std::endl;
                    }
                } else {
                    std::cout << "[PredictionManager] in cooldown" << std::endl;
                }
            }
        }

        auto elapsed = std::chrono::steady_clock::now() - start;
        auto sleep_ms = std::chrono::milliseconds(CHECK_MS) - std::chrono::duration_cast<std::chrono::milliseconds>(elapsed);
        if (sleep_ms.count() > 0) std::this_thread::sleep_for(sleep_ms);
    }
}

bool PredictionManager::hasSignificantChange(const std::vector<std::vector<DetectionBox>>& frames) {
    // Check last two frames existence of ball and centroid move > threshold
    int n = (int)frames.size();
    if (n < 2) return false;
    int i = n - 1; // last
    int j = n - 2; // prev

    auto pick = [](const std::vector<DetectionBox>& boxes)->std::optional<cv::Point2f> {
        float best_conf = 0.0f;
        std::optional<cv::Point2f> out;
        for (const auto &b : boxes) {
            std::string name = b.class_name;
            if (!PredictionManager::isBall(name)) continue;
            if (b.confidence > best_conf) {
                best_conf = b.confidence;
                out = PredictionManager::centroid(b);
            }
        }
        return out;
    };

    auto p1 = pick(frames[i]);
    auto p0 = pick(frames[j]);
    if (!p1 && !p0) {
        std::cout << "[PredictionManager] no ball in last two frames" << std::endl;
        return false;
    }
    if (p1 && !p0) {
        std::cout << "[PredictionManager] ball appeared" << std::endl;
        return true; // appear
    }
    if (!p1 && p0) {
        std::cout << "[PredictionManager] ball disappeared" << std::endl;
        return true; // disappear
    }

    // both exist -> 累积位移判定（使用最后 WINDOW 帧）
    std::vector<cv::Point2f> pts;
    for (int k = std::max(0, n - WINDOW); k < n; ++k) {
        auto pick_k = [&]()->std::optional<cv::Point2f> {
            float best_conf = 0.0f;
            std::optional<cv::Point2f> out;
            for (const auto &b : frames[k]) {
                if (!PredictionManager::isBall(b.class_name)) continue;
                if (b.confidence > best_conf) {
                    best_conf = b.confidence;
                    out = PredictionManager::centroid(b);
                }
            }
            return out;
        };
        auto pk = pick_k();
        if (pk) pts.push_back(*pk);
    }

    if (pts.size() < 2) {
        std::cout << "[PredictionManager] not enough ball points for cumulative check" << std::endl;
        return false;
    }

    double sum_dist = 0.0;
    for (size_t t = 1; t < pts.size(); ++t) {
        double dx = pts[t].x - pts[t-1].x;
        double dy = pts[t].y - pts[t-1].y;
        sum_dist += std::sqrt(dx*dx + dy*dy);
    }
    // std::cout << "[PredictionManager] cumulative dist=" << sum_dist << " (sum_thresh=" << sum_move_thresh_ << ")" << std::endl;
    return sum_dist >= sum_move_thresh_;
}

cv::Point2f PredictionManager::centroid(const DetectionBox& b) {
    return cv::Point2f((b.x1 + b.x2) * 0.5f, (b.y1 + b.y2) * 0.5f);
}

bool PredictionManager::isBall(const std::string& s) {
    std::string low = s;
    std::transform(low.begin(), low.end(), low.begin(), [](unsigned char c){ return std::tolower(c); });
    bool ret = (low.find("ball") != std::string::npos) || (low.find("football") != std::string::npos);
    // Debug: 打印 class_name 判断
    // std::cout << "[PredictionManager] isBall("<<s<<") -> " << (ret?"YES":"NO") << std::endl;
    return ret;
}

// 返回平均预测时间（ms）
double PredictionManager::get_avg_prediction_time_ms() const {
    int count = prediction_count_.load();
    if (count == 0) return 0.0;
    // total stored in microseconds -> convert to milliseconds
    double avg_us = (double)total_prediction_time_us_.load() / (double)count;
    return avg_us / 1000.0;
}