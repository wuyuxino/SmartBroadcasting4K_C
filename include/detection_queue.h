// detection_queue.h
#pragma once

#include <mutex>
#include <condition_variable>
#include <deque>
#include <atomic>
#include "common.h"

class DetectionResultQueue {
private:
    std::deque<std::vector<DetectionBox>> queue_;
    size_t capacity_;
    
    mutable std::mutex mutex_;  // 添加 mutable
    std::condition_variable not_empty_;
    std::condition_variable not_full_;
    
public:
    explicit DetectionResultQueue(size_t capacity) : capacity_(capacity) {}
    
    // 推送检测结果
    bool push(std::vector<DetectionBox>&& result) {
        std::unique_lock<std::mutex> lock(mutex_);
        
        if (queue_.size() >= capacity_) {
            // 队列满，丢弃最旧的结果
            queue_.pop_front();
        }
        
        queue_.push_back(std::move(result));
        not_empty_.notify_one();
        return true;
    }
    
    // 弹出检测结果
    bool pop(std::vector<DetectionBox>& result) {
        std::unique_lock<std::mutex> lock(mutex_);
        
        if (queue_.empty()) {
            return false;
        }
        
        result = std::move(queue_.front());
        queue_.pop_front();
        not_full_.notify_one();
        return true;
    }
    
    // 获取最新结果（不移除）
    bool peek_latest(std::vector<DetectionBox>& result) {
        std::unique_lock<std::mutex> lock(mutex_);
        
        if (queue_.empty()) {
            return false;
        }
        
        result = queue_.back();
        return true;
    }
    
    // 清空队列
    void clear() {
        std::unique_lock<std::mutex> lock(mutex_);
        queue_.clear();
        not_full_.notify_all();
    }
    
    size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);  // 使用 lock_guard
        return queue_.size();
    }
    
    bool empty() const {
        std::lock_guard<std::mutex> lock(mutex_);  // 使用 lock_guard
        return queue_.empty();
    }
    
    bool full() const {
        std::lock_guard<std::mutex> lock(mutex_);  // 使用 lock_guard
        return queue_.size() >= capacity_;
    }
};