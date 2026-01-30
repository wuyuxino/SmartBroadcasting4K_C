#pragma once

#include <mutex>
#include <condition_variable>
#include <vector>
#include <atomic>
#include "common.h"

template<typename T>
class RingBuffer {
private:
    std::vector<T> buffer_;
    size_t capacity_;
    size_t head_ = 0;
    size_t tail_ = 0;
    std::atomic<size_t> count_ = 0;
    
    std::mutex mutex_;
    std::condition_variable not_empty_;
    std::condition_variable not_full_;
    
public:
    explicit RingBuffer(size_t capacity) 
        : capacity_(capacity), buffer_(capacity) {}
    
    // 非阻塞推送（覆盖最旧数据）
    bool push_nonblock(T&& item) {
        std::unique_lock<std::mutex> lock(mutex_);
        
        if (count_ < capacity_) {
            buffer_[tail_] = std::move(item);
            tail_ = (tail_ + 1) % capacity_;
            count_++;
            not_empty_.notify_one();
            return true;
        } else {
            // 缓冲区满，覆盖最旧数据
            buffer_[head_] = std::move(item);
            head_ = (head_ + 1) % capacity_;
            tail_ = head_;  // 尾指针跟随头指针
            not_empty_.notify_one();
            return true;
        }
    }
    
    // 阻塞推送
    bool push(T&& item) {
        std::unique_lock<std::mutex> lock(mutex_);
        not_full_.wait(lock, [this]() { return count_ < capacity_; });
        
        buffer_[tail_] = std::move(item);
        tail_ = (tail_ + 1) % capacity_;
        count_++;
        not_empty_.notify_one();
        return true;
    }
    
    // 非阻塞弹出
    bool pop_nonblock(T& item) {
        std::unique_lock<std::mutex> lock(mutex_);
        
        if (count_ == 0) {
            return false;
        }
        
        item = std::move(buffer_[head_]);
        head_ = (head_ + 1) % capacity_;
        count_--;
        not_full_.notify_one();
        return true;
    }
    
    // 阻塞弹出
    bool pop(T& item) {
        std::unique_lock<std::mutex> lock(mutex_);
        not_empty_.wait(lock, [this]() { return count_ > 0; });
        
        item = std::move(buffer_[head_]);
        head_ = (head_ + 1) % capacity_;
        count_--;
        not_full_.notify_one();
        return true;
    }
    
    // 获取最新数据（不移除）
    bool peek_latest(T& item) {
        std::unique_lock<std::mutex> lock(mutex_);
        
        if (count_ == 0) {
            return false;
        }
        
        size_t latest_index = (tail_ == 0) ? capacity_ - 1 : tail_ - 1;
        item = buffer_[latest_index];
        return true;
    }
    
    // 清空缓冲区
    void clear() {
        std::unique_lock<std::mutex> lock(mutex_);
        head_ = 0;
        tail_ = 0;
        count_ = 0;
        not_full_.notify_all();
    }
    
    size_t size() const { return count_; }
    size_t capacity() const { return capacity_; }
    bool empty() const { return count_ == 0; }
    bool full() const { return count_ == capacity_; }
};

using FrameRingBuffer = RingBuffer<FrameData>;