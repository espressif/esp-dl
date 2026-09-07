#pragma once

#include <condition_variable>
#include <mutex>
#include <vector>

#include "dl_fft.h"
#include "dl_rfft.h"

namespace dl {
class FFT {
private:
    // Private constructor for singleton
    FFT() = default;
    ~FFT() = default;

    // Delete copy constructor and assignment operator
    FFT(const FFT &) = delete;
    FFT &operator=(const FFT &) = delete;

    // Four handle vectors for different FFT types
    std::vector<dl_fft_f32_t *> fft_f32_handles;
    std::vector<dl_fft_s16_t *> fft_s16_handles;
    std::vector<dl_fft_f32_t *> rfft_f32_handles;
    std::vector<dl_fft_s16_t *> rfft_s16_handles;

    std::mutex mutex_;
    std::condition_variable cv_;
    int in_flight_ = 0;
    bool clearing_ = false;

    uint32_t m_caps = MALLOC_CAP_8BIT; // Default memory allocation capabilities

    // Called with mutex_ held.
    template <typename HandleType, typename InitFunc>
    HandleType *get_or_create_handle(int fft_length, std::vector<HandleType *> &handles, InitFunc init_func)
    {
        for (auto *handle : handles) {
            if (handle->fft_point == fft_length) {
                return handle;
            }
        }

        HandleType *new_handle = init_func(fft_length, m_caps);
        if (new_handle) {
            handles.push_back(new_handle);
        }
        return new_handle;
    }

    // Lookup/create under lock, then run without holding mutex so concurrent FFTs stay parallel.
    // clear() waits until in_flight_ drops to 0, so it cannot free a handle still in use.
    template <typename HandleType, typename InitFunc, typename RunFunc>
    esp_err_t run_with_handle(int fft_length, std::vector<HandleType *> &handles, InitFunc init_func, RunFunc run_func)
    {
        HandleType *handle = nullptr;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            cv_.wait(lock, [this] { return !clearing_; });
            handle = get_or_create_handle(fft_length, handles, init_func);
            if (!handle) {
                return ESP_FAIL;
            }
            in_flight_++;
        }

        esp_err_t result = run_func(handle);

        {
            std::lock_guard<std::mutex> lock(mutex_);
            in_flight_--;
            if (in_flight_ == 0) {
                cv_.notify_all();
            }
        }
        return result;
    }

public:
    // Get singleton instance
    static FFT *get_instance()
    {
        static FFT instance;
        return &instance;
    }

    uint32_t get_caps() { return m_caps; }

    void set_caps(uint32_t caps) { m_caps = caps; }

    // FFT for float32
    esp_err_t fft(float *data, int fft_length)
    {
        return run_with_handle(
            fft_length,
            fft_f32_handles,
            [](int len, uint32_t caps) { return dl_fft_f32_init(len, caps); },
            [data](dl_fft_f32_t *handle) { return dl_fft_f32_run(handle, data); });
    }

    // IFFT for float32
    esp_err_t ifft(float *data, int fft_length)
    {
        return run_with_handle(
            fft_length,
            fft_f32_handles,
            [](int len, uint32_t caps) { return dl_fft_f32_init(len, caps); },
            [data](dl_fft_f32_t *handle) { return dl_ifft_f32_run(handle, data); });
    }

    // RFFT for float32
    esp_err_t rfft(float *data, int fft_length)
    {
        return run_with_handle(
            fft_length,
            rfft_f32_handles,
            [](int len, uint32_t caps) { return dl_rfft_f32_init(len, caps); },
            [data](dl_fft_f32_t *handle) { return dl_rfft_f32_run(handle, data); });
    }

    // IRFFT for float32
    esp_err_t irfft(float *data, int fft_length)
    {
        return run_with_handle(
            fft_length,
            rfft_f32_handles,
            [](int len, uint32_t caps) { return dl_rfft_f32_init(len, caps); },
            [data](dl_fft_f32_t *handle) { return dl_irfft_f32_run(handle, data); });
    }

    // FFT for int16
    esp_err_t fft(int16_t *data, int fft_length, int in_exponent = 0, int *out_exponent = nullptr)
    {
        return run_with_handle(
            fft_length,
            fft_s16_handles,
            [](int len, uint32_t caps) { return dl_fft_s16_init(len, caps); },
            [data, in_exponent, out_exponent](dl_fft_s16_t *handle) {
                int temp_out_exp = 0;
                return dl_fft_s16_run(handle, data, in_exponent, out_exponent ? out_exponent : &temp_out_exp);
            });
    }

    // IFFT for int16
    esp_err_t ifft(int16_t *data, int fft_length, int in_exponent = 0, int *out_exponent = nullptr)
    {
        return run_with_handle(
            fft_length,
            fft_s16_handles,
            [](int len, uint32_t caps) { return dl_fft_s16_init(len, caps); },
            [data, in_exponent, out_exponent](dl_fft_s16_t *handle) {
                int temp_out_exp = 0;
                return dl_ifft_s16_run(handle, data, in_exponent, out_exponent ? out_exponent : &temp_out_exp);
            });
    }

    // RFFT for int16
    esp_err_t rfft(int16_t *data, int fft_length, int in_exponent = 0, int *out_exponent = nullptr)
    {
        return run_with_handle(
            fft_length,
            rfft_s16_handles,
            [](int len, uint32_t caps) { return dl_rfft_s16_init(len, caps); },
            [data, in_exponent, out_exponent](dl_fft_s16_t *handle) {
                int temp_out_exp = 0;
                return dl_rfft_s16_run(handle, data, in_exponent, out_exponent ? out_exponent : &temp_out_exp);
            });
    }

    // IRFFT for int16
    esp_err_t irfft(int16_t *data, int fft_length, int in_exponent = 0, int *out_exponent = nullptr)
    {
        return run_with_handle(
            fft_length,
            rfft_s16_handles,
            [](int len, uint32_t caps) { return dl_rfft_s16_init(len, caps); },
            [data, in_exponent, out_exponent](dl_fft_s16_t *handle) {
                int temp_out_exp = 0;
                return dl_irfft_s16_run(handle, data, in_exponent, out_exponent ? out_exponent : &temp_out_exp);
            });
    }

    // FFT with high precision for int16
    esp_err_t fft_hp(int16_t *data, int fft_length, int in_exponent = 0, int *out_exponent = nullptr)
    {
        return run_with_handle(
            fft_length,
            fft_s16_handles,
            [](int len, uint32_t caps) { return dl_fft_s16_init(len, caps); },
            [data, in_exponent, out_exponent](dl_fft_s16_t *handle) {
                int temp_out_exp = 0;
                return dl_fft_s16_hp_run(handle, data, in_exponent, out_exponent ? out_exponent : &temp_out_exp);
            });
    }

    // IFFT with high precision for int16
    esp_err_t ifft_hp(int16_t *data, int fft_length, int in_exponent = 0, int *out_exponent = nullptr)
    {
        return run_with_handle(
            fft_length,
            fft_s16_handles,
            [](int len, uint32_t caps) { return dl_fft_s16_init(len, caps); },
            [data, in_exponent, out_exponent](dl_fft_s16_t *handle) {
                int temp_out_exp = 0;
                return dl_ifft_s16_hp_run(handle, data, in_exponent, out_exponent ? out_exponent : &temp_out_exp);
            });
    }

    // RFFT with high precision for int16
    esp_err_t rfft_hp(int16_t *data, int fft_length, int in_exponent = 0, int *out_exponent = nullptr)
    {
        return run_with_handle(
            fft_length,
            rfft_s16_handles,
            [](int len, uint32_t caps) { return dl_rfft_s16_init(len, caps); },
            [data, in_exponent, out_exponent](dl_fft_s16_t *handle) {
                int temp_out_exp = 0;
                return dl_rfft_s16_hp_run(handle, data, in_exponent, out_exponent ? out_exponent : &temp_out_exp);
            });
    }

    // IRFFT with high precision for int16
    esp_err_t irfft_hp(int16_t *data, int fft_length, int in_exponent = 0, int *out_exponent = nullptr)
    {
        return run_with_handle(
            fft_length,
            rfft_s16_handles,
            [](int len, uint32_t caps) { return dl_rfft_s16_init(len, caps); },
            [data, in_exponent, out_exponent](dl_fft_s16_t *handle) {
                int temp_out_exp = 0;
                return dl_irfft_s16_hp_run(handle, data, in_exponent, out_exponent ? out_exponent : &temp_out_exp);
            });
    }

    // Waits for in-flight FFT/IFFT/RFFT calls, then frees cached handles.
    // Concurrent FFT calls block until clear() finishes, then allocate new handles.
    void clear()
    {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this] { return !clearing_; });
        clearing_ = true;
        cv_.wait(lock, [this] { return in_flight_ == 0; });

        // Clear FFT float32 handles
        for (auto *handle : fft_f32_handles) {
            dl_fft_f32_deinit(handle);
        }
        fft_f32_handles.clear();
        std::vector<dl_fft_f32_t *>().swap(fft_f32_handles);

        // Clear FFT int16 handles
        for (auto *handle : fft_s16_handles) {
            dl_fft_s16_deinit(handle);
        }
        fft_s16_handles.clear();
        std::vector<dl_fft_s16_t *>().swap(fft_s16_handles);

        // Clear RFFT float32 handles
        for (auto *handle : rfft_f32_handles) {
            dl_rfft_f32_deinit(handle);
        }
        rfft_f32_handles.clear();
        std::vector<dl_fft_f32_t *>().swap(rfft_f32_handles);

        // Clear RFFT int16 handles
        for (auto *handle : rfft_s16_handles) {
            dl_rfft_s16_deinit(handle);
        }
        rfft_s16_handles.clear();
        std::vector<dl_fft_s16_t *>().swap(rfft_s16_handles);

        clearing_ = false;
        cv_.notify_all();
    }

    // Get handle count for debugging
    size_t get_handle_count()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return fft_f32_handles.size() + fft_s16_handles.size() + rfft_f32_handles.size() + rfft_s16_handles.size();
    }
};

} // namespace dl
