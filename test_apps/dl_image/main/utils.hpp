#include <concepts>
#include <limits>
#include <random>
#include <type_traits>

template <typename T>
    requires std::is_integral_v<T>
T get_random_value(T range_min, T range_max)
{
    assert(range_min < range_max);
    static std::mt19937 rng(std::random_device{}());
    std::uniform_int_distribution<int> dist(range_min, range_max);
    return static_cast<T>(dist(rng));
}

template <typename T>
    requires std::is_integral_v<T>
void fill_random_value(T *buf, std::size_t n, T range_min, T range_max)
{
    assert(range_min < range_max);
    static std::mt19937 rng(std::random_device{}());
    std::uniform_int_distribution<int> dist(range_min, range_max);
    for (int i = 0; i < n; i++) {
        buf[i] = static_cast<T>(dist(rng));
    }
}

inline bool is_align(void *ptr, int n = 16)
{
    return !(reinterpret_cast<uintptr_t>(ptr) & (n - 1));
}
