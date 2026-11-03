#pragma once

#include <cstddef>
#include <string_view>

namespace wray_bench
{
    template<class T>
    using function = void (*)(const T *, const T *, T *, T *, std::size_t);

    template<class T>
    function<T> libm(std::string_view name);

    extern template function<float> libm<float>(std::string_view);
    extern template function<double> libm<double>(std::string_view);

#if defined(WRAY_BENCH_HAS_SLEEF)
    enum class sleef_accuracy { u10, u35 };

    template<class T>
    function<T> sleef(std::string_view name, sleef_accuracy accuracy);

    extern template function<float> sleef<float>(std::string_view, sleef_accuracy);
    extern template function<double> sleef<double>(std::string_view, sleef_accuracy);
#endif

#if defined(WRAY_BENCH_HAS_LIBMVEC)
    template<class T>
    function<T> libmvec(std::string_view name);

    extern template function<float> libmvec<float>(std::string_view);
    extern template function<double> libmvec<double>(std::string_view);
#endif

#if defined(__APPLE__)
    template<class T>
    function<T> vforce(std::string_view name);

    extern template function<float> vforce<float>(std::string_view);
    extern template function<double> vforce<double>(std::string_view);
#endif
}
