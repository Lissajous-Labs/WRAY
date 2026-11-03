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

#if defined(__APPLE__)
    template<class T>
    function<T> vforce(std::string_view name);

    extern template function<float> vforce<float>(std::string_view);
    extern template function<double> vforce<double>(std::string_view);
#endif
}
