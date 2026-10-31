#pragma once

#include <cstdio>

namespace wray::test
{
    inline int failures = 0;
    inline int checks = 0;

    inline int finish(const char* name)
    {
        std::printf("%s: %d checks, %d failures\n", name, checks, failures);
        return failures == 0 ? 0 : 1;
    }
}

#define CHECK(cond)                                                                \
    do                                                                             \
    {                                                                              \
        ++wray::test::checks;                                                      \
        if (!(cond))                                                               \
        {                                                                          \
            ++wray::test::failures;                                                \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);            \
        }                                                                          \
    } while (0)
