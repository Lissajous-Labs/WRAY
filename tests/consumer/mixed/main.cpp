#include <cstdint>
#include <cstdio>

extern "C" const char *abi_a();
extern "C" const char *abi_b();
extern "C" std::uint64_t hash_a();
extern "C" std::uint64_t hash_b();

int main()
{
    std::printf("%s %016llx %s %016llx\n", abi_a(), static_cast<unsigned long long>(hash_a()), abi_b(), static_cast<unsigned long long>(hash_b()));
    return 0;
}
