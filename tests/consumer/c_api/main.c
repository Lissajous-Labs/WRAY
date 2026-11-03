#include <wray/c/wray.h>

#include <stdio.h>
#include <string.h>

int main(void)
{
    const float in[5] = {0.0f, 0.5f, 1.0f, 2.0f, -3.0f};
    float s[5];
    float c[5];
    float e[5];
    double d[5] = {0.0, 1.0, 2.0, 3.0, 4.0};
    wray_sincos_f32(in, s, c, 5);
    wray_exp_f32(in, e, 5);
    wray_exp2_f64(d, d, 5);
    if (wray_c_abi_version() != WRAY_C_ABI_VERSION || strlen(wray_c_backend()) == 0)
        return 1;
    if (s[0] != 0.0f || c[0] != 1.0f || e[0] != 1.0f || d[0] != 1.0 || d[4] != 16.0)
        return 1;
    wray_balanced_tanh_f32(in, s, 5);
    wray_balanced_tanh_f64(d, d, 5);
    if (s[0] != 0.0f || s[1] <= 0.0f || s[1] >= 1.0f || s[4] >= 0.0f || s[4] < -1.0f ||
        d[0] <= 0.0 || d[0] >= 1.0 || d[4] <= 0.0 || d[4] > 1.0)
        return 1;
    printf("wray_c %s abi %d\n", wray_c_backend(), wray_c_abi_version());
    return 0;
}
