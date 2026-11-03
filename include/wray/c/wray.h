#ifndef WRAY_C_H
#define WRAY_C_H

#include <stddef.h>

#define WRAY_C_ABI_VERSION 1

#if defined(_WIN32)
#if defined(WRAY_C_BUILD)
#define WRAY_C_API __declspec(dllexport)
#elif defined(WRAY_C_SHARED)
#define WRAY_C_API __declspec(dllimport)
#else
#define WRAY_C_API
#endif
#elif defined(__GNUC__) || defined(__clang__)
#define WRAY_C_API __attribute__((visibility("default")))
#else
#define WRAY_C_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

WRAY_C_API int wray_c_abi_version(void);
WRAY_C_API const char *wray_c_backend(void);
WRAY_C_API int wray_c_has_fma(void);

WRAY_C_API void wray_exp_f32(const float *in, float *out, size_t n);
WRAY_C_API void wray_expm1_f32(const float *in, float *out, size_t n);
WRAY_C_API void wray_exp2_f32(const float *in, float *out, size_t n);
WRAY_C_API void wray_log_f32(const float *in, float *out, size_t n);
WRAY_C_API void wray_log1p_f32(const float *in, float *out, size_t n);
WRAY_C_API void wray_log10_f32(const float *in, float *out, size_t n);
WRAY_C_API void wray_log2_f32(const float *in, float *out, size_t n);
WRAY_C_API void wray_wrap_turns_f32(const float *in, float *out, size_t n);
WRAY_C_API void wray_sin_turns_f32(const float *in, float *out, size_t n);
WRAY_C_API void wray_sin_f32(const float *in, float *out, size_t n);
WRAY_C_API void wray_cos_turns_f32(const float *in, float *out, size_t n);
WRAY_C_API void wray_cos_f32(const float *in, float *out, size_t n);
WRAY_C_API void wray_tanpi_f32(const float *in, float *out, size_t n);
WRAY_C_API void wray_tan_turns_f32(const float *in, float *out, size_t n);
WRAY_C_API void wray_tan_f32(const float *in, float *out, size_t n);
WRAY_C_API void wray_atan_f32(const float *in, float *out, size_t n);
WRAY_C_API void wray_sinh_f32(const float *in, float *out, size_t n);
WRAY_C_API void wray_cosh_f32(const float *in, float *out, size_t n);
WRAY_C_API void wray_asinh_f32(const float *in, float *out, size_t n);
WRAY_C_API void wray_cbrt_f32(const float *in, float *out, size_t n);
WRAY_C_API void wray_erf_f32(const float *in, float *out, size_t n);
WRAY_C_API void wray_midi_to_hz_f32(const float *in, float *out, size_t n);
WRAY_C_API void wray_hz_to_midi_f32(const float *in, float *out, size_t n);
WRAY_C_API void wray_ratio_from_cents_f32(const float *in, float *out, size_t n);
WRAY_C_API void wray_omega_f32(const float *in, float *out, size_t n);
WRAY_C_API void wray_product_log_f32(const float *in, float *out, size_t n);
WRAY_C_API void wray_tanh_f32(const float *in, float *out, size_t n);
WRAY_C_API void wray_sigmoid_f32(const float *in, float *out, size_t n);
WRAY_C_API void wray_db_to_gain_f32(const float *in, float *out, size_t n);
WRAY_C_API void wray_gain_to_db_f32(const float *in, float *out, size_t n);
WRAY_C_API void wray_sincos_f32(const float *in, float *sin_out, float *cos_out, size_t n);
WRAY_C_API void wray_sincos_turns_f32(const float *in, float *sin_out, float *cos_out, size_t n);
WRAY_C_API void wray_smoothing_coefficient_f32(const float *seconds, const float *sample_rate, float *out, size_t n);
WRAY_C_API void wray_pole_from_cutoff_f32(const float *hz, const float *sample_rate, float *out, size_t n);
WRAY_C_API void wray_prewarp_f32(const float *hz, const float *sample_rate, float *out, size_t n);
WRAY_C_API void wray_atan2_f32(const float *y, const float *x, float *out, size_t n);
WRAY_C_API void wray_pow_f32(const float *x, const float *y, float *out, size_t n);

WRAY_C_API void wray_exp_f64(const double *in, double *out, size_t n);
WRAY_C_API void wray_expm1_f64(const double *in, double *out, size_t n);
WRAY_C_API void wray_exp2_f64(const double *in, double *out, size_t n);
WRAY_C_API void wray_log_f64(const double *in, double *out, size_t n);
WRAY_C_API void wray_log1p_f64(const double *in, double *out, size_t n);
WRAY_C_API void wray_log10_f64(const double *in, double *out, size_t n);
WRAY_C_API void wray_log2_f64(const double *in, double *out, size_t n);
WRAY_C_API void wray_wrap_turns_f64(const double *in, double *out, size_t n);
WRAY_C_API void wray_sin_turns_f64(const double *in, double *out, size_t n);
WRAY_C_API void wray_sin_f64(const double *in, double *out, size_t n);
WRAY_C_API void wray_cos_turns_f64(const double *in, double *out, size_t n);
WRAY_C_API void wray_cos_f64(const double *in, double *out, size_t n);
WRAY_C_API void wray_tanpi_f64(const double *in, double *out, size_t n);
WRAY_C_API void wray_tan_turns_f64(const double *in, double *out, size_t n);
WRAY_C_API void wray_tan_f64(const double *in, double *out, size_t n);
WRAY_C_API void wray_atan_f64(const double *in, double *out, size_t n);
WRAY_C_API void wray_sinh_f64(const double *in, double *out, size_t n);
WRAY_C_API void wray_cosh_f64(const double *in, double *out, size_t n);
WRAY_C_API void wray_asinh_f64(const double *in, double *out, size_t n);
WRAY_C_API void wray_cbrt_f64(const double *in, double *out, size_t n);
WRAY_C_API void wray_erf_f64(const double *in, double *out, size_t n);
WRAY_C_API void wray_midi_to_hz_f64(const double *in, double *out, size_t n);
WRAY_C_API void wray_hz_to_midi_f64(const double *in, double *out, size_t n);
WRAY_C_API void wray_ratio_from_cents_f64(const double *in, double *out, size_t n);
WRAY_C_API void wray_omega_f64(const double *in, double *out, size_t n);
WRAY_C_API void wray_product_log_f64(const double *in, double *out, size_t n);
WRAY_C_API void wray_tanh_f64(const double *in, double *out, size_t n);
WRAY_C_API void wray_sigmoid_f64(const double *in, double *out, size_t n);
WRAY_C_API void wray_db_to_gain_f64(const double *in, double *out, size_t n);
WRAY_C_API void wray_gain_to_db_f64(const double *in, double *out, size_t n);
WRAY_C_API void wray_sincos_f64(const double *in, double *sin_out, double *cos_out, size_t n);
WRAY_C_API void wray_sincos_turns_f64(const double *in, double *sin_out, double *cos_out, size_t n);
WRAY_C_API void wray_smoothing_coefficient_f64(const double *seconds, const double *sample_rate, double *out, size_t n);
WRAY_C_API void wray_pole_from_cutoff_f64(const double *hz, const double *sample_rate, double *out, size_t n);
WRAY_C_API void wray_prewarp_f64(const double *hz, const double *sample_rate, double *out, size_t n);
WRAY_C_API void wray_atan2_f64(const double *y, const double *x, double *out, size_t n);
WRAY_C_API void wray_pow_f64(const double *x, const double *y, double *out, size_t n);

WRAY_C_API void wray_fast_exp_f32(const float *in, float *out, size_t n);
WRAY_C_API void wray_fast_exp2_f32(const float *in, float *out, size_t n);
WRAY_C_API void wray_fast_log_f32(const float *in, float *out, size_t n);
WRAY_C_API void wray_fast_log2_f32(const float *in, float *out, size_t n);
WRAY_C_API void wray_fast_sin_turns_f32(const float *in, float *out, size_t n);
WRAY_C_API void wray_fast_sin_f32(const float *in, float *out, size_t n);
WRAY_C_API void wray_fast_cos_turns_f32(const float *in, float *out, size_t n);
WRAY_C_API void wray_fast_cos_f32(const float *in, float *out, size_t n);
WRAY_C_API void wray_fast_tanpi_f32(const float *in, float *out, size_t n);
WRAY_C_API void wray_fast_tan_turns_f32(const float *in, float *out, size_t n);
WRAY_C_API void wray_fast_tan_f32(const float *in, float *out, size_t n);
WRAY_C_API void wray_fast_atan_f32(const float *in, float *out, size_t n);
WRAY_C_API void wray_fast_omega_f32(const float *in, float *out, size_t n);
WRAY_C_API void wray_fast_product_log_f32(const float *in, float *out, size_t n);
WRAY_C_API void wray_fast_tanh_f32(const float *in, float *out, size_t n);
WRAY_C_API void wray_fast_sigmoid_f32(const float *in, float *out, size_t n);
WRAY_C_API void wray_fast_db_to_gain_f32(const float *in, float *out, size_t n);
WRAY_C_API void wray_fast_gain_to_db_f32(const float *in, float *out, size_t n);
WRAY_C_API void wray_fast_sincos_f32(const float *in, float *sin_out, float *cos_out, size_t n);
WRAY_C_API void wray_fast_sincos_turns_f32(const float *in, float *sin_out, float *cos_out, size_t n);
WRAY_C_API void wray_fast_atan2_f32(const float *y, const float *x, float *out, size_t n);
WRAY_C_API void wray_fast_pow_f32(const float *x, const float *y, float *out, size_t n);

WRAY_C_API void wray_balanced_sin_f32(const float *in, float *out, size_t n);
WRAY_C_API void wray_balanced_cos_f32(const float *in, float *out, size_t n);
WRAY_C_API void wray_balanced_sincos_f32(const float *in, float *sin_out, float *cos_out, size_t n);
WRAY_C_API void wray_balanced_sin_f64(const double *in, double *out, size_t n);
WRAY_C_API void wray_balanced_cos_f64(const double *in, double *out, size_t n);
WRAY_C_API void wray_balanced_sincos_f64(const double *in, double *sin_out, double *cos_out, size_t n);
WRAY_C_API void wray_balanced_tanh_f32(const float *in, float *out, size_t n);
WRAY_C_API void wray_balanced_tanh_f64(const double *in, double *out, size_t n);

#ifdef __cplusplus
}
#endif

#endif
