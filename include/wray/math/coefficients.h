#pragma once

#include <array>

namespace wray::math::coeff
{
    struct atan_fast
    {
        using value_type = float;
        static constexpr std::array<float, 7> p{ -0x1.55552cp-2f, 0x1.997942p-3f, -0x1.226a7ep-3f, 0x1.aa6e36p-4f, -0x1.1343c4p-4f, 0x1.ed8e72p-6f, -0x1.a59a0ep-8f };
        static constexpr std::array<float, 1> q{ 0x1p+0f };
    };

    struct atan_rest_f32
    {
        using value_type = float;
        static constexpr std::array<float, 9> p{ 0x1.999998p-3f, -0x1.24911ep-3f, 0x1.c6db9ap-4f, -0x1.7198a2p-4f, 0x1.2b6cdap-4f, -0x1.b96494p-5f, 0x1.00ba18p-5f, -0x1.8625e4p-7f, 0x1.1586c2p-9f };
        static constexpr std::array<float, 1> q{ 0x1p+0f };
    };

    struct atan_rest_f64
    {
        using value_type = double;
        static constexpr std::array<double, 6> p{ 0x1.9999999999999p-3, 0x1.e4db31f7fab7fp-2, 0x1.9a71ed1583d0ap-2, 0x1.2b6089d7be1adp-3, 0x1.635089b9b4aabp-6, 0x1.f1ea1e7045038p-11 };
        static constexpr std::array<double, 7> q{ 0x1p+0, 0x1.8a76b6166a678p+1, 0x1.d32d4476596aep+1, 0x1.0a49d3b9442e5p+1, 0x1.2a746f5ea88dbp-1, 0x1.2895b039e09cep-4, 0x1.768b68be7db37p-9 };
    };

    struct cos_f32
    {
        using value_type = float;
        static constexpr std::array<float, 3> p{ 0x1.555554p-5f, -0x1.6c12dp-10f, 0x1.9bd72cp-16f };
        static constexpr std::array<float, 1> q{ 0x1p+0f };
    };

    struct cos_f64
    {
        using value_type = double;
        static constexpr std::array<double, 6> p{ 0x1.5555555555555p-5, -0x1.6c16c16c16962p-10, 0x1.a01a019f4dca3p-16, -0x1.27e4fa16d5722p-22, 0x1.1eeb67f7fe4c8p-29, -0x1.907d070e3b1bfp-37 };
        static constexpr std::array<double, 1> q{ 0x1p+0 };
    };

    struct cos_fast
    {
        using value_type = float;
        static constexpr std::array<float, 2> p{ 0x1.5551f2p-5f, -0x1.676b56p-10f };
        static constexpr std::array<float, 1> q{ 0x1p+0f };
    };

    struct cos_turns_f32
    {
        using value_type = float;
        static constexpr std::array<float, 3> p{ 0x1.03c1fp+6f, -0x1.55d014p+6f, 0x1.dd058cp+5f };
        static constexpr std::array<float, 1> q{ 0x1p+0f };
    };

    struct cos_turns_f64
    {
        using value_type = double;
        static constexpr std::array<double, 6> p{ 0x1.03c1f081b5ac4p+6, -0x1.55d3c7e3cbd7p+6, 0x1.e1f506882f221p+5, -0x1.a6d1f0aa04a5ap+4, 0x1.f9cfbecdf0482p+2, -0x1.b38296efe1282p+0 };
        static constexpr std::array<double, 1> q{ 0x1p+0 };
    };

    struct cosh_f32
    {
        using value_type = float;
        static constexpr std::array<float, 3> p{ 0x1.55555ap-5f, 0x1.6c0c4cp-10f, 0x1.a714b6p-16f };
        static constexpr std::array<float, 1> q{ 0x1p+0f };
    };

    struct cosh_f64
    {
        using value_type = double;
        static constexpr std::array<double, 6> p{ 0x1.5555555555552p-5, 0x1.6c16c16c18a56p-10, 0x1.a01a019a92206p-16, 0x1.27e50160ce274p-22, 0x1.1ee7da5443778p-29, 0x1.98a916b471a96p-37 };
        static constexpr std::array<double, 1> q{ 0x1p+0 };
    };

    struct exp_f32
    {
        using value_type = float;
        static constexpr std::array<float, 5> p{ 0x1p-1f, 0x1.5554dep-3f, 0x1.5555bap-5f, 0x1.120b1cp-7f, 0x1.6c69f2p-10f };
        static constexpr std::array<float, 1> q{ 0x1p+0f };
    };

    struct exp_f64
    {
        using value_type = double;
        static constexpr std::array<double, 10> p{ 0x1.0000000000001p-1, 0x1.555555555554dp-3, 0x1.5555555553d82p-5, 0x1.1111111114483p-7, 0x1.6c16c1788176p-10, 0x1.a01a018c3310ep-13, 0x1.a019b930d47c7p-16, 0x1.71de5a3b96f46p-19, 0x1.28915f2743aa3p-22, 0x1.aeaaf677a4269p-26 };
        static constexpr std::array<double, 1> q{ 0x1p+0 };
    };

    struct exp_fast
    {
        using value_type = float;
        static constexpr std::array<float, 3> p{ 0x1.0007bap-1f, 0x1.57639ep-3f, 0x1.52ae5ep-5f };
        static constexpr std::array<float, 1> q{ 0x1p+0f };
    };

    struct log_f32
    {
        using value_type = float;
        static constexpr std::array<float, 3> p{ 0x1.55555cp-1f, 0x1.997c26p-2f, 0x1.2ee78ap-2f };
        static constexpr std::array<float, 1> q{ 0x1p+0f };
    };

    struct log_f64
    {
        using value_type = double;
        static constexpr std::array<double, 7> p{ 0x1.5555555555558p-1, 0x1.99999999952a7p-2, 0x1.2492492df708p-2, 0x1.c71c62defbc08p-3, 0x1.7462b65697063p-3, 0x1.39fe2df00adfap-3, 0x1.2b5a8673095a6p-3 };
        static constexpr std::array<double, 1> q{ 0x1p+0 };
    };

    struct log_fast
    {
        using value_type = float;
        static constexpr std::array<float, 2> p{ 0x1.55512cp-1f, 0x1.a26506p-2f };
        static constexpr std::array<float, 1> q{ 0x1p+0f };
    };

    struct sin_f32
    {
        using value_type = float;
        static constexpr std::array<float, 3> p{ -0x1.555552p-3f, 0x1.110c22p-7f, -0x1.9ac6d8p-13f };
        static constexpr std::array<float, 1> q{ 0x1p+0f };
    };

    struct sin_f64
    {
        using value_type = double;
        static constexpr std::array<double, 6> p{ -0x1.5555555555555p-3, 0x1.1111111110ba5p-7, -0x1.a01a019e80e58p-13, 0x1.71de3793660e8p-19, -0x1.ae60081aa0821p-26, 0x1.5e0a28e64bcc3p-33 };
        static constexpr std::array<double, 1> q{ 0x1p+0 };
    };

    struct sin_fast
    {
        using value_type = float;
        static constexpr std::array<float, 2> p{ -0x1.554e8cp-3f, 0x1.0c671ep-7f };
        static constexpr std::array<float, 1> q{ 0x1p+0f };
    };

    struct sin_turns_f32
    {
        using value_type = float;
        static constexpr std::array<float, 4> p{ -0x1.4abbcep+5f, 0x1.466bc2p+6f, -0x1.32ce1ap+6f, 0x1.4cb5c6p+5f };
        static constexpr std::array<float, 1> q{ 0x1p+0f };
    };

    struct sin_turns_f64
    {
        using value_type = double;
        static constexpr std::array<double, 7> p{ -0x1.4abbce625be53p+5, 0x1.466bc6775aae1p+6, -0x1.32d2cce62b868p+6, 0x1.50783486f970fp+5, -0x1.e3074d237aa4fp+3, 0x1.e8f0346d3ac48p+1, -0x1.6cc4b2c7ad5dp-1 };
        static constexpr std::array<double, 1> q{ 0x1p+0 };
    };

    struct sinh_f32
    {
        using value_type = float;
        static constexpr std::array<float, 4> p{ 0x1.555556p-3f, 0x1.11111ep-7f, 0x1.a0091cp-13f, 0x1.78a24cp-19f };
        static constexpr std::array<float, 1> q{ 0x1p+0f };
    };

    struct sinh_f64
    {
        using value_type = double;
        static constexpr std::array<double, 7> p{ 0x1.5555555555555p-3, 0x1.11111111110fep-7, 0x1.a01a01a01ed75p-13, 0x1.71de3a4e25c12p-19, 0x1.ae6460eaa9872p-26, 0x1.611cba693ad22p-33, 0x1.b40fca07893b3p-41 };
        static constexpr std::array<double, 1> q{ 0x1p+0 };
    };

    struct tan_f32
    {
        using value_type = float;
        static constexpr std::array<float, 7> p{ 0x1.555556p-2f, 0x1.111088p-3f, 0x1.ba52a8p-5f, 0x1.623ccep-6f, 0x1.467574p-7f, 0x1.368096p-10f, 0x1.f7c9ep-9f };
        static constexpr std::array<float, 1> q{ 0x1p+0f };
    };

    struct tan_f64
    {
        using value_type = double;
        static constexpr std::array<double, 4> p{ 0x1.5555555555555p-2, -0x1.5f4ac0a642804p-6, 0x1.00b406b5670fbp-12, -0x1.32e6f7b83fa8ep-26 };
        static constexpr std::array<double, 4> q{ 0x1p+0, -0x1.db779db8c6119p-2, 0x1.926030f8ce96cp-6, -0x1.0c6559eb2ca66p-12 };
    };

    struct tan_fast
    {
        using value_type = float;
        static constexpr std::array<float, 5> p{ 0x1.5555b6p-2f, 0x1.10d556p-3f, 0x1.c59822p-5f, 0x1.0692dcp-6f, 0x1.283c82p-6f };
        static constexpr std::array<float, 1> q{ 0x1p+0f };
    };

    struct tan_turns_f32
    {
        using value_type = float;
        static constexpr std::array<float, 7> p{ 0x1.4abbcep+3f, 0x1.466b22p+5f, 0x1.46288cp+7f, 0x1.42401ap+9f, 0x1.6e626cp+11f, 0x1.adea66p+11f, 0x1.ae468p+16f };
        static constexpr std::array<float, 1> q{ 0x1p+0f };
    };

    struct tan_turns_f64
    {
        using value_type = double;
        static constexpr std::array<double, 4> p{ 0x1.4abbce625be53p+3, -0x1.a3ee448b39e6bp+2, 0x1.7a92e09cf4531p-1, -0x1.17307d5e521bcp-11 };
        static constexpr std::array<double, 4> q{ 0x1p+0, -0x1.254ac7571b21cp+2, 0x1.32361c9105e3bp+1, -0x1.f7f875e4624fap-3 };
    };

    struct tanh_f32
    {
        using value_type = float;
        static constexpr std::array<float, 5> p{ -0x1.555552p-2f, 0x1.110e1ep-3f, -0x1.b925a2p-5f, 0x1.58054cp-6f, -0x1.8f91b4p-8f };
        static constexpr std::array<float, 1> q{ 0x1p+0f };
    };

    struct tanh_f64
    {
        using value_type = double;
        static constexpr std::array<double, 4> p{ -0x1.5555555555555p-2, -0x1.5ef506ed1a94ep-6, -0x1.ff53d96c31d54p-13, -0x1.2c1dff260e4fep-26 };
        static constexpr std::array<double, 4> q{ 0x1p+0, 0x1.db678ae60e958p-2, 0x1.91ecd8e2a71f9p-6, 0x1.0b450ff775464p-12 };
    };

    struct tanh_range_fast
    {
        using value_type = float;
        static constexpr std::array<float, 4> p{ 0x1.fffe9p-1f, 0x1.c0b9a8p-4f, 0x1.0b733ap-10f, -0x1.5d001ep-19f };
        static constexpr std::array<float, 3> q{ 0x1p+0f, 0x1.c57038p-2f, 0x1.f7e174p-7f };
    };
}

namespace wray::math
{
    template <class T>
    struct constants;

    template <>
    struct constants<float>
    {
        static constexpr float ln2 = 0x1.62e43p-1f;
        static constexpr float ln2_hi = 0x1.62e4p-1f;
        static constexpr float ln2_lo = 0x1.7f7d1cp-20f;
        static constexpr float inv_ln2 = 0x1.715476p+0f;
        static constexpr float inv_ln2_hi = 0x1.715476p+0f;
        static constexpr float inv_ln2_lo = 0x1.4ae0cp-26f;
        static constexpr float veltkamp = 0x1.001p+12f;
        static constexpr float sqrt2 = 0x1.6a09e6p+0f;
        static constexpr float sqrt_half = 0x1.6a09e6p-1f;
        static constexpr float pio2_1 = 0x1.922p+0f;
        static constexpr float pio2_2 = -0x1.2aep-18f;
        static constexpr float pio2_3 = -0x1.deap-31f;
        static constexpr float pio2_4 = 0x1.184698p-44f;
        static constexpr float two_over_pi = 0x1.45f306p-1f;
        static constexpr float pi_hi = 0x1.921fb6p+1f;
        static constexpr float pi_lo = -0x1.777a5cp-24f;
        static constexpr float pio2_hi = 0x1.921fb6p+0f;
        static constexpr float pio2_lo = -0x1.777a5cp-25f;
        static constexpr float third_hi = 0x1.555556p-2f;
        static constexpr float third_lo = -0x1.555556p-27f;
        static constexpr float max_arg = 0x1.62e43p+6f;
        static constexpr float min_arg = -0x1.9fe368p+6f;
        static constexpr float exp2_max = 0x1p+7f;
        static constexpr float exp2_min = -0x1.2cp+7f;
        static constexpr float trig_arg_max = 0x1.921fb6p+12f;
        static constexpr float tanh_clamp = 0x1.28p+3f;
        static constexpr float tanh_clamp_fast = 0x1.79999ap+2f;
        static constexpr float tanh_switch = 0x1.4p-1f;
        static constexpr float db_per_log2 = 0x1.815182p+2f;
        static constexpr float log2_per_db = 0x1.542a5ap-3f;
        static constexpr float log2_per_db_hi = 0x1.542a5ap-3f;
        static constexpr float log2_per_db_lo = 0x1.2e1c5ap-31f;
        static constexpr float db_per_log2_hi = 0x1.8152p+2f;
        static constexpr float db_per_log2_lo = -0x1.f6ce2ap-16f;
        static constexpr float db_per_ln_hi = 0x1.15f2cep+3f;
        static constexpr float db_per_ln_lo = 0x1.a709e6p-22f;
        static constexpr float log10_per_log2_hi = 0x1.3442p-2f;
        static constexpr float log10_per_log2_lo = -0x1.95ec1p-19f;
        static constexpr float log10_per_ln_hi = 0x1.bcb7b2p-2f;
        static constexpr float log10_per_ln_lo = -0x1.5b235ep-27f;
        static constexpr float two_pi_sq_hi = 0x1.3bd3ccp+4f;
        static constexpr float two_pi_sq_lo = 0x1.37c8bcp-21f;
    };

    template <>
    struct constants<double>
    {
        static constexpr double ln2 = 0x1.62e42fefa39efp-1;
        static constexpr double ln2_hi = 0x1.62e42fefa38p-1;
        static constexpr double ln2_lo = 0x1.ef35793c7673p-45;
        static constexpr double inv_ln2 = 0x1.71547652b82fep+0;
        static constexpr double inv_ln2_hi = 0x1.71547652b82fep+0;
        static constexpr double inv_ln2_lo = 0x1.777d0ffda0d24p-56;
        static constexpr double veltkamp = 0x1.0000002p+27;
        static constexpr double sqrt2 = 0x1.6a09e667f3bcdp+0;
        static constexpr double sqrt_half = 0x1.6a09e667f3bcdp-1;
        static constexpr double pio2_1 = 0x1.921fb544p+0;
        static constexpr double pio2_2 = 0x1.0b4611a8p-34;
        static constexpr double pio2_3 = -0x1.d9cceba3f91f2p-66;
        static constexpr double two_over_pi = 0x1.45f306dc9c883p-1;
        static constexpr double pi_hi = 0x1.921fb54442d18p+1;
        static constexpr double pi_lo = 0x1.1a62633145c07p-53;
        static constexpr double pio2_hi = 0x1.921fb54442d18p+0;
        static constexpr double pio2_lo = 0x1.1a62633145c07p-54;
        static constexpr double third_hi = 0x1.5555555555555p-2;
        static constexpr double third_lo = 0x1.5555555555555p-56;
        static constexpr double max_arg = 0x1.62e42fefa39efp+9;
        static constexpr double min_arg = -0x1.74910d52d3052p+9;
        static constexpr double exp2_max = 0x1p+10;
        static constexpr double exp2_min = -0x1.0ccp+10;
        static constexpr double trig_arg_max = 0x1.921fb54442d18p+22;
        static constexpr double tanh_clamp = 0x1.34p+4;
        static constexpr double tanh_clamp_fast = 0x1.34p+4;
        static constexpr double tanh_switch = 0x1.4p-1;
        static constexpr double db_per_log2 = 0x1.8151824c7587fp+2;
        static constexpr double log2_per_db = 0x1.542a5a12e1c5bp-3;
        static constexpr double log2_per_db_hi = 0x1.542a5a12e1c5bp-3;
        static constexpr double log2_per_db_lo = -0x1.33e2bb36cd142p-57;
        static constexpr double db_per_log2_hi = 0x1.8151824c758p+2;
        static constexpr double db_per_log2_lo = 0x1.fabf59b5d80b8p-44;
        static constexpr double db_per_ln_hi = 0x1.15f2ced384f29p+3;
        static constexpr double db_per_ln_lo = -0x1.02bea6b55233cp-52;
        static constexpr double log10_per_log2_hi = 0x1.34413509f78p-2;
        static constexpr double log10_per_log2_lo = 0x1.fef311f12b358p-46;
        static constexpr double log10_per_ln_hi = 0x1.bcb7b1526e50ep-2;
        static constexpr double log10_per_ln_lo = 0x1.95355baaafad3p-57;
        static constexpr double two_pi_sq_hi = 0x1.3bd3cc9be45dep+4;
        static constexpr double two_pi_sq_lo = 0x1.692b71366cc04p-50;
    };
}
