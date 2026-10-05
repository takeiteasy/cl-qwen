#include "q8.h"
#include <math.h>
#include <stdint.h>
#include <string.h>
#if defined(__aarch64__) && !defined(CQ_SCALAR)
#include <arm_neon.h>
#endif

static float block_scale(const uint8_t *block) {
    uint16_t bits = (uint16_t)(block[0] | (uint16_t)block[1] << 8);
    unsigned exponent = (bits >> 10) & 31, fraction = bits & 1023;
    if (!exponent) return (bits & 32768 ? -1.0f : 1.0f) * ldexpf((float)fraction, -24);
    uint32_t wide = ((uint32_t)(bits & 32768) << 16) |
                    ((exponent == 31 ? 255u : exponent + 112u) << 23) | (fraction << 13);
    float scale;
    memcpy(&scale, &wide, sizeof(scale));
    return scale;
}

static uint16_t positive_half(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    int exponent = (int)((bits >> 23) & 255) - 112;
    uint32_t fraction = bits & 0x7fffff;
    if (exponent >= 31) return 0x7c00;
    if (exponent < -10) return 0;
    unsigned shift = exponent <= 0 ? (unsigned)(14 - exponent) : 13;
    if (exponent <= 0) fraction |= 0x800000;
    uint32_t result = (exponent <= 0 ? 0 : (uint32_t)exponent << 10) + (fraction >> shift);
    uint32_t remainder = fraction & ((1u << shift) - 1);
    uint32_t halfway = 1u << (shift - 1);
    if (remainder > halfway || (remainder == halfway && (result & 1))) ++result;
    return (uint16_t)result;
}

static int nearest_even(float value) {
    int result = (int)value;
    float remainder = value - result;
    if (remainder > 0.5f || (remainder == 0.5f && (result & 1))) ++result;
    if (remainder < -0.5f || (remainder == -0.5f && (result & 1))) --result;
    return result;
}

int cq_q8_pack(const float *x, uint8_t *blocks, size_t width) {
    if (width % 32 || (width && (!x || !blocks)) || width / 32 > SIZE_MAX / 34) return 1;
    for (size_t start = 0; start < width; start += 32) {
        float maximum = 0;
        for (size_t j = 0; j < 32; ++j) {
            if (!isfinite(x[start + j])) return 1;
            maximum = fmaxf(maximum, fabsf(x[start + j]));
        }
        float scale = maximum / 127.0f;
        float inverse = scale ? 1.0f / scale : 0;
        uint16_t half = positive_half(scale);
        if (half == 0x7c00) return 1;
        uint8_t *block = blocks + (start / 32) * 34;
        block[0] = (uint8_t)half; block[1] = (uint8_t)(half >> 8);
        for (size_t j = 0; j < 32; ++j) {
            float scaled = isfinite(inverse) ? x[start + j] * inverse : x[start + j] / scale;
            block[2 + j] = (uint8_t)nearest_even(scaled);
        }
    }
    return 0;
}

#if defined(__aarch64__) && !defined(CQ_SCALAR)
static int32x4_t dot16(int8x16_t a, int8x16_t b) {
#if defined(__ARM_FEATURE_DOTPROD)
    return vdotq_s32(vdupq_n_s32(0), a, b);
#else
    int16x8_t lo = vmull_s8(vget_low_s8(a), vget_low_s8(b));
    int16x8_t hi = vmull_s8(vget_high_s8(a), vget_high_s8(b));
    return vpaddq_s32(vpaddlq_s16(lo), vpaddlq_s16(hi));
#endif
}
#endif

int cq_q8_matvec(const uint8_t *weights, const uint8_t *activation, float *out,
                 size_t rows, size_t width) {
    if (width % 32 || (rows && !out) || (rows && width && (!weights || !activation))) return 1;
    if (width / 32 > SIZE_MAX / 34 ||
        (width && rows > SIZE_MAX / ((width / 32) * 34))) return 1;
    size_t count = width / 32;
    for (size_t row = 0; row < rows; ++row) {
        float sum = 0;
        for (size_t block_index = 0; block_index < count; ++block_index) {
            const uint8_t *w = weights + (row * count + block_index) * 34;
            const uint8_t *x = activation + block_index * 34;
            int dot = 0;
#if defined(__aarch64__) && !defined(CQ_SCALAR)
            int32x4_t lanes = vaddq_s32(dot16(vld1q_s8((const int8_t *)(w + 2)), vld1q_s8((const int8_t *)(x + 2))),
                                      dot16(vld1q_s8((const int8_t *)(w + 18)), vld1q_s8((const int8_t *)(x + 18))));
            dot = vaddvq_s32(lanes);
#else
            for (size_t i = 0; i < 32; ++i) {
                int wi = w[i + 2] < 128 ? w[i + 2] : (int)w[i + 2] - 256;
                int xi = x[i + 2] < 128 ? x[i + 2] : (int)x[i + 2] - 256;
                dot += wi * xi;
            }
#endif
            sum = fmaf((float)dot, block_scale(w) * block_scale(x), sum);
        }
        out[row] = sum;
    }
    return 0;
}
