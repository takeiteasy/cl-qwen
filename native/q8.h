#ifndef CL_QWEN_Q8_H
#define CL_QWEN_Q8_H
#include <stddef.h>
#include <stdint.h>
int cq_q8_pack(const float *x, uint8_t *blocks, size_t width);
int cq_q8_matvec(const uint8_t *weights, const uint8_t *activation, float *out,
                 size_t rows, size_t width);
#endif
