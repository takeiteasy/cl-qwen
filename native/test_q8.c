#include "q8.h"
#include <math.h>
#include <stdio.h>
#include <string.h>

int main(void) {
    unsigned char blocks[4 * 3 * 34];
    float x[96], out[6];
    unsigned char activation[3 * 34];
    for (size_t i = 0; i < 96; ++i) x[i] = ((int)(i % 13) - 6) * 0.1f;
    for (size_t b = 0; b < 12; ++b) {
        blocks[b * 34] = 0;
        blocks[b * 34 + 1] = 0x38;
        for (size_t i = 0; i < 32; ++i) blocks[b * 34 + 2 + i] = (unsigned char)((int)((b * 11 + i) % 256) - 128);
    }
    for (size_t width = 32; width <= 96; width += 32) {
        out[0] = out[5] = 12345;
        if (cq_q8_pack(x, activation, width)) return 11;
        if (cq_q8_matvec(blocks, activation, out + 1, 4, width)) return 1;
        if (out[0] != 12345 || out[5] != 12345) return 2;
        for (size_t r = 0; r < 4; ++r) {
            double sum = 0, magnitude = 0;
            for (size_t i = 0; i < width; ++i) {
                size_t b = r * (width / 32) + i / 32;
                int byte = blocks[b * 34 + 2 + i % 32];
                int q = byte < 128 ? byte : byte - 256;
                size_t block = i / 32;
                unsigned h = activation[block * 34] | ((unsigned)activation[block * 34 + 1] << 8);
                double scale = ldexp(1.0 + (h & 1023) / 1024.0, (int)((h >> 10) & 31) - 15);
                int byte_x = activation[block * 34 + 2 + i % 32];
                int qx = byte_x < 128 ? byte_x : byte_x - 256;
                double value = q * 0.5 * qx * scale;
                sum += value; magnitude += fabs(value);
            }
            if (fabs(out[r + 1] - sum) > 1e-4 + 1e-4 * magnitude) return 3;
        }
    }
    if (!cq_q8_matvec(blocks, activation, out, 1, 31)) return 4;
    if (cq_q8_matvec(NULL, NULL, NULL, 0, 32)) return 5;
    if (cq_q8_matvec(NULL, NULL, out, 4, 0)) return 6;
    for (size_t i = 0; i < 4; ++i) if (out[i] != 0) return 7;
    if (!cq_q8_matvec(NULL, activation, out, 1, 32)) return 8;
    memset(blocks, 0, sizeof(blocks));
    blocks[0] = 1; blocks[2] = 127;
    for (size_t i = 0; i < 32; ++i) x[i] = 1;
    if (cq_q8_pack(x, activation, 32)) return 12;
    if (cq_q8_matvec(blocks, activation, out, 1, 32)) return 9;
    if (fabsf(out[0] - ldexpf(127.0f, -24)) > 1e-8f) return 10;
    x[0] = 127; x[1] = 0.5f; x[2] = -0.5f; x[3] = 1.5f; x[4] = -1.5f;
    if (cq_q8_pack(x, activation, 32)) return 13;
    if (activation[2] != 127 || activation[3] != 0 || activation[4] != 0 || activation[5] != 2 || activation[6] != 254) return 14;
    if (!cq_q8_pack(x, activation, 31)) return 15;
    x[0] = INFINITY;
    if (!cq_q8_pack(x, activation, 32)) return 16;
    puts("Q8_0 numerical and boundary checks pass");
    return 0;
}
