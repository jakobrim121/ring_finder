/* Transpose an 8 by 8 block using only bit shuffles, with no arithmetic.
 *
 * Integration uses rows of derivatives and columns of quadrature nodes;
 * math tables use rows of inputs and columns of coefficients. Both need the
 * same operation. Keeping it here avoids three copies of the instruction map.
 * Include only where Vec is an eight-double vector and AVX-512 is enabled.
 */
#ifndef WCTERING_VECTOR_TRANSPOSE_H
#define WCTERING_VECTOR_TRANSPOSE_H
static __attribute__((always_inline)) inline void transpose8(const Vec *row,
                                                             Vec *column) {
    Vec pairs[8], blocks[8];
    for (int pair = 0; pair < 4; pair++) {
        pairs[2 * pair] = _mm512_unpacklo_pd(row[2 * pair], row[2 * pair + 1]);
        pairs[2 * pair + 1] = _mm512_unpackhi_pd(row[2 * pair], row[2 * pair + 1]);
    }
    for (int block = 0; block < 2; block++) {
        int i = 4 * block;
        blocks[i] = _mm512_shuffle_f64x2(pairs[i], pairs[i + 2], 0x44);
        blocks[i + 1] = _mm512_shuffle_f64x2(pairs[i + 1], pairs[i + 3], 0x44);
        blocks[i + 2] = _mm512_shuffle_f64x2(pairs[i], pairs[i + 2], 0xee);
        blocks[i + 3] = _mm512_shuffle_f64x2(pairs[i + 1], pairs[i + 3], 0xee);
    }
    column[0] = _mm512_shuffle_f64x2(blocks[0], blocks[4], 0x88);
    column[1] = _mm512_shuffle_f64x2(blocks[1], blocks[5], 0x88);
    column[2] = _mm512_shuffle_f64x2(blocks[0], blocks[4], 0xdd);
    column[3] = _mm512_shuffle_f64x2(blocks[1], blocks[5], 0xdd);
    column[4] = _mm512_shuffle_f64x2(blocks[2], blocks[6], 0x88);
    column[5] = _mm512_shuffle_f64x2(blocks[3], blocks[7], 0x88);
    column[6] = _mm512_shuffle_f64x2(blocks[2], blocks[6], 0xdd);
    column[7] = _mm512_shuffle_f64x2(blocks[3], blocks[7], 0xdd);
}
#endif
