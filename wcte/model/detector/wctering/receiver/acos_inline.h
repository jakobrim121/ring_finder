/* Exact glibc 2.37 acos coefficient/arithmetic adaptation.
 * Copyright (C) 2001-2023 Free Software Foundation, Inc.
 * IBM Accurate Mathematical Library, LGPL-2.1-or-later.
 * See COPYING.LIBM and LIBM_ORIGIN.txt. Enable only for the verified libm hash.
 */
#include "acos_coefficients.h"
/* Glibc's common acos table branches use one Horner recurrence with different
 * lengths. Zero padding preserves each original operation, including its FMA
 * contractions. Near zero and the singular endpoints keep scalar acos. */
typedef long long IndexVec __attribute__((vector_size(8 * LANES)));
static inline Vec acos_coefficient(IndexVec row, int column) {
#if LANES == 8
    /* Quadrature nodes often use the same table row. Broadcast that row's
     * coefficient instead of issuing eight identical gather loads. */
    if (_mm512_cmpeq_epi64_mask(row, _mm512_set1_epi64(row[0])) == 255)
        return fill(acos_table[row[0]][column].value);
#endif
    IndexVec index = row * 13;
    for (int lane = 0; lane < LANES; lane++)
        index[lane] += column;
#if LANES == 8
    return _mm512_i64gather_pd(index, acos_table, 8);
#else
    return _mm256_i64gather_pd((const double *)acos_table, index, 8);
#endif
}
#if LANES == 8
#include "vector_transpose.h"
#endif

/*
 * IBM Accurate Mathematical Library
 * Copyright (C) 2001-2023 Free Software Foundation, Inc.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation; either version 2.1 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, see <https://www.gnu.org/licenses/>.
 */

/****************************************************************/
/* TABLES FOR THE usqrt() FUNCTION                              */
/****************************************************************/

static const double acos_inroot[128] = {
    1.40872145012100, 1.39792649065766, 1.38737595123859, 1.37706074531819,
    1.36697225234682, 1.35710228748795, 1.34744307370643, 1.33798721601135,
    1.32872767765984, 1.31965775814772, 1.31077107283046, 1.30206153403386,
    1.29352333352711, 1.28515092624400, 1.27693901514820, 1.26888253714903,
    1.26097664998256, 1.25321671998073, 1.24559831065844, 1.23811717205462,
    1.23076923076923, 1.22355058064300, 1.21645747403153, 1.20948631362953,
    1.20263364480453, 1.19589614840310, 1.18927063399547, 1.18275403352732,
    1.17634339535009, 1.17003587860341, 1.16382874792529, 1.15771936846787,
    1.15170520119791, 1.14578379846309, 1.13995279980655, 1.13420992801334,
    1.12855298537376, 1.12297985014975, 1.11748847323133, 1.11207687497107,
    1.10674314218572, 1.10148542531442, 1.09630193572405, 1.09119094315276,
    1.08615077328341, 1.08117980543918, 1.07627647039410, 1.07143924829188,
    1.06666666666667, 1.06195729855996, 1.05730976072814, 1.05272271193563,
    1.04819485132867, 1.04372491688551, 1.03931168393861, 1.03495396376504,
    1.03065060224133, 1.02640047855933, 1.02220250399990, 1.01805562076124,
    1.01395880083916, 1.00991104495649, 1.00591138153909, 1.00195886573624,
    0.99611649018350, 0.98848330114434, 0.98102294317595, 0.97372899112030,
    0.96659534932828, 0.95961623024651, 0.95278613468066, 0.94609983358253,
    0.93955235122353, 0.93313894963169, 0.92685511418159, 0.92069654023750,
    0.91465912076005, 0.90873893479530, 0.90293223677296, 0.89723544654727,
    0.89164514012056, 0.88615804099474, 0.88077101210109, 0.87548104826333,
    0.87028526915267, 0.86518091269740, 0.86016532891275, 0.85523597411976,
    0.85039040552437, 0.84562627613070, 0.84094132996422, 0.83633339758291,
    0.83180039185606, 0.82734030399203, 0.82295119979782, 0.81863121615464,
    0.81437855769486, 0.81019149366693, 0.80606835497581, 0.80200753138734,
    0.79800746888611, 0.79406666717674, 0.79018367731967, 0.78635709949278,
    0.78258558087123, 0.77886781361798, 0.77520253297841, 0.77158851547266,
    0.76802457717971, 0.76450957210799, 0.76104239064719, 0.75762195809661,
    0.75424723326565, 0.75091720714229, 0.74763090162560, 0.74438736831878,
    0.74118568737933, 0.73802496642311, 0.73490433947940, 0.73182296599416,
    0.72878002987884, 0.72577473860242, 0.72280632232420, 0.71987403306536,
    0.71697714391715, 0.71411494828392, 0.71128675915902, 0.70849190843208};
/*
 * IBM Accurate Mathematical Library
 * Copyright (C) 2001-2023 Free Software Foundation, Inc.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation; either version 2.1 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, see <https://www.gnu.org/licenses/>.
 */

/****************************************************************/
/* TABLES FOR THE upow() FUNCTION                               */
/****************************************************************/

static const double acos_powtwo[] = {
    1.0,       2.0,       4.0,       8.0,        16.0,       32.0,       64.0,
    128.0,     256.0,     512.0,     1024.0,     2048.0,     4096.0,     8192.0,
    16384.0,   32768.0,   65536.0,   131072.0,   262144.0,   524288.0,   1048576.0,
    2097152.0, 4194304.0, 8388608.0, 16777216.0, 33554432.0, 67108864.0, 134217728.0};

static inline Vec ordered_acos(Vec x) {
    Vec a = maximum(x, -x);
    Vec ordinary = (Vec)((a >= fill(0.125)) & (a < fill(0.96875)));
    Vec small = (Vec)(a < fill(0.125));
    Vec result = fill(0x1.921fb54442d18p0);
    if (any_lane(ordinary)) {
        Vec safe = choose(ordinary, a, fill(0.25));
        IndexVec row;
        int largest_row = 0;
#if LANES == 8
        __m512i high = _mm512_srli_epi64(_mm512_castpd_si512(safe), 32);
        __m512i mantissa = _mm512_and_si512(high, _mm512_set1_epi64(0xfffff));
        __m512i low = _mm512_srli_epi64(mantissa, 15);
        __m512i middle =
            _mm512_add_epi64(_mm512_set1_epi64(32), _mm512_srli_epi64(mantissa, 14));
        __m512i upper = _mm512_add_epi64(
            _mm512_set1_epi64(96),
            _mm512_srli_epi64(_mm512_and_si512(high, _mm512_set1_epi64(0xfe000)), 13));
        row = _mm512_mask_blend_epi64(
            _mm512_cmplt_epi64_mask(high, _mm512_set1_epi64(0x3fe00000)), upper,
            middle);
        row = _mm512_mask_blend_epi64(
            _mm512_cmplt_epi64_mask(high, _mm512_set1_epi64(0x3fd00000)), row, low);
        largest_row = (int)_mm512_reduce_max_epi64(row);
#else
        for (int lane = 0; lane < LANES; lane++) {
            AtanCoefficient number = {.value = safe[lane]};
            uint32_t k = (uint32_t)(number.bits >> 32);
            if (k < 0x3fd00000)
                row[lane] = (k & 0x000fffff) >> 15;
            else if (k < 0x3fe00000)
                row[lane] = 32 + ((k & 0x000fffff) >> 14);
            else
                row[lane] = 96 + ((k & 0x000fe000) >> 13);
            if (row[lane] > largest_row)
                largest_row = row[lane];
        }
#endif
#if LANES == 8
        /* Load the same thirteen doubles of each row in two adjacent packets;
         * the final three lanes are masked. Transposition changes no bits. */
        Vec rows[8], coefficients[16];
        for (int i = 0; i < 8; i++)
            rows[i] = _mm512_loadu_pd(&acos_table[row[i]][0]);
        transpose8(rows, coefficients);
        for (int i = 0; i < 8; i++)
            rows[i] = _mm512_maskz_loadu_pd(0x1f, &acos_table[row[i]][8]);
        transpose8(rows, coefficients + 8);
#define ACOS_COEFF(column) coefficients[column]
#else
#define ACOS_COEFF(column) acos_coefficient(row, column)
#endif
        Vec reduced = a - ACOS_COEFF(0);
        /* Every lane keeps its own coefficients; absent leading terms are zero.
         * Omit only the padding above the largest degree needed by this batch. */
        int degree = largest_row < 96    ? 6
                     : largest_row < 160 ? 7
                     : largest_row < 204 ? 8
                     : largest_row < 212 ? 9
                                         : 10;
        Vec polynomial = ACOS_COEFF(degree);
        switch (degree) {
        case 10:
            polynomial = table_fma(reduced, polynomial, ACOS_COEFF(9));
        case 9:
            polynomial = table_fma(reduced, polynomial, ACOS_COEFF(8));
        case 8:
            polynomial = table_fma(reduced, polynomial, ACOS_COEFF(7));
        case 7:
            polynomial = table_fma(reduced, polynomial, ACOS_COEFF(6));
        }
        for (int column = 5; column >= 2; column--)
            polynomial = table_fma(reduced, polynomial, ACOS_COEFF(column));
        Vec tail = table_fma(reduced * reduced, polynomial, ACOS_COEFF(11));
        tail = table_fma(reduced, ACOS_COEFF(1), tail);
        Vec angle = ACOS_COEFF(12);
        Vec positive = (Vec)(x > fill(0.));
        Vec head = choose(positive, fill(0x1.921fb54442d18p0) - angle,
                          fill(0x1.921fb54442d18p0) + angle);
        Vec correction = choose(positive, fill(0x1.1a62633145c07p-54) - tail,
                                fill(0x1.1a62633145c07p-54) + tail);
        result = head + correction;
    }
    if (any_lane(small)) {
        /* glibc 2.37 e_asin.c's |x|<1/8 acos branch. Preserve both the
         * polynomial FMAs and final fused subtraction from the compensated sum. */
        Vec x2 = x * x;
        Vec polynomial = table_fma(fill(1.81382903404565056280372531963613E-02), x2,
                                   fill(2.23551211026525610742786300334557E-02));
        polynomial =
            table_fma(polynomial, x2, fill(3.03821268582119319911193410625235E-02));
        polynomial =
            table_fma(polynomial, x2, fill(4.46428561421059750978517350006940E-02));
        polynomial =
            table_fma(polynomial, x2, fill(7.50000000026122686814431784722623E-02));
        polynomial =
            table_fma(polynomial, x2, fill(1.66666666666664110590506577996662E-01));
        Vec cube = x2 * x;
        Vec head = fill(0x1.921fb54442d18p0) - x;
        Vec correction =
            ((fill(0x1.921fb54442d18p0) - head) - x) + fill(0x1.1a62633145c07p-54);
        correction = table_fma(-polynomial, cube, correction);
        Vec small_result = choose((Vec)(a < fill(0x1.8p-55)), fill(0x1.921fb54442d18p0),
                                  head + correction);
        result = choose(small, small_result, result);
    }
    Vec endpoint = (Vec)((a >= fill(0.96875)) & (a < fill(1.)));
    if (any_lane(endpoint)) {
        /* The endpoint reduction and every rounded/fused operation follow
         * the verified glibc 2.37 FMA implementation. Inactive lanes use a
         * harmless in-range argument before the two small table lookups. */
        Vec z = fill(.5) * (fill(1.) - choose(endpoint, a, fill(.98)));
        Vec estimate;
        for (int lane = 0; lane < LANES; lane++) {
            AtanCoefficient bits = {.value = z[lane]};
            estimate[lane] = acos_inroot[(bits.bits >> 46) & 127] *
                             acos_powtwo[511 - (bits.bits >> 53)];
        }
        Vec error = table_fma(-(estimate * estimate), z, fill(1.));
        Vec correction = table_fma(fill(3.12523626554518656309172508769531E-01), error,
                                   fill(3.75017500867345182581453026130850E-01));
        correction =
            table_fma(correction, error, fill(4.99999999495955425917856814202739E-01));
        correction =
            table_fma(correction, error, fill(9.99999999859990725855365213134618E-01));
        estimate = correction * estimate;
        Vec c = z * estimate;
        Vec factor = table_fma(-(fill(.5) * estimate), c, fill(1.5));
        Vec high = table_fma(fill(134217728.), c, c);
        Vec y = table_fma(fill(-134217728.), c, high);
        Vec remainder = table_fma(-y, y, z) / table_fma(factor, c, y);
        Vec polynomial = table_fma(fill(1.81382903404565056280372531963613E-02), z,
                                   fill(2.23551211026525610742786300334557E-02));
        polynomial =
            table_fma(polynomial, z, fill(3.03821268582119319911193410625235E-02));
        polynomial =
            table_fma(polynomial, z, fill(4.46428561421059750978517350006940E-02));
        polynomial =
            table_fma(polynomial, z, fill(7.50000000026122686814431784722623E-02));
        polynomial =
            table_fma(polynomial, z, fill(1.66666666666664110590506577996662E-01));
        Vec product = (polynomial * z) * (y + remainder);
        Vec positive = y + (remainder + product);
        Vec negative = (fill(0x1.921fb54442d18p0) - y) +
                       ((fill(0x1.1a62633145c07p-54) - remainder) - product);
        Vec half = choose((Vec)(x > fill(0.)), positive, negative);
        result = choose(endpoint, half + half, result);
    }
    int mask = any_lane(ordinary) | any_lane(small) | any_lane(endpoint);
    for (int lane = 0; lane < LANES; lane++)
        if (!(mask & (1 << lane)))
            result[lane] = acos(x[lane]);
    return result;
}

#undef ACOS_COEFF
