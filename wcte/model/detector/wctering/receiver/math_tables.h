/* Coefficients and arithmetic adapted from glibc 2.37.
 * Copyright (C) 2001-2023 Free Software Foundation, Inc.
 * atan2: IBM Accurate Mathematical Library; exp: Arm Limited.
 * LGPL-2.1-or-later; see COPYING.LIBM and LIBM_ORIGIN.txt.
 */
#ifndef WCTERING_MATH_TABLES_H
#define WCTERING_MATH_TABLES_H
#include <stdint.h>
typedef union { uint64_t bits; double value; } AtanCoefficient;
extern const AtanCoefficient atan_table[241][7];
extern const uint64_t exp_table[256];
#endif
