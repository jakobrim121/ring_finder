/* Call the running NumPy's own numeric loop, not a replacement approximation.
 * ABI: NumPy 1.26.4's documented numpy_1.24_ufunc_call_info capsule.
 * Python retains the capsule; all addresses arrive as runtime arguments.
 * The caller supplies aligned float64 buffers and the original byte strides.
 */
#include <stdint.h>

typedef int (*numpy_loop)(void *, char *const *, const intptr_t *, const intptr_t *,
                          void *);

int call_numpy_loop(const uintptr_t *info, uintptr_t *data, const intptr_t *dimensions,
                    const intptr_t *strides) {
    numpy_loop function = (numpy_loop)info[0];
    return function((void *)info[1], (char *const *)data, dimensions, strides,
                    (void *)info[2]);
}
