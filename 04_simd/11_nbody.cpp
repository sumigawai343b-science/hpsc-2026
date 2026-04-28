#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <immintrin.h>

int main() {
  const int N = 16;
  float x[N], y[N], m[N], fx[N], fy[N];
  for(int i=0; i<N; i++) {
    x[i] = drand48();
    y[i] = drand48();
    m[i] = drand48();
    fx[i] = fy[i] = 0;
  }
  for(int i=0; i<N; i++) {
    __m512 xi = _mm512_set1_ps(x[i]);
    __m512 yi = _mm512_set1_ps(y[i]);
    __m512 fxi = _mm512_setzero_ps();
    __m512 fyi = _mm512_setzero_ps();

    for(int j=0; j<N; j+=16) {
      __m512 xj = _mm512_loadu_ps(&x[j]);
      __m512 yj = _mm512_loadu_ps(&y[j]);
      __m512 mj = _mm512_loadu_ps(&m[j]);

      __m512 rx = _mm512_sub_ps(xi, xj);
      __m512 ry = _mm512_sub_ps(yi, yj);

      __mmask16 mask = _mm512_cmp_ps_mask(
        _mm512_set1_ps((float)i),
        _mm512_set_ps(
          (float)(j+15),(float)(j+14),(float)(j+13),(float)(j+12),
          (float)(j+11),(float)(j+10),(float)(j+9),(float)(j+8),
          (float)(j+7),(float)(j+6),(float)(j+5),(float)(j+4),
          (float)(j+3),(float)(j+2),(float)(j+1),(float)(j)
        ),
        _CMP_NEQ_OQ
      );

      __m512 r2 = _mm512_add_ps(_mm512_mul_ps(rx, rx),_mm512_mul_ps(ry, ry));

      __m512 inv_r = _mm512_rsqrt14_ps(r2);

      __m512 inv_r3 = _mm512_mul_ps(inv_r, _mm512_mul_ps(inv_r, inv_r));

      __m512 fx_part= _mm512_mul_ps(rx, _mm512_mul_ps(mj, inv_r3));
      __m512 fy_part = _mm512_mul_ps(ry, _mm512_mul_ps(mj, inv_r3));

      fxi = _mm512_mask_sub_ps(fxi, mask, fxi, fx_part);
      fyi = _mm512_mask_sub_ps(fyi, mask, fyi, fy_part);
    }
    fx[i] = _mm512_reduce_add_ps(fxi);
    fy[i] = _mm512_reduce_add_ps(fyi);

    printf("%d %g %g\n",i,fx[i],fy[i]);
  }
}
