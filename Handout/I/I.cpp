#include <iostream>
#include <chrono>

#include<omp.h>
#include<thread>
#include<immintrin.h>


double it(double r, double x, int64_t itn) {
    for (int64_t i = 0; i < itn; i++) {
        x = r * x * (1.0 - x);
    }
    return x;
}

void itv(double r, double* x, int64_t n, int64_t itn) {
    for (int64_t i = 0; i < n; i++) {
        x[i] = it(r, x[i], itn);
    }
}

#define i64 int64_t
#define v512 __m512d

void process(double r,double* x, i64 n,i64 itn){
    v512 v_r=_mm512_set1_pd(r);
    v512 v_mid=_mm512_set1_pd(1.0);

#pragma omp parallel for
    for(i64 i=0;i<n;i+=16){
        v512 v_x=_mm512_load_pd(x+i);
        for(i64 j=0;j<itn;j++){
            v512 v_m=_mm512_mul_pd(v_r, v_x),v_s=_mm512_sub_pd(v_mid, v_x);
            v_x = _mm512_mul_pd(v_m,v_s);
        }
        _mm512_store_pd(x+i,v_x);
    }
}   


int main(){
    FILE* fi;
    fi = fopen("conf.data", "rb");

    int64_t itn;
    double r;
    int64_t n;
    double* x;

    fread(&itn, 1, 8, fi);
    fread(&r, 1, 8, fi);
    fread(&n, 1, 8, fi);
    x = (double*)_mm_malloc(n * sizeof(double),64);
    fread(x, 1, n * 8, fi);
    fclose(fi);


    auto t1 = std::chrono::steady_clock::now();

omp_set_num_threads(8);


    process(r,x,n,itn);



    auto t2 = std::chrono::steady_clock::now();
    int d1 = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
    printf("%d\n", d1);

    fi = fopen("out.data", "wb");
    fwrite(x, 1, n * 8, fi);
    fclose(fi);
    _mm_free(x);

    return 0;
}