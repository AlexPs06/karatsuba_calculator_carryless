#include <iostream>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <immintrin.h>
using namespace std;


void print_m256i_hex(__m256i var) {
    alignas(32) uint8_t values[32];
    _mm256_store_si256((__m256i*)values, var);

    for (int i = 31; i >= 0; --i) {
        // std::cout << std::hex << static_cast<int>(values[i]) << " ";
        printf("%02x ", values[i]);

        // std::cout << static_cast<int>(values[i]) << " ";
    }
    std::cout << std::endl;
}
void print_m256i(__m256i var) {
    alignas(32) int32_t values[8];
    _mm256_store_si256((__m256i*)values, var);

    for (int i = 0; i < 8; ++i) {
        std::cout << std::hex << values[i] << " ";
    }
    std::cout << std::endl;
}


bool check_overflow_avx(__m256i a, __m256i b) {
    __m256i sum = _mm256_add_epi64(a, b);
    
    // Comparar elementos del resultado con los originales
    __m256i cmp1 = _mm256_cmpgt_epi64(a, sum);
    __m256i cmp2 = _mm256_cmpgt_epi64(b, sum);
    __m256i overflow_mask = _mm256_or_si256(cmp1, cmp2);

    // Verificar si se produce un desbordamiento
    return _mm256_testz_si256(overflow_mask, overflow_mask) != 1;
}
void karatsuba_multiply_avx_64(const uint32_t* num1, const uint32_t* num2, uint32_t* result) {

    __m256i uno = _mm256_set_epi32(0,0,0,0 ,0,1,0,0);

    // Dividir num1 y num2 en partes más pequeñas
    __m256i a = _mm256_set_epi32(0,0,0,0,0,0,0,num1[0]);
    __m256i b = _mm256_set_epi32(0,0,0,0,0,0,0,num1[1]);
    __m256i c = _mm256_set_epi32(0,0,0,0,0,0,0,num2[0]);
    __m256i d = _mm256_set_epi32(0,0,0,0,0,0,0,num2[1]);


    // Multiplicar las partes más pequeñas
    __m256i ac = _mm256_mul_epu32(a, c);
    __m256i bd = _mm256_mul_epu32(b, d);

    // Calcular (a + b) * (c + d)
    __m256i sum_ab = _mm256_add_epi64(a, b);
    __m256i sum_cd = _mm256_add_epi64(c, d);

    __m256i sum_abcd = _mm256_mul_epu32(sum_ab, sum_cd);

    // Restar las multiplicaciones intermedias
    __m256i adbc = _mm256_sub_epi64(sum_abcd, ac);
    adbc = _mm256_sub_epi64(adbc, bd);

    // Desplazar y combinar los resultados
    __m256i adbc_shifted = _mm256_slli_si256(adbc, 4);
    __m256i ac_shifted = _mm256_slli_si256(ac, 8);

     __m256i final_result;
    __m256i result_lo = _mm256_add_epi64(adbc_shifted, bd);
     bool carry = check_overflow_avx(adbc_shifted, bd);
    if(carry){
        final_result = _mm256_add_epi64(result_lo, ac_shifted);
        final_result = _mm256_add_epi64(final_result, uno);
    }else{
        final_result = _mm256_add_epi64(result_lo, ac_shifted);
    }

    _mm256_storeu_si256((__m256i*)result, final_result);
}





void karatsuba_multiply_avx(const uint32_t* num1, const uint32_t* num2, uint32_t* result) {


    uint32_t X1_t[2];
    uint32_t Y1_t[2];
    uint32_t X0_t[2];
    uint32_t Y0_t[2];

    uint32_t X1Y1_t[8] = {0,0,0,0,0,0,0,0};
    uint32_t X0Y0_t[8] = {0,0,0,0,0,0,0,0};

    X1_t[0]=num1[0];
    X1_t[1]=num1[1];

    Y1_t[0]=num2[0];
    Y1_t[1]=num2[1];

    X0_t[0]=num1[2];
    X0_t[1]=num1[3];

    Y0_t[0]=num2[2];
    Y0_t[1]=num2[3];

    

    karatsuba_multiply_avx_64(X1_t,Y1_t, X1Y1_t);
    karatsuba_multiply_avx_64(X0_t,Y0_t, X0Y0_t);

    __m256i X1Y1 =  _mm256_loadu_si256((__m256i*)X1Y1_t);
    __m256i X0Y0 =  _mm256_loadu_si256((__m256i*)X0Y0_t);

    // Dividir num1 y num2 en partes más pequeñas
    __m256i X1 =  _mm256_set_epi32(0,0,0,0,0,0,X1_t[0],X1_t[1]);
    __m256i Y1 =  _mm256_set_epi32(0,0,0,0,0,0,Y1_t[0],Y1_t[1]);
    __m256i X0 =  _mm256_set_epi32(0,0,0,0,0,0,X0_t[0],X0_t[1]);
    __m256i Y0 =  _mm256_set_epi32(0,0,0,0,0,0,Y0_t[0],Y0_t[1]);

    __m256i uno = _mm256_set_epi32(0,0,0,0 ,0,1,0,0);

    __m256i sum_X1X0 = _mm256_add_epi64(X1, X0);
    __m256i sum_Y1Y0 = _mm256_add_epi64(Y1, Y0);
    // 12 BB 62 2F 66 59 F9 8A

    uint32_t sum_X1X0_t[8];
    uint32_t sum_Y1Y0_t[8];
    uint32_t mul_X1X0_Y1Y0_t[8];

    uint32_t temp[8];

    _mm256_storeu_si256((__m256i*)&sum_X1X0_t, sum_X1X0);
    _mm256_storeu_si256((__m256i*)&sum_Y1Y0_t, sum_Y1Y0);

    temp[0]=sum_X1X0_t[1];
    temp[1]=sum_X1X0_t[0];

    sum_X1X0_t[0]=temp[0];
    sum_X1X0_t[1]=temp[1];

    temp[0]=sum_Y1Y0_t[1];
    temp[1]=sum_Y1Y0_t[0];

    sum_Y1Y0_t[0]=temp[0];
    sum_Y1Y0_t[1]=temp[1];


    karatsuba_multiply_avx_64(sum_X1X0_t,sum_Y1Y0_t, mul_X1X0_Y1Y0_t);

    __m256i mul_X1X0_Y1Y0 =  _mm256_loadu_si256((__m256i*)mul_X1X0_Y1Y0_t);
    mul_X1X0_Y1Y0 = _mm256_sub_epi64(mul_X1X0_Y1Y0,uno);


    mul_X1X0_Y1Y0 = _mm256_sub_epi64(mul_X1X0_Y1Y0, X1Y1);
    mul_X1X0_Y1Y0 = _mm256_sub_epi64(mul_X1X0_Y1Y0, X0Y0);

    // print_m256i_hex(mul_X1X0_Y1Y0);

    _mm256_storeu_si256((__m256i*)&temp, mul_X1X0_Y1Y0);
    __m256i mul_X1X0_Y1Y0_shifted = _mm256_set_epi32(0,0, temp[3], temp[2], temp[1], temp[0], 0,0);
    

    _mm256_storeu_si256((__m256i*)&temp, X1Y1);
    __m256i X1Y1_shifted = _mm256_set_epi32(temp[3], temp[2], temp[1], temp[0], 0,0,0,0);

    // print_m256i_hex(mul_X1X0_Y1Y0_shifted);

    __m256i final_result = _mm256_add_epi64(mul_X1X0_Y1Y0_shifted, X0Y0);
    final_result = _mm256_add_epi64(final_result, X1Y1_shifted);

    _mm256_storeu_si256((__m256i*)result, final_result);

}




int main() {
// https://www.rapidtables.com/convert/number/decimal-to-hex.html
    alignas(32) uint32_t num1[8] =  {0x95db117, 0xb32cfcc5, 0x95db117, 0xb32cfcc5, 0,0,0,0};
    alignas(32) uint32_t num2[8] =  {0x95db117, 0xb32cfcc5, 0x95db117, 0xb32cfcc5, 0,0,0,0};
    alignas(32) uint32_t result[8]= {0, 0, 0, 0, 0, 0, 0, 0};

    karatsuba_multiply_avx(num1,  num2,  result);
    // _mm256_store_si256((__m256i*)result, product);
    printf("Multiplicación con karatsuba: \n");
    
    printf("Numero 1: \n");
    __m256i product =  _mm256_loadu_si256((__m256i*)num1);
    print_m256i_hex(product);

    printf("Numero 2: \n");
    product =  _mm256_loadu_si256((__m256i*)num2);
    print_m256i_hex(product);
    
    
    printf("Resultado: \n");
    product =  _mm256_loadu_si256((__m256i*)result);
    print_m256i_hex(product);

    return 0;
}

