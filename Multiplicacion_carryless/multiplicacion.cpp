#include <iostream>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <immintrin.h>
#include <wmmintrin.h>
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

void print_m128i_hex(__m128i var) {
    alignas(16) uint8_t values[16];
    _mm_store_si128((__m128i*)values, var);

    for (int i = 15; i >= 0; --i) {
        // std::cout << std::hex << static_cast<int>(values[i]) << " ";
        printf("%02x ", values[i]);

        // std::cout << static_cast<int>(values[i]) << " ";
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




void multiply_128bits_avx(const uint32_t* num1, const uint32_t* num2, uint32_t* result) {

    // __m256i a1 =  _mm256_set_epi32(0, num1[0], 0, num1[1], 0,num1[2], 0,num1[3]);

    __m128i a0 = _mm_set_epi32(0,num1[0], 0,num1[1]);
    __m128i a1 = _mm_set_epi32(0,num1[2], 0,num1[3]);
    __m128i b3 = _mm_set_epi32(0,num2[3], 0,num2[3]);
    __m128i b2 = _mm_set_epi32(0,num2[2], 0,num2[2]);
    __m128i b1 = _mm_set_epi32(0,num2[1], 0,num2[1]);
    __m128i b0 = _mm_set_epi32(0,num2[0], 0,num2[0]);
    
    union { __m256i block256; __m128i block128; uint32_t u32[8];} temp;
    
    // printf("a: ");
    // print_m256i_hex(a);
    // printf("b: ");
    // print_m256i_hex(b3);

    __m128i mul_ab3_1 = _mm_clmulepi64_si128(a0, b3, 0);
    __m128i mul_ab3_2 = _mm_clmulepi64_si128(a1, b3, 0);

    __m128i mul_ab2_1 = _mm_clmulepi64_si128(a0, b2, 0);
    __m128i mul_ab2_2 = _mm_clmulepi64_si128(a1, b2, 0);

    __m128i mul_ab1_1 = _mm_clmulepi64_si128(a0, b1, 0);
    __m128i mul_ab1_2 = _mm_clmulepi64_si128(a1, b1, 0);
    
    __m128i mul_ab0_1 = _mm_clmulepi64_si128(a0, b0, 0);
    __m128i mul_ab0_2 = _mm_clmulepi64_si128(a1, b0, 0);
    __m256i final_result = _mm256_setzero_si256();
    __m128i zero = _mm_setzero_si128();
    
    printf("mul_ab0_1: ");
    print_m128i_hex(mul_ab0_1);
    printf("mul_ab0_2: ");
    print_m128i_hex(mul_ab0_2);
    // printf("mul_ab3_2: ");
    // print_m128i_hex(mul_ab3_2);
    // printf("mul_ab1_2: ");
    // print_m128i_hex(mul_ab1_2);
    // printf("mul_ab0_1: ");
    // print_m128i_hex(mul_ab0_1);

    temp.block256 = _mm256_set_m128i( mul_ab3_1,mul_ab3_2);

    printf("temp.block256: ");
    print_m256i_hex(temp.block256);
    

    __m256i temp_a = _mm256_set_epi32(0, 0, 0, 0, temp.u32[5],temp.u32[4], temp.u32[1],temp.u32[0]);
    __m256i temp_b = _mm256_set_epi32( 0, 0, 0, temp.u32[7],temp.u32[6], temp.u32[3],temp.u32[2], 0);

    // 00 00 00 00-00 00 00 00-45 05 04 50-55 50 50 11-00 00 00 00-00 00 00 00-45 05 04 50-55 50 50 11
    __m256i ab3 =  _mm256_xor_si256(temp_b,temp_a);

    temp.block256 =  _mm256_set_m128i(mul_ab2_1, mul_ab2_2);
    temp_a = _mm256_set_epi32( 0, 0, 0, temp.u32[5],temp.u32[4], temp.u32[1],temp.u32[0],0);
    temp_b = _mm256_set_epi32( 0, 0, temp.u32[7],temp.u32[6], temp.u32[3],temp.u32[2], 0,0);

    __m256i ab2 =  _mm256_xor_si256(temp_b,temp_a);

    temp.block256 =  _mm256_set_m128i(mul_ab1_1, mul_ab1_2);
    temp_a = _mm256_set_epi32( 0, 0, temp.u32[5],temp.u32[4], temp.u32[1],temp.u32[0],0,0);
    temp_b = _mm256_set_epi32( 0,  temp.u32[7],temp.u32[6], temp.u32[3],temp.u32[2], 0,0,0);

    __m256i ab1 =  _mm256_xor_si256(temp_b,temp_a);

    temp.block256 =  _mm256_set_m128i(mul_ab0_1, mul_ab0_2);
    temp_a = _mm256_set_epi32( 0, temp.u32[5],temp.u32[4], temp.u32[1],temp.u32[0],0,0,0);
    temp_b = _mm256_set_epi32( temp.u32[7],temp.u32[6], temp.u32[3],temp.u32[2], 0,0,0,0);

    __m256i ab0 =  _mm256_xor_si256(temp_b,temp_a);

    // final_result = _mm256_xor_si256(ab3, ab2);
    // final_result = _mm256_xor_si256(final_result, ab1);
    final_result = _mm256_xor_si256(final_result, ab0);


    _mm256_storeu_si256((__m256i*)result, final_result);


}



int main() {
    // alignas(32) uint32_t num1[8] =  {0x95db117, 0xb32cfcc5, x, x,   0, 0, 0, 0};
    // alignas(32) uint32_t num2[8] =  {        0,          2, x, x,   0, 0, 0, 0};
// https://www.rapidtables.com/convert/number/decimal-to-hex.html
    alignas(16) uint32_t num1[4] =  {0x95db117, 0xb32cfcc5, 0x95db117, 0xb32cfcc5};
    alignas(16) uint32_t num2[4] =  {0x95db117, 0xb32cfcc5, 0x95db117, 0xb32cfcc5};
    alignas(32) uint32_t result[8]= {0, 0, 0, 0, 0, 0, 0, 0};

    multiply_128bits_avx(num1, num2, result);
    __m256i product =  _mm256_loadu_si256((__m256i*)result);
    print_m256i_hex(product);
    return 0;
}

