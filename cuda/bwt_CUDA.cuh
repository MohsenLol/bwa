#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <stddef.h>
#include "../bwt.h"
#include "CUDAKernel_memmgnt.cuh"
#include "../kmers_index/hashKMerIndex.h"

typedef struct // same as bwtintv_t, but use 32-bit for query position (info)
{
    __int128_t x0:35, x1:35, x2:35, start:11, end:11;
    // bwtint_t x[3];
    // uint32_t info;
} bwtintv_lite_t;

// use kMer hash to determine the interval of length K starting from position i. Return true if success. Write interval to *interval
extern __device__ bool bwt_KMerHashInit(int qlen, const uint8_t *q, int i, kmers_bucket_t *d_kmersHashTab, bwtintv_lite_t *interval);

// extend 1 to the right, write result in-place. interval is input and output. Return true if successfully extended, false otherwise
extern __device__ bool bwt_extend_right1(const bwt_t *bwt, int qlen, const uint8_t *q, int min_intv, uint64_t max_intv, bwtintv_lite_t *interval);

extern __device__ void bwt_smem_left(const bwt_t *bwt, int len, const uint8_t *q, int x, int min_intv, uint64_t max_intv, int min_seed_len, bwtintv_v *mem);

extern __device__ void bwt_smem_right(const bwt_t* __restrict__ bwt, const int len, const uint8_t* __restrict__ q, const int x, int min_intv, uint64_t max_intv, int min_seed_len, bwtintv_t* __restrict__ mem_a, const kmers_bucket_t* __restrict__ d_kmersHashTab) ;

extern __device__ int bwt_smem1a_gpu(const bwt_t *bwt, int len, const uint8_t *q, int x, int min_intv, uint64_t max_intv, bwtintv_v *mem, bwtintv_v *tmpvec[2], void* d_buffer_ptr);

extern __device__ int bwt_seed_strategy1_gpu(const bwt_t *bwt, int len, const uint8_t *q, int x, int min_len, int max_intv, bwtintv_t *mem);


#define bwt_bwt(b, k) ((b)->bwt[((k)>>7<<4) + sizeof(bwtint_t) + (((k)&0x7f)>>4)])
#define bwt_B0(b, k) (bwt_bwt(b, k)>>((~(k)&0xf)<<1)&3)
#define bwt_occ_intv(b, k) ((b)->bwt + ((k)>>7<<4))
#define OCC_INTV_SHIFT 7
#define OCC_INTERVAL   (1LL<<OCC_INTV_SHIFT)
#define OCC_INTV_MASK  (OCC_INTERVAL - 1)

__device__ static inline int __occ_aux(uint64_t y, const int c)
{
	// reduce nucleotide counting to bits counting
	y = ((c&2)? y : ~y) >> 1 & ((c&1)? y : ~y) & 0x5555555555555555ull;
	// count the number of 1s in y
	y = (y & 0x3333333333333333ull) + (y >> 2 & 0x3333333333333333ull);
	return ((y + (y >> 4)) & 0xf0f0f0f0f0f0f0full) * 0x101010101010101ull >> 56;
}
__device__ __forceinline__ static bwtint_t bwt_occ_gpu(const bwt_t* __restrict__ bwt, bwtint_t k, ubyte_t c)
{
	bwtint_t n;
	uint32_t *p, *end;

	if (k == bwt->seq_len) return bwt->L2[c+1] - bwt->L2[c];
	if (k == (bwtint_t)(-1)) return 0;
	k -= (k >= bwt->primary); // because $ is not in bwt

	// retrieve Occ at k/OCC_INTERVAL
	n = ((bwtint_t*)(p = bwt_occ_intv(bwt, k)))[c];
	p += sizeof(bwtint_t); // jump to the start of the first BWT cell

	// calculate Occ up to the last k/32
	end = p + (((k>>5) - ((k&~OCC_INTV_MASK)>>5))<<1);
	for (; p < end; p += 2) n += __occ_aux((uint64_t)p[0]<<32 | p[1], c);

	// calculate Occ
	n += __occ_aux(((uint64_t)p[0]<<32 | p[1]) & ~((1ull<<((~k&31)<<1)) - 1), c);
	if (c == 0) n -= ~k&31; // corrected for the masked bits

	return n;
}

// what is the previous position in bwt for position k  Ψ⁻¹ 
__forceinline__ __device__ static bwtint_t bwt_invPsi(const bwt_t* __restrict__ bwt, bwtint_t k) // compute inverse CSA
{
	// bwt->primary : position of $ in bwt
	// convert to compacted bwt used in bwt_B0
	bwtint_t x = k - (k > bwt->primary);
	// 2-bit character code of x => x
	x = bwt_B0(bwt, x);
	// bwt->L2[x] : number of characters smaller than x in the original text
	// bwt_occ_gpu(bwt, k, x) : number of character x in bwt[0..k-1]
	x = bwt->L2[x] + bwt_occ_gpu(bwt, k, x);
	// => x : postion of the previous character in bwt
// return 0 if k is the primary index
	return k == bwt->primary? 0 : x;
}

__forceinline__ __device__ bwtint_t bwt_sa_gpu(const bwt_t* __restrict__ bwt, bwtint_t k)
{

	// sa how many steps we go backward
	// sa_int : sa saved interval for bwt
	bwtint_t sa = 0, mask = bwt->sa_intv - 1;
	while (k & mask) /*while mod(k, mask) != 0 (not saved sd)*/ { 
		++sa;
		k = bwt_invPsi(bwt, k);
	}
	/* without setting bwt->sa[0] = -1, the following line should be
	   changed to (sa + bwt->sa[k/bwt->sa_intv]) % (bwt->seq_len + 1) */
	return sa + bwt->sa[k/bwt->sa_intv];
}
