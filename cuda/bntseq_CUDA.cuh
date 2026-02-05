#include <stdint.h>
#include "../bntseq.h"



__forceinline__ __device__ int bns_pos2rid_gpu(const bntseq_t *bns, int64_t pos_f);
__forceinline__ __device__ static int64_t  bns_intv2rid_gpu(const bntseq_t *bns, int64_t rb, int64_t re);
extern __device__ uint8_t *bns_fetch_seq_gpu(const bntseq_t *bns, const uint8_t *pac, int64_t *beg, int64_t mid, int64_t *end, int *rid, void* d_buffer_ptr);
extern __device__ uint8_t *bns_get_seq_gpu(int64_t l_pac, const uint8_t *pac, int64_t beg, int64_t end, int64_t *len, void* d_buffer_ptr);


__forceinline__ __device__ int bns_pos2rid_gpu(const bntseq_t *bns, int64_t pos_f)
{
	int left, mid, right;
	if (pos_f >= bns->l_pac) return -1;
	left = 0; mid = 0; right = bns->n_seqs;
	while (left < right) { // binary search
		mid = (left + right) >> 1;
		if (pos_f >= bns->anns[mid].offset) {
			if (mid == bns->n_seqs - 1) break;
			if (pos_f < bns->anns[mid+1].offset) break; // bracketed
			left = mid + 1;
		} else right = mid;
	}
	return mid;
}

__forceinline__ __device__ static  int64_t bns_depos_gpu(const bntseq_t *bns, int64_t pos, int *is_rev)
{
	return (*is_rev = (pos >= bns->l_pac))? (bns->l_pac<<1) - 1 - pos : pos;
}

