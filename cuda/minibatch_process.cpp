#include "minibatch_process.h"
#include "../bwa.h"
#include <locale.h>
#include "bwamem_GPU.cuh"
#include "batch_config.h"
#include "streams.cuh"
#include <future>
#include <sys/time.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/**
 * @brief convert current host addresses on a minibatch's transfer_data to their (future) addresses on GPU
 * assuming name, seq, comment, qual pointers on trasnfer_data still points to host memory
 *
 * @param seqs
 * @param n_seqs
 * @param transfer_data transfer_data_t object where these reads reside
 */
void convert2DevAddr(transfer_data_t *transfer_data)
{
	auto reads = transfer_data->h_seqs;
	auto n_reads = transfer_data->n_seqs;
	auto first_read = reads[0];
	for (int i = 0; i < n_reads; i++)
	{
		reads[i].name = reads[i].name - first_read.name + transfer_data->d_seq_name_ptr;
		reads[i].seq = reads[i].seq - first_read.seq + transfer_data->d_seq_seq_ptr;
		reads[i].comment = reads[i].comment - first_read.comment + transfer_data->d_seq_comment_ptr;
		reads[i].qual = reads[i].qual - first_read.qual + transfer_data->d_seq_qual_ptr;
	}
}

/**
 * @brief copy a minibatch of n_reads from superbatch to transfer_data minibatch's pinned memory, starting from firstReadId. 
 * Read info are contiguous, but name, comment, seq, qual are not
 * 
 * @param superbatch_data
 * @param transfer_data 
 * @param firstReadId 
 * @param n_reads 
 */
void copyReads2PinnedMem(superbatch_data_t *superbatch_data, transfer_data_t *transfer_data, int firstReadId, int n_reads){
	int lastReadId = firstReadId + n_reads - 1; 
	// copy name, comment, seq, qual one by one

	/*
	typedef struct {
	int l_seq, id;
	char *name, *comment, *seq, *qual, *sam;
	int8_t l_name, l_comment;
	int16_t l_qual;
} bseq1_t;


typedef struct {
    // reads on device
	int n_seqs;			// number of reads
	int64_t total_input;	// number of reads input prior to this batch
	int64_t total_output;	// number of reads output prior to this batch
	bseq1_t *d_seqs;		// reads
    char *d_seq_name_ptr, *d_seq_comment_ptr, *d_seq_seq_ptr, *d_seq_qual_ptr, *d_seq_sam_ptr;  // name, comment, seq, qual, sam output
	int *d_seq_sam_size;	// length of sam on device
    // pre-allocated reads on host
    bseq1_t *h_seqs;		// reads
    char *h_seq_name_ptr, *h_seq_comment_ptr, *h_seq_seq_ptr, *h_seq_qual_ptr, *h_seq_sam_ptr;  // name, comment, seq, qual, sam output
	int h_seq_name_size, h_seq_comment_size, h_seq_seq_size, h_seq_qual_size; // total char length of name, comment, seq, qual in a batch
    // pointers to CUDA stream, using generic pointers for compatibility with C
    void *CUDA_stream;   // transfer stream
} transfer_data_t;


typedef struct superbatch_data_t
{
	int n_reads;	   // number of reads
	bseq1_t *reads;	   // read info with pointers to the ones below
	char *name;		   // big chunk of all names
	char *comment;	   // big chunk of all comments
	char *seqs;		   // big chunk of all seqs
	char *qual;		   // big chunk of all qual
	long name_size;	   // total length of name strings
	long comment_size; // total length of comment strings
	long seqs_size;	   // total length of seq strings
	long qual_size;	   // total length of qual strings
} superbatch_data_t;

	*/
	for (int i = firstReadId; i <= lastReadId; i++){
		// h_seq_x_size keeps track of current size used in each buffer
		// and grow as we copy each read
		bseq1_t *read = &(superbatch_data->reads[i]);
		char *toAddr = transfer_data->h_seq_name_ptr + transfer_data->h_seq_name_size;
		memcpy(toAddr, read->name, read->l_name + 1); // size + 1 for null-terminating char
		read->name = toAddr;
		transfer_data->h_seq_name_size += read->l_name + 1;
		
		toAddr = transfer_data->h_seq_comment_ptr + transfer_data->h_seq_comment_size;
		memcpy(toAddr, read->comment, read->l_comment + 1); // size + 1 for null-terminating char
		read->comment = toAddr;
		transfer_data->h_seq_comment_size += read->l_comment + 1;

		toAddr = transfer_data->h_seq_seq_ptr + transfer_data->h_seq_seq_size;
		memcpy(toAddr, read->seq, read->l_seq + 1); // size + 1 for null-terminating char
		read->seq = toAddr;
		transfer_data->h_seq_seq_size += read->l_seq + 1;

		toAddr = transfer_data->h_seq_qual_ptr + transfer_data->h_seq_qual_size;
		memcpy(toAddr, read->qual, read->l_qual + 1); // size + 1 for null-terminating char
		read->qual = toAddr;
		transfer_data->h_seq_qual_size += read->l_qual + 1;
	}
	// copy read metadata array into pinned memory
	memcpy(transfer_data->h_seqs, &superbatch_data->reads[firstReadId], n_reads * sizeof(bseq1_t));
}

/**
 * @brief load a small batch from superbatch to transfer_data, up to MB_MAX_COUNT. 
 * Return number of reads loaded into transfer_data->n_seqs. return 0 if no read loaded
 * after loading, translate reads' addresses to GPU and transfer to GPU,
 * @param transfer_data
 * @param superbatch_data
 * @param n_loaded number of reads loaded from this superbatch before this minibatch
 * @return int number of reads loaded into transfer_data->n_seqs
 */
static void loadInputMiniBatch(transfer_data_t *transfer_data, superbatch_data_t *superbatch_data, int n_loaded)
{
	struct timespec timing_start, timing_stop; // variables for printing timings
	// number of reads to be loaded = min (remaining reads in superbatch, MB_MAX_COUNT)
	int n_reads_loading = superbatch_data->n_reads - n_loaded;
	if (n_reads_loading <= 0){
		transfer_data->n_seqs = 0;
		return;
	}
	if (n_reads_loading > MB_MAX_COUNT)
		n_reads_loading = MB_MAX_COUNT;

	// set all sizes to 0
	resetTransfer(transfer_data);

	transfer_data->n_seqs = n_reads_loading;
	// up to this point, 0 to n_loaded - 1 reads have been loaded in previous minibatches
	int first_readId = n_loaded;

	// copy data from superbatch to minibatch
	if (bwa_verbose >= 3)
		clock_gettime(CLOCK_MONOTONIC_RAW, &timing_start);
	copyReads2PinnedMem(superbatch_data, transfer_data, first_readId, n_reads_loading);

	// at this point, all pointers on transfer_data still point to name, seq, comment, qual addresses on superbatch_data
	if (bwa_verbose >= 3)
		clock_gettime(CLOCK_MONOTONIC_RAW, &timing_stop);
	if (bwa_verbose >= 3){
		fprintf(stderr, "[M::%-25s] ***loaded %'d reads from superbatch to minibatch\n", __func__, n_reads_loading);
		fprintf(stderr, "[M::%-25s] ***transfer from superbatch to minibatch took %lu ms\n", __func__, (timing_stop.tv_nsec - timing_start.tv_nsec) / 1000000);
	}

	if (bwa_verbose >= 3)
		clock_gettime(CLOCK_MONOTONIC_RAW, &timing_start);
	// translate reads' addresses to GPU addresses.
	// d_address = h_address - h_base_address + d_base_address for name, seq, comment, qual
	convert2DevAddr(transfer_data);
	// copy data to GPU
	CUDATransferSeqsIn(transfer_data);
	if (bwa_verbose >= 3)
		clock_gettime(CLOCK_MONOTONIC_RAW, &timing_stop);
	if (bwa_verbose >= 3)
		fprintf(stderr, "[M::%-25s] ***transfer to GPU took %lu ms\n", __func__, (timing_stop.tv_nsec - timing_start.tv_nsec) / 1000000);
	
}



/**
 * @brief output the previous batch of reads.
 * first transfer device's seqio to host's seq_io.
 * then write from host's seq_io to output
 * 
 * @param first_batch 
 * @param transfer_data 
 */
static void writeOutputMiniBatch(transfer_data_t *transfer_data)
{
	int n_seqs = transfer_data->n_seqs;
	if (n_seqs==0) return;
	struct timespec timing_start, timing_stop; // variables for printing timings
	// transfer from device's to host's
	if (bwa_verbose >= 3) clock_gettime(CLOCK_MONOTONIC_RAW, &timing_start);

	CUDATransferSamOut(transfer_data);
	if (bwa_verbose >= 3) clock_gettime(CLOCK_MONOTONIC_RAW, &timing_stop);
	if (bwa_verbose >= 3) fprintf(stderr, "[M::%-25s] ***transfer SAMs to host took %lu ms\n", __func__, (timing_stop.tv_nsec - timing_start.tv_nsec) / 1000000);

	// write from host's seq_io to output
	if (bwa_verbose >= 3) clock_gettime(CLOCK_MONOTONIC_RAW, &timing_start);
	bseq1_t *seqs = transfer_data->h_seqs;
	for (int i = 0; i < n_seqs; ++i)
		if (seqs[i].sam) err_fputs(seqs[i].sam, stdout);
	if (bwa_verbose >= 3) clock_gettime(CLOCK_MONOTONIC_RAW, &timing_stop);
	if (bwa_verbose >= 3) fprintf(stderr, "[M::%-25s] ***write SAMs to stdout took %lu ms\n", __func__, (timing_stop.tv_nsec - timing_start.tv_nsec) / 1000000);
	if (bwa_verbose >= 3)
		fprintf(stderr, "[M::%-25s] *** wrote output for sequences %'ld - %'ld\n", __func__, transfer_data->total_output, transfer_data->total_output+n_seqs);
	transfer_data->total_output += n_seqs;
	fprintf(stderr, "[M::%-25s] finised %'ld read\n", __func__, transfer_data->total_output);
}

/**
 * @brief process all reads in a superbatch by using mini batches
 * Async in a kernel:
 * - (1) if there is any processed seqs on GPU (reside in transfer_data), transfer them to host then to stdout
 * - (2) process reads stored on process_data
 * - (3) wait for (1) then load new minibatch to GPU
 * wait for (2) and (3)
 * swap process_data <-> transfer_data
 * swap corresponding 2 sets of pointers on device
 *
 * @param data all the reads in a superbatch
 */

void miniBatchMain(superbatch_data_t *superbatch_data, transfer_data_t *transfer_data, process_data_t *process_data)
{
	int n_loaded = 0; // number of reads we have loaded in this superbatch
	// process while we still have data in superbatch or in process minibatch
	while (n_loaded < superbatch_data->n_reads || process_data->n_seqs>0)
	{
		if (bwa_verbose >= 4)
			fprintf(stderr, "[M::%-25s] **** seqs to output=%'d, seqs to process=%'d \n", __func__, transfer_data->n_seqs, process_data->n_seqs);
		// async output previous batch
		auto outputAsync = std::async(std::launch::async, writeOutputMiniBatch, transfer_data); // does nothing if transfer_data->n_seqs == 0

		// async process current batch
		auto processAsync = std::async(std::launch::async, mem_align_GPU, process_data); // does nothing if process_data->n_seqs is 0

		// wait for output task to finish before doing next batch input
		outputAsync.wait();
		// async input next batch
		auto inputAsync = std::async(std::launch::async, loadInputMiniBatch, transfer_data, superbatch_data, n_loaded);
		
		inputAsync.wait();
		processAsync.wait();
		n_loaded += transfer_data->n_seqs;

		// process_data->n_seqs is now the number of seqs to output in next iteration
		// transfer_data->n_seqs is now the number of seqs to process in next interation
		// swap output of current processed batch to "transfer", and "transfer" to process for next loop
		swapData(process_data, transfer_data);
		// after this swap, process_data->n_seqs is the number of seqs to process in next iteration
		// 					transfer_data->n_seqs is the number of seqs to output in next iteration
		if (bwa_verbose >= 4)
			fprintf(stderr, "[M::%-25s] **** data swapped \n", __func__);

	}

	// output the final minibatch
	writeOutputMiniBatch(transfer_data);
	resetTransfer(transfer_data);
	return;
}