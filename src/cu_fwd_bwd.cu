/*
 *  File: cu_fwd_bwd.cu
 *
 *  forward-backward algorithm
 *
 *  Modified from fwd_bwd.c
 *
 */

#include <stdio.h>
#include <time.h>
#include <math.h>
extern "C" {
  #include "nrutil.h"
  #include "hmm.h"
}
#include "logmath.h"
#include <omp.h>

__device__ double d_logadd(const double p, const double q) {
  return (p > q) ? p + log1p(exp(q - p)) : q + log1p(exp(p - q));
}

__global__ void cuda_dev_Forward_P(double * pi, int N, double * alpha,
                                   size_t alpha_row_size, int * peakPos,
                                   double * log_A_matrix,
                                   size_t log_A_matrix_row_size,
                                   double * emission_matrix,
                                   size_t emission_matrix_row_size,
                                   int TFlist_length, int * TFlist, int TF) {
  int k = blockIdx.x;
  int stateIdx = threadIdx.x;
  int peakStart = peakPos[k] - 1;
  int peakEnd = peakPos[k + 1] - 2;
  double sum;
  double log_a;
  int pwm_last;

  bool motif_start = false;
  for (int n = 0; n < TFlist_length; n++) {
    if (stateIdx == TFlist[n]) {
      motif_start = true;
        break;
    }
  }

  /* 1. Initialization */
  if (pi[stateIdx] == 0.0) {
    alpha[stateIdx * alpha_row_size + peakStart] = -INFINITY;
  } else {
    alpha[stateIdx * alpha_row_size + peakStart] = log(pi[stateIdx]) +
      emission_matrix[stateIdx * emission_matrix_row_size + peakPos[k] - 1];
  }

  __syncthreads();

  /* 2. Induction */
  for (int t = peakStart; t < peakEnd; t++) {
    if ((stateIdx > 0) && (stateIdx <= TF)) {
      alpha[stateIdx * alpha_row_size + t + 1] =
        alpha[(stateIdx - 1) * alpha_row_size + t] +
        log_A_matrix[(stateIdx - 1) * log_A_matrix_row_size + stateIdx] +
        emission_matrix[stateIdx * emission_matrix_row_size + t + 1];
    }
    if (motif_start) {
      sum = -INFINITY;
      for (int i = TF + 1; i < N; i++) {
        log_a = log_A_matrix[i * log_A_matrix_row_size + stateIdx];
        if (alpha[i * alpha_row_size + t] != -INFINITY && log_a != -INFINITY) {
          if (sum != -INFINITY) {
            sum = d_logadd(sum, alpha[i * alpha_row_size + t] + log_a);
          } else {
            sum = alpha[i * alpha_row_size + t] + log_a;
          }
        }
      }
      alpha[stateIdx * alpha_row_size + t + 1] = sum +
        emission_matrix[stateIdx * emission_matrix_row_size + t + 1];
    }
    if ((stateIdx > TF) && (stateIdx < N)) {
      sum = -INFINITY;
      for (int n = 1; n < TFlist_length; n++) {
        pwm_last = TFlist[n] - 1;
        log_a = log_A_matrix[pwm_last * log_A_matrix_row_size + stateIdx];
        if (alpha[pwm_last * alpha_row_size + t] != -INFINITY
            && log_a != -INFINITY) {
          if (sum != -INFINITY) {
            sum = d_logadd(sum, alpha[pwm_last * alpha_row_size + t] + log_a);
          } else {
            sum = alpha[pwm_last * alpha_row_size + t] + log_a;
          }
        }
      }
      for (int i = 0; i < N; i++) {
        log_a = log_A_matrix[i * log_A_matrix_row_size + stateIdx];
        if (alpha[i * alpha_row_size + t] != -INFINITY && log_a != -INFINITY) {
          if (sum != -INFINITY) {
            sum = d_logadd(sum, alpha[i * alpha_row_size + t] + log_a);
          } else {
            sum = alpha[i * alpha_row_size + t] + log_a;
          }
        }
      }
      alpha[stateIdx * alpha_row_size + t + 1] = sum +
        emission_matrix[stateIdx * emission_matrix_row_size + t + 1];
    }

    __syncthreads();
  }
}

extern "C"
void cuda_host_Forward_P(HMM * phmm, int T, double ** alpha, double * pprob,
                         int P, int * peakPos, gsl_matrix * emission_matrix) {
  int * TFlist, TF;

  TFlist = ivector(phmm->M * (phmm->inactive + 1));
  TF = 0;
  for (int j = 0; j < phmm->M; j++) {
    TFlist[j * (phmm->inactive + 1)] = TF;
    TF += phmm->D[j];
    if (phmm->inactive == 1) {
      TFlist[j * (phmm->inactive + 1) + 1] = TF;
      TF += phmm->D[j];
    }
  }
  TF -= 1;

  double * d_pi;
  cudaMalloc(&d_pi, phmm->N * sizeof(double));
  cudaMemcpy(d_pi, phmm->pi, phmm->N * sizeof(double), cudaMemcpyHostToDevice);
  double * d_alpha;
  cudaMalloc(&d_alpha, phmm->N * T * sizeof(double));
  int * d_peakPos;
  cudaMalloc(&d_peakPos, (P + 1) * sizeof(int));
  cudaMemcpy(d_peakPos, peakPos, (P + 1) * sizeof(int), cudaMemcpyHostToDevice);
  double * d_log_A_matrix;
  cudaMalloc(&d_log_A_matrix,
             phmm->log_A_matrix->size1 *
               phmm->log_A_matrix->tda *
               sizeof(double));
  cudaMemcpy(d_log_A_matrix,
             phmm->log_A_matrix->data,
             phmm->log_A_matrix->size1 *
               phmm->log_A_matrix->tda *
               sizeof(double),
             cudaMemcpyHostToDevice);
  double * d_emission_matrix;
  cudaMalloc(&d_emission_matrix,
             emission_matrix->size1 * emission_matrix->tda * sizeof(double));
  cudaMemcpy(d_emission_matrix,
             emission_matrix->data,
             emission_matrix->size1 * emission_matrix->tda * sizeof(double),
             cudaMemcpyHostToDevice);
  int * d_TFlist;
  cudaMalloc(&d_TFlist, phmm->M * (phmm->inactive + 1) * sizeof(int));
  cudaMemcpy(d_TFlist, TFlist, phmm->M * (phmm->inactive + 1) * sizeof(int),
             cudaMemcpyHostToDevice);
  free_ivector(TFlist, MAX(phmm->M, 2));

  cuda_dev_Forward_P<<<P, phmm->N>>>(d_pi, phmm->N, d_alpha, T, d_peakPos,
                                     d_log_A_matrix, phmm->log_A_matrix->tda,
                                     d_emission_matrix, emission_matrix->tda,
                                     phmm->M * (phmm->inactive + 1), d_TFlist,
                                     TF);

  double * h_alpha = (double*)malloc(phmm->N * T * sizeof(double));
  cudaMemcpy(h_alpha, d_alpha, phmm->N * T * sizeof(double),
             cudaMemcpyDeviceToHost);
  cudaFree(d_pi);
  cudaFree(d_alpha);
  cudaFree(d_peakPos);
  cudaFree(d_log_A_matrix);
  cudaFree(d_emission_matrix);
  cudaFree(d_TFlist);

  /* 3. Termination */
#pragma omp parallel num_threads(THREAD_NUM)
{
#pragma omp for collapse(2)
  for (int i = 0; i < phmm->N; i++) {
    for (int j = 0; j < T; j++) {
      alpha[i][j] = h_alpha[i * T + j];
    }
  }
}
  free(h_alpha);
#pragma omp parallel num_threads(THREAD_NUM)
{
#pragma omp for
  for (int k = 0; k < P; k++) {
    pprob[k] = -INFINITY;
    int endIdx = peakPos[k+1] - 2;
    for (int i = 0; i < phmm->N; i++) {
      if (alpha[i][endIdx] != -INFINITY &&
            alpha[i][endIdx] == alpha[i][endIdx]) {
        if (pprob[k] != -INFINITY) {
          pprob[k] = logadd(pprob[k], alpha[i][endIdx]);
        } else {
          pprob[k] = alpha[i][endIdx];
        }
      }
    }
  }
}
}

