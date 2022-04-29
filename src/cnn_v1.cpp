/**********
Copyright (c) 2019, Xilinx, Inc.
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
may be used to endorse or promote products derived from this software
without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED.
IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
BUSINESS INTERRUPTION)
HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
THIS SOFTWARE,
EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
**********/

/*******************************************************************************
Description:
    HLS pragmas can be used to optimize the design : improve throughput, reduce
latency and
    device resource utilization of the resulting RTL code
    This is vector addition example to demonstrate how HLS optimizations are
used in kernel.
*******************************************************************************/

#include "hls_stream.h"
#include "ap_int.h"

#include "cnn.h"

extern "C" {
	
void readInput(DTYPE *input, DTYPE local_input[][kInImSize][kNum])
{
	//read in input data from host (buffer_A)
    for (int h = 0; h < kInImSize; ++h) {
        for (int w = 0; w < kInImSize; ++w){
            for (int i = 0; i < kNum; ++i) {
				#pragma HLS PIPELINE II=1
				local_input[h][w][i] = input[((h*kInImSize+w)*kNum+i)];
            }
        }
    }
}

void readWeight(DTYPE *weight, DTYPE local_weight[][kKernel][kNum][kNum])
{
	//read in weight data from host (buffer_B)
    for (int p = 0; p < kKernel; ++p) {
        for (int q = 0; q < kKernel; ++q){
            for (int i = 0; i < kNum; ++i) {
                for (int j = 0; j < kNum; ++j) {
					#pragma HLS PIPELINE II=1
                    local_weight[p][q][i][j] = weight[((p*kKernel+q)*kNum+i)*kNum+j];
                }
            }
        }
    }
}

void comp(DTYPE local_input[][kInImSize][kNum], DTYPE local_weight[kKernel][kKernel][kNum][kNum], DTYPE local_output[][kOutImSize][kNum])
{	
	//initialize the local output values to 0 (buffer_AB on host)
    for (int h = 0; h < kOutImSize; ++h) {
        for (int w = 0; w < kOutImSize; ++w){
			#pragma HLS PIPELINE II=1
            for (int i = 0; i < kNum; ++i) {
				#pragma HLS unroll
                local_output[h][w][i] = 0.0f;
            }
        }
    }
	
	// Convolution
	//calculate final result that will be copied back to host
    
        
	for (int j = 0; j < kNum; ++j) {
		for (int p = 0; p < kKernel; ++p) {
			for (int q = 0; q < kKernel; ++q){
				for (int h = 0; h < kOutImSize; ++h) {
					for (int w = 0; w < kOutImSize; ++w) {
						#pragma HLS PIPELINE II=1
						for (int i = 0; i < kNum; ++i) {
							#pragma HLS unroll
							local_output[h][w][i] += local_input[h+p][w+q][j] * local_weight[p][q][i][j];
						}
                    }
				}
			}
		}
	}
}

void writeOutput(DTYPE *output, DTYPE local_output[][kOutImSize][kNum])
{
	//copy final result back to host (buffer_AB)
    for (int h = 0; h < kOutImSize; ++h) {
        for (int w = 0; w < kOutImSize; ++w){
            for (int i = 0; i < kNum; ++i) {
				#pragma HLS PIPELINE II=1
                output[(h*kOutImSize + w)*kNum+i] = local_output[h][w][i];
            }
        }
    }
}

void cnn(DTYPE *input,  DTYPE *weight, DTYPE *output)
{
	#pragma HLS INTERFACE m_axi port = input offset = slave bundle = m0
	#pragma HLS INTERFACE m_axi port = weight offset = slave bundle = m1
	#pragma HLS INTERFACE m_axi port = output offset = slave bundle = m1

	#pragma HLS INTERFACE s_axilite port = input bundle = control
	#pragma HLS INTERFACE s_axilite port = weight bundle = control
	#pragma HLS INTERFACE s_axilite port = output bundle = control
	#pragma HLS INTERFACE s_axilite port = return bundle = control
	
    DTYPE local_input[kInImSize][kInImSize][kNum];
	#pragma HLS RESOURCE variable=local_input core=RAM_1P_URAM
    DTYPE local_output[kOutImSize][kOutImSize][kNum];
	#pragma HLS RESOURCE variable=local_output core=RAM_2P_URAM
    DTYPE local_weight[kKernel][kKernel][kNum][kNum];
	
	//#pragma HLS ARRAY_PARTITION variable=local_input complete dim=3  
	#pragma HLS ARRAY_PARTITION variable=local_output complete dim=3 
	#pragma HLS ARRAY_PARTITION variable=local_weight complete dim=3 
	
	//#pragma HLS DATAFLOW
	
	readInput(input, local_input);
	readWeight(weight, local_weight);
	comp(local_input, local_weight, local_output);
	writeOutput(output, local_output);

}

}
