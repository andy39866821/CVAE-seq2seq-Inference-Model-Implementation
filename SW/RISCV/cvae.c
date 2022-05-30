/* Copyright (c) 2011-2021 Columbia University, System Level Design Group */
/* SPDX-License-Identifier: Apache-2.0 */

#include <stdio.h>
#include <math.h>
#ifndef __riscv
#include <stdlib.h>
#endif

// Include data
#include "weight.h"
#include "bias.h"
#include "inputs.h"
#include "outputs.h"
#include "seq_lens.h"

typedef int32_t DT;
typedef int64_t DOUBLE_DT;
#define MAX_LEN 60
#define QUAN 24
#define DATA_WIDTH 32
#define SIGMOID 0
#define TANH 1

static inline uint64_t get_counter()
{
	uint64_t counter;
	asm volatile (
		"li t0, 0;"
		"csrr t0, mcycle;"
		"mv %0, t0"
		: "=r" ( counter )
		:
		: "t0"
	);
	return counter;
}

int i, j, k;
DT cordic(DT in, int mode){
	const int CORDIC_QUAN = 16;
	int M = 3;
	int N = 15;
	int P = 18;
	DT M_ATANH[4];
	DT ATANH[16];
	DT z0;
	DT X[30];
	DT Y[30];
	DT Z[30];
		
	DT MX[30];
	DT MY[30];
	DT MZ[30];

	// Inv torque for (m, n) = (3, 15)
	DT inv_torque = 704813465;
	
	//cout << "INV_TORQUE: " << inv_torque;

	// arc-tanh LUT for k <= 0
	M_ATANH[0] = 386121;
	M_ATANH[1] = 204353;
	M_ATANH[2] = 112524;
	M_ATANH[3] = 63763;

	// arc-tanh LUT for k > 0
	ATANH[0] = 0; // not used
	ATANH[1] = 35999;
	ATANH[2] = 16738;
	ATANH[3] = 8235;
	ATANH[4] = 4101;
	ATANH[5] = 2048;
	ATANH[6] = 1024;
	ATANH[7] = 512;
	ATANH[8] = 256;
	ATANH[9] = 128;
	ATANH[10] = 64;
	ATANH[11] = 32;
	ATANH[12] = 16;
	ATANH[13] = 8;
	ATANH[14] = 4;
	ATANH[15] = 2;

	if(mode == SIGMOID) // D2 selection
		z0 = -in;
	else	
		z0 = in << 1;

	MX[0] = inv_torque;
	MY[0] = 0;
	MZ[0] = z0 ;
	int i;
	for(i = 0 ; i <= M ; i++){
		int k = -M+i;
		int shift = 1 << (1-k);
		//cout << "Shift: " << shift  << " " << k << endl;
		if(MZ[i] < 0) {
			MX[i+1] = MX[i] - (MY[i] - (MY[i] >> shift));
			MY[i+1] = MY[i] - (MX[i] - (MX[i] >> shift));
			MZ[i+1] = MZ[i] + M_ATANH[i];
		}   
		else{
			MX[i+1] = MX[i] + (MY[i] - (MY[i] >> shift));
			MY[i+1] = MY[i] + (MX[i] - (MX[i] >> shift));
			MZ[i+1] = MZ[i] - M_ATANH[i];
		}	
	}

	X[1] = MX[M+1];
	Y[1] = MY[M+1];
	Z[1] = MZ[M+1];


	for(k = 1 ; k <= N ; k++){
		if(Z[k] < 0) {
			X[k+1] = X[k] - (Y[k] >> k);
			Y[k+1] = Y[k] - (X[k] >> k);
			Z[k+1] = Z[k] + ATANH[k];
		}   
		else{
			X[k+1] = X[k] + (Y[k] >> k);
			Y[k+1] = Y[k] + (X[k] >> k);
			Z[k+1] = Z[k] - ATANH[k];
		}	   
			
		if(k==4 || k==13){ // accumulate again if k equals to specific value
			DT tempX = X[k+1];
			DT tempY = Y[k+1];
			DT tempZ = Z[k+1];
			if(tempZ < 0) {
				X[k+1] = tempX - (tempY >> k);
				Y[k+1] = tempY - (tempX >> k);
				Z[k+1] = tempZ + ATANH[k];
			}   
			else{
				X[k+1] = tempX + (tempY >> k);
				Y[k+1] = tempY + (tempX >> k);
				Z[k+1] = tempZ - ATANH[k];
			}	
		}

	}
	
		
	X[0] = X[N+1] + Y[N+1] + (1 << CORDIC_QUAN);
	Y[0] = (1 << CORDIC_QUAN);
	Z[0] = 0;

	for(k = 0 ; k < P ; k++) {
		if(Y[k] < 0) {
			X[k+1] = X[k];
			Y[k+1] = Y[k] + (X[k] >> k);
			Z[k+1] = Z[k] - ((1 << CORDIC_QUAN) >> k );
		}   
		else{
			X[k+1] = X[k];
			Y[k+1] = Y[k] - (X[k] >> k);
			Z[k+1] = Z[k] + ((1 << CORDIC_QUAN) >> k );
				
		}	   
	}


	DT result;
	if(mode == SIGMOID) {
		result = Z[18]; // result = S
	}
	else {
		result = (1 << CORDIC_QUAN) - (Z[18] << 1); // result = 1 - 2S
	}
		
	return result;
		

}

DT fc_l2h_weights[32*4];
DT fc_l2h_bias[32];
DT fc_conf_weights[2*38];
DT fc_conf_bias[2];
DT fc_state_weights[13*32];
DT fc_state_bias[13];
DT gru_ir_weights[32*19];
DT gru_iz_weights[32*19];
DT gru_in_weights[32*19];
DT gru_hr_weights[32*32];
DT gru_hz_weights[32*32];
DT gru_hn_weights[32*32];
DT gru_ir_bias[32];
DT gru_iz_bias[32];
DT gru_in_bias[32];
DT gru_hr_bias[32];
DT gru_hz_bias[32];
DT gru_hn_bias[32];

DT state_buffer[60][13];
DT hidden_buffer[60][32];
DT conf_buffer[60][2];

int seq_lens = 0;

void load_data(){
	int index = 0;
	for(i = 0 ; i < 32*19 ; i++)
		gru_in_weights[i] = weight[index++];
	for(i = 0 ; i < 32*19 ; i++)
		gru_ir_weights[i] = weight[index++];
	for(i = 0 ; i < 32*19 ; i++)
		gru_iz_weights[i] = weight[index++];
		
	for(i = 0 ; i < 32*32 ; i++)
		gru_hn_weights[i] = weight[index++];
	for(i = 0 ; i < 32*32 ; i++)
		gru_hr_weights[i] = weight[index++];
	for(i = 0 ; i < 32*32 ; i++)
		gru_hz_weights[i] = weight[index++];

	for (i = 0; i < 32*13; i++) 
		fc_state_weights[i] = weight[index++];
	for (i = 0; i < 38*2; i++) 
		fc_conf_weights[i] = weight[index++];
	for (i = 0; i < 32*4; i++) 
		fc_l2h_weights[i] = weight[index++];

	index = 0;
	for(i = 0 ; i < 32 ; i++)
		gru_in_bias[i] = bias[index++];
	for(i = 0 ; i < 32 ; i++)
		gru_ir_bias[i] = bias[index++];
	for(i = 0 ; i < 32 ; i++)
		gru_iz_bias[i] = bias[index++];
		
	for(i = 0 ; i < 32 ; i++)
		gru_hn_bias[i] = bias[index++];
	for(i = 0 ; i < 32 ; i++)
		gru_hr_bias[i] = bias[index++];
	for(i = 0 ; i < 32 ; i++)
		gru_hz_bias[i] = bias[index++];

	for (i= 0; i < 13; i++) 
		fc_state_bias[i] = bias[index++];
	for (i= 0; i < 2; i++) 
		fc_conf_bias[i] = bias[index++];
	for (i= 0; i < 32; i++) 
		fc_l2h_bias[i] = bias[index++];
}

void L2H_FC() {

	for (i= 0; i < 32; i++) {
		hidden_buffer[0][i] = 0;
		for (j = 0; j < 4; j++) {
			DOUBLE_DT source = z_in[j];
			DOUBLE_DT weight = fc_l2h_weights[i * 4 + j];
			DOUBLE_DT partial_sum = (source * weight);
			hidden_buffer[0][i] += partial_sum >> QUAN;
			//printf("Buffer[%d]: %08x = %d * %d\n", i, hidden_buffer[0][i], source, weight);
		}
		hidden_buffer[0][i] += fc_l2h_bias[i];
		
		
		//printf("Hidden[%d]: %08x\n", i, hidden_buffer[0][i]);
	}

	for (i= 0; i < 13; i++)
		state_buffer[0][i] = s_in[i]; // Copy to initial state

}
void STATE_FC() {
	for (i= 0; i < 13; i++) {
		state_buffer[seq_lens+1][i] = 0;
		for (j = 0; j < 32; j++) {
			DOUBLE_DT source = hidden_buffer[seq_lens + 1][j];
			DOUBLE_DT weight = fc_state_weights[i * 32 + j] ;
			
			DOUBLE_DT partial_sum = (source * weight);
			state_buffer[seq_lens + 1][i] += partial_sum >> QUAN;
		}

		state_buffer[seq_lens + 1][i] += fc_state_bias[i];

		//printf("Hidden[%d]: %e\n", i, hidden_buffer[0][i]);
	}
}
void CONF_FC() {
	DT sources[38];
	// concate out[32] & goal[6]
	for (i= 0; i < 32; i++)
		sources[i] = hidden_buffer[seq_lens + 1][i];
	for (i= 0; i < 6; i++)
		sources[32+i] = s_goal[i];
	
	for (i= 0; i < 2; i++) {
		conf_buffer[seq_lens + 1][i] = 0;
		for (j = 0; j < 38; j++) {
			DOUBLE_DT source = sources[j];
			DOUBLE_DT weight = fc_conf_weights[i * 38 + j];
			DOUBLE_DT partial_sum = (source * weight);
			conf_buffer[seq_lens + 1][i] += partial_sum >> QUAN;
		}

		conf_buffer[seq_lens + 1][i] += fc_conf_bias[i];

		//printf("CONF[%d]: %08x\n", i, conf_buffer[seq_lens + 1][i]);
	}
}
void GRU() {
	DT R[32];
	DT Z[32];
	DT N[32];
	DT X[19];

	// concate X
	for (i= 0; i < 13; i++)
		X[i] = state_buffer[seq_lens][i];
	for (i= 0; i < 6; i++)
		X[13 + i] = s_goal[i];

	
	//Rt
	for (i= 0; i < 32; i++) {
		DOUBLE_DT acc_HR = 0;
		for (j = 0; j < 32; j++) {
			DOUBLE_DT source = hidden_buffer[seq_lens][j];
			DOUBLE_DT weight = gru_hr_weights[i * 32 + j] ;
			DOUBLE_DT partial_sum = (source * weight);
			acc_HR += partial_sum >> QUAN;
		}
		acc_HR += gru_hr_bias[i];

		DT acc_IR = 0;
		for (j = 0; j < 19; j++) {
			DOUBLE_DT source = X[j];
			DOUBLE_DT weight = gru_ir_weights[i * 19 + j];
			DOUBLE_DT partial_sum = (source * weight);
			acc_IR += partial_sum >> QUAN;
		}
		acc_IR += gru_ir_bias[i];
		R[i] = acc_IR + acc_HR;
		R[i] = R[i] >> (QUAN - 16);
		R[i] = cordic(R[i], SIGMOID);
		R[i] = R[i] << (QUAN - 16);
	}

	// Zt
	for (i= 0; i < 32; i++) {
		DOUBLE_DT acc_HZ = 0;
		for (j = 0; j < 32; j++) {
			DOUBLE_DT source = hidden_buffer[seq_lens][j];
			DOUBLE_DT weight = gru_hz_weights[i * 32 + j];
			DOUBLE_DT partial_sum = (source * weight);
			acc_HZ += partial_sum >> QUAN;
		}
		acc_HZ += gru_hz_bias[i];

		DOUBLE_DT acc_IZ = 0;
		for (j = 0; j < 19; j++) {
			DOUBLE_DT source = X[j];
			DOUBLE_DT weight = gru_iz_weights[i * 19 + j];
			DOUBLE_DT partial_sum = (source * weight);
			acc_IZ += partial_sum >> QUAN;
		}
		acc_IZ += gru_iz_bias[i];


		Z[i] = acc_IZ + acc_HZ;
		Z[i] = Z[i] >> (QUAN - 16);
		Z[i] = cordic(Z[i], SIGMOID);
		Z[i] = Z[i] << (QUAN - 16);
	}

	// Nt
	for (i= 0; i < 32; i++) {
		N[i] = 0;

		DOUBLE_DT acc_HN = 0;
		for (j = 0; j < 32; j++) {
			DOUBLE_DT source = hidden_buffer[seq_lens][j];
			DOUBLE_DT weight = gru_hn_weights[i * 32 + j];
			DOUBLE_DT partial_sum = (source * weight);
			acc_HN += partial_sum >> QUAN;
		}
		acc_HN += gru_hn_bias[i];

		DOUBLE_DT partial_sum = R[i] * acc_HN;
		N[i] += partial_sum >> QUAN ;

		DOUBLE_DT acc_IN = 0;
		for (j = 0; j < 19; j++) {
			DOUBLE_DT source = X[j];
			DOUBLE_DT weight = gru_in_weights[i * 19 + j];
			DOUBLE_DT partial_sum = (source * weight);
			acc_IN += partial_sum >> QUAN;
		}
		acc_IN += gru_in_bias[i];

		N[i] += acc_IN;
		N[i] = N[i] >> (QUAN - 16);
		N[i] = cordic(N[i], TANH);
		N[i] = N[i] << (QUAN - 16);
	}

	for (i= 0; i < 32; i++) {
		hidden_buffer[seq_lens + 1][i] = 0;

		DOUBLE_DT source1 = ((1 << QUAN) - Z[i]);
		DOUBLE_DT source2 = N[i];
		
		DOUBLE_DT partial_sum = (source1 * source2);
		
		hidden_buffer[seq_lens + 1][i] += partial_sum >> QUAN;

		source1 = Z[i] ;
		source2 = hidden_buffer[seq_lens][i];
		partial_sum = (source1 * source2);
		hidden_buffer[seq_lens + 1][i] += partial_sum >> QUAN;


		//printf("hidden_buffer[%d]: %08x\n", i, hidden_buffer[seq_lens + 1][i]);
	}

}



int main(){
	seq_lens = 0;
	load_data();
	uint64_t start_time = get_counter();
	L2H_FC();
	while (seq_lens < MAX_LEN - 1) {
		GRU();
		CONF_FC();
		STATE_FC();
		printf("Current: %d\n", seq_lens);
		seq_lens++;
		if (conf_buffer[seq_lens][0] < conf_buffer[seq_lens][1])
			break;
		
	}
	uint64_t end_time = get_counter();

	printf("--------Result-------\n");
	
	if (golden_seq_lens != seq_lens)
		printf("[ERROR] gold: %d , yours: %d\n", golden_seq_lens, seq_lens);
	else
		printf("[PASS] gold: %d , yours: %d\n", golden_seq_lens, seq_lens);

	int pass = 1;
	for(i = 0 ; i < seq_lens*13 ; i++)
		if(golden_states[i] != state_buffer[i/13][i%13]){
			
			printf("[ERROR] gold[%d]: %d , yours[%d]: %d\n",i, golden_states[i],i, state_buffer[i/13][i%13]);
			pass = 0;
		}
	
	if (!pass)
		printf("[ERROR] State has errors\n");
	else
		printf("[PASS] Congratulation!\n");

	printf("Total cycle count: %d\n", end_time - start_time);
	return 0;
}