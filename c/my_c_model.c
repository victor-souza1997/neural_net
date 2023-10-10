#include <math.h> 
 #include <string.h> 
#include "./include/k2c_include.h" 
#include "./include/k2c_tensor_include.h" 

 


void my_c_model(k2c_tensor* dense_input_input, k2c_tensor* dense_3_output,float* dense_output_array,float* dense_kernel_array,float* dense_bias_array,float* dense_1_output_array,float* dense_1_kernel_array,float* dense_1_bias_array,float* dense_2_output_array,float* dense_2_kernel_array,float* dense_2_bias_array,float* dense_3_kernel_array,float* dense_3_bias_array) { 

k2c_tensor dense_output = {dense_output_array,1,10,{10, 1, 1, 1, 1}}; 
k2c_tensor dense_kernel = {dense_kernel_array,2,10,{ 1,10, 1, 1, 1}}; 
k2c_tensor dense_bias = {dense_bias_array,1,10,{10, 1, 1, 1, 1}}; 
float dense_fwork[11] = {0}; 

 
k2c_tensor dense_1_output = {dense_1_output_array,1,30,{30, 1, 1, 1, 1}}; 
k2c_tensor dense_1_kernel = {dense_1_kernel_array,2,300,{10,30, 1, 1, 1}}; 
k2c_tensor dense_1_bias = {dense_1_bias_array,1,30,{30, 1, 1, 1, 1}}; 
float dense_1_fwork[310] = {0}; 

 
k2c_tensor dense_2_output = {dense_2_output_array,1,10,{10, 1, 1, 1, 1}}; 
k2c_tensor dense_2_kernel = {dense_2_kernel_array,2,300,{30,10, 1, 1, 1}}; 
k2c_tensor dense_2_bias = {dense_2_bias_array,1,10,{10, 1, 1, 1, 1}}; 
float dense_2_fwork[330] = {0}; 

 
k2c_tensor dense_3_kernel = {dense_3_kernel_array,2,10,{10, 1, 1, 1, 1}}; 
k2c_tensor dense_3_bias = {dense_3_bias_array,1,1,{1,1,1,1,1}}; 
float dense_3_fwork[20] = {0}; 

 
k2c_dense(&dense_output,dense_input_input,&dense_kernel, 
	&dense_bias,k2c_sigmoid,dense_fwork); 
k2c_dense(&dense_1_output,&dense_output,&dense_1_kernel, 
	&dense_1_bias,k2c_sigmoid,dense_1_fwork); 
k2c_dense(&dense_2_output,&dense_1_output,&dense_2_kernel, 
	&dense_2_bias,k2c_sigmoid,dense_2_fwork); 
k2c_dense(dense_3_output,&dense_2_output,&dense_3_kernel, 
	&dense_3_bias,k2c_sigmoid,dense_3_fwork); 

 } 

void my_c_model_initialize(float** dense_output_array 
,float** dense_kernel_array 
,float** dense_bias_array 
,float** dense_1_output_array 
,float** dense_1_kernel_array 
,float** dense_1_bias_array 
,float** dense_2_output_array 
,float** dense_2_kernel_array 
,float** dense_2_bias_array 
,float** dense_3_kernel_array 
,float** dense_3_bias_array 
) { 

*dense_output_array = k2c_read_array("my_c_modeldense_output_array.csv",10); 
*dense_kernel_array = k2c_read_array("my_c_modeldense_kernel_array.csv",10); 
*dense_bias_array = k2c_read_array("my_c_modeldense_bias_array.csv",10); 
*dense_1_output_array = k2c_read_array("my_c_modeldense_1_output_array.csv",30); 
*dense_1_kernel_array = k2c_read_array("my_c_modeldense_1_kernel_array.csv",300); 
*dense_1_bias_array = k2c_read_array("my_c_modeldense_1_bias_array.csv",30); 
*dense_2_output_array = k2c_read_array("my_c_modeldense_2_output_array.csv",10); 
*dense_2_kernel_array = k2c_read_array("my_c_modeldense_2_kernel_array.csv",300); 
*dense_2_bias_array = k2c_read_array("my_c_modeldense_2_bias_array.csv",10); 
*dense_3_kernel_array = k2c_read_array("my_c_modeldense_3_kernel_array.csv",10); 
*dense_3_bias_array = k2c_read_array("my_c_modeldense_3_bias_array.csv",1); 
} 

void my_c_model_terminate(float* dense_output_array,float* dense_kernel_array,float* dense_bias_array,float* dense_1_output_array,float* dense_1_kernel_array,float* dense_1_bias_array,float* dense_2_output_array,float* dense_2_kernel_array,float* dense_2_bias_array,float* dense_3_kernel_array,float* dense_3_bias_array) { 

free(dense_output_array); 
free(dense_kernel_array); 
free(dense_bias_array); 
free(dense_1_output_array); 
free(dense_1_kernel_array); 
free(dense_1_bias_array); 
free(dense_2_output_array); 
free(dense_2_kernel_array); 
free(dense_2_bias_array); 
free(dense_3_kernel_array); 
free(dense_3_bias_array); 
} 

