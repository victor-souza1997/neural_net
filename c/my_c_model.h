#pragma once 
#include "./include/k2c_tensor_include.h" 
void my_c_model(k2c_tensor* dense_input_input, k2c_tensor* dense_3_output,float* dense_output_array,float* dense_kernel_array,float* dense_bias_array,float* dense_1_output_array,float* dense_1_kernel_array,float* dense_1_bias_array,float* dense_2_output_array,float* dense_2_kernel_array,float* dense_2_bias_array,float* dense_3_kernel_array,float* dense_3_bias_array); 
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
); 
void my_c_model_terminate(float* dense_output_array,float* dense_kernel_array,float* dense_bias_array,float* dense_1_output_array,float* dense_1_kernel_array,float* dense_1_bias_array,float* dense_2_output_array,float* dense_2_kernel_array,float* dense_2_bias_array,float* dense_3_kernel_array,float* dense_3_bias_array); 
