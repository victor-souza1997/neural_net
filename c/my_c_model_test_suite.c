#include <stdio.h> 
#include <math.h> 
#include <time.h> 
#include "./include/k2c_include.h" 
#include "my_c_model.h" 

float maxabs(k2c_tensor *tensor1, k2c_tensor *tensor2);
struct timeval GetTimeStamp(); 
 
float test1_dense_input_input_array[1] = {
+5.38619176e-01f,}; 
k2c_tensor test1_dense_input_input = {&test1_dense_input_input_array[0],1,1,{1,1,1,1,1}}; 
float keras_dense_3_test1_array[1] = {
+9.76756990e-01f,}; 
k2c_tensor keras_dense_3_test1 = {&keras_dense_3_test1_array[0],1,1,{1,1,1,1,1}}; 
float c_dense_3_test1_array[1] = {0}; 
k2c_tensor c_dense_3_test1 = {&c_dense_3_test1_array[0],1,1,{1,1,1,1,1}}; 
float test2_dense_input_input_array[1] = {
-6.02107028e-01f,}; 
k2c_tensor test2_dense_input_input = {&test2_dense_input_input_array[0],1,1,{1,1,1,1,1}}; 
float keras_dense_3_test2_array[1] = {
+9.95252192e-01f,}; 
k2c_tensor keras_dense_3_test2 = {&keras_dense_3_test2_array[0],1,1,{1,1,1,1,1}}; 
float c_dense_3_test2_array[1] = {0}; 
k2c_tensor c_dense_3_test2 = {&c_dense_3_test2_array[0],1,1,{1,1,1,1,1}}; 
float test3_dense_input_input_array[1] = {
+5.97967822e-01f,}; 
k2c_tensor test3_dense_input_input = {&test3_dense_input_input_array[0],1,1,{1,1,1,1,1}}; 
float keras_dense_3_test3_array[1] = {
+9.72255230e-01f,}; 
k2c_tensor keras_dense_3_test3 = {&keras_dense_3_test3_array[0],1,1,{1,1,1,1,1}}; 
float c_dense_3_test3_array[1] = {0}; 
k2c_tensor c_dense_3_test3 = {&c_dense_3_test3_array[0],1,1,{1,1,1,1,1}}; 
float test4_dense_input_input_array[1] = {
-1.51966062e+00f,}; 
k2c_tensor test4_dense_input_input = {&test4_dense_input_input_array[0],1,1,{1,1,1,1,1}}; 
float keras_dense_3_test4_array[1] = {
+9.96097684e-01f,}; 
k2c_tensor keras_dense_3_test4 = {&keras_dense_3_test4_array[0],1,1,{1,1,1,1,1}}; 
float c_dense_3_test4_array[1] = {0}; 
k2c_tensor c_dense_3_test4 = {&c_dense_3_test4_array[0],1,1,{1,1,1,1,1}}; 
float test5_dense_input_input_array[1] = {
-3.39507429e-01f,}; 
k2c_tensor test5_dense_input_input = {&test5_dense_input_input_array[0],1,1,{1,1,1,1,1}}; 
float keras_dense_3_test5_array[1] = {
+9.94525790e-01f,}; 
k2c_tensor keras_dense_3_test5 = {&keras_dense_3_test5_array[0],1,1,{1,1,1,1,1}}; 
float c_dense_3_test5_array[1] = {0}; 
k2c_tensor c_dense_3_test5 = {&c_dense_3_test5_array[0],1,1,{1,1,1,1,1}}; 
float test6_dense_input_input_array[1] = {
+1.19161290e+00f,}; 
k2c_tensor test6_dense_input_input = {&test6_dense_input_input_array[0],1,1,{1,1,1,1,1}}; 
float keras_dense_3_test6_array[1] = {
+7.85606921e-01f,}; 
k2c_tensor keras_dense_3_test6 = {&keras_dense_3_test6_array[0],1,1,{1,1,1,1,1}}; 
float c_dense_3_test6_array[1] = {0}; 
k2c_tensor c_dense_3_test6 = {&c_dense_3_test6_array[0],1,1,{1,1,1,1,1}}; 
float test7_dense_input_input_array[1] = {
-1.76289813e+00f,}; 
k2c_tensor test7_dense_input_input = {&test7_dense_input_input_array[0],1,1,{1,1,1,1,1}}; 
float keras_dense_3_test7_array[1] = {
+9.96173620e-01f,}; 
k2c_tensor keras_dense_3_test7 = {&keras_dense_3_test7_array[0],1,1,{1,1,1,1,1}}; 
float c_dense_3_test7_array[1] = {0}; 
k2c_tensor c_dense_3_test7 = {&c_dense_3_test7_array[0],1,1,{1,1,1,1,1}}; 
float test8_dense_input_input_array[1] = {
-1.62925028e+00f,}; 
k2c_tensor test8_dense_input_input = {&test8_dense_input_input_array[0],1,1,{1,1,1,1,1}}; 
float keras_dense_3_test8_array[1] = {
+9.96135354e-01f,}; 
k2c_tensor keras_dense_3_test8 = {&keras_dense_3_test8_array[0],1,1,{1,1,1,1,1}}; 
float c_dense_3_test8_array[1] = {0}; 
k2c_tensor c_dense_3_test8 = {&c_dense_3_test8_array[0],1,1,{1,1,1,1,1}}; 
float test9_dense_input_input_array[1] = {
+1.04698808e+00f,}; 
k2c_tensor test9_dense_input_input = {&test9_dense_input_input_array[0],1,1,{1,1,1,1,1}}; 
float keras_dense_3_test9_array[1] = {
+8.70482564e-01f,}; 
k2c_tensor keras_dense_3_test9 = {&keras_dense_3_test9_array[0],1,1,{1,1,1,1,1}}; 
float c_dense_3_test9_array[1] = {0}; 
k2c_tensor c_dense_3_test9 = {&c_dense_3_test9_array[0],1,1,{1,1,1,1,1}}; 
float test10_dense_input_input_array[1] = {
+4.16800241e-02f,}; 
k2c_tensor test10_dense_input_input = {&test10_dense_input_input_array[0],1,1,{1,1,1,1,1}}; 
float keras_dense_3_test10_array[1] = {
+9.91969883e-01f,}; 
k2c_tensor keras_dense_3_test10 = {&keras_dense_3_test10_array[0],1,1,{1,1,1,1,1}}; 
float c_dense_3_test10_array[1] = {0}; 
k2c_tensor c_dense_3_test10 = {&c_dense_3_test10_array[0],1,1,{1,1,1,1,1}}; 
int main(){
 float errors[10];
 size_t num_tests = 10; 
size_t num_outputs = 1; 
float* dense_output_array; 
float* dense_kernel_array; 
float* dense_bias_array; 
float* dense_1_output_array; 
float* dense_1_kernel_array; 
float* dense_1_bias_array; 
float* dense_2_output_array; 
float* dense_2_kernel_array; 
float* dense_2_bias_array; 
float* dense_3_kernel_array; 
float* dense_3_bias_array; 
my_c_model_initialize(&dense_output_array,&dense_kernel_array,&dense_bias_array,&dense_1_output_array,&dense_1_kernel_array,&dense_1_bias_array,&dense_2_output_array,&dense_2_kernel_array,&dense_2_bias_array,&dense_3_kernel_array,&dense_3_bias_array); 
clock_t t0 = clock(); 
my_c_model(&test1_dense_input_input,&c_dense_3_test1,dense_output_array,dense_kernel_array,dense_bias_array,dense_1_output_array,dense_1_kernel_array,dense_1_bias_array,dense_2_output_array,dense_2_kernel_array,dense_2_bias_array,dense_3_kernel_array,dense_3_bias_array); 
my_c_model(&test2_dense_input_input,&c_dense_3_test2,dense_output_array,dense_kernel_array,dense_bias_array,dense_1_output_array,dense_1_kernel_array,dense_1_bias_array,dense_2_output_array,dense_2_kernel_array,dense_2_bias_array,dense_3_kernel_array,dense_3_bias_array); 
my_c_model(&test3_dense_input_input,&c_dense_3_test3,dense_output_array,dense_kernel_array,dense_bias_array,dense_1_output_array,dense_1_kernel_array,dense_1_bias_array,dense_2_output_array,dense_2_kernel_array,dense_2_bias_array,dense_3_kernel_array,dense_3_bias_array); 
my_c_model(&test4_dense_input_input,&c_dense_3_test4,dense_output_array,dense_kernel_array,dense_bias_array,dense_1_output_array,dense_1_kernel_array,dense_1_bias_array,dense_2_output_array,dense_2_kernel_array,dense_2_bias_array,dense_3_kernel_array,dense_3_bias_array); 
my_c_model(&test5_dense_input_input,&c_dense_3_test5,dense_output_array,dense_kernel_array,dense_bias_array,dense_1_output_array,dense_1_kernel_array,dense_1_bias_array,dense_2_output_array,dense_2_kernel_array,dense_2_bias_array,dense_3_kernel_array,dense_3_bias_array); 
my_c_model(&test6_dense_input_input,&c_dense_3_test6,dense_output_array,dense_kernel_array,dense_bias_array,dense_1_output_array,dense_1_kernel_array,dense_1_bias_array,dense_2_output_array,dense_2_kernel_array,dense_2_bias_array,dense_3_kernel_array,dense_3_bias_array); 
my_c_model(&test7_dense_input_input,&c_dense_3_test7,dense_output_array,dense_kernel_array,dense_bias_array,dense_1_output_array,dense_1_kernel_array,dense_1_bias_array,dense_2_output_array,dense_2_kernel_array,dense_2_bias_array,dense_3_kernel_array,dense_3_bias_array); 
my_c_model(&test8_dense_input_input,&c_dense_3_test8,dense_output_array,dense_kernel_array,dense_bias_array,dense_1_output_array,dense_1_kernel_array,dense_1_bias_array,dense_2_output_array,dense_2_kernel_array,dense_2_bias_array,dense_3_kernel_array,dense_3_bias_array); 
my_c_model(&test9_dense_input_input,&c_dense_3_test9,dense_output_array,dense_kernel_array,dense_bias_array,dense_1_output_array,dense_1_kernel_array,dense_1_bias_array,dense_2_output_array,dense_2_kernel_array,dense_2_bias_array,dense_3_kernel_array,dense_3_bias_array); 
my_c_model(&test10_dense_input_input,&c_dense_3_test10,dense_output_array,dense_kernel_array,dense_bias_array,dense_1_output_array,dense_1_kernel_array,dense_1_bias_array,dense_2_output_array,dense_2_kernel_array,dense_2_bias_array,dense_3_kernel_array,dense_3_bias_array); 

clock_t t1 = clock(); 
printf("Average time over 10 tests: %e s \n", 
 ((double)t1-t0)/(double)CLOCKS_PER_SEC/(double)10); 
errors[0] = maxabs(&keras_dense_3_test1,&c_dense_3_test1); 
errors[1] = maxabs(&keras_dense_3_test2,&c_dense_3_test2); 
errors[2] = maxabs(&keras_dense_3_test3,&c_dense_3_test3); 
errors[3] = maxabs(&keras_dense_3_test4,&c_dense_3_test4); 
errors[4] = maxabs(&keras_dense_3_test5,&c_dense_3_test5); 
errors[5] = maxabs(&keras_dense_3_test6,&c_dense_3_test6); 
errors[6] = maxabs(&keras_dense_3_test7,&c_dense_3_test7); 
errors[7] = maxabs(&keras_dense_3_test8,&c_dense_3_test8); 
errors[8] = maxabs(&keras_dense_3_test9,&c_dense_3_test9); 
errors[9] = maxabs(&keras_dense_3_test10,&c_dense_3_test10); 
float maxerror = errors[0]; 
for(size_t i=1; i< num_tests*num_outputs;i++){ 
if (errors[i] > maxerror) { 
maxerror = errors[i];}} 
printf("Max absolute error for 10 tests: %e \n", maxerror);
my_c_model_terminate(dense_output_array,dense_kernel_array,dense_bias_array,dense_1_output_array,dense_1_kernel_array,dense_1_bias_array,dense_2_output_array,dense_2_kernel_array,dense_2_bias_array,dense_3_kernel_array,dense_3_bias_array); 
if (maxerror > 1e-05) { 
return 1;} 
return 0;
} 

float maxabs(k2c_tensor *tensor1, k2c_tensor *tensor2){ 

    float x = 0; 

    float y = 0; 

    for(size_t i=0; i<tensor1->numel; i++){

    y = fabsf(tensor1->array[i]-tensor2->array[i]);
    if (y>x) {x=y;}}
    return x;}

