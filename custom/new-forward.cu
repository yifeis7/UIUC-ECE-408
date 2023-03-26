#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"

#define TILE_WIDTH 8
#define TILE_WIDTH_1 8
#define TILE_WIDTH_2 16
#define TILE_WIDTH_3 16
#define TILE_WIDTH_4 8
#define CEIL(x,y) (((x - 1) / (y)) + 1)
/*Function paramter definitions:
y - output
x - input
k - kernel
B - batch_size (number of images in x)     

M - number of output feature maps        4  16
C - number of input feature maps         1  4


H - input height dimension               86 40
W - input width dimension                86 40
K - kernel height and width (K x K)      7 * 7
*/

//**************************************************************************
//Basic implementation
__constant__ float Mask[4000];
__global__ void conv_forward_kernel(float *y, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K)
{
    // (void)H_out; // silence declared but never referenced warning. remove this line when you start working
    // (void)W_out; // silence declared but never referenced warning. remove this line when you start working

    // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    // An example use of these macros:
    // float a = y4d(0,0,0,0)
    // y4d(0,0,0,0) = a

    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    

    #define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
    #define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
    #define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    // Insert your GPU convolution kernel code here
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int W_Grid = ceil(W_out/(TILE_WIDTH*1.0));
    int h = (blockIdx.z/W_Grid) * TILE_WIDTH +threadIdx.y;
    int w = (blockIdx.z%W_Grid) * TILE_WIDTH + threadIdx.x;
    if((w < W_out) && (h < H_out)){
        float acc = 0;
        for(int c=0; c<C; c++){
            for(int i =0; i<K; i++){
                for(int j=0; j< K; j++){
                    acc += x4d(bx, c, h+i, w+j) * k4d(by, c, i, j);
                }
            }
        }
        y4d(bx, by, h, w) = acc;
    }
#undef y4d
#undef x4d
#undef k4d
}


//**************************************************************************
// Tiled shared memory convolution
__global__ void conv_forward_kernel_tiled(float *y, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K)
{
#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define k4d(i3, i2, i1, i0) Mask[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]	
    

    const int H_out = H - K + 1;
    const int W_out = W - K + 1;

    int W_grid = ceil(1.0*W_out/TILE_WIDTH);
    int X_out_width = TILE_WIDTH + K -1;

    extern __shared__ float shmem[];

    float* X_shared=&shmem[0];
    float* W_shared=&shmem[X_out_width * X_out_width];

    int n = blockIdx.x; 
    int m=blockIdx.y; 
    int w0=threadIdx.x;
    int h0=threadIdx.y;

    int h_base=(blockIdx.z/W_grid)*TILE_WIDTH; 
    int w_base=(blockIdx.z % W_grid)*TILE_WIDTH; 

    int h=h_base+h0; 
    int w=w_base+w0;

    float acc=0;

    for (int c=0; c<C; c++){
        if ((h0<K) && (w0<K)){
            W_shared[h0*K+w0]=k4d(m,c,h0,w0);
            }
        __syncthreads();
	    for (int i=h; i<h_base+X_out_width; i+=TILE_WIDTH){
		    for (int j=w; j<w_base+X_out_width; j+=TILE_WIDTH){
			    if (i<H && j<W){
				    X_shared[(i-h_base)*(X_out_width)+(j-w_base)]=x4d(n,c,i,j);
			    }
			    else{
				    X_shared[(i-h_base)*(X_out_width)+(j-w_base)]=0;
			    }
		    }
	    }
	    __syncthreads();
	    for (int p=0; p<K; p++){
		    for (int q=0; q<K; q++){
			    if(((h0+p) < X_out_width) && ((w0+q) < X_out_width)){
				    acc+=X_shared[(h0+p)*(X_out_width) + (w0+q)] * W_shared[p*K+q];
            // acc+=X_shared[(h0+p)*(X_out_width) + (w0+q)] *k4d(m, c, p, q);
			    }
		    }
	    }
	    __syncthreads();
    }
    if (n<B && m<M && h<H_out && w<W_out){
	    y4d(n,m,h,w)=acc;
    }

#undef y4d
#undef x4d
#undef k4d
}
//**************************************************************************
// Tiled shared memory convolution with constant
__global__ void conv_forward_kernel_tiled_constant(float *y, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K)
{
  const int H_out = H - K + 1;
  const int W_out = W - K + 1;

#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define k4d(i3, i2, i1, i0) Mask[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

  extern __shared__ float mem[];

  int bx = blockIdx.x;
  int by = blockIdx.y;
  int bz = blockIdx.z;
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int tile_width = blockDim.x;

  int tile_x = bz % (int)CEIL(W_out, (float)tile_width);
  int tile_y = bz / (int)CEIL(W_out, (float)tile_width);
  int x1 = tile_x * tile_width + tx;
  int y1 = tile_y * tile_width + ty;

  int b = bx;
  int m = by;

  for (int cc = 0; cc < C; cc++)
  {

      int SL = tile_width + K - 1;
      for (int tile_move_x = 0; tile_move_x < CEIL(SL, tile_width); tile_move_x++)
      {
          for (int tile_move_y = 0; tile_move_y < CEIL(SL, tile_width); tile_move_y++)
          {
              int xm = tile_move_x * tile_width + tx;
              int ym = tile_move_y * tile_width + ty;
              int xi = xm + tile_x * tile_width;
              int yi = ym + tile_y * tile_width;

              if (xm < SL && ym < SL)
              {
                  if (xi < W && yi < H)
                      mem[ym * SL + xm] =
                          x4d(b, cc, yi, xi);
                  else
                      mem[ym * SL + xm] = 0.0f;
              }
          }
      }

      __syncthreads();

      float result = 0;
      int xw = x1;
      int yw = y1;
      for (int i = 0; i < K; i++)
      {
          for (int j = 0; j < K; j++)
          {
              int x_load_sh_memo = tx + j;
              int y_load_sh_memo = ty + i;
              result += mem[y_load_sh_memo * SL + x_load_sh_memo] * k4d(m, cc, i, j);
          }
      }

      if (xw < W_out && yw < H_out)
      {
          if (cc == 0)
              y4d(b, m, yw, xw) = result;
          else
              y4d(b, m, yw, xw) += result;
      }

      __syncthreads();
      
}
#undef y4d
#undef x4d
#undef k4d
}


//**************************************************************************
//Weight matrix (kernel values) in constant memory

__global__ void conv_forward_kernel_weight_in_constant(float *y, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K)
{
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    

    #define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
    #define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
    #define k4d(i3, i2, i1, i0) Mask[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    // Insert your GPU convolution kernel code here
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int W_Grid = ceil(W_out/(TILE_WIDTH*1.0));
    int h = (blockIdx.z/W_Grid) * TILE_WIDTH +threadIdx.y;
    int w = (blockIdx.z%W_Grid) * TILE_WIDTH + threadIdx.x;
    if((w < W_out) && (h < H_out)){
        float acc = 0;
        for(int c=0; c<C; c++){
            for(int i =0; i<K; i++){
                for(int j=0; j< K; j++){
                    acc += x4d(bx, c, h+i, w+j) * k4d(by, c, i, j);
                }
            }
        }
        y4d(bx, by, h, w) = acc;
    }
#undef y4d
#undef x4d
#undef k4d
}
__global__ void conv_forward_kernel_weight_in_constant_TW3(float *y, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K)
{
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    

    #define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
    #define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
    #define k4d(i3, i2, i1, i0) Mask[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    // Insert your GPU convolution kernel code here
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int W_Grid = ceil(W_out/(TILE_WIDTH_3*1.0));
    int h = (blockIdx.z/W_Grid) * TILE_WIDTH_3 +threadIdx.y;
    int w = (blockIdx.z%W_Grid) * TILE_WIDTH_3 + threadIdx.x;
    if((w < W_out) && (h < H_out)){
        float acc = 0;
        for(int c=0; c<C; c++){
            for(int i =0; i<K; i++){
                for(int j=0; j< K; j++){
                    acc += x4d(bx, c, h+i, w+j) * k4d(by, c, i, j);
                }
            }
        }
        y4d(bx, by, h, w) = acc;
    }
#undef y4d
#undef x4d
#undef k4d
}
__global__ void conv_forward_kernel_weight_in_constant_TW4(float *y, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K)
{
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    

    #define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
    #define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
    #define k4d(i3, i2, i1, i0) Mask[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    // Insert your GPU convolution kernel code here
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int W_Grid = ceil(W_out/(TILE_WIDTH_4*1.0));
    int h = (blockIdx.z/W_Grid) * TILE_WIDTH_4 +threadIdx.y;
    int w = (blockIdx.z%W_Grid) * TILE_WIDTH_4 + threadIdx.x;
    if((w < W_out) && (h < H_out)){
        float acc = 0;
        for(int c=0; c<C; c++){
            for(int i =0; i<K; i++){
                for(int j=0; j< K; j++){
                    acc += x4d(bx, c, h+i, w+j) * k4d(by, c, i, j);
                }
            }
        }
        y4d(bx, by, h, w) = acc;
    }
#undef y4d
#undef x4d
#undef k4d
}

//**************************************************************************
// Kernel fusion for unrolling and matrix-multiplication (requires previous optimization) 
__global__ void conv_forward_kernel_unroll(float *y, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K){
  __shared__ float mask[TILE_WIDTH][TILE_WIDTH];
  __shared__ float input[TILE_WIDTH][TILE_WIDTH];

  const int H_out = H - K + 1;
  const int W_out = W - K + 1;

  #define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
  #define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
  #define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
  float acc = 0.0;  
  int bz = blockIdx.z;
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int y0 = blockIdx.y * TILE_WIDTH + ty;
  int x0 = blockIdx.x * TILE_WIDTH + tx;
  int unroll_col = C*K*K;
  int iter_num = ceil(unroll_col/(1.0*TILE_WIDTH));

  for (int i = 0; i < iter_num; i++) {
    int lx = i*TILE_WIDTH + tx;
    int ly = i*TILE_WIDTH + ty;
    int w0 = y0;
    int w1 = lx/(K*K);
    int w2 = (lx%(K*K))/K;
    int w3 = (lx%(K*K))%K;

    mask[ty][tx] = 0;
    input[ty][tx] = 0;

    if ((lx < unroll_col) && (y0 < M)){
      mask[ty][tx] = k4d(w0, w1, w2, w3);
    }
    else{
      mask[ty][tx] = 0;
    }

    int X_b = bz;
    int X_c = ly/(K*K);
    int X_p = (ly%(K*K))/K;
    int X_q = (ly%(K*K))%K;
    int X_h = x0/W_out;
    int X_w = x0%W_out;

    if (ly < unroll_col && x0 < H_out*W_out){
      input[ty][tx] = x4d(X_b, X_c, X_h + X_p, X_w + X_q);
    }
    else{
      input[ty][tx] = 0;
    }
    __syncthreads();

    for (int q = 0; q < TILE_WIDTH; q++){
      acc += mask[ty][q] * input[q][tx];
    }
    __syncthreads();
  }
  int Y_b = bz;
  int Y_m = y0;
  int Y_h = x0 / W_out;
  int Y_w = x0 % W_out;

  if (y0 < M && x0 < W_out*H_out){
        y4d(Y_b, Y_m, Y_h, Y_w) = acc;
  }
#undef y4d
#undef x4d
#undef k4d
}


//**************************************************************************
// Multiple kernels for different layers
__global__ void conv_forward_kernel_unroll_1(float *y, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K){
  __shared__ float mask[TILE_WIDTH_1][TILE_WIDTH_1];
  __shared__ float input[TILE_WIDTH_1][TILE_WIDTH_1];

  const int H_out = H - K + 1;
  const int W_out = W - K + 1;

  #define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
  #define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
  #define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
  float acc = 0.0;  
  int bz = blockIdx.z;
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int y0 = blockIdx.y * TILE_WIDTH_1 + ty;
  int x0 = blockIdx.x * TILE_WIDTH_1 + tx;
  int unroll_col = C*K*K;
  int iter_num = ceil(unroll_col/(1.0*TILE_WIDTH_1));

  for (int i = 0; i < iter_num; i++) {
    int lx = i*TILE_WIDTH_1 + tx;
    int ly = i*TILE_WIDTH_1 + ty;
    int w0 = y0;
    int w1 = lx/(K*K);
    int w2 = (lx%(K*K))/K;
    int w3 = (lx%(K*K))%K;

    mask[ty][tx] = 0;
    input[ty][tx] = 0;

    if ((lx < unroll_col) && (y0 < M)){
      mask[ty][tx] = k4d(w0, w1, w2, w3);
    }
    else{
      mask[ty][tx] = 0;
    }

    int X_b = bz;
    int X_c = ly/(K*K);
    int X_p = (ly%(K*K))/K;
    int X_q = (ly%(K*K))%K;
    int X_h = x0/W_out;
    int X_w = x0%W_out;

    if (ly < unroll_col && x0 < H_out*W_out){
      input[ty][tx] = x4d(X_b, X_c, X_h + X_p, X_w + X_q);
    }
    else{
      input[ty][tx] = 0;
    }
    __syncthreads();

    for (int q = 0; q < TILE_WIDTH_1; q++){
      acc += mask[ty][q] * input[q][tx];
    }


    __syncthreads();
  }
  int Y_b = bz;
  int Y_m = y0;
  int Y_h = x0 / W_out;
  int Y_w = x0 % W_out;

  if (y0 < M && x0 < W_out*H_out){
        y4d(Y_b, Y_m, Y_h, Y_w) = acc;
  }
#undef y4d
#undef x4d
#undef k4d
}
__global__ void conv_forward_kernel_unroll_2(float *y, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K){
  __shared__ float mask[TILE_WIDTH_2][TILE_WIDTH_2];
  __shared__ float input[TILE_WIDTH_2][TILE_WIDTH_2];

  const int H_out = H - K + 1;
  const int W_out = W - K + 1;

  #define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
  #define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
  #define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
  float acc = 0.0;  
  int bz = blockIdx.z;
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int y0 = blockIdx.y * TILE_WIDTH_2 + ty;
  int x0 = blockIdx.x * TILE_WIDTH_2 + tx;
  int unroll_col = C*K*K;
  int iter_num = ceil(unroll_col/(1.0*TILE_WIDTH_2));

  for (int i = 0; i < iter_num; i++) {
    int lx = i*TILE_WIDTH_2 + tx;
    int ly = i*TILE_WIDTH_2 + ty;
    int w0 = y0;
    int w1 = lx/(K*K);
    int w2 = (lx%(K*K))/K;
    int w3 = (lx%(K*K))%K;

    mask[ty][tx] = 0;
    input[ty][tx] = 0;

    if ((lx < unroll_col) && (y0 < M)){
      mask[ty][tx] = k4d(w0, w1, w2, w3);
    }
    else{
      mask[ty][tx] = 0;
    }

    int X_b = bz;
    int X_c = ly/(K*K);
    int X_p = (ly%(K*K))/K;
    int X_q = (ly%(K*K))%K;
    int X_h = x0/W_out;
    int X_w = x0%W_out;

    if (ly < unroll_col && x0 < H_out*W_out){
      input[ty][tx] = x4d(X_b, X_c, X_h + X_p, X_w + X_q);
    }
    else{
      input[ty][tx] = 0;
    }
    __syncthreads();

    for (int q = 0; q < TILE_WIDTH_2; q++){
      acc += mask[ty][q] * input[q][tx];
    }


    __syncthreads();
  }
  int Y_b = bz;
  int Y_m = y0;
  int Y_h = x0 / W_out;
  int Y_w = x0 % W_out;

  if (y0 < M && x0 < W_out*H_out){
        y4d(Y_b, Y_m, Y_h, Y_w) = acc;
  }
#undef y4d
#undef x4d
#undef k4d
}





__host__ void GPUInterface::conv_forward_gpu_prolog(float *host_y, const float *host_x, const float *host_k, 
    float **device_y_ptr, float **device_x_ptr, float **device_k_ptr, 
    const int B, const int M, const int C, const int H, const int W, const int K)
{
    //---------------basic------------------------------------------------------
    // // Allocate memory and copy over the relevant data structures to the GPU
    cudaMalloc((void**) device_y_ptr, B*M*(H-K+1)*(W-K+1)*sizeof(float));
    cudaMalloc((void**) device_x_ptr, B*C*H*W*sizeof(float));
    cudaMalloc((void**) device_k_ptr, M*C*K*K*sizeof(float));
    // // We pass double pointers for you to initialize the relevant device pointers,
    // //  which are passed to the other two functions.



    // cudaHostRegister((float *)host_k, M*C*K*K*sizeof(float), cudaHostAllocDefault);
    // cudaHostRegister((float *)host_x, B*C*H*W*sizeof(float), cudaHostAllocDefault);

    // cudaHostRegister(host_y, B*M*(H-K+1)*(W-K+1)*sizeof(float), cudaHostAllocDefault);



    
    cudaMemcpy(*device_x_ptr, host_x, B*C*H*W*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(*device_k_ptr, host_k, M*C*K*K*sizeof(float), cudaMemcpyHostToDevice);
    //---------------basic---------------------------------------------------------


    // //---------------using streams for data transfer-----------------------------------

    // int H_out = H - K + 1;
    // int W_out = W - K + 1;
    // int SegSize = 100;
    // int W_Grid = ceil(W_out/(TILE_WIDTH*1.0));
    // int H_Grid = ceil(H_out/(TILE_WIDTH*1.0));
    // int Z = W_Grid * H_Grid;
    // cudaStream_t stream0; 
    // cudaStream_t stream1; 
    // cudaStream_t stream2; 
    // cudaStream_t stream3; 

    // cudaStream_t stream4; 
    // cudaStream_t stream5; 
    // cudaStream_t stream6; 
    // cudaStream_t stream7;
    // cudaStream_t stream8;
    // cudaStream_t stream9;


    // cudaStreamCreate( &stream0);
    // cudaStreamCreate( &stream1);
    // cudaStreamCreate( &stream2);
    // cudaStreamCreate( &stream3);

    // cudaStreamCreate( &stream4);
    // cudaStreamCreate( &stream5);
    // cudaStreamCreate( &stream6);
    // cudaStreamCreate( &stream7);
    // cudaStreamCreate( &stream8);
    // cudaStreamCreate( &stream9);
    
    // int input_size = SegSize*C*H*W;
    // int output_size = SegSize*M*(H-K+1)*(W-K+1);

    // // // // //For tiled convolution
    // // // int out_size = TILE_WIDTH + K - 1;

    

    // // // // // //other-------
    // // dim3 dimGrid(SegSize, M, Z);
    // // // // // //unroll------
    // dim3 dimGrid(ceil(H_out*W_out/(1.0*TILE_WIDTH)), ceil(M/(1.0*TILE_WIDTH)), SegSize);

    // dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);






    // cudaMemcpyToSymbol(Mask, *device_k_ptr, sizeof(float) * M * C * K * K);
    
    // for(int i=0; i < B; i += 8*SegSize){
    //     cudaMemcpyAsync(*device_x_ptr + i*C*H*W, host_x + i*C*H*W, input_size*sizeof(float),cudaMemcpyHostToDevice, stream0);
    //     cudaMemcpyAsync(*device_x_ptr + (i+SegSize)*C*H*W, host_x + (i+SegSize)*C*H*W, input_size*sizeof(float), cudaMemcpyHostToDevice, stream1);
    //     cudaMemcpyAsync(*device_x_ptr + (i+SegSize*2)*C*H*W, host_x + (i+SegSize*2)*C*H*W, input_size*sizeof(float), cudaMemcpyHostToDevice, stream2);
    //     cudaMemcpyAsync(*device_x_ptr + (i+SegSize*3)*C*H*W, host_x + (i+SegSize*3)*C*H*W, input_size*sizeof(float), cudaMemcpyHostToDevice, stream3);
    //     cudaMemcpyAsync(*device_x_ptr + (i+SegSize*4)*C*H*W, host_x + (i+SegSize*4)*C*H*W, input_size*sizeof(float), cudaMemcpyHostToDevice, stream4);
    //     cudaMemcpyAsync(*device_x_ptr + (i+SegSize*5)*C*H*W, host_x + (i+SegSize*5)*C*H*W, input_size*sizeof(float), cudaMemcpyHostToDevice, stream5);
    //     cudaMemcpyAsync(*device_x_ptr + (i+SegSize*6)*C*H*W, host_x + (i+SegSize*6)*C*H*W, input_size*sizeof(float), cudaMemcpyHostToDevice, stream6);
    //     cudaMemcpyAsync(*device_x_ptr + (i+SegSize*7)*C*H*W, host_x + (i+SegSize*7)*C*H*W, input_size*sizeof(float), cudaMemcpyHostToDevice, stream7);

    //     conv_forward_kernel_unroll<<<dimGrid, dimBlock, 0, stream0>>>(*device_y_ptr + i*M*(H-K+1)*(W-K+1), *device_x_ptr + i*C*H*W, *device_k_ptr, B, M, C, H, W, K);
    //     conv_forward_kernel_unroll<<<dimGrid, dimBlock, 0, stream1>>>(*device_y_ptr + (i+SegSize*1)*M*(H-K+1)*(W-K+1), *device_x_ptr + (i+SegSize*1)*C*H*W, *device_k_ptr, B, M, C, H, W, K);
    //     conv_forward_kernel_unroll<<<dimGrid, dimBlock, 0, stream2>>>(*device_y_ptr + (i+SegSize*2)*M*(H-K+1)*(W-K+1), *device_x_ptr + (i+SegSize*2)*C*H*W, *device_k_ptr, B, M, C, H, W, K);
    //     conv_forward_kernel_unroll<<<dimGrid, dimBlock, 0, stream3>>>(*device_y_ptr + (i+SegSize*3)*M*(H-K+1)*(W-K+1), *device_x_ptr + (i+SegSize*3)*C*H*W, *device_k_ptr, B, M, C, H, W, K);
    //     conv_forward_kernel_unroll<<<dimGrid, dimBlock, 0, stream4>>>(*device_y_ptr + (i+SegSize*4)*M*(H-K+1)*(W-K+1), *device_x_ptr + (i+SegSize*4)*C*H*W, *device_k_ptr, B, M, C, H, W, K);
    //     conv_forward_kernel_unroll<<<dimGrid, dimBlock, 0, stream5>>>(*device_y_ptr + (i+SegSize*5)*M*(H-K+1)*(W-K+1), *device_x_ptr + (i+SegSize*5)*C*H*W, *device_k_ptr, B, M, C, H, W, K);
    //     conv_forward_kernel_unroll<<<dimGrid, dimBlock, 0, stream6>>>(*device_y_ptr + (i+SegSize*6)*M*(H-K+1)*(W-K+1), *device_x_ptr + (i+SegSize*6)*C*H*W, *device_k_ptr, B, M, C, H, W, K);
    //     conv_forward_kernel_unroll<<<dimGrid, dimBlock, 0, stream7>>>(*device_y_ptr + (i+SegSize*7)*M*(H-K+1)*(W-K+1), *device_x_ptr + (i+SegSize*7)*C*H*W, *device_k_ptr, B, M, C, H, W, K);

    //     cudaMemcpyAsync(host_y + i*M*(H-K+1)*(W-K+1), *device_y_ptr + i*M*(H-K+1)*(W-K+1), output_size*sizeof(float),cudaMemcpyDeviceToHost, stream0);
    //     cudaMemcpyAsync(host_y + (i+SegSize*1)*M*(H-K+1)*(W-K+1), *device_y_ptr + (i+SegSize*1)*M*(H-K+1)*(W-K+1), output_size*sizeof(float), cudaMemcpyDeviceToHost, stream1);
    //     cudaMemcpyAsync(host_y + (i+SegSize*2)*M*(H-K+1)*(W-K+1), *device_y_ptr + (i+SegSize*2)*M*(H-K+1)*(W-K+1), output_size*sizeof(float), cudaMemcpyDeviceToHost, stream2);
    //     cudaMemcpyAsync(host_y + (i+SegSize*3)*M*(H-K+1)*(W-K+1), *device_y_ptr + (i+SegSize*3)*M*(H-K+1)*(W-K+1), output_size*sizeof(float), cudaMemcpyDeviceToHost, stream3);
    //     cudaMemcpyAsync(host_y + (i+SegSize*4)*M*(H-K+1)*(W-K+1), *device_y_ptr + (i+SegSize*4)*M*(H-K+1)*(W-K+1), output_size*sizeof(float), cudaMemcpyDeviceToHost, stream4);
    //     cudaMemcpyAsync(host_y + (i+SegSize*5)*M*(H-K+1)*(W-K+1), *device_y_ptr + (i+SegSize*5)*M*(H-K+1)*(W-K+1), output_size*sizeof(float), cudaMemcpyDeviceToHost, stream5);
    //     cudaMemcpyAsync(host_y + (i+SegSize*6)*M*(H-K+1)*(W-K+1), *device_y_ptr + (i+SegSize*6)*M*(H-K+1)*(W-K+1), output_size*sizeof(float), cudaMemcpyDeviceToHost, stream6);
    //     cudaMemcpyAsync(host_y + (i+SegSize*7)*M*(H-K+1)*(W-K+1), *device_y_ptr + (i+SegSize*7)*M*(H-K+1)*(W-K+1), output_size*sizeof(float), cudaMemcpyDeviceToHost, stream7);    
    //   }
      






    // if(M == 4){
    //   dim3 dimBlock_1(TILE_WIDTH, TILE_WIDTH,1);
    //   dim3 dimGrid_1(SegSize, M, Z);
    //   for(int i=0; i < B; i += 10*SegSize){
    //     cudaMemcpyAsync(*device_x_ptr + i*C*H*W, host_x + i*C*H*W, input_size*sizeof(float),cudaMemcpyHostToDevice, stream0);
    //     cudaMemcpyAsync(*device_x_ptr + (i+SegSize)*C*H*W, host_x + (i+SegSize)*C*H*W, input_size*sizeof(float), cudaMemcpyHostToDevice, stream1);
    //     cudaMemcpyAsync(*device_x_ptr + (i+SegSize*2)*C*H*W, host_x + (i+SegSize*2)*C*H*W, input_size*sizeof(float), cudaMemcpyHostToDevice, stream2);
    //     cudaMemcpyAsync(*device_x_ptr + (i+SegSize*3)*C*H*W, host_x + (i+SegSize*3)*C*H*W, input_size*sizeof(float), cudaMemcpyHostToDevice, stream3);
        
    //     cudaMemcpyAsync(*device_x_ptr + (i+SegSize*4)*C*H*W, host_x + (i+SegSize*4)*C*H*W, input_size*sizeof(float), cudaMemcpyHostToDevice, stream4);
    //     cudaMemcpyAsync(*device_x_ptr + (i+SegSize*5)*C*H*W, host_x + (i+SegSize*5)*C*H*W, input_size*sizeof(float), cudaMemcpyHostToDevice, stream5);
    //     cudaMemcpyAsync(*device_x_ptr + (i+SegSize*6)*C*H*W, host_x + (i+SegSize*6)*C*H*W, input_size*sizeof(float), cudaMemcpyHostToDevice, stream6);
    //     cudaMemcpyAsync(*device_x_ptr + (i+SegSize*7)*C*H*W, host_x + (i+SegSize*7)*C*H*W, input_size*sizeof(float), cudaMemcpyHostToDevice, stream7);
    //     cudaMemcpyAsync(*device_x_ptr + (i+SegSize*8)*C*H*W, host_x + (i+SegSize*8)*C*H*W, input_size*sizeof(float), cudaMemcpyHostToDevice, stream8);
    //     cudaMemcpyAsync(*device_x_ptr + (i+SegSize*9)*C*H*W, host_x + (i+SegSize*9)*C*H*W, input_size*sizeof(float), cudaMemcpyHostToDevice, stream9);



    //     //----------------
        
    //     conv_forward_kernel_weight_in_constant<<<dimGrid_1, dimBlock_1, 0, stream0>>>(*device_y_ptr + i*M*(H-K+1)*(W-K+1), *device_x_ptr + i*C*H*W, *device_k_ptr, B, M, C, H, W, K);
    //     conv_forward_kernel_weight_in_constant<<<dimGrid_1, dimBlock_1, 0, stream1>>>(*device_y_ptr + (i+SegSize*1)*M*(H-K+1)*(W-K+1), *device_x_ptr + (i+SegSize*1)*C*H*W, *device_k_ptr, B, M, C, H, W, K);
    //     conv_forward_kernel_weight_in_constant<<<dimGrid_1, dimBlock_1, 0, stream2>>>(*device_y_ptr + (i+SegSize*2)*M*(H-K+1)*(W-K+1), *device_x_ptr + (i+SegSize*2)*C*H*W, *device_k_ptr, B, M, C, H, W, K);
    //     conv_forward_kernel_weight_in_constant<<<dimGrid_1, dimBlock_1, 0, stream3>>>(*device_y_ptr + (i+SegSize*3)*M*(H-K+1)*(W-K+1), *device_x_ptr + (i+SegSize*3)*C*H*W, *device_k_ptr, B, M, C, H, W, K);
        
    //     conv_forward_kernel_weight_in_constant<<<dimGrid_1, dimBlock_1, 0, stream4>>>(*device_y_ptr + (i+SegSize*4)*M*(H-K+1)*(W-K+1), *device_x_ptr + (i+SegSize*4)*C*H*W, *device_k_ptr, B, M, C, H, W, K);
    //     conv_forward_kernel_weight_in_constant<<<dimGrid_1, dimBlock_1, 0, stream5>>>(*device_y_ptr + (i+SegSize*5)*M*(H-K+1)*(W-K+1), *device_x_ptr + (i+SegSize*5)*C*H*W, *device_k_ptr, B, M, C, H, W, K);
    //     conv_forward_kernel_weight_in_constant<<<dimGrid_1, dimBlock_1, 0, stream6>>>(*device_y_ptr + (i+SegSize*6)*M*(H-K+1)*(W-K+1), *device_x_ptr + (i+SegSize*6)*C*H*W, *device_k_ptr, B, M, C, H, W, K);
    //     conv_forward_kernel_weight_in_constant<<<dimGrid_1, dimBlock_1, 0, stream7>>>(*device_y_ptr + (i+SegSize*7)*M*(H-K+1)*(W-K+1), *device_x_ptr + (i+SegSize*7)*C*H*W, *device_k_ptr, B, M, C, H, W, K);
    //     conv_forward_kernel_weight_in_constant<<<dimGrid_1, dimBlock_1, 0, stream8>>>(*device_y_ptr + (i+SegSize*8)*M*(H-K+1)*(W-K+1), *device_x_ptr + (i+SegSize*8)*C*H*W, *device_k_ptr, B, M, C, H, W, K);
    //     conv_forward_kernel_weight_in_constant<<<dimGrid_1, dimBlock_1, 0, stream9>>>(*device_y_ptr + (i+SegSize*9)*M*(H-K+1)*(W-K+1), *device_x_ptr + (i+SegSize*9)*C*H*W, *device_k_ptr, B, M, C, H, W, K);


        
        
    //     //----------------

    //     cudaMemcpyAsync(host_y + i*M*(H-K+1)*(W-K+1), *device_y_ptr + i*M*(H-K+1)*(W-K+1), output_size*sizeof(float),cudaMemcpyDeviceToHost, stream0);
    //     cudaMemcpyAsync(host_y + (i+SegSize*1)*M*(H-K+1)*(W-K+1), *device_y_ptr + (i+SegSize*1)*M*(H-K+1)*(W-K+1), output_size*sizeof(float), cudaMemcpyDeviceToHost, stream1);
    //     cudaMemcpyAsync(host_y + (i+SegSize*2)*M*(H-K+1)*(W-K+1), *device_y_ptr + (i+SegSize*2)*M*(H-K+1)*(W-K+1), output_size*sizeof(float), cudaMemcpyDeviceToHost, stream2);
    //     cudaMemcpyAsync(host_y + (i+SegSize*3)*M*(H-K+1)*(W-K+1), *device_y_ptr + (i+SegSize*3)*M*(H-K+1)*(W-K+1), output_size*sizeof(float), cudaMemcpyDeviceToHost, stream3);
        
    //     cudaMemcpyAsync(host_y + (i+SegSize*4)*M*(H-K+1)*(W-K+1), *device_y_ptr + (i+SegSize*4)*M*(H-K+1)*(W-K+1), output_size*sizeof(float), cudaMemcpyDeviceToHost, stream4);
    //     cudaMemcpyAsync(host_y + (i+SegSize*5)*M*(H-K+1)*(W-K+1), *device_y_ptr + (i+SegSize*5)*M*(H-K+1)*(W-K+1), output_size*sizeof(float), cudaMemcpyDeviceToHost, stream5);
    //     cudaMemcpyAsync(host_y + (i+SegSize*6)*M*(H-K+1)*(W-K+1), *device_y_ptr + (i+SegSize*6)*M*(H-K+1)*(W-K+1), output_size*sizeof(float), cudaMemcpyDeviceToHost, stream6);
    //     cudaMemcpyAsync(host_y + (i+SegSize*7)*M*(H-K+1)*(W-K+1), *device_y_ptr + (i+SegSize*7)*M*(H-K+1)*(W-K+1), output_size*sizeof(float), cudaMemcpyDeviceToHost, stream7);  
        
    //     cudaMemcpyAsync(host_y + (i+SegSize*8)*M*(H-K+1)*(W-K+1), *device_y_ptr + (i+SegSize*8)*M*(H-K+1)*(W-K+1), output_size*sizeof(float), cudaMemcpyDeviceToHost, stream8);
    //     cudaMemcpyAsync(host_y + (i+SegSize*9)*M*(H-K+1)*(W-K+1), *device_y_ptr + (i+SegSize*9)*M*(H-K+1)*(W-K+1), output_size*sizeof(float), cudaMemcpyDeviceToHost, stream9); 
    //   }

    // }else{
    //   dim3 dimBlock_2(TILE_WIDTH_3, TILE_WIDTH_3,1);
    //   dim3 dimGrid_2(SegSize, M, Z);
    //   for(int i=0; i < B; i += 10*SegSize){
    //     cudaMemcpyAsync(*device_x_ptr + i*C*H*W, host_x + i*C*H*W, input_size*sizeof(float),cudaMemcpyHostToDevice, stream0);
    //     cudaMemcpyAsync(*device_x_ptr + (i+SegSize)*C*H*W, host_x + (i+SegSize)*C*H*W, input_size*sizeof(float), cudaMemcpyHostToDevice, stream1);
    //     cudaMemcpyAsync(*device_x_ptr + (i+SegSize*2)*C*H*W, host_x + (i+SegSize*2)*C*H*W, input_size*sizeof(float), cudaMemcpyHostToDevice, stream2);
    //     cudaMemcpyAsync(*device_x_ptr + (i+SegSize*3)*C*H*W, host_x + (i+SegSize*3)*C*H*W, input_size*sizeof(float), cudaMemcpyHostToDevice, stream3);
        
    //     cudaMemcpyAsync(*device_x_ptr + (i+SegSize*4)*C*H*W, host_x + (i+SegSize*4)*C*H*W, input_size*sizeof(float), cudaMemcpyHostToDevice, stream4);
    //     cudaMemcpyAsync(*device_x_ptr + (i+SegSize*5)*C*H*W, host_x + (i+SegSize*5)*C*H*W, input_size*sizeof(float), cudaMemcpyHostToDevice, stream5);
    //     cudaMemcpyAsync(*device_x_ptr + (i+SegSize*6)*C*H*W, host_x + (i+SegSize*6)*C*H*W, input_size*sizeof(float), cudaMemcpyHostToDevice, stream6);
    //     cudaMemcpyAsync(*device_x_ptr + (i+SegSize*7)*C*H*W, host_x + (i+SegSize*7)*C*H*W, input_size*sizeof(float), cudaMemcpyHostToDevice, stream7);

    //     cudaMemcpyAsync(*device_x_ptr + (i+SegSize*8)*C*H*W, host_x + (i+SegSize*8)*C*H*W, input_size*sizeof(float), cudaMemcpyHostToDevice, stream8);
    //     cudaMemcpyAsync(*device_x_ptr + (i+SegSize*9)*C*H*W, host_x + (i+SegSize*9)*C*H*W, input_size*sizeof(float), cudaMemcpyHostToDevice, stream9);


    //     //----------------
    //     conv_forward_kernel_weight_in_constant_TW3<<<dimGrid_2, dimBlock_2, 0, stream0>>>(*device_y_ptr + i*M*(H-K+1)*(W-K+1), *device_x_ptr + i*C*H*W, *device_k_ptr, B, M, C, H, W, K);
    //     conv_forward_kernel_weight_in_constant_TW3<<<dimGrid_2, dimBlock_2, 0, stream1>>>(*device_y_ptr + (i+SegSize*1)*M*(H-K+1)*(W-K+1), *device_x_ptr + (i+SegSize*1)*C*H*W, *device_k_ptr, B, M, C, H, W, K);
    //     conv_forward_kernel_weight_in_constant_TW3<<<dimGrid_2, dimBlock_2, 0, stream2>>>(*device_y_ptr + (i+SegSize*2)*M*(H-K+1)*(W-K+1), *device_x_ptr + (i+SegSize*2)*C*H*W, *device_k_ptr, B, M, C, H, W, K);
    //     conv_forward_kernel_weight_in_constant_TW3<<<dimGrid_2, dimBlock_2, 0, stream3>>>(*device_y_ptr + (i+SegSize*3)*M*(H-K+1)*(W-K+1), *device_x_ptr + (i+SegSize*3)*C*H*W, *device_k_ptr, B, M, C, H, W, K);
        
    //     conv_forward_kernel_weight_in_constant_TW3<<<dimGrid_2, dimBlock_2, 0, stream4>>>(*device_y_ptr + (i+SegSize*4)*M*(H-K+1)*(W-K+1), *device_x_ptr + (i+SegSize*4)*C*H*W, *device_k_ptr, B, M, C, H, W, K);
    //     conv_forward_kernel_weight_in_constant_TW3<<<dimGrid_2, dimBlock_2, 0, stream5>>>(*device_y_ptr + (i+SegSize*5)*M*(H-K+1)*(W-K+1), *device_x_ptr + (i+SegSize*5)*C*H*W, *device_k_ptr, B, M, C, H, W, K);
    //     conv_forward_kernel_weight_in_constant_TW3<<<dimGrid_2, dimBlock_2, 0, stream6>>>(*device_y_ptr + (i+SegSize*6)*M*(H-K+1)*(W-K+1), *device_x_ptr + (i+SegSize*6)*C*H*W, *device_k_ptr, B, M, C, H, W, K);
    //     conv_forward_kernel_weight_in_constant_TW3<<<dimGrid_2, dimBlock_2, 0, stream7>>>(*device_y_ptr + (i+SegSize*7)*M*(H-K+1)*(W-K+1), *device_x_ptr + (i+SegSize*7)*C*H*W, *device_k_ptr, B, M, C, H, W, K);

    //     conv_forward_kernel_weight_in_constant_TW3<<<dimGrid_2, dimBlock_2, 0, stream8>>>(*device_y_ptr + (i+SegSize*8)*M*(H-K+1)*(W-K+1), *device_x_ptr + (i+SegSize*8)*C*H*W, *device_k_ptr, B, M, C, H, W, K);
    //     conv_forward_kernel_weight_in_constant_TW3<<<dimGrid_2, dimBlock_2, 0, stream9>>>(*device_y_ptr + (i+SegSize*9)*M*(H-K+1)*(W-K+1), *device_x_ptr + (i+SegSize*9)*C*H*W, *device_k_ptr, B, M, C, H, W, K);
       
    //     //----------------
    //     cudaMemcpyAsync(host_y + i*M*(H-K+1)*(W-K+1), *device_y_ptr + i*M*(H-K+1)*(W-K+1), output_size*sizeof(float),cudaMemcpyDeviceToHost, stream0);
    //     cudaMemcpyAsync(host_y + (i+SegSize*1)*M*(H-K+1)*(W-K+1), *device_y_ptr + (i+SegSize*1)*M*(H-K+1)*(W-K+1), output_size*sizeof(float), cudaMemcpyDeviceToHost, stream1);
    //     cudaMemcpyAsync(host_y + (i+SegSize*2)*M*(H-K+1)*(W-K+1), *device_y_ptr + (i+SegSize*2)*M*(H-K+1)*(W-K+1), output_size*sizeof(float), cudaMemcpyDeviceToHost, stream2);
    //     cudaMemcpyAsync(host_y + (i+SegSize*3)*M*(H-K+1)*(W-K+1), *device_y_ptr + (i+SegSize*3)*M*(H-K+1)*(W-K+1), output_size*sizeof(float), cudaMemcpyDeviceToHost, stream3);
        
    //     cudaMemcpyAsync(host_y + (i+SegSize*4)*M*(H-K+1)*(W-K+1), *device_y_ptr + (i+SegSize*4)*M*(H-K+1)*(W-K+1), output_size*sizeof(float), cudaMemcpyDeviceToHost, stream4);
    //     cudaMemcpyAsync(host_y + (i+SegSize*5)*M*(H-K+1)*(W-K+1), *device_y_ptr + (i+SegSize*5)*M*(H-K+1)*(W-K+1), output_size*sizeof(float), cudaMemcpyDeviceToHost, stream5);
    //     cudaMemcpyAsync(host_y + (i+SegSize*6)*M*(H-K+1)*(W-K+1), *device_y_ptr + (i+SegSize*6)*M*(H-K+1)*(W-K+1), output_size*sizeof(float), cudaMemcpyDeviceToHost, stream6);
    //     cudaMemcpyAsync(host_y + (i+SegSize*7)*M*(H-K+1)*(W-K+1), *device_y_ptr + (i+SegSize*7)*M*(H-K+1)*(W-K+1), output_size*sizeof(float), cudaMemcpyDeviceToHost, stream7);   
        
    //     cudaMemcpyAsync(host_y + (i+SegSize*8)*M*(H-K+1)*(W-K+1), *device_y_ptr + (i+SegSize*8)*M*(H-K+1)*(W-K+1), output_size*sizeof(float), cudaMemcpyDeviceToHost, stream8);
    //     cudaMemcpyAsync(host_y + (i+SegSize*9)*M*(H-K+1)*(W-K+1), *device_y_ptr + (i+SegSize*9)*M*(H-K+1)*(W-K+1), output_size*sizeof(float), cudaMemcpyDeviceToHost, stream9); 
    //   }
    // }


















    // if(M == 4){
    //   dim3 dimBlock_1(TILE_WIDTH_1, TILE_WIDTH_1,1);
    //   dim3 dimGrid_1(ceil(H_out*W_out/(1.0*TILE_WIDTH)), ceil(M/(1.0*TILE_WIDTH)), SegSize);
    //   for(int i=0; i < B; i += 10*SegSize){
    //     cudaMemcpyAsync(*device_x_ptr + i*C*H*W, host_x + i*C*H*W, input_size*sizeof(float),cudaMemcpyHostToDevice, stream0);
    //     cudaMemcpyAsync(*device_x_ptr + (i+SegSize)*C*H*W, host_x + (i+SegSize)*C*H*W, input_size*sizeof(float), cudaMemcpyHostToDevice, stream1);
    //     cudaMemcpyAsync(*device_x_ptr + (i+SegSize*2)*C*H*W, host_x + (i+SegSize*2)*C*H*W, input_size*sizeof(float), cudaMemcpyHostToDevice, stream2);
    //     cudaMemcpyAsync(*device_x_ptr + (i+SegSize*3)*C*H*W, host_x + (i+SegSize*3)*C*H*W, input_size*sizeof(float), cudaMemcpyHostToDevice, stream3);
        
    //     cudaMemcpyAsync(*device_x_ptr + (i+SegSize*4)*C*H*W, host_x + (i+SegSize*4)*C*H*W, input_size*sizeof(float), cudaMemcpyHostToDevice, stream4);
    //     cudaMemcpyAsync(*device_x_ptr + (i+SegSize*5)*C*H*W, host_x + (i+SegSize*5)*C*H*W, input_size*sizeof(float), cudaMemcpyHostToDevice, stream5);
    //     cudaMemcpyAsync(*device_x_ptr + (i+SegSize*6)*C*H*W, host_x + (i+SegSize*6)*C*H*W, input_size*sizeof(float), cudaMemcpyHostToDevice, stream6);
    //     cudaMemcpyAsync(*device_x_ptr + (i+SegSize*7)*C*H*W, host_x + (i+SegSize*7)*C*H*W, input_size*sizeof(float), cudaMemcpyHostToDevice, stream7);
    //     cudaMemcpyAsync(*device_x_ptr + (i+SegSize*8)*C*H*W, host_x + (i+SegSize*8)*C*H*W, input_size*sizeof(float), cudaMemcpyHostToDevice, stream8);
    //     cudaMemcpyAsync(*device_x_ptr + (i+SegSize*9)*C*H*W, host_x + (i+SegSize*9)*C*H*W, input_size*sizeof(float), cudaMemcpyHostToDevice, stream9);



    //     //----------------
        
    //     conv_forward_kernel_unroll_1<<<dimGrid_1, dimBlock_1, 0, stream0>>>(*device_y_ptr + i*M*(H-K+1)*(W-K+1), *device_x_ptr + i*C*H*W, *device_k_ptr, B, M, C, H, W, K);
    //     conv_forward_kernel_unroll_1<<<dimGrid_1, dimBlock_1, 0, stream1>>>(*device_y_ptr + (i+SegSize*1)*M*(H-K+1)*(W-K+1), *device_x_ptr + (i+SegSize*1)*C*H*W, *device_k_ptr, B, M, C, H, W, K);
    //     conv_forward_kernel_unroll_1<<<dimGrid_1, dimBlock_1, 0, stream2>>>(*device_y_ptr + (i+SegSize*2)*M*(H-K+1)*(W-K+1), *device_x_ptr + (i+SegSize*2)*C*H*W, *device_k_ptr, B, M, C, H, W, K);
    //     conv_forward_kernel_unroll_1<<<dimGrid_1, dimBlock_1, 0, stream3>>>(*device_y_ptr + (i+SegSize*3)*M*(H-K+1)*(W-K+1), *device_x_ptr + (i+SegSize*3)*C*H*W, *device_k_ptr, B, M, C, H, W, K);
        
    //     conv_forward_kernel_unroll_1<<<dimGrid_1, dimBlock_1, 0, stream4>>>(*device_y_ptr + (i+SegSize*4)*M*(H-K+1)*(W-K+1), *device_x_ptr + (i+SegSize*4)*C*H*W, *device_k_ptr, B, M, C, H, W, K);
    //     conv_forward_kernel_unroll_1<<<dimGrid_1, dimBlock_1, 0, stream5>>>(*device_y_ptr + (i+SegSize*5)*M*(H-K+1)*(W-K+1), *device_x_ptr + (i+SegSize*5)*C*H*W, *device_k_ptr, B, M, C, H, W, K);
    //     conv_forward_kernel_unroll_1<<<dimGrid_1, dimBlock_1, 0, stream6>>>(*device_y_ptr + (i+SegSize*6)*M*(H-K+1)*(W-K+1), *device_x_ptr + (i+SegSize*6)*C*H*W, *device_k_ptr, B, M, C, H, W, K);
    //     conv_forward_kernel_unroll_1<<<dimGrid_1, dimBlock_1, 0, stream7>>>(*device_y_ptr + (i+SegSize*7)*M*(H-K+1)*(W-K+1), *device_x_ptr + (i+SegSize*7)*C*H*W, *device_k_ptr, B, M, C, H, W, K);
    //     conv_forward_kernel_unroll_1<<<dimGrid_1, dimBlock_1, 0, stream8>>>(*device_y_ptr + (i+SegSize*8)*M*(H-K+1)*(W-K+1), *device_x_ptr + (i+SegSize*8)*C*H*W, *device_k_ptr, B, M, C, H, W, K);
    //     conv_forward_kernel_unroll_1<<<dimGrid_1, dimBlock_1, 0, stream9>>>(*device_y_ptr + (i+SegSize*9)*M*(H-K+1)*(W-K+1), *device_x_ptr + (i+SegSize*9)*C*H*W, *device_k_ptr, B, M, C, H, W, K);


        
        
    //     //----------------

    //     cudaMemcpyAsync(host_y + i*M*(H-K+1)*(W-K+1), *device_y_ptr + i*M*(H-K+1)*(W-K+1), output_size*sizeof(float),cudaMemcpyDeviceToHost, stream0);
    //     cudaMemcpyAsync(host_y + (i+SegSize*1)*M*(H-K+1)*(W-K+1), *device_y_ptr + (i+SegSize*1)*M*(H-K+1)*(W-K+1), output_size*sizeof(float), cudaMemcpyDeviceToHost, stream1);
    //     cudaMemcpyAsync(host_y + (i+SegSize*2)*M*(H-K+1)*(W-K+1), *device_y_ptr + (i+SegSize*2)*M*(H-K+1)*(W-K+1), output_size*sizeof(float), cudaMemcpyDeviceToHost, stream2);
    //     cudaMemcpyAsync(host_y + (i+SegSize*3)*M*(H-K+1)*(W-K+1), *device_y_ptr + (i+SegSize*3)*M*(H-K+1)*(W-K+1), output_size*sizeof(float), cudaMemcpyDeviceToHost, stream3);
        
    //     cudaMemcpyAsync(host_y + (i+SegSize*4)*M*(H-K+1)*(W-K+1), *device_y_ptr + (i+SegSize*4)*M*(H-K+1)*(W-K+1), output_size*sizeof(float), cudaMemcpyDeviceToHost, stream4);
    //     cudaMemcpyAsync(host_y + (i+SegSize*5)*M*(H-K+1)*(W-K+1), *device_y_ptr + (i+SegSize*5)*M*(H-K+1)*(W-K+1), output_size*sizeof(float), cudaMemcpyDeviceToHost, stream5);
    //     cudaMemcpyAsync(host_y + (i+SegSize*6)*M*(H-K+1)*(W-K+1), *device_y_ptr + (i+SegSize*6)*M*(H-K+1)*(W-K+1), output_size*sizeof(float), cudaMemcpyDeviceToHost, stream6);
    //     cudaMemcpyAsync(host_y + (i+SegSize*7)*M*(H-K+1)*(W-K+1), *device_y_ptr + (i+SegSize*7)*M*(H-K+1)*(W-K+1), output_size*sizeof(float), cudaMemcpyDeviceToHost, stream7);  
        
    //     cudaMemcpyAsync(host_y + (i+SegSize*8)*M*(H-K+1)*(W-K+1), *device_y_ptr + (i+SegSize*8)*M*(H-K+1)*(W-K+1), output_size*sizeof(float), cudaMemcpyDeviceToHost, stream8);
    //     cudaMemcpyAsync(host_y + (i+SegSize*9)*M*(H-K+1)*(W-K+1), *device_y_ptr + (i+SegSize*9)*M*(H-K+1)*(W-K+1), output_size*sizeof(float), cudaMemcpyDeviceToHost, stream9); 
    //   }

    // }else{
    //   dim3 dimBlock_2(TILE_WIDTH_2, TILE_WIDTH_2,1);
    //   dim3 dimGrid_2(ceil(H_out*W_out/(1.0*TILE_WIDTH_2)), ceil(M/(1.0*TILE_WIDTH_2)), SegSize);
    //   for(int i=0; i < B; i += 10*SegSize){
    //     cudaMemcpyAsync(*device_x_ptr + i*C*H*W, host_x + i*C*H*W, input_size*sizeof(float),cudaMemcpyHostToDevice, stream0);
    //     cudaMemcpyAsync(*device_x_ptr + (i+SegSize)*C*H*W, host_x + (i+SegSize)*C*H*W, input_size*sizeof(float), cudaMemcpyHostToDevice, stream1);
    //     cudaMemcpyAsync(*device_x_ptr + (i+SegSize*2)*C*H*W, host_x + (i+SegSize*2)*C*H*W, input_size*sizeof(float), cudaMemcpyHostToDevice, stream2);
    //     cudaMemcpyAsync(*device_x_ptr + (i+SegSize*3)*C*H*W, host_x + (i+SegSize*3)*C*H*W, input_size*sizeof(float), cudaMemcpyHostToDevice, stream3);
        
    //     cudaMemcpyAsync(*device_x_ptr + (i+SegSize*4)*C*H*W, host_x + (i+SegSize*4)*C*H*W, input_size*sizeof(float), cudaMemcpyHostToDevice, stream4);
    //     cudaMemcpyAsync(*device_x_ptr + (i+SegSize*5)*C*H*W, host_x + (i+SegSize*5)*C*H*W, input_size*sizeof(float), cudaMemcpyHostToDevice, stream5);
    //     cudaMemcpyAsync(*device_x_ptr + (i+SegSize*6)*C*H*W, host_x + (i+SegSize*6)*C*H*W, input_size*sizeof(float), cudaMemcpyHostToDevice, stream6);
    //     cudaMemcpyAsync(*device_x_ptr + (i+SegSize*7)*C*H*W, host_x + (i+SegSize*7)*C*H*W, input_size*sizeof(float), cudaMemcpyHostToDevice, stream7);

    //     cudaMemcpyAsync(*device_x_ptr + (i+SegSize*8)*C*H*W, host_x + (i+SegSize*8)*C*H*W, input_size*sizeof(float), cudaMemcpyHostToDevice, stream8);
    //     cudaMemcpyAsync(*device_x_ptr + (i+SegSize*9)*C*H*W, host_x + (i+SegSize*9)*C*H*W, input_size*sizeof(float), cudaMemcpyHostToDevice, stream9);


    //     //----------------
    //     conv_forward_kernel_unroll_2<<<dimGrid_2, dimBlock_2, 0, stream0>>>(*device_y_ptr + i*M*(H-K+1)*(W-K+1), *device_x_ptr + i*C*H*W, *device_k_ptr, B, M, C, H, W, K);
    //     conv_forward_kernel_unroll_2<<<dimGrid_2, dimBlock_2, 0, stream1>>>(*device_y_ptr + (i+SegSize*1)*M*(H-K+1)*(W-K+1), *device_x_ptr + (i+SegSize*1)*C*H*W, *device_k_ptr, B, M, C, H, W, K);
    //     conv_forward_kernel_unroll_2<<<dimGrid_2, dimBlock_2, 0, stream2>>>(*device_y_ptr + (i+SegSize*2)*M*(H-K+1)*(W-K+1), *device_x_ptr + (i+SegSize*2)*C*H*W, *device_k_ptr, B, M, C, H, W, K);
    //     conv_forward_kernel_unroll_2<<<dimGrid_2, dimBlock_2, 0, stream3>>>(*device_y_ptr + (i+SegSize*3)*M*(H-K+1)*(W-K+1), *device_x_ptr + (i+SegSize*3)*C*H*W, *device_k_ptr, B, M, C, H, W, K);
        
    //     conv_forward_kernel_unroll_2<<<dimGrid_2, dimBlock_2, 0, stream4>>>(*device_y_ptr + (i+SegSize*4)*M*(H-K+1)*(W-K+1), *device_x_ptr + (i+SegSize*4)*C*H*W, *device_k_ptr, B, M, C, H, W, K);
    //     conv_forward_kernel_unroll_2<<<dimGrid_2, dimBlock_2, 0, stream5>>>(*device_y_ptr + (i+SegSize*5)*M*(H-K+1)*(W-K+1), *device_x_ptr + (i+SegSize*5)*C*H*W, *device_k_ptr, B, M, C, H, W, K);
    //     conv_forward_kernel_unroll_2<<<dimGrid_2, dimBlock_2, 0, stream6>>>(*device_y_ptr + (i+SegSize*6)*M*(H-K+1)*(W-K+1), *device_x_ptr + (i+SegSize*6)*C*H*W, *device_k_ptr, B, M, C, H, W, K);
    //     conv_forward_kernel_unroll_2<<<dimGrid_2, dimBlock_2, 0, stream7>>>(*device_y_ptr + (i+SegSize*7)*M*(H-K+1)*(W-K+1), *device_x_ptr + (i+SegSize*7)*C*H*W, *device_k_ptr, B, M, C, H, W, K);

    //     conv_forward_kernel_unroll_2<<<dimGrid_2, dimBlock_2, 0, stream8>>>(*device_y_ptr + (i+SegSize*8)*M*(H-K+1)*(W-K+1), *device_x_ptr + (i+SegSize*8)*C*H*W, *device_k_ptr, B, M, C, H, W, K);
    //     conv_forward_kernel_unroll_2<<<dimGrid_2, dimBlock_2, 0, stream9>>>(*device_y_ptr + (i+SegSize*9)*M*(H-K+1)*(W-K+1), *device_x_ptr + (i+SegSize*9)*C*H*W, *device_k_ptr, B, M, C, H, W, K);
       
    //     //----------------
    //     cudaMemcpyAsync(host_y + i*M*(H-K+1)*(W-K+1), *device_y_ptr + i*M*(H-K+1)*(W-K+1), output_size*sizeof(float),cudaMemcpyDeviceToHost, stream0);
    //     cudaMemcpyAsync(host_y + (i+SegSize*1)*M*(H-K+1)*(W-K+1), *device_y_ptr + (i+SegSize*1)*M*(H-K+1)*(W-K+1), output_size*sizeof(float), cudaMemcpyDeviceToHost, stream1);
    //     cudaMemcpyAsync(host_y + (i+SegSize*2)*M*(H-K+1)*(W-K+1), *device_y_ptr + (i+SegSize*2)*M*(H-K+1)*(W-K+1), output_size*sizeof(float), cudaMemcpyDeviceToHost, stream2);
    //     cudaMemcpyAsync(host_y + (i+SegSize*3)*M*(H-K+1)*(W-K+1), *device_y_ptr + (i+SegSize*3)*M*(H-K+1)*(W-K+1), output_size*sizeof(float), cudaMemcpyDeviceToHost, stream3);
        
    //     cudaMemcpyAsync(host_y + (i+SegSize*4)*M*(H-K+1)*(W-K+1), *device_y_ptr + (i+SegSize*4)*M*(H-K+1)*(W-K+1), output_size*sizeof(float), cudaMemcpyDeviceToHost, stream4);
    //     cudaMemcpyAsync(host_y + (i+SegSize*5)*M*(H-K+1)*(W-K+1), *device_y_ptr + (i+SegSize*5)*M*(H-K+1)*(W-K+1), output_size*sizeof(float), cudaMemcpyDeviceToHost, stream5);
    //     cudaMemcpyAsync(host_y + (i+SegSize*6)*M*(H-K+1)*(W-K+1), *device_y_ptr + (i+SegSize*6)*M*(H-K+1)*(W-K+1), output_size*sizeof(float), cudaMemcpyDeviceToHost, stream6);
    //     cudaMemcpyAsync(host_y + (i+SegSize*7)*M*(H-K+1)*(W-K+1), *device_y_ptr + (i+SegSize*7)*M*(H-K+1)*(W-K+1), output_size*sizeof(float), cudaMemcpyDeviceToHost, stream7);   
        
    //     cudaMemcpyAsync(host_y + (i+SegSize*8)*M*(H-K+1)*(W-K+1), *device_y_ptr + (i+SegSize*8)*M*(H-K+1)*(W-K+1), output_size*sizeof(float), cudaMemcpyDeviceToHost, stream8);
    //     cudaMemcpyAsync(host_y + (i+SegSize*9)*M*(H-K+1)*(W-K+1), *device_y_ptr + (i+SegSize*9)*M*(H-K+1)*(W-K+1), output_size*sizeof(float), cudaMemcpyDeviceToHost, stream9); 
    //   }
    // }
  

    
    //---------------using streams for data transfer-----------------------------------



    
    // Useful snippet for error checking
    // cudaError_t error = cudaGetLastError();
    // if(error != cudaSuccess)
    // {
    //     std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;
    //     exit(-1);
    // }
}

//------------host----------------------------------
__host__ void GPUInterface::conv_forward_gpu(float *device_y, const float *device_x, const float *device_k, 
    const int B, const int M, const int C, const int H, const int W, const int K)
{
    // Set the kernel dimensions and call the kernel

    //------------------------------------------------------------------
    int H_out = H - K + 1;
    int W_out = W - K + 1;
    int W_Grid = ceil(W_out/(TILE_WIDTH*1.0));
    int H_Grid = ceil(H_out/(TILE_WIDTH*1.0));
    int Z = W_Grid * H_Grid;
    dim3 dimGrid(B, M, Z);
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
    //------------------------------------------------------------------
    
    //1. Basic
    // conv_forward_kernel<<<dimGrid, dimBlock>>>(device_y, device_x, device_k, B, M, C, H, W, K);

    //2. Tiled shared memory convolution
    // int out_size = TILE_WIDTH + K - 1;
    // cudaMemcpyToSymbol(Mask, device_k, sizeof(float) * M * C * K * K);
    // conv_forward_kernel_tiled_constant<<<dimGrid, dimBlock, (sizeof(float) * (out_size*out_size + K*K))>>>(device_y, device_x, device_k, B, M, C, H, W, K);
    //**********************************

    //3.weight in constant
    cudaMemcpyToSymbol(Mask, device_k, sizeof(float) * M * C * K * K);
    // conv_forward_kernel_weight_in_constant<<<dimGrid, dimBlock>>>(device_y, device_x, device_k, B, M, C, H, W, K);

    //4. Shared memory matrix multiplication and input matrix unrolling
    //changed to kernel fusion


    //5. Kernel fusion for unrolling and matrix-multiplication
    // dim3 dimGrid_4(ceil(H_out*W_out/(1.0*TILE_WIDTH)), ceil(M/(1.0*TILE_WIDTH)), B);
    // conv_forward_kernel_unroll<<<dimGrid_4, dimBlock>>>(device_y, device_x, device_k, B, M, C, H, W, K);

    // 6. Multiple kernel implementations for different layer sizes
    if(M == 4){
        dim3 dimBlock_1(TILE_WIDTH_3, TILE_WIDTH_3,1);
        dim3 dimGrid_1(B, M, Z);
        conv_forward_kernel_weight_in_constant_TW3<<<dimGrid_1, dimBlock_1>>>(device_y, device_x, device_k, B, M, C, H, W, K);
    }else{
      dim3 dimGrid_4(ceil(H_out*W_out/(1.0*TILE_WIDTH)), ceil(M/(1.0*TILE_WIDTH)), B);
      conv_forward_kernel_unroll<<<dimGrid_4, dimBlock>>>(device_y, device_x, device_k, B, M, C, H, W, K);
  

    }

    //6. Using Streams to overlap computation with data transfer
    //Shown in the function  conv_forward_gpu_prolog




    //loat *y, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K
}
//------------------------------------------------------------

__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_y, float *device_y, float *device_x, float *device_k, 
    const int B, const int M, const int C, const int H, const int W, const int K)
{
    // Copy the output back to host
    //delete this line when using overlapping kernel excution.

    //comment when using overlap
    cudaMemcpy(host_y, device_y, B*M*(H-K+1)*(W-K+1)*sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(device_k);
    cudaFree(device_x);
    cudaFree(device_y);
}


__host__ void GPUInterface::get_device_properties()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for(int dev = 0; dev < deviceCount; dev++)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        std::cout<<"Device "<<dev<<" name: "<<deviceProp.name<<std::endl;
        std::cout<<"Computational capabilities: "<<deviceProp.major<<"."<<deviceProp.minor<<std::endl;
        std::cout<<"Max Global memory size: "<<deviceProp.totalGlobalMem<<std::endl;
        std::cout<<"Max Constant memory size: "<<deviceProp.totalConstMem<<std::endl;
        std::cout<<"Max Shared memory size per block: "<<deviceProp.sharedMemPerBlock<<std::endl;
        std::cout<<"Max threads per block: "<<deviceProp.maxThreadsPerBlock<<std::endl;
        std::cout<<"Max block dimensions: "<<deviceProp.maxThreadsDim[0]<<" x, "<<deviceProp.maxThreadsDim[1]<<" y, "<<deviceProp.maxThreadsDim[2]<<" z"<<std::endl;
        std::cout<<"Max grid dimensions: "<<deviceProp.maxGridSize[0]<<" x, "<<deviceProp.maxGridSize[1]<<" y, "<<deviceProp.maxGridSize[2]<<" z"<<std::endl;
        std::cout<<"Warp Size: "<<deviceProp.warpSize<<std::endl;
    }
}
