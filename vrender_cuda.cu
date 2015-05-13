#include "vrender_cuda_kernel.cuh"

extern "C"
void initializeVRender( float           *data,
                        int3            size,
                        float           max,
                        float           min,
                        cudaExtent      volumeSize,
                        uint            imageW,
                        uint            imageH )
{
    size_t float_size = size.x * size.y * size.z * sizeof(float);

    float *ddata;
    checkCudaErrors( cudaMalloc( (void**) &ddata, float_size ) );
    checkCudaErrors( cudaMemset( ddata, 0, float_size ) );

    dim3 block(size.x);
    dim3 grid(size.y,size.z);
    size_t data_size = size.x * size.y * size.z * sizeof(unsigned char);

    checkCudaErrors( cudaMalloc( (void**) &d_charvol, data_size ) );
    checkCudaErrors( cudaMemset( d_charvol, 0, data_size ) );

    checkCudaErrors( cudaMemcpy( ddata, data, float_size, cudaMemcpyHostToDevice ) );
    checkCudaErrors( cudaMemcpyToSymbol( dmax, &max, sizeof(float) ) );
    checkCudaErrors( cudaMemcpyToSymbol( dmin, &min, sizeof(float) ) );

    deviceFloat2Char<<<grid,block>>>( ddata, d_charvol );
    cudaThreadSynchronize();

    checkCudaErrors( cudaFree(ddata) );


    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<unsigned char>();
    checkCudaErrors( cudaMalloc3DArray( &d_volumeArray, &channelDesc, volumeSize ) );

    cudaMemcpy3DParms copyParams;
    copyParams.srcPtr   =   make_cudaPitchedPtr( d_charvol, volumeSize.width * sizeof(unsigned char), volumeSize.width, volumeSize.height );
    copyParams.dstArray =   d_volumeArray;
    copyParams.extent   =   volumeSize;
    copyParams.kind     =   cudaMemcpyDeviceToDevice;
    checkCudaErrors( cudaMemcpy3D( &copyParams ) );

    tex.normalized = true;
    tex.filterMode = cudaFilterModeLinear;
    tex.addressMode[0] = cudaAddressModeClamp;
    tex.addressMode[1] = cudaAddressModeClamp;

    checkCudaErrors( cudaBindTextureToArray( tex, d_volumeArray, channelDesc ) );

    float4 transferFunc[] = {
        { 0, 0, 0, 0 },
        { 0.2, 0.2, 0.2, 1 },
        { 0.4, 0.4, 0.4, 1 },
        { 0.6, 0.6, 0.6, 1 },
        { 0.8, 0.8, 0.8, 1 },
        { 1, 1, 1, 1 }
    };

    cudaChannelFormatDesc channelDesc2 = cudaCreateChannelDesc<float4>();
    cudaArray *d_transferFuncArray;

    checkCudaErrors( cudaMallocArray( &d_transferFuncArray, &channelDesc2, sizeof(transferFunc) / sizeof(float4), 1 ) );
    checkCudaErrors( cudaMemcpyToArray( d_transferFuncArray, 0, 0, transferFunc, sizeof(transferFunc), cudaMemcpyHostToDevice ) );

    transferTex.filterMode = cudaFilterModeLinear;
    transferTex.normalized = true;
    transferTex.addressMode[0] = cudaAddressModeClamp;

    checkCudaErrors( cudaBindTextureToArray( transferTex, d_transferFuncArray, channelDesc2 ) );

    checkCudaErrors( cudaMalloc( (void**) &d_volume, imageW * imageH * 3 * sizeof(unsigned char) ) );
}


extern "C"
void freeCudaBuffers()
{
    checkCudaErrors( cudaUnbindTexture(tex) );
    checkCudaErrors( cudaFreeArray(d_volumeArray) );

    checkCudaErrors( cudaUnbindTexture(transferTex) );
    checkCudaErrors( cudaFreeArray(d_transferFuncArray) );

    checkCudaErrors( cudaFree( d_charvol ) );
    checkCudaErrors( cudaFree( d_volume ) );
}


extern "C"
void render_kernel( dim3 gridSize, dim3 blockSize,
                    unsigned char *buffer,
                    uint imageW, uint imageH,
                    float dens, float bright, float offset, float scale, float weight )
{
    checkCudaErrors( cudaMemset( d_volume, 0, imageW * imageH * 3 * sizeof(unsigned char) ) );
    d_render<<<gridSize,blockSize>>>( d_volume,
                                      imageW, imageH,
                                      dens, bright, offset, scale, weight );
    checkCudaErrors( cudaMemcpy( buffer, d_volume, imageW * imageH * 3 * sizeof(unsigned char), cudaMemcpyDeviceToHost ) );
}


extern "C"
void copyInvViewMatrix( float *invViewMatrix, size_t sizeofMatrix )
{
    checkCudaErrors( cudaMemcpyToSymbol( c_invViewMatrix, invViewMatrix, sizeofMatrix, 0, cudaMemcpyHostToDevice ) );
}




