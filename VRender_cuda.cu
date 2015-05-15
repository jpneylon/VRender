#include "VRender_cuda_kernel.cuh"

extern "C"
void initializeVRender( void  *red_map,
                        void  *green_map,
                        void  *blue_map,
                        cudaExtent      volumeSize,
                        uint            imageW,
                        uint            imageH )
{
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uchar>();

    // RED
    checkCudaErrors( cudaMalloc3DArray( &d_redArray, &channelDesc, volumeSize ) );

    redParams.srcPtr   =   make_cudaPitchedPtr( red_map, volumeSize.width*sizeof(uchar), volumeSize.width, volumeSize.height );
    redParams.dstArray =   d_redArray;
    redParams.extent   =   volumeSize;
    redParams.kind     =   cudaMemcpyHostToDevice;
    checkCudaErrors( cudaMemcpy3D( &redParams ) );

    texRed.normalized = true;
    texRed.filterMode = cudaFilterModeLinear;
    texRed.addressMode[0] = cudaAddressModeClamp;
    texRed.addressMode[1] = cudaAddressModeClamp;
    texRed.addressMode[2] = cudaAddressModeClamp;

    checkCudaErrors( cudaBindTextureToArray( texRed, d_redArray, channelDesc ) );

    // GREEN
    checkCudaErrors( cudaMalloc3DArray( &d_greenArray, &channelDesc, volumeSize ) );

    greenParams.srcPtr   =   make_cudaPitchedPtr( green_map, volumeSize.width*sizeof(uchar), volumeSize.width, volumeSize.height );
    greenParams.dstArray =   d_greenArray;
    greenParams.extent   =   volumeSize;
    greenParams.kind     =   cudaMemcpyHostToDevice;
    checkCudaErrors( cudaMemcpy3D( &greenParams ) );

    texGreen.normalized = true;
    texGreen.filterMode = cudaFilterModeLinear;
    texGreen.addressMode[0] = cudaAddressModeClamp;
    texGreen.addressMode[1] = cudaAddressModeClamp;
    texGreen.addressMode[2] = cudaAddressModeClamp;

    checkCudaErrors( cudaBindTextureToArray( texGreen, d_greenArray, channelDesc ) );

    // BLUE
    checkCudaErrors( cudaMalloc3DArray( &d_blueArray, &channelDesc, volumeSize ) );

    blueParams.srcPtr   =   make_cudaPitchedPtr( blue_map, volumeSize.width*sizeof(uchar), volumeSize.width, volumeSize.height );
    blueParams.dstArray =   d_blueArray;
    blueParams.extent   =   volumeSize;
    blueParams.kind     =   cudaMemcpyHostToDevice;
    checkCudaErrors( cudaMemcpy3D( &blueParams ) );

    texBlue.normalized = true;
    texBlue.filterMode = cudaFilterModeLinear;
    texBlue.addressMode[0] = cudaAddressModeClamp;
    texBlue.addressMode[1] = cudaAddressModeClamp;
    texBlue.addressMode[2] = cudaAddressModeClamp;

    checkCudaErrors( cudaBindTextureToArray( texBlue, d_blueArray, channelDesc ) );

    // OUTPUT BUFFER
    checkCudaErrors( cudaMalloc( (void**) &d_volume, imageW * imageH * 3 * sizeof(uchar) ) );
    checkCudaErrors( cudaMemset( d_volume, 0, imageW * imageH * 3 * sizeof(uchar) ) );
}


extern "C"
void freeCudaBuffers()
{
    checkCudaErrors( cudaUnbindTexture(texRed) );
    checkCudaErrors( cudaFreeArray(d_redArray) );

    checkCudaErrors( cudaUnbindTexture(texGreen) );
    checkCudaErrors( cudaFreeArray(d_greenArray) );

    checkCudaErrors( cudaUnbindTexture(texBlue) );
    checkCudaErrors( cudaFreeArray(d_blueArray) );

    checkCudaErrors( cudaFree( d_volume ) );
}


extern "C"
void render_kernel( dim3 gridSize, dim3 blockSize,
                    unsigned char *buffer,
                    uint imageW, uint imageH,
                    float dens, float bright, float offset, float scale )
{
    checkCudaErrors( cudaMemset( d_volume, 0, imageW * imageH * 3 * sizeof(unsigned char) ) );
    d_render<<<gridSize,blockSize>>>( d_volume,
                                      imageW, imageH,
                                      dens, bright, offset, scale );
    checkCudaErrors( cudaMemcpy( buffer, d_volume, imageW * imageH * 3 * sizeof(unsigned char), cudaMemcpyDeviceToHost ) );
}


extern "C"
void copyInvViewMatrix( float *invViewMatrix, size_t sizeofMatrix )
{
    checkCudaErrors( cudaMemcpyToSymbol( c_invViewMatrix, invViewMatrix, sizeofMatrix, 0, cudaMemcpyHostToDevice ) );
}




