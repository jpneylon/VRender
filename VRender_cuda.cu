#include <algorithm>
#include "VRender_cuda_kernel.cuh"
#include "Cloud.h"

extern "C"
void createVRenderColorMaps( Cloud * cloud )
{
    cudaSetDevice(1);

    float cudatime;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    cudaDeviceProp devProp;
    cudaGetDeviceProperties( &devProp, 1 );
    //printf("\n Device: %s", devProp.name );

    checkCudaErrors( cudaMalloc( (void**) &d_red, cloud->world_count ) );
    checkCudaErrors( cudaMalloc( (void**) &d_green, cloud->world_count ) );
    checkCudaErrors( cudaMalloc( (void**) &d_blue, cloud->world_count ) );

    checkCudaErrors( cudaMemset( d_red, 0, cloud->world_count ) );
    checkCudaErrors( cudaMemset( d_green, 0, cloud->world_count ) );
    checkCudaErrors( cudaMemset( d_blue, 0, cloud->world_count ) );

    float3 *h_pos;
    h_pos = new float3[cloud->position.size()];
    std::copy( cloud->position.begin(), cloud->position.end(), h_pos );

    float3 *d_pos;
    checkCudaErrors( cudaMalloc( (void**) &d_pos, cloud->position.size() * sizeof(float3) ) );
    checkCudaErrors( cudaMemcpy( d_pos, h_pos, cloud->position.size() * sizeof(float3), cudaMemcpyHostToDevice ) );

    uint3 *h_color;
    h_color = new uint3[cloud->rgb.size()];
    std::copy( cloud->rgb.begin(), cloud->rgb.end(), h_color );

    uint3 *d_color;
    checkCudaErrors( cudaMalloc( (void**) &d_color, cloud->rgb.size() * sizeof(uint3) ) );
    checkCudaErrors( cudaMemcpy( d_color, h_color, cloud->rgb.size() * sizeof(uint3), cudaMemcpyHostToDevice ) );

    WORLD h_world;
    h_world.npoints = cloud->count;
    h_world.count = cloud->world_count;
    h_world.start.x = cloud->world_start.x;
    h_world.start.y = cloud->world_start.y;
    h_world.start.z = cloud->world_start.z;
    h_world.resolution = cloud->world_res;
    h_world.size = cloud->world_size;

    checkCudaErrors( cudaMemcpyToSymbol( d_world, &h_world, sizeof(WORLD) ) );

    dim3 block(devProp.maxThreadsPerBlock / 4);
    uint sizer = cloud->count;
    int3 tempGridExtent;
    tempGridExtent.x = sizer / block.x;
    tempGridExtent.y = 1;
    tempGridExtent.z = 1;
    if (sizer % block.x > 0) tempGridExtent.x++;
    if (tempGridExtent.x > devProp.maxGridSize[1])
    {
        tempGridExtent.y = tempGridExtent.x / devProp.maxGridSize[1];
        if (tempGridExtent.x % devProp.maxGridSize[1] > 0) tempGridExtent.y++;
        tempGridExtent.x = devProp.maxGridSize[1];
        if (tempGridExtent.y > devProp.maxGridSize[1])
        {
            tempGridExtent.z = tempGridExtent.y / devProp.maxGridSize[1];
            if (tempGridExtent.y % devProp.maxGridSize[1] > 0) tempGridExtent.z++;
            tempGridExtent.y = devProp.maxGridSize[1];
        }
    }
    dim3 grid(tempGridExtent.x,tempGridExtent.y,tempGridExtent.z);

    cuda_create_color_maps<<<grid,block>>> ( d_pos,
                                             d_color,
                                             d_red,
                                             d_green,
                                             d_blue );
    cudaThreadSynchronize();

    checkCudaErrors( cudaFree( d_pos ) );
    checkCudaErrors( cudaFree( d_color ) );

    delete [] h_pos;
    delete [] h_color;

    cudaEventRecord( stop, 0 );
    cudaEventSynchronize( stop );
    cudaEventElapsedTime( &cudatime, start, stop );
    printf("\n ||| TIME - Create Color Maps: %f ms\n", cudatime);
}


extern "C"
void initializeVRender( cudaExtent      volumeSize,
                        uint            imageW,
                        uint            imageH )
{
    float cudatime;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uchar>();

    // RED
    checkCudaErrors( cudaMalloc3DArray( &d_redArray, &channelDesc, volumeSize ) );
    redParams.srcPtr   =   make_cudaPitchedPtr( d_red, volumeSize.width, volumeSize.width, volumeSize.height );
    redParams.dstArray =   d_redArray;
    redParams.extent   =   volumeSize;
    redParams.kind     =   cudaMemcpyDeviceToDevice;
    checkCudaErrors( cudaMemcpy3D( &redParams ) );

    texRed.normalized = true;
    texRed.filterMode = cudaFilterModeLinear;
    texRed.addressMode[0] = cudaAddressModeClamp;
    texRed.addressMode[1] = cudaAddressModeClamp;
    texRed.addressMode[2] = cudaAddressModeClamp;

    checkCudaErrors( cudaBindTextureToArray( texRed, d_redArray, channelDesc ) );

    // GREEN
    checkCudaErrors( cudaMalloc3DArray( &d_greenArray, &channelDesc, volumeSize ) );
    greenParams.srcPtr   =   make_cudaPitchedPtr( d_green, volumeSize.width, volumeSize.width, volumeSize.height );
    greenParams.dstArray =   d_greenArray;
    greenParams.extent   =   volumeSize;
    greenParams.kind     =   cudaMemcpyDeviceToDevice;
    checkCudaErrors( cudaMemcpy3D( &greenParams ) );

    texGreen.normalized = true;
    texGreen.filterMode = cudaFilterModeLinear;
    texGreen.addressMode[0] = cudaAddressModeClamp;
    texGreen.addressMode[1] = cudaAddressModeClamp;
    texGreen.addressMode[2] = cudaAddressModeClamp;

    checkCudaErrors( cudaBindTextureToArray( texGreen, d_greenArray, channelDesc ) );

    // BLUE
    checkCudaErrors( cudaMalloc3DArray( &d_blueArray, &channelDesc, volumeSize ) );
    blueParams.srcPtr   =   make_cudaPitchedPtr( d_blue, volumeSize.width, volumeSize.width, volumeSize.height );
    blueParams.dstArray =   d_blueArray;
    blueParams.extent   =   volumeSize;
    blueParams.kind     =   cudaMemcpyDeviceToDevice;
    checkCudaErrors( cudaMemcpy3D( &blueParams ) );

    texBlue.normalized = true;
    texBlue.filterMode = cudaFilterModeLinear;
    texBlue.addressMode[0] = cudaAddressModeClamp;
    texBlue.addressMode[1] = cudaAddressModeClamp;
    texBlue.addressMode[2] = cudaAddressModeClamp;

    checkCudaErrors( cudaBindTextureToArray( texBlue, d_blueArray, channelDesc ) );

    // OUTPUT BUFFER
    checkCudaErrors( cudaMalloc( (void**) &d_volume, imageW * imageH * 3 ) );
    checkCudaErrors( cudaMemset( d_volume, 0, imageW * imageH * 3 ) );

    cudaEventRecord( stop, 0 );
    cudaEventSynchronize( stop );
    cudaEventElapsedTime( &cudatime, start, stop );
    printf("\n ||| TIME - Initalize GPU Memory: %f ms\n", cudatime);

    size_t freeMem, totalMem;
    checkCudaErrors(cudaMemGetInfo(&freeMem,&totalMem));
    printf("\n Free Memory: %lu / %lu\n",freeMem,totalMem);
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

    checkCudaErrors( cudaFree( d_red ) );
    checkCudaErrors( cudaFree( d_green ) );
    checkCudaErrors( cudaFree( d_blue ) );

    checkCudaErrors( cudaFree( d_volume ) );
}


extern "C"
void render_kernel( dim3 gridSize, dim3 blockSize,
                    unsigned char *buffer,
                    uint imageW, uint imageH,
                    float dens, float bright, float offset, float scale,
                    float *fps )
{
    float cudatime;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    checkCudaErrors( cudaMemset( d_volume, 0, imageW * imageH * 3 ) );
    d_render<<<gridSize,blockSize>>>( d_volume,
                                      imageW, imageH,
                                      dens, bright, offset, scale );
    cudaThreadSynchronize();
    checkCudaErrors( cudaMemcpy( buffer, d_volume, imageW * imageH * 3, cudaMemcpyDeviceToHost ) );

    cudaEventRecord( stop, 0 );
    cudaEventSynchronize( stop );
    cudaEventElapsedTime( &cudatime, start, stop );
    *fps = cudatime / 1000.f;
    //printf(" ||| TIME - Render Update: %f ms\n", cudatime);
}


extern "C"
void copyInvViewMatrix( float *invViewMatrix, size_t sizeofMatrix )
{
    checkCudaErrors( cudaMemcpyToSymbol( c_invViewMatrix, invViewMatrix, sizeofMatrix, 0, cudaMemcpyHostToDevice ) );
}




