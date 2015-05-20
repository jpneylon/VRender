#include "Cloud.h"


int iDivUp( int a, int b );


extern "C"
void updateVRenderColorMaps( Cloud * cloud, unsigned char cycle, float *fps );


extern "C"
void createVRenderColorMaps( Cloud * cloud );


extern "C"
    void initializeVRender( cudaExtent volumeSize,
                            uint imageW, uint imageH );

extern "C"
    void freeCudaBuffers();


extern "C"
    void render_kernel( dim3 gridSize, dim3 blockSize,
                        unsigned char *buffer,
                        uint imageW, uint imageH,
                        float dens, float bright, float offset, float scale,
                        float *fps );


extern "C"
    void copyInvViewMatrix( float *invViewMatrix,
                            size_t sizeofMatrix);

