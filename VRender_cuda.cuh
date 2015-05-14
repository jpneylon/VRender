
extern "C"
    void initializeVRender( char *data,
                            uint size,
                            uint3 *color_map,
                            cudaExtent volumeSize,
                            uint imageW, uint imageH );

extern "C"
    void freeCudaBuffers();

extern "C"
    void render_kernel( dim3 gridSize, dim3 blockSize,
                        unsigned char *buffer,
                        uint imageW, uint imageH,
                        float dens, float bright, float offset, float scale, float weight );

extern "C"
    void copyInvViewMatrix( float *invViewMatrix,
                            size_t sizeofMatrix);

