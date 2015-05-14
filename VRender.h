#ifndef __VRENDER_CLASS_H__
#define __VRENDER_CLASS_H__

// CUDA utilities and system includes
#include <cuda_runtime_api.h>
#include <helper_cuda.h>
#include <helper_math.h>

// CUDA Includes
#include <vector_functions.h>
#include <driver_functions.h>

#define PI 3.14159f



class VRender
{
    public:
        VRender();
        ~VRender();

        //functions
        unsigned char *get_vrender_buffer();

        void set_vrender_parameters( float r_dens, float r_bright, float r_offset, float r_scale );
        void set_vrender_rotation( float dx, float dy );
        void set_vrender_translation( float dx, float dy );
        void set_vrender_zoom( float dy );

        int init_vrender( char *data,
                          int3 data_size,
                          uint3 *color_map );

        int get_width()
        {
            return width;
        };
        int get_height()
        {
            return height;
        };
        float get_density()
        {
            return density;
        };
        float get_brightness()
        {
            return brightness;
        };
        float get_offset()
        {
            return transferOffset;
        };
        float get_scale()
        {
            return transferScale;
        };
        float get_last_x()
        {
            return last_x;
        };
        float get_last_y()
        {
            return last_y;
        };

        void set_width( int i )
        {
            width = i;
        };
        void set_height( int i )
        {
            height = i;
        };
        void set_density( float v )
        {
            density = v;
        };
        void set_brightness( float v )
        {
            brightness = v;
        };
        void set_offset( float v )
        {
            transferOffset = v;
        };
        void set_scale( float v )
        {
            transferScale = v;
        };
        void set_last_x( float v )
        {
            last_x = v;
        };
        void set_last_y( float v )
        {
            last_y = v;
        };

    protected:
       //functions
        void setInvViewMatrix();
        void render();

        void translateMat( float *matrix, float3 translation );
        void rotMat( float *matrix, float3 axis, float theta, float3 center );
        void multiplyModelViewMatrix( float *trans );
        void transformModelViewMatrix();

        int iDivUp( int a, int b ){ return (a % b != 0) ? (a / b + 1) : (a / b); }

        //variables
        uint width, height;

        int3 size;

        dim3 blockSize;
        dim3 gridSize;

        cudaExtent volumeSize;

        float3 viewRotation;
        float3 viewTranslation;

        float invViewMatrix[12];
        float identityMatrix[16];
        float modelViewMatrix[16];

        float density;
        float brightness;
        float transferOffset;
        float transferScale;
        float weight;
        float last_x, last_y;

        unsigned char *render_buf;
        unsigned char *d_volume;
        unsigned char *h_volume;
};


#endif // __VRENDER_CLASS_H__
