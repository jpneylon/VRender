#include <cuda_runtime_api.h>
#include <helper_cuda.h>
#include <helper_math.h>

typedef unsigned char uchar;

uchar *d_volume;

cudaArray *d_redArray = 0;
cudaArray *d_blueArray = 0;
cudaArray *d_greenArray = 0;

cudaMemcpy3DParms redParams = {0};
cudaMemcpy3DParms greenParams = {0};
cudaMemcpy3DParms blueParams = {0};

texture<uchar, 3, cudaReadModeNormalizedFloat> texRed;
texture<uchar, 3, cudaReadModeNormalizedFloat> texGreen;
texture<uchar, 3, cudaReadModeNormalizedFloat> texBlue;

typedef struct {
    float4 m[3];
} float3x4;

__constant__ float3x4 c_invViewMatrix;
__constant__ int dID;
__constant__ float dmax, dmin;

struct Ray {
    float3 o;
    float3 d;
};



__device__
int intersectBox( Ray r, float3 boxmin, float3 boxmax, float *tnear, float *tfar )
{
    float3 invR = make_float3(1.f) / r.d;
    float3 tbot = invR * (boxmin - r.o);
    float3 ttop = invR * (boxmax - r.o);

    float3 tmin = fminf(ttop, tbot);
    float3 tmax = fmaxf(ttop, tbot);

    float largest_tmin = fmaxf(fmaxf(tmin.x,tmin.y), fminf(tmin.x,tmin.z));
    float smallest_tmax = fminf(fminf(tmax.x,tmax.y), fminf(tmax.x,tmax.z));

    *tnear = largest_tmin;
    *tfar = smallest_tmax;

    return smallest_tmax - largest_tmin;
}

__device__
float3 get_eye_ray_direction( const float3x4 &M, const float3 &v )
{
    float3 r;
    r.x = dot(v, make_float3(M.m[0]));
    r.y = dot(v, make_float3(M.m[1]));
    r.z = dot(v, make_float3(M.m[2]));
    return r;
}

__device__
float4 get_eye_ray_origin( const float3x4 &M, const float4 &v )
{
    float4 r;
    r.x = dot(v, M.m[0]);
    r.x = dot(v, M.m[1]);
    r.x = dot(v, M.m[2]);
    r.w = 1.f;
    return r;
}

__device__
float4 get_pix_val( int maxSteps, Ray eyeRay, float tstep, float tnear, float tfar,
                    float offset, float scale, float dens, float opacity )
{
    float t = tnear;
    float3 pos = eyeRay.o + eyeRay.d * tnear;
    float3 step = eyeRay.d * tstep;
    float4 sum = make_float4( 0 );

    for( int i=0; i<maxSteps; i++)
    {
        float red, green, blue;
        float4 col;
        col.w = dens;

        blue = tex3D( texRed, pos.x * 0.5f + 0.5f, pos.y * 0.5f + 0.5f, pos.z * 0.5f + 0.5f );
        green = tex3D( texGreen, pos.x * 0.5f + 0.5f, pos.y * 0.5f + 0.5f, pos.z * 0.5f + 0.5f );
        red = tex3D( texBlue, pos.x * 0.5f + 0.5f, pos.y * 0.5f + 0.5f, pos.z * 0.5f + 0.5f );

        col.x = ((red - offset) * scale) * col.w;
        col.y = ((green - offset) * scale) * col.w;
        col.z = ((blue - offset) * scale) * col.w;

        sum += col * (1.f - sum.w);

        t += tstep;
        if (t > tfar) break;

        pos += step;
    }

    return sum;
}

__global__
void d_render( unsigned char *d_output,
               uint imageW,
               uint imageH,
               float dens,
               float bright,
               float offset,
               float scale )
{
    const int maxSteps = 500;
    const float tstep = 0.01f;
    const float opacityThreshold = 0.95f;
    const float3 boxmin = make_float3( -1.f, -1.f, -1.f );
    const float3 boxmax = make_float3( 1.f, 1.f, 1.f );

    uint x = blockIdx.x * blockDim.x + threadIdx.x;
    uint y = blockIdx.y * blockDim.y + threadIdx.y;
    if ( x >= imageW || y >= imageH ) return;

    float u = (x / (float)imageW) * 2.f - 1.f;
    float v = (y / (float)imageH) * 2.f - 1.f;

    Ray eyeRay;
    eyeRay.o = make_float3( get_eye_ray_origin( c_invViewMatrix, make_float4( 0.f, 0.f, 0.f, -1.f ) ) );
    eyeRay.d = normalize( make_float3( u, v, 2.f ) );
    eyeRay.d = get_eye_ray_direction( c_invViewMatrix, eyeRay.d );

    float tnear, tfar;
    int hit = intersectBox( eyeRay, boxmin, boxmax, &tnear, &tfar );
    if (!hit) return;
    if (tnear < 0.f) tnear = 0.f;

    float4 sum = make_float4(0.f);
    sum += get_pix_val( maxSteps, eyeRay, tstep, tnear, tfar, offset, scale, dens, opacityThreshold );
    sum *= bright;

    float3 rgba;
    rgba.x = __saturatef(sum.x);
    rgba.y = __saturatef(sum.y);
    rgba.z = __saturatef(sum.z);

    d_output[ 3*(x + imageW * y) + 0] = (unsigned char) (255.f * rgba.x);
    d_output[ 3*(x + imageW * y) + 1] = (unsigned char) (255.f * rgba.y);
    d_output[ 3*(x + imageW * y) + 2] = (unsigned char) (255.f * rgba.z);
}




