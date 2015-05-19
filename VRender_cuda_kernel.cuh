#include <cuda_runtime_api.h>
#include <helper_cuda.h>
#include <helper_math.h>

#define tstep       0.005f
#define maxSteps    500
#define opacity     0.95f

typedef unsigned char uchar;

uchar *d_volume;

uchar *d_red, *d_green, *d_blue;

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
    uint npoints;
    uint count;
    float3 start;
    float resolution;
    uint size;
} WORLD;

typedef struct {
    float4 m[3];
} float3x4;

__constant__ WORLD d_world;
__constant__ float3x4 c_invViewMatrix;
__constant__ int dID;
__constant__ float dmax, dmin;

struct Ray {
    float3 o;
    float3 d;
};



__device__
int intersectBox(Ray r, float3 boxmin, float3 boxmax, float *tnear, float *tfar)
{
    // compute intersection of ray with all six bbox planes
    float3 invR = make_float3(1.0f) / r.d;
    float3 tbot = invR * (boxmin - r.o);
    float3 ttop = invR * (boxmax - r.o);

    // re-order intersections to find smallest and largest on each axis
    float3 tmin = fminf(ttop, tbot);
    float3 tmax = fmaxf(ttop, tbot);

    // find the largest tmin and the smallest tmax
    float largest_tmin = fmaxf(fmaxf(tmin.x, tmin.y), fmaxf(tmin.x, tmin.z));
    float smallest_tmax = fminf(fminf(tmax.x, tmax.y), fminf(tmax.x, tmax.z));

	*tnear = largest_tmin;
	*tfar = smallest_tmax;

	return smallest_tmax > largest_tmin;
}

// transform vector by matrix (no translation)
__device__
float3 get_eye_ray_direction(const float3x4 &M, const float3 &v)
{
    float3 r;
    r.x = dot(v, make_float3(M.m[0]));
    r.y = dot(v, make_float3(M.m[1]));
    r.z = dot(v, make_float3(M.m[2]));
    return r;
}

// transform vector by matrix with translation
__device__
float4 get_eye_ray_origin(const float3x4 &M, const float4 &v)
{
    float4 r;
    r.x = dot(v, M.m[0]);
    r.y = dot(v, M.m[1]);
    r.z = dot(v, M.m[2]);
    r.w = 1.0f;
    return r;
}

__device__
float4 get_pix_val( Ray eyeRay, float tnear, float tfar,
                    float offset, float scale, float dens )
{
    float t = tnear;
    float3 pos = eyeRay.o + eyeRay.d * tnear;
    float3 step = eyeRay.d * tstep;
    float4 sum = make_float4( 0 );

    for( int i=0; i<maxSteps; i++)
    {
        float red, green, blue;
        float4 col;

        blue = tex3D( texRed, pos.x * 0.5f + 0.5f, pos.y * 0.5f + 0.5f, pos.z * 0.5f + 0.5f );
        green = tex3D( texGreen, pos.x * 0.5f + 0.5f, pos.y * 0.5f + 0.5f, pos.z * 0.5f + 0.5f );
        red = tex3D( texBlue, pos.x * 0.5f + 0.5f, pos.y * 0.5f + 0.5f, pos.z * 0.5f + 0.5f );

        int zero_check = (red + green + blue) != 0;
        col.w = dens * __int2float_rn(zero_check);

        col.x = ((red - offset) * scale) * col.w;
        col.y = ((green - offset) * scale) * col.w;
        col.z = ((blue - offset) * scale) * col.w;

        sum += col * (1.f - sum.w);

        // exit early if opaque
        if (sum.w > opacity)
            break;

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
    const float3 boxMin = make_float3(-1.0f, -1.0f, -1.0f);
    const float3 boxMax = make_float3(1.0f, 1.0f, 1.0f);

	uint x = blockIdx.x*blockDim.x + threadIdx.x;
    uint y = blockIdx.y*blockDim.y + threadIdx.y;
    if ((x >= imageW) || (y >= imageH)) return;

    float u = (x / (float) imageW)*2.0f-1.0f;
    float v = (y / (float) imageH)*2.0f-1.0f;

    // calculate eye ray in world space
    Ray eyeRay;
    eyeRay.o = make_float3( get_eye_ray_origin(c_invViewMatrix, make_float4(0.0f, 0.0f, 0.0f, 1.0f)) );
    eyeRay.d = normalize(make_float3(u, v, 2.0f));
    eyeRay.d = get_eye_ray_direction(c_invViewMatrix, eyeRay.d);

    // find intersection with box
	float tnear, tfar;
	int hit = intersectBox(eyeRay, boxMin, boxMax, &tnear, &tfar);
    if (!hit) return;

    // clamp to near plane
	if (tnear < 0.0f) tnear = 0.0f;

    // march along ray from front to back, accumulating color
    float4 sum = make_float4(0.0f);
    sum += get_pix_val( eyeRay, tnear, tfar, offset, scale, dens );
    sum *= bright;

    // clamp to [0.0, 1.0]
    float4 rgba;
    rgba.x = __saturatef(sum.x);
    rgba.y = __saturatef(sum.y);
    rgba.z = __saturatef(sum.z);
    rgba.w = __saturatef(sum.w);

    // write output color
    d_output[3*(x + imageW * y) + 0] = (unsigned char) (255 * rgba.x);
    d_output[3*(x + imageW * y) + 1] = (unsigned char) (255 * rgba.y);
    d_output[3*(x + imageW * y) + 2] = (unsigned char) (255 * rgba.z);
}


__global__
void cuda_create_color_maps( float3 *position,
                             uint3  *color,
                             uchar  *red,
                             uchar  *green,
                             uchar  *blue )
{
    uint bIdx = blockIdx.x + gridDim.x * ( blockIdx.y + gridDim.y * blockIdx.z );
    uint in = threadIdx.x + blockDim.x * bIdx;
    if (in >= d_world.npoints) return;

    float3 pos = position[in];
    uint3 rgb = color[in];

    float3 floc = (pos - d_world.start) / d_world.resolution;

    int x = __float2int_rd(floc.x);
    int y = __float2int_rd(floc.y);
    int z = __float2int_rd(floc.z);

    float3 plus_frac = make_float3( floc.x - __int2float_rn(x),
                                    floc.y - __int2float_rn(y),
                                    floc.z - __int2float_rn(z) );
    float3 min_frac = make_float3(1,1,1) - plus_frac;

    if (x < 0 || y < 0 || z < 0 || x >= d_world.size || y >= d_world.size || z >= d_world.size ) return;
    uint out = x + d_world.size * (y + d_world.size * z);

    red[out] = rgb.x;
    green[out] = rgb.y;
    blue[out] = rgb.z;
/*
    int nx = x - 1;
    int ny = y;
    int nz = z;

    if (nx >= 0 || ny >= 0 || nz >= 0 || nx < d_world.size || ny < d_world.size || nz < d_world.size )
    {
        out = nx + d_world.size * (ny + d_world.size * nz);
        red[out] = rgb.x * min_frac.x;
        green[out] = rgb.y * min_frac.x;
        blue[out] = rgb.z * min_frac.x;
    }
    nx = x + 1;
    if (nx >= 0 || ny >= 0 || nz >= 0 || nx < d_world.size || ny < d_world.size || nz < d_world.size )
    {
        out = nx + d_world.size * (ny + d_world.size * nz);
        red[out] = rgb.x * plus_frac.x;
        green[out] = rgb.y * plus_frac.x;
        blue[out] = rgb.z * plus_frac.x;
    }

    nx = x;
    ny = y - 1;
    if (nx >= 0 || ny >= 0 || nz >= 0 || nx < d_world.size || ny < d_world.size || nz < d_world.size )
    {
        out = nx + d_world.size * (ny + d_world.size * nz);
        red[out] = rgb.x * min_frac.y;
        green[out] = rgb.y * min_frac.y;
        blue[out] = rgb.z * min_frac.y;
    }
    ny = y + 1;
    if (nx >= 0 || ny >= 0 || nz >= 0 || nx < d_world.size || ny < d_world.size || nz < d_world.size )
    {
        out = nx + d_world.size * (ny + d_world.size * nz);
        red[out] = rgb.x * plus_frac.y;
        green[out] = rgb.y * plus_frac.y;
        blue[out] = rgb.z * plus_frac.y;
    }

    ny = y;
    nz = z - 1;
    if (nx >= 0 || ny >= 0 || nz >= 0 || nx < d_world.size || ny < d_world.size || nz < d_world.size )
    {
        out = nx + d_world.size * (ny + d_world.size * nz);
        red[out] = rgb.x * min_frac.z;
        green[out] = rgb.y * min_frac.z;
        blue[out] = rgb.z * min_frac.z;
    }
    nz = z + 1;
    if (nx >= 0 || ny >= 0 || nz >= 0 || nx < d_world.size || ny < d_world.size || nz < d_world.size )
    {
        out = nx + d_world.size * (ny + d_world.size * nz);
        red[out] = rgb.x * plus_frac.z;
        green[out] = rgb.y * plus_frac.z;
        blue[out] = rgb.z * plus_frac.z;
    }
*/
}


