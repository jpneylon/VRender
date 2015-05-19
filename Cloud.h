#ifndef __CLOUD_CLASS_H__
#define __CLOUD_CLASS_H__

#include <vector>

class Cloud
{
  public:
    Cloud()
    {
        count = 0;
        max_pos.x = -999999;
        max_pos.y = -999999;
        max_pos.z = -999999;
        min_pos.x = 999999;
        min_pos.y = 999999;
        min_pos.z = 999999;
    }

    std::vector<float3> position;
    std::vector<uint3> rgb;
    uint count;
    float3 max_pos;
    float3 min_pos;
    float3 data_dim;
    float3 world_origin;
    float3 world_start;
    uint   world_size;
    float  world_res;
    uint   world_count;
    float  world_dim;
};


#endif // __CLOUD_CLASS_H__
