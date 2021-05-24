// CUDA kernels for accelerating data generation

__global__ void makePointCloud (
    float3 *dest, float *depthTexture, float2 *intrinsics
) {
    float2 f = intrinsics[0];
    float2 c = intrinsics[1];
    float2 res = intrinsics[2];

    int u = (blockIdx.x * blockDim.x) + threadIdx.x;
    int v = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (u > (res.x - 1) || v > (res.y - 1)) return;

    int linearIdx = v * res.x + u;

    float depth = depthTexture[linearIdx];

    float x = (u - c.x) * depth / f.x;
    float y = (v - c.y) * depth / f.y;
    float z = -depth;

    float3 position;
    position.x = x;
    position.y = y;
    position.z = z;

    dest[linearIdx] = position;
}

__global__ void cameraAdjustment (
  float3 *pointCloud,
  float *camToWorld, float *rotation, float2 *intrinsics
) {
  float2 res = intrinsics[2];
  int u = (blockIdx.x * blockDim.x) + threadIdx.x;
  int v = (blockIdx.y * blockDim.y) + threadIdx.y;

  if (u > (res.x - 1) || v > (res.y - 1)) return;

  int linearIdx = v * res.x + u;

  float3 point = pointCloud[linearIdx];

  float3 position;

  position.x = camToWorld[0] * point.x + camToWorld[1] * point.y +\
   camToWorld[2] * point.z;
  position.y = camToWorld[4] * point.x + camToWorld[5] * point.y +\
   camToWorld[6] * point.z;
  position.z = camToWorld[8] * point.x + camToWorld[9] * point.y +\
   camToWorld[10] * point.z;

  float3 rPos;

  rPos.x = rotation[0] * position.x +\
    rotation[1] * position.y + rotation[2] * position.z;
  rPos.y = rotation[3] * position.x +\
    rotation[4] * position.y + rotation[5] * position.z;
  rPos.z = rotation[6] * position.x +\
    rotation[7] * position.y + rotation[8] * position.z;

  pointCloud[linearIdx] = rPos;
}

// Floating point numbers are not associative
// (a + b) + c != a + (b + c)
// running this function multiple times will result in
// different outputs, but the difference is acceptable.
__global__ void makeSHCoefficients (
    double3 *ldrDest, double3 *hdrDest, float *basis,
    float3 *cubemapColorLDR, float3 *cubemapColorHDR
) {
    int faceId = blockIdx.x;
    int pointId = (blockIdx.y * blockDim.y) + threadIdx.y;
    if (pointId >= 128 * 128) return;

    int cubemapIdx = faceId * 128 * 128 + pointId;

    float3 ldrRgb = cubemapColorLDR[cubemapIdx];
    float3 hdrRgb = cubemapColorHDR[cubemapIdx];

    for (int i = 0; i < 9; i++)
    {
        int id = cubemapIdx * 9 + i;
        atomicAdd(&ldrDest[i].x, ldrRgb.x * basis[id]);
        atomicAdd(&ldrDest[i].y, ldrRgb.y * basis[id]);
        atomicAdd(&ldrDest[i].z, ldrRgb.z * basis[id]);

        atomicAdd(&hdrDest[i].x, hdrRgb.x * basis[id]);
        atomicAdd(&hdrDest[i].y, hdrRgb.y * basis[id]);
        atomicAdd(&hdrDest[i].z, hdrRgb.z * basis[id]);
    }
}
