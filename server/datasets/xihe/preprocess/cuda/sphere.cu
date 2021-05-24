#define CACHE_SIZE 1024
#define ANCHOR_SIZE 1280

__global__ void nn_search(
    float2 *dest, float3 *points, float3 *anchors
)
{
    const uint pointIdx = blockIdx.x;
    const uint anchorIdx = threadIdx.x;
    const uint threadSize = blockDim.x;

    float3 point = points[pointIdx];

    __shared__ uint idx[CACHE_SIZE];
    __shared__ float cosines[CACHE_SIZE];


    uint i = 0;
    uint rIdx = i * threadSize + anchorIdx;
    float pointLength = sqrt(point.x * point.x + point.y * point.y + point.z * point.z);

    // Initialize cache array
    // By default, CUDA shared array is uninitialized
    cosines[anchorIdx] = 0;

    // Compute angels for each anchor point and sphere point
    while (rIdx < ANCHOR_SIZE) {
        float3 anchor = anchors[rIdx];

        float p_cos = (point.x * anchor.x +\
            point.y * anchor.y + point.z * anchor.z) / pointLength;

        if (p_cos > cosines[anchorIdx]) {
            idx[anchorIdx] = rIdx;
            cosines[anchorIdx] = p_cos;
        }

        rIdx = ++i * threadSize + anchorIdx;
    }

    __syncthreads();

    // Begin reduction to find the maximum cosine (smallest angel)
    i = CACHE_SIZE / 2;

    while (i != 0) {
        if (anchorIdx < i) {
            rIdx = anchorIdx + i;
            if (cosines[anchorIdx] < cosines[rIdx]) {
                idx[anchorIdx] = idx[rIdx];
                cosines[anchorIdx] = cosines[rIdx];
            }
        }

        __syncthreads();

        i /= 2;
    }

    int selectedAnchorIdx = idx[0];

    atomicMin((int *)(&dest[selectedAnchorIdx].y), __float_as_int(pointLength));

    __syncwarp();

    if (dest[selectedAnchorIdx].y == pointLength) {
        dest[selectedAnchorIdx].x = pointIdx;
    }
}
