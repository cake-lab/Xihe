using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using Unity.Mathematics;

namespace Xihe.Library
{
    internal struct AnchorPointInGridPool
    {
        public int Index;
        public float CosSim;
    }

    internal static class RealitySharp
    {
        public enum CoordinateType
        {
            Cartesian = 0,
            Spherical = 1
        }

        public static float3[] FibonacciSphere(int samples, CoordinateType coordinateType)
        {
            // Vector3[] results = new Vector3[samples];
            // Vector2[] results = new Vector2[samples];
            var results = new float3[samples];

            var phi = Math.PI * (3.0f - Math.Sqrt(5.0f));

            for (var i = 0; i < samples; i++)
            {
                var y = 1.0f - ((float)i / (samples - 1.0f)) * 2.0f;
                var radius = (float)(Math.Sqrt(1 - y * y));
                var theta = (float)(phi * (float)i);

                // results[i] = new Vector2((float)theta, (float)phi);

                var x = (float)(Math.Cos(theta) * radius);
                var z = (float)(Math.Sin(theta) * radius);

                switch (coordinateType)
                {
                    case CoordinateType.Cartesian:
                        results[i] = new float3(x, (float)y, z);
                        break;
                    case CoordinateType.Spherical:
                        results[i] = RealitySharp.CartesianToSpherical(new float3(x, (float)y, z));
                        break;
                    default:
                        throw new ArgumentOutOfRangeException(nameof(coordinateType), coordinateType, null);
                }
            }

            return results;
        }

        public static IEnumerable<uint> MakeCacheGrid(uint gridSize, float3[] anchorsSpherical)
        {
            var r = 2 * math.PI / gridSize;

            var cacheGrid = new uint[gridSize * gridSize * 3];
            for (uint i = 0; i < cacheGrid.Length; i++) cacheGrid[i] = (uint)(anchorsSpherical.Length);

            for (uint i = 0; i < anchorsSpherical.Length; i++)
            {
                var u = (int)(anchorsSpherical[i].x / (r / 2.0f));
                var v = (int)(anchorsSpherical[i].y / r + gridSize / 2.0f);

                var idx = (int)((v * gridSize + u) * 3);

                for (var j = 0; j < 3; j++)
                {
                    if (cacheGrid[idx + j] < anchorsSpherical.Length) continue;
                    cacheGrid[idx + j] = i;
                    break;
                }
            }

            return cacheGrid;
        }

        internal static int[] MakePoolingGrid(float3[] sphericalAnchors, int numNeighbors = 4)
        {
            var result = new int[sphericalAnchors.Length * numNeighbors];

            for (var i = 0; i < sphericalAnchors.Length; i++)
            {
                var buffer = new List<AnchorPointInGridPool>();

                for (var j = 0; j < sphericalAnchors.Length; j++)
                {
                    if (i == j) continue;

                    var p = new AnchorPointInGridPool
                    {
                        Index = j, CosSim = math.dot(sphericalAnchors[i], sphericalAnchors[j])
                    };

                    buffer.Add(p);
                }

                buffer.Sort((a, b) => 
                    Math.Abs(a.CosSim - b.CosSim) < float.Epsilon
                    ? 0
                    : (a.CosSim < b.CosSim ? 1 : -1));

                var sortedBuffer = buffer.GetRange(0, 9);

                for (var k = 0; k < numNeighbors; k++)
                {
                    result[i * numNeighbors + k] = sortedBuffer[k].Index;
                }
            }

            return result;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static float3 CartesianToSpherical(float3 cord)
        {
            var res = new float3 {z = math.length(cord)};
            res.x = math.acos(cord.z / res.z);
            res.y = math.atan2(cord.y, cord.x);
            return res;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static int FindClosestSpherePoint(float3 point, IReadOnlyList<float3> searchPool)
        {
            var idx = 0;
            float maxCos = -2;

            for (var i = 0; i < searchPool.Count; i++)
            {
                var cos = math.dot(point, searchPool[i]);

                if (!(cos > maxCos)) continue;
                maxCos = cos;
                idx = i;
            }

            return idx;
        }

        public static int[][] BuildIndexBetweenPointCloud(float3[] source, float3[] target)
        {
            var map = new List<int>[source.Length];
            for (var i = 0; i < map.Length; i++) map[i] = new List<int>();

            for (var i = 0; i < target.Length; i++)
            {
                var idx = FindClosestSpherePoint(target[i], source);
                map[idx].Add(i);
            }

            var res = new int[source.Length][];

            for (var i = 0; i < map.Length; i++)
            {
                res[i] = map[i].ToArray();
            }

            return res;
        }
    }
}
