using System.Linq;
using Unity.Mathematics;
using UnityEngine;

namespace Xihe.DataProvider
{
    public struct CameraIntrinsics
    {
        public float DepthScaler;
        public float2 FocalLength;
        public float2 PrinciplePoint;
        public float2 Resolution;

        public float[] ToArray()
        {
            return new[]
            {
                DepthScaler,
                FocalLength.x, FocalLength.y,
                PrinciplePoint.x, PrinciplePoint.y,
                Resolution.x, Resolution.y
            };
        }

        public static CameraIntrinsics CreateFromString(string data)
        {
            var numbers = data
                .Split(',')
                .Select(float.Parse)
                .ToArray();

            return new CameraIntrinsics
            {
                DepthScaler = numbers[0],
                FocalLength = new float2(numbers[1], numbers[2]),
                PrinciplePoint = new float2(numbers[3], numbers[4]),
                Resolution = new float2(numbers[5], numbers[6])
            };
        }
    }

    public interface IXiheDataProvider
    {
        public Texture2D AcquireCameraColorImage();
        public Texture2D AcquireCameraDepthImage();
        public CameraIntrinsics AcquireCameraIntrinsic();
        public float[] AcquireCameraExtrinsic();
    }
}