using Unity.Mathematics;
using UnityEngine;

namespace Xihe.DataProvider
{
    public class SimulatorDataProvider : IXiheDataProvider
    {
        public Texture2D AcquireCameraColorImage()
        {
            var asset = Resources.Load<TextAsset>("Images/Simulation/Sample/test_97_color_scaled");
            var texture = new Texture2D(256, 192, TextureFormat.RGB24, false);

            texture.LoadRawTextureData(asset.bytes);
            texture.Apply();

            return texture;
        }

        public Texture2D AcquireCameraDepthImage()
        {
            var asset = Resources.Load<TextAsset>("Images/Simulation/Sample/test_97_depth_scaled");
            var texture = new Texture2D(256, 192, TextureFormat.R16, false);

            texture.LoadRawTextureData(asset.bytes);
            texture.Apply();

            return texture;
        }

        public CameraIntrinsics AcquireCameraIntrinsic()
        {
            const float sx = 256f / 1280f;
            const float sy = 192f / 1024f;

            var intrinsics = new CameraIntrinsics
            {
                DepthScaler = 65535f / 4000f,
                FocalLength = new float2(1076.51f * sx, 1076.92f * sy),
                PrinciplePoint = new float2(629.969f * sx, 515.181f * sy),
                Resolution = new float2(256, 192)
            };

            return intrinsics;
        }

        public float[] AcquireCameraExtrinsic()
        {
            // return new float[16] {
            //     0.7378529906272888f, -0.46791398525238037f, 0.4864409863948822f, 0f,
            //     0.6748499870300293f, 0.49855300784111023f, -0.54407799243927f, 0f,
            //     0.012066000141203403f, 0.7297260165214539f, 0.6836299896240234f, 0f,
            //     0.0f, 0.0f, 0.0f, 1.0f
            // };

            return new float[16]
            {
                0.7378529906272888f, 0.6748499870300293f, 0.012066000141203403f, 0.0f,
                -0.46791398525238037f, 0.49855300784111023f, 0.7297260165214539f, 0.0f,
                0.4864409863948822f, -0.54407799243927f, 0.6836299896240234f, 0.0f,
                0f, 0f, 0f, 1.0f
            };
        }
    }
}