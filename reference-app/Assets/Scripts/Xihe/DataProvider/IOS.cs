using System;
using System.Collections.Generic;
using System.Linq;
using Unity.Collections;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.XR.ARFoundation;
using UnityEngine.XR.ARSubsystems;

namespace Xihe.DataProvider
{
    public class IOSDataProvider : IXiheDataProvider
    {
        private readonly Camera _arMainCamera;
        private readonly ARCameraManager _cameraManager;
        private readonly XROcclusionSubsystem _occlusionSubsystem;

        private int _textureWidth;
        private int _textureHeight;

        public IOSDataProvider(ARCameraManager cameraManager)
        {
            _cameraManager = cameraManager;
            _arMainCamera = _cameraManager.GetComponent<Camera>();
            _occlusionSubsystem = CreateOcclusionSubsystem();
        }

        private static Texture2D EncodeCPUImageToTexture2D(
            XRCpuImage image,
            XRCpuImage.ConversionParams conversionParams,
            bool dispose = true
        )
        {
            var size = image.GetConvertedDataSize(conversionParams);
            var buffer = new NativeArray<byte>(size, Allocator.Temp);
            image.Convert(conversionParams, buffer);

            var width = conversionParams.outputDimensions.x;
            var height = conversionParams.outputDimensions.y;

            var t = new Texture2D(width, height, conversionParams.outputFormat, false);
            t.LoadRawTextureData(buffer);
            t.Apply();

            if (dispose) image.Dispose();

            return t;
        }

        public Texture2D AcquireCameraColorImage()
        {
            var result = _cameraManager
                .TryAcquireLatestCpuImage(out var image);

            if (!result) throw new Exception("No color image acquired");

            // Pass converted data to Xihe
            var conversionParams = new XRCpuImage.ConversionParams(
                image, TextureFormat.RGB24,
                XRCpuImage.Transformation.MirrorY
            )
            {
                outputDimensions = new Vector2Int(_textureWidth, _textureHeight)
            };

            return EncodeCPUImageToTexture2D(image, conversionParams);
        }

        public Texture2D AcquireCameraDepthImage()
        {
            var result = _occlusionSubsystem
                .TryAcquireEnvironmentDepthCpuImage(out var image);

            if (!result) throw new Exception("No depth image acquired");

            // Pass converted data to Xihe
            var depthTexture = EncodeCPUImageToTexture2D(
                image, new XRCpuImage.ConversionParams(
                    image, image.format.AsTextureFormat(),
                    XRCpuImage.Transformation.MirrorY
                ));

            _textureWidth = image.width;
            _textureHeight = image.height;

            return depthTexture;
        }

        public CameraIntrinsics AcquireCameraIntrinsic()
        {
            var result = _cameraManager.TryGetIntrinsics(out var arCamIntrinsics);
            if (!result) throw new Exception("TryGetIntrinsics Failed");

            AcquireCameraDepthImage();

            var sx = (float) _textureWidth / arCamIntrinsics.resolution.x;
            var sy = (float) _textureHeight / arCamIntrinsics.resolution.y;

            var intrinsics = new CameraIntrinsics
            {
                DepthScaler = 1.0f,
                FocalLength = new float2(
                    arCamIntrinsics.focalLength.x * sx,
                    arCamIntrinsics.focalLength.y * sy),
                PrinciplePoint = new float2(
                    arCamIntrinsics.principalPoint.x * sx,
                    arCamIntrinsics.principalPoint.y * sy),
                Resolution = new int2(_textureWidth, _textureHeight),
            };

            return intrinsics;
        }

        public float[] AcquireCameraExtrinsic()
        {
            var c = _arMainCamera.cameraToWorldMatrix;
            return new[]
            {
                c[0, 0], c[1, 0], c[2, 0], c[3, 0],
                c[0, 1], c[1, 1], c[2, 1], c[3, 1],
                c[0, 2], c[1, 2], c[2, 2], c[3, 2],
                c[0, 3], c[1, 3], c[2, 3], c[3, 3]
            };
        }

        private static XROcclusionSubsystem CreateOcclusionSubsystem()
        {
            var descriptors = new List<XROcclusionSubsystemDescriptor>();
            SubsystemManager.GetSubsystemDescriptors(descriptors);

            return (from descriptor in descriptors
                where descriptor.environmentDepthImageSupported == Supported.Supported
                select descriptor.Create()).FirstOrDefault();
        }

        public static bool CheckAvailability()
        {
            var descriptors = new List<XROcclusionSubsystemDescriptor>();
            SubsystemManager.GetSubsystemDescriptors(descriptors);

            return descriptors
                .Any(descriptor => descriptor
                    .environmentDepthImageSupported == Supported.Supported);
        }
    }
}