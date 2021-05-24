using System;
using UnityEngine;
using Xihe.DataProvider;

namespace Xihe
{
    public struct EnvironmentScan : IDisposable
    {
        public Texture2D ColorTexture;
        public Texture2D DepthTexture;
        public float[] MatrixCamToWorld;

        public void Dispose()
        {
            UnityEngine.Object.Destroy(ColorTexture);
            UnityEngine.Object.Destroy(DepthTexture);
        }
    }

    internal class EnvironmentScanner
    {
        private readonly bool _acquireHighResColor;
        private readonly IXiheDataProvider _dataProvider;

        internal EnvironmentScanner(IXiheDataProvider dataProvider, bool acquireHighResColor = false)
        {
            _dataProvider = dataProvider;
            _acquireHighResColor = acquireHighResColor;
        }

        internal EnvironmentScan? AcquireEnvironmentScan()
        {
            EnvironmentScan scan;

            try
            {
                scan = new EnvironmentScan
                {
                    DepthTexture = _dataProvider.AcquireCameraDepthImage(),
                    ColorTexture = _dataProvider.AcquireCameraColorImage(),
                    MatrixCamToWorld = _dataProvider.AcquireCameraExtrinsic()
                };
            }
            catch (Exception e)
            {
                Debug.Log(e.Message);
                return null;
            }

            return scan;
        }
    }
}