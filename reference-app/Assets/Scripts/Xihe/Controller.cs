using System;
using System.Collections.Generic;
using UnityEngine;
using Xihe.DataProvider;

namespace Xihe
{
    public struct XiheLightingEstimationConfigs
    {
        public bool Debug;
        public bool Enabled;
        public int NumAnchors;
        public int NumTriggerPoolingNeighbors;
        public IXiheDataProvider DataProvider;
    }

    public class XiheController
    {
        internal XiheLightingEstimationConfigs Configs;
        internal readonly EnvironmentScanner Scanner;
        internal readonly GPUDataProcessor GPUDataProcessor;
        internal readonly EdgeInferenceBackend InferenceBackend;
        internal readonly List<XiheLightProbe> Probes = new List<XiheLightProbe>();

        internal int TriggerCounter;

        public Action<float[]> OnEstimatedSHCoefficientsReceived;

        #region Basic Lifecycle

        public XiheController(XiheLightingEstimationConfigs configs)
        {
            Configs = configs;

            // Initialize system components
            Scanner = new EnvironmentScanner(configs.DataProvider);
            InferenceBackend = new EdgeInferenceBackend(configs.NumAnchors);
            GPUDataProcessor = new GPUDataProcessor(new GPUDataProcessorConfigs
            {
                NumAnchors = Configs.NumAnchors,
                NumTriggerPoolingNeighbors = Configs.NumTriggerPoolingNeighbors,
                CameraIntrinsics = configs.DataProvider.AcquireCameraIntrinsic()
            });
        }

        public void Update()
        {
            // Only one estimation is allowed at one frame
            if (!Configs.Enabled) return;

            // Start estimation process
            using var scan = Scanner.AcquireEnvironmentScan();

            if (!scan.HasValue) return;
            foreach (var probe in Probes) XiheEstimation(probe, scan.Value);

            scan.Value.Dispose();
        }

        private void XiheEstimation(XiheLightProbe probe, EnvironmentScan scan)
        {
            var encodedBuffer = GPUDataProcessor.RunGPUPipeline(scan, probe, Configs.Debug);
            if (encodedBuffer == null || encodedBuffer.Length <= 0) return;
            TriggerNewEstimation(probe, encodedBuffer);
            TriggerCounter++;
        }

        public void Dispose()
        {
            GPUDataProcessor.Dispose();
            Probes.ForEach(v => { v.Dispose(); });
        }

        #endregion

        public void PlaceXiheLightProbe(Vector3 position)
        {
            var probe = new XiheLightProbe(Configs.NumAnchors, position);
            Probes.Add(probe);
        }

        internal void TriggerNewEstimation(XiheLightProbe probe, byte[] anchorBuffer)
        {
            var coefficients = InferenceBackend.Inference(anchorBuffer);
            OnEstimatedSHCoefficientsReceived(coefficients);
            var bakedProbes = LightmapSettings.lightProbes.bakedProbes;

            for (var i = 0; i < LightmapSettings.lightProbes.count; i++)
            {
                for (var c = 0; c < 3; c++)
                {
                    for (var b = 0; b < 9; b++)
                    {
                        bakedProbes[i][c, b] = coefficients[c * 9 + b];
                    }
                }
            }

            LightmapSettings.lightProbes.bakedProbes = bakedProbes;
        }
    }
}