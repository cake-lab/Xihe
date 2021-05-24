using System;
using System.Linq;
using UnityEngine;
using Unity.Mathematics;
using Xihe.GPU;
using Xihe.Library;
using Xihe.DataProvider;


namespace Xihe
{
    internal struct GPUDataProcessorConfigs
    {
        public int NumAnchors;
        public int NumTriggerPoolingNeighbors;
        public CameraIntrinsics CameraIntrinsics;
    }

    internal class GPUDataProcessor
    {
        private readonly GPUDataProcessorConfigs _configs;

        private readonly int _pointCloudSize;
        private readonly ManagedShader _shader;

        internal GPUDataProcessor(GPUDataProcessorConfigs configs)
        {
            _configs = configs;

            _pointCloudSize = (int) (configs.CameraIntrinsics.Resolution.x * configs.CameraIntrinsics.Resolution.y);

            var anchorsCartesian = RealitySharp.FibonacciSphere(
                _configs.NumAnchors, RealitySharp.CoordinateType.Cartesian);
            var poolingGrid = RealitySharp.MakePoolingGrid(
                anchorsCartesian, _configs.NumTriggerPoolingNeighbors);

            var cacheGridBinary = Resources.Load<TextAsset>("Binary/grid_1024_512");
            var cacheGridData = new uint[cacheGridBinary.bytes.Length / 4];
            Buffer.BlockCopy(cacheGridBinary.bytes, 0, cacheGridData, 0, cacheGridBinary.bytes.Length);

            #region Register Compute Buffer

            // Init shader
            _shader = new ManagedShader("Shaders/PointCloudProcessing");

            // Make buffers
            _shader.MakeBuffer(PcpShaderBuffers.PcgResultPointCloud, _pointCloudSize * 6, sizeof(float));
            _shader.MakeBuffer(PcpShaderBuffers.PcgCameraIntrinsics, 7, 4); // 4 bytes per unit, 7 units
            _shader.MakeBuffer(PcpShaderBuffers.PcgCameraToWorldMatrix, 16, 4);
            _shader.MakeBuffer(PcpShaderBuffers.LpsAnchors, _configs.NumAnchors * 3, sizeof(float));
            _shader.MakeBuffer(PcpShaderBuffers.LpnCacheGrid, cacheGridData.Length, sizeof(int));
            _shader.MakeBuffer(PcpShaderBuffers.LpnResultBuffer, _pointCloudSize * 2, sizeof(float));
            _shader.MakeBuffer(PcpShaderBuffers.LprResult, _configs.NumAnchors * 4, sizeof(float));
            _shader.MakeBuffer(PcpShaderBuffers.MtdDecisionBuffer, 1, sizeof(int));
            _shader.MakeBuffer(PcpShaderBuffers.MtdPoolingGrid,
                _configs.NumAnchors * _configs.NumTriggerPoolingNeighbors, sizeof(int));


            // Make kernels
            _shader.MakeKernel(PcpShaderKernels.PointCloudGeneration);
            _shader.MakeKernel(PcpShaderKernels.LightProbeNnSearch);
            _shader.MakeKernel(PcpShaderKernels.LightProbeNnSearchAcc);
            _shader.MakeKernel(PcpShaderKernels.LightProbeNnReduce);
            _shader.MakeKernel(PcpShaderKernels.MakeTriggerDecision);
            _shader.MakeKernel(PcpShaderKernels.MergeBuffers);


            // Bind Buffer
            _shader.BindBuffer(PcpShaderKernels.PointCloudGeneration, PcpShaderBuffers.PcgResultPointCloud);
            _shader.BindBuffer(PcpShaderKernels.PointCloudGeneration, PcpShaderBuffers.PcgCameraIntrinsics);
            _shader.BindBuffer(PcpShaderKernels.PointCloudGeneration, PcpShaderBuffers.PcgCameraToWorldMatrix);

            _shader.BindBuffer(PcpShaderKernels.LightProbeNnSearchAcc, PcpShaderBuffers.PcgResultPointCloud);
            _shader.BindBuffer(PcpShaderKernels.LightProbeNnReduce, PcpShaderBuffers.PcgResultPointCloud);

            _shader.BindBuffer(PcpShaderKernels.LightProbeNnSearch, PcpShaderBuffers.LpsAnchors);
            _shader.BindBuffer(PcpShaderKernels.LightProbeNnSearch, PcpShaderBuffers.LpnResultBuffer);
            _shader.BindBuffer(PcpShaderKernels.LightProbeNnSearchAcc, PcpShaderBuffers.LpnCacheGrid);
            _shader.BindBuffer(PcpShaderKernels.LightProbeNnSearchAcc, PcpShaderBuffers.LpnResultBuffer);
            _shader.BindBuffer(PcpShaderKernels.LightProbeNnReduce, PcpShaderBuffers.LpnResultBuffer);
            _shader.BindBuffer(PcpShaderKernels.LightProbeNnReduce, PcpShaderBuffers.LprResult);
            _shader.BindBuffer(PcpShaderKernels.MakeTriggerDecision, PcpShaderBuffers.MtdPoolingGrid);
            _shader.BindBuffer(PcpShaderKernels.MakeTriggerDecision, PcpShaderBuffers.MtdDecisionBuffer);

            // Set buffer data
            _shader.SetInts(PcpShaderNumbers.IntLprNumPoints, _pointCloudSize);
            _shader.WriteBuffer(PcpShaderBuffers.PcgCameraIntrinsics, _configs.CameraIntrinsics.ToArray());
            _shader.WriteBuffer(PcpShaderBuffers.LpsAnchors, anchorsCartesian);
            _shader.WriteBuffer(PcpShaderBuffers.LpnCacheGrid, cacheGridData);
            _shader.WriteBuffer(PcpShaderBuffers.MtdPoolingGrid, poolingGrid);
            _shader.WriteBuffer(PcpShaderBuffers.MtdDecisionBuffer, new[] {0});

            #endregion
        }

        internal void Dispose()
        {
            _shader.Dispose();
        }

        internal byte[] RunGPUPipeline(EnvironmentScan scan, XiheLightProbe probe, bool forceTrigger = false)
        {
            // Generate point cloud data for the incoming scan
            GeneratePointCloud(scan);
            // pointCloudBuffer.Dump("point-cloud", "point_cloud_column_major");


            // Uni-sphere sample
            var sampleResultBuffer = UniSphereSample(probe);
            // Network.XiheHttpSession.DumpPointCloud(
            //     "point-cloud-sampled", "point_cloud_float4_no_stripe",
            //     sampleResultBuffer, _controller.Configs.NumAnchors);


            // Merge sample result and temporary buffer
            MergeProbeBuffer(sampleResultBuffer, probe.TemporaryBuffer);
            // Network.XiheHttpSession.DumpPointCloud(
            //     "point-cloud-temporary", "point_cloud_float4_no_stripe",
            //     probe.TemporaryBuffer, _controller.Configs.NumAnchors);

            // Network.XiheHttpSession.DumpPointCloud(
            //     "point-cloud-fib", "point_cloud_fib_sphere",
            //     probe.TemporaryBuffer, _configs.NumAnchors);

            // Make decision
            var triggerDecision = MakeTriggerDecision(probe);
            if (!triggerDecision && !forceTrigger) return null;
            // _controller.OnVisLog($"Triggered estimation!");

            // Merge buffer if decision is true
            MergeProbeBuffer(probe.TemporaryBuffer, probe.PersistBuffer);
            // Network.XiheHttpSession.DumpPointCloud(
            //     "point-cloud-persist", "point_cloud_float4_no_stripe",
            //     probe.PersistBuffer, _configs.NumAnchors);

            // Encode data
            // var anchorBuffer = EncodeProbeBufferCPUFloat4(probe.PersistBuffer);
            var data = new float[_configs.NumAnchors * 4];
            probe.PersistBuffer.GetData(data);
            var anchorBuffer = EncodeProbeBufferCPUOptimizedStriped(data);

            // pointCloudBuffer.Dispose();
            // sampleResultBuffer.Dispose();

            return anchorBuffer;
        }

        private void GeneratePointCloud(EnvironmentScan scan)
        {
            // Set Extrinsic
            _shader.WriteBuffer(PcpShaderBuffers.PcgCameraToWorldMatrix, scan.MatrixCamToWorld);

            // Set textures
            _shader.BindTexture(
                PcpShaderKernels.PointCloudGeneration,
                PcpShaderTextures.PcgDepthTexture,
                scan.DepthTexture);

            _shader.BindTexture(
                PcpShaderKernels.PointCloudGeneration,
                PcpShaderTextures.PcgColorTexture,
                scan.ColorTexture);

            // Dispatch
            _shader.DispatchKernel(
                PcpShaderKernels.PointCloudGeneration,
                scan.DepthTexture.width / 32,
                scan.DepthTexture.height / 32,
                1
            );

            // return pointCloudBuffer;
        }

        // Generate light probe patch for each incoming probe
        private ComputeBuffer UniSphereSample(XiheLightProbe probe)
        {
            // Set global variables for two kernels
            _shader.SetFloats(
                PcpShaderNumbers.Float3LpsProbePosition,
                probe.Position.x, probe.Position.y, probe.Position.z);

            _shader.WriteBuffer(PcpShaderBuffers.LprResult, new float[_configs.NumAnchors * 4]);

            // Dispatch Kernels
            _shader.DispatchKernel(
                PcpShaderKernels.LightProbeNnSearchAcc,
                (_pointCloudSize + 1024 - 1) / 1024, 1, 1);
            _shader.DispatchKernel(
                PcpShaderKernels.LightProbeNnReduce,
                1, 1, 1);

            return _shader.GetBuffer(PcpShaderBuffers.LprResult);
        }

        private void MergeProbeBuffer(ComputeBuffer inputBuffer, ComputeBuffer baseBuffer)
        {
            _shader.BindExternalBuffer(
                PcpShaderKernels.MergeBuffers,
                PcpShaderBuffers.MbInputBuffer,
                inputBuffer);

            _shader.BindExternalBuffer(
                PcpShaderKernels.MergeBuffers,
                PcpShaderBuffers.MbBaseBuffer,
                baseBuffer);

            _shader.DispatchKernel(
                PcpShaderKernels.MergeBuffers,
                (_configs.NumAnchors + 32 - 1) / 32, 1, 1);
        }

        private bool MakeTriggerDecision(XiheLightProbe probe)
        {
            _shader.BindExternalBuffer(
                PcpShaderKernels.MakeTriggerDecision,
                PcpShaderBuffers.MtdInputBuffer,
                probe.TemporaryBuffer);

            _shader.BindExternalBuffer(
                PcpShaderKernels.MakeTriggerDecision,
                PcpShaderBuffers.MtdBaseBuffer,
                probe.PersistBuffer);

            _shader.WriteBuffer(PcpShaderBuffers.MtdDecisionBuffer, new[] {0});

            _shader.DispatchKernel(
                PcpShaderKernels.MakeTriggerDecision,
                (_configs.NumAnchors + 32 - 1) / 32, 1, 1
            );

            var decision = _shader.ReadBuffer<int>(PcpShaderBuffers.MtdDecisionBuffer)[0] > 0;

            return decision;
        }

        internal byte[] EncodeProbeBufferCPUFloat4(float[] data)
        {
            // var data = new float[_controller.Configs.NumAnchors * 4];
            // buffer.GetData(data);

            var result = new byte[_configs.NumAnchors * 4 * 4];
            Buffer.BlockCopy(data, 0, result, 0, data.Length);

            return result;
        }

        internal byte[] EncodeProbeBufferCPUFloat4Striped(float[] data)
        {
            // var data = new float[_controller.Configs.NumAnchors * 4];
            // buffer.GetData(data);

            var result = new byte[_configs.NumAnchors * 18];

            var counter = 0;
            for (var i = 0; i < _configs.NumAnchors; i++)
            {
                var r = data[i * 4 + 0];
                var g = data[i * 4 + 1];
                var b = data[i * 4 + 2];
                var w = data[i * 4 + 3];

                if (r + b + g < 0.05) continue;

                var indexBytes = BitConverter.GetBytes((ushort) i);

                var rBytes = BitConverter.GetBytes(r);
                var gBytes = BitConverter.GetBytes(g);
                var bBytes = BitConverter.GetBytes(b);
                var wBytes = BitConverter.GetBytes(w);

                result[i * 7 + 0] = indexBytes[0];
                result[i * 7 + 1] = indexBytes[1];

                result[i * 7 + 2] = rBytes[0];
                result[i * 7 + 3] = rBytes[1];
                result[i * 7 + 4] = rBytes[2];
                result[i * 7 + 5] = rBytes[3];

                result[i * 7 + 6] = gBytes[0];
                result[i * 7 + 7] = gBytes[1];
                result[i * 7 + 8] = gBytes[2];
                result[i * 7 + 9] = gBytes[3];

                result[i * 7 + 10] = bBytes[0];
                result[i * 7 + 11] = bBytes[1];
                result[i * 7 + 12] = bBytes[2];
                result[i * 7 + 13] = bBytes[3];

                result[i * 7 + 14] = wBytes[0];
                result[i * 7 + 15] = wBytes[1];
                result[i * 7 + 16] = wBytes[2];
                result[i * 7 + 17] = wBytes[3];

                counter++;
            }

            var segment = new ArraySegment<byte>(result, 0, counter * 18);
            return segment.ToArray();
        }

        internal byte[] EncodeProbeBufferCPUOptimized(float[] data)
        {
            // var data = new float[_controller.Configs.NumAnchors * 4];
            // buffer.GetData(data);
            var result = new byte[_configs.NumAnchors * 7];

            for (var i = 0; i < _configs.NumAnchors; i++)
            {
                var r = data[i * 4 + 0];
                var g = data[i * 4 + 1];
                var b = data[i * 4 + 2];
                var w = data[i * 4 + 3];

                var indexBytes = BitConverter.GetBytes((ushort) i);
                var wHalfBytes = BitConverter.GetBytes(((half) (w)).value);

                result[i * 7 + 0] = indexBytes[0];
                result[i * 7 + 1] = indexBytes[1];
                result[i * 7 + 2] = (byte) (r * 255);
                result[i * 7 + 3] = (byte) (g * 255);
                result[i * 7 + 4] = (byte) (b * 255);
                result[i * 7 + 5] = wHalfBytes[0];
                result[i * 7 + 6] = wHalfBytes[1];
            }

            return result;
        }

        // Optimized probe anchor buffer encoding
        // color -> uint8 * 3
        // distance -> half
        // optimization rate: (2 + 1 * 3 + 2) / (4 * 3)
        internal byte[] EncodeProbeBufferCPUOptimizedStriped(float[] data)
        {
            // var data = new float[_controller.Configs.NumAnchors * 4];
            // buffer.GetData(data);
            var result = new byte[_configs.NumAnchors * 7];

            var counter = 0;
            for (var i = 0; i < _configs.NumAnchors; i++)
            {
                var r = data[i * 4 + 0];
                var g = data[i * 4 + 1];
                var b = data[i * 4 + 2];
                var w = data[i * 4 + 3];

                if (r + b + g < 0.05) continue;

                var indexBytes = BitConverter.GetBytes((ushort) i);
                var wHalfBytes = BitConverter.GetBytes(((half) (w)).value);

                result[i * 7 + 0] = indexBytes[0];
                result[i * 7 + 1] = indexBytes[1];
                result[i * 7 + 2] = (byte) (r * 255);
                result[i * 7 + 3] = (byte) (g * 255);
                result[i * 7 + 4] = (byte) (b * 255);
                result[i * 7 + 5] = wHalfBytes[0];
                result[i * 7 + 6] = wHalfBytes[1];

                counter++;
            }

            var segment = new ArraySegment<byte>(result, 0, counter * 7);
            return segment.ToArray();
        }
    }
}