using System;
using System.Linq;
using UnityEngine;

namespace Xihe.Utilities
{
    // This profiler is used for testing and debugging purpose only
    // It will be removed in the future
    public static class XiheDebugger
    {
        public static void ProfileEncoding(XiheController xiheController)
        {
            const int numAnchors = 1280;
            var package = Resources.Load<TextAsset>($"Simulation/sphere_package_{numAnchors}");
            var sphereData = new float[2037 * numAnchors * 4];
            Buffer.BlockCopy(package.bytes, 0, sphereData, 0, package.bytes.Length);

            var buffer = new ArraySegment<float>(
                sphereData, 0 * numAnchors * 4, numAnchors * 4
            ).ToArray();
            
            xiheController.GPUDataProcessor.EncodeProbeBufferCPUFloat4(buffer);
            xiheController.GPUDataProcessor.EncodeProbeBufferCPUOptimized(buffer);
            xiheController.GPUDataProcessor.EncodeProbeBufferCPUFloat4Striped(buffer);
            xiheController.GPUDataProcessor.EncodeProbeBufferCPUOptimizedStriped(buffer);
        }

        public static string ProfileNetworking(XiheController xiheController)
        {
            var log = "data_idx,n_anchors,encoding,n_uspc,network_time\n";

            var stopwatch = new System.Diagnostics.Stopwatch();

            // var anchorGroup = new [] {512, 768, 1024, 1280, 1536, 1792, 2048};
            // var anchorGroup = new[] {512};
            // var anchorGroup = new[] {1280};
            var anchorGroup = new[] {2048};


            foreach (var numAnchors in anchorGroup)
            {
                var package = Resources.Load<TextAsset>($"Simulation/sphere_package_{numAnchors}");
                var sphereData = new float[2037 * numAnchors * 4];
                Buffer.BlockCopy(package.bytes, 0, sphereData, 0, package.bytes.Length);

                xiheController.Configs.NumAnchors = numAnchors;
                for (var i = 0; i < 2037; i++)
                {
                    var buffer = new ArraySegment<float>(sphereData, i * numAnchors * 4, numAnchors * 4);

                    // EncodeProbeBufferCPUFloat4
                    // var encodedBuffer = _xiheController._pointCloudProcessor.EncodeProbeBufferCPUFloat4(buffer.ToArray());

                    // stopwatch.Restart();
                    // _xiheController._inferenceBackend.Inference(encodedBuffer);
                    // stopwatch.Stop();
                    // log += $"{i},{numAnchors},EncodeProbeBufferCPUFloat4,1,{stopwatch.ElapsedMilliseconds}\n";
                    //
                    //
                    // encodedBuffer = encodedBuffer.Concat(encodedBuffer).ToArray();
                    // stopwatch.Restart();
                    // _xiheController._inferenceBackend.Inference(encodedBuffer);
                    // stopwatch.Stop();
                    // log += $"{i},{numAnchors},EncodeProbeBufferCPUFloat4,2,{stopwatch.ElapsedMilliseconds}\n";
                    //
                    // encodedBuffer = encodedBuffer.Concat(encodedBuffer).ToArray();
                    // stopwatch.Restart();
                    // _xiheController._inferenceBackend.Inference(encodedBuffer);
                    // stopwatch.Stop();
                    // log += $"{i},{numAnchors},EncodeProbeBufferCPUFloat4,4,{stopwatch.ElapsedMilliseconds}\n";
                    //
                    //
                    // EncodeProbeBufferCPUFloat4Striped
                    // var encodedBuffer = _xiheController._pointCloudProcessor.EncodeProbeBufferCPUFloat4Striped(buffer.ToArray());
                    //
                    // stopwatch.Restart();
                    // _xiheController._inferenceBackend.Inference(encodedBuffer);
                    // stopwatch.Stop();
                    // log += $"{i},{numAnchors},EncodeProbeBufferCPUFloat4Striped,1,{stopwatch.ElapsedMilliseconds}\n";
                    //
                    //
                    // encodedBuffer = encodedBuffer.Concat(encodedBuffer).ToArray();
                    // stopwatch.Restart();
                    // _xiheController._inferenceBackend.Inference(encodedBuffer);
                    // stopwatch.Stop();
                    // log += $"{i},{numAnchors},EncodeProbeBufferCPUFloat4Striped,2,{stopwatch.ElapsedMilliseconds}\n";
                    //
                    // encodedBuffer = encodedBuffer.Concat(encodedBuffer).ToArray();
                    // stopwatch.Restart();
                    // _xiheController._inferenceBackend.Inference(encodedBuffer);
                    // stopwatch.Stop();
                    // log += $"{i},{numAnchors},EncodeProbeBufferCPUFloat4Striped,4,{stopwatch.ElapsedMilliseconds}\n";
                    //
                    //
                    //
                    // // EncodeProbeBufferCPUOptimized
                    // encodedBuffer = _xiheController._pointCloudProcessor.EncodeProbeBufferCPUOptimized(buffer.ToArray());
                    //
                    // stopwatch.Restart();
                    // _xiheController._inferenceBackend.Inference(encodedBuffer);
                    // stopwatch.Stop();
                    // log += $"{i},{numAnchors},EncodeProbeBufferCPUOptimized,1,{stopwatch.ElapsedMilliseconds}\n";
                    //
                    //
                    // encodedBuffer = encodedBuffer.Concat(encodedBuffer).ToArray();
                    // stopwatch.Restart();
                    // _xiheController._inferenceBackend.Inference(encodedBuffer);
                    // stopwatch.Stop();
                    // log += $"{i},{numAnchors},EncodeProbeBufferCPUOptimized,2,{stopwatch.ElapsedMilliseconds}\n";
                    //
                    // encodedBuffer = encodedBuffer.Concat(encodedBuffer).ToArray();
                    // stopwatch.Restart();
                    // _xiheController._inferenceBackend.Inference(encodedBuffer);
                    // stopwatch.Stop();
                    // log += $"{i},{numAnchors},EncodeProbeBufferCPUOptimized,4,{stopwatch.ElapsedMilliseconds}\n";


                    // EncodeProbeBufferCPUOptimizedStriped
                    var encodedBuffer =
                        xiheController.GPUDataProcessor.EncodeProbeBufferCPUOptimizedStriped(buffer.ToArray());

                    stopwatch.Restart();
                    xiheController.InferenceBackend.Inference(encodedBuffer);
                    stopwatch.Stop();
                    log += $"{i},{numAnchors},EncodeProbeBufferCPUOptimizedStriped,1,{stopwatch.ElapsedMilliseconds}\n";

                    // encodedBuffer = encodedBuffer.Concat(encodedBuffer).ToArray();
                    // stopwatch.Restart();
                    // _xiheController._inferenceBackend.Inference(encodedBuffer);
                    // stopwatch.Stop();
                    // log += $"{i},{numAnchors},EncodeProbeBufferCPUOptimizedStriped,2,{stopwatch.ElapsedMilliseconds}\n";
                    //
                    // encodedBuffer = encodedBuffer.Concat(encodedBuffer).ToArray();
                    // stopwatch.Restart();
                    // _xiheController._inferenceBackend.Inference(encodedBuffer);
                    // stopwatch.Stop();
                    // log += $"{i},{numAnchors},EncodeProbeBufferCPUOptimizedStriped,4,{stopwatch.ElapsedMilliseconds}\n";
                }
            }

            return log;
        }

        public static void ProfileGPUPipeline(XiheController xiheController)
        {
            var scan = xiheController.Scanner.AcquireEnvironmentScan();

            if (!scan.HasValue) return;

            foreach (var probe in xiheController.Probes)
            {
                var encodedBuffer = xiheController.GPUDataProcessor.RunGPUPipeline(scan.Value, probe);

                if (encodedBuffer != null && encodedBuffer.Length > 0)
                {
                    xiheController.TriggerNewEstimation(probe, encodedBuffer);
                }
            }

            scan.Value.Dispose();
        }
    }
}