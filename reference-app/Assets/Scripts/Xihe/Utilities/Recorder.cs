using System;
using System.IO;
using System.IO.Compression;
using UnityEngine;
using Xihe.DataProvider;
using CompressionLevel = System.IO.Compression.CompressionLevel;

namespace Xihe.Utilities
{
    public class XiheRecorder : IDisposable
    {
        public bool InRecording;

        private bool _enabled;
        private int _currentFrameNum = -1;
        private ZipArchive _recArchive;
        private CameraIntrinsics _intrinsic;
        private readonly EnvironmentScanner _envScanner;

        public XiheRecorder(IXiheDataProvider dataProvider)
        {
            _intrinsic = dataProvider.AcquireCameraIntrinsic();
            _envScanner = new EnvironmentScanner(dataProvider, true);
        }

        public void Update()
        {
            if (!_enabled) return;

            if (InRecording) _currentFrameNum += 1;
            else _currentFrameNum = -1;

            using var scan = _envScanner.AcquireEnvironmentScan();

            if (!scan.HasValue) return;
            SaveFrame(scan.Value);
        }

        private void SaveFrame(EnvironmentScan scan)
        {
            // Color bytes
            var colorBytes = scan.ColorTexture.GetRawTextureData();
            var writer = new BinaryWriter(_recArchive
                .CreateEntry($"{_currentFrameNum}/color.bytes", CompressionLevel.Optimal)
                .Open());
            writer.Write(colorBytes, 0, colorBytes.Length);
            writer.Close();

            // Depth bytes
            var depthBytes = scan.DepthTexture.GetRawTextureData();
            writer = new BinaryWriter(_recArchive
                .CreateEntry($"{_currentFrameNum}/depth.bytes", CompressionLevel.Optimal)
                .Open());
            writer.Write(depthBytes, 0, depthBytes.Length);
            writer.Close();

            // Extrinsic bytes
            var extrinsic = scan.MatrixCamToWorld;
            var extrinsicBytes = new byte[extrinsic.Length * 4];
            Buffer.BlockCopy(extrinsic, 0, extrinsicBytes, 0, extrinsicBytes.Length);
            writer = new BinaryWriter(_recArchive
                .CreateEntry($"{_currentFrameNum}/extrinsic.bytes", CompressionLevel.Optimal)
                .Open());
            writer.Write(extrinsicBytes, 0, extrinsicBytes.Length);
            writer.Close();
            
            scan.Dispose();
        }

        public void SaveAdditionalInfo(string fileName, byte[] content)
        {
            var writer = new BinaryWriter(_recArchive
                .CreateEntry($"{_currentFrameNum}/{fileName}.bytes", CompressionLevel.Optimal)
                .Open());
            writer.Write(content, 0, content.Length);
            writer.Close();
        }
        
        public void SaveAdditionalInfo(string fileName, string content)
        {
            var writer = new StreamWriter(_recArchive
                .CreateEntry($"{_currentFrameNum}/{fileName}.txt", CompressionLevel.Optimal)
                .Open());
            writer.Write(content);
            writer.Close();
        }

        /// <summary>
        /// Start recording
        /// </summary>
        public void Start(RecordingConfig config)
        {
            _enabled = true;

            _recArchive = new ZipArchive(File.Create(Path.Combine(
                    Application.persistentDataPath,
                    DateTime.Now.ToString("MM_dd_yyyy-HH_mm_ss") + ".zip")),
                ZipArchiveMode.Update);

            var infoFile = _recArchive.CreateEntry("info.txt", CompressionLevel.Fastest);

            var p = config.ObjectPose.position;
            var r = config.ObjectPose.rotation;

            var sw = new StreamWriter(infoFile.Open());
            sw.WriteLine($"{p.x},{p.y},{p.z}");
            sw.WriteLine($"{r.x},{r.y},{r.z},{r.w}");
            sw.WriteLine(string.Join(",", _intrinsic.ToArray()));
            sw.Close();

            InRecording = true;
        }

        /// <summary>
        /// End recording
        /// </summary>
        public void Stop()
        {
            _enabled = false;
            InRecording = false;
            _recArchive.Dispose();
        }

        public void Dispose()
        {
            _recArchive?.Dispose();
        }
    }

    public struct RecordingConfig
    {
        public Pose ObjectPose;
    }
}