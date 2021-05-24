using System;
using System.IO;
using System.IO.Compression;
using UnityEngine;


namespace Xihe.DataProvider
{
    public class ReplayDataProvider : IXiheDataProvider, IDisposable
    {
        private int _numFrame = 0;
        private readonly ZipArchive _recArchive;
        private readonly CameraIntrinsics _intrinsics;
        
        public ReplayDataProvider(ZipArchive recArchive, CameraIntrinsics intrinsics)
        {
            _recArchive = recArchive;
            _intrinsics = intrinsics;
        }

        public Texture2D AcquireCameraColorImage()
        {
            var entry = _recArchive.GetEntry($"{_numFrame}/color.bytes");
            if (entry == null) throw new Exception("No color image acquired");

            var t = new Texture2D(256, 192, TextureFormat.RGB24, false);
            
            using var sr = new BinaryReader(entry.Open());
            t.LoadRawTextureData(sr.ReadBytes((int) entry.Length));
            t.Apply();
            
            return t;
        }

        public Texture2D AcquireCameraDepthImage()
        {
            _numFrame += 1;
            
            var entry = _recArchive.GetEntry($"{_numFrame}/depth.bytes");
            if (entry == null) throw new Exception("No depth image acquired");

            var t = new Texture2D(256, 192, TextureFormat.RFloat, false);

            using var sr = new BinaryReader(entry.Open());
            t.LoadRawTextureData(sr.ReadBytes((int) entry.Length));
            t.Apply();
            
            return t;
        }

        public CameraIntrinsics AcquireCameraIntrinsic()
        {
            return _intrinsics;
        }

        public float[] AcquireCameraExtrinsic()
        {
            var entry = _recArchive.GetEntry($"{_numFrame}/extrinsic.bytes");
            if (entry == null) throw new Exception("No extrinsic acquired");

            using var sr = new BinaryReader(entry.Open());
            var extrinsicBytes = sr.ReadBytes((int) entry.Length);
            var extrinsic = new float[16];

            Buffer.BlockCopy(extrinsicBytes, 0, extrinsic, 0, extrinsicBytes.Length);
            return extrinsic;
        }

        public string FetchAdditionalData(string dataName)
        {
            var entry = _recArchive.GetEntry($"{_numFrame}/{dataName}");
            if (entry == null) throw new Exception($"No {dataName} acquired");

            using var sr = new StreamReader(entry.Open());
            var data = sr.ReadToEnd();
            sr.Close();
            
            return data;
        }

        public void Dispose()
        {
            _recArchive?.Dispose();
        }
    }
}