using System;
using System.Net.Http;
using System.Collections.Generic;
using Unity.Mathematics;
using UnityEngine;
using Xihe.Network;

namespace Xihe.GPU
{
    internal static class ComputeBufferExtension
    {
        public static ByteArrayContent ToHttpBytesPayload(this ComputeBuffer buffer)
        {
            var bufferBytes = new byte[buffer.count * buffer.stride];
            buffer.GetData(bufferBytes);
            return new ByteArrayContent(bufferBytes);
        }

        public static void Dump(this ComputeBuffer buffer, string fileName, string fileType)
        {
            var payload = buffer.ToHttpBytesPayload();
            
            payload.Headers.Add("File-Name", fileName);
            payload.Headers.Add("File-Type", fileType);
            
            XiheHttpSession.Dump(payload);
        }
    }

    internal class ComputeKernel
    {
        private readonly int _kernelIndex;
        private readonly ComputeShader _shader;

        internal ComputeKernel(ComputeShader shader, string kernelName)
        {
            _shader = shader;
            _kernelIndex = shader.FindKernel(kernelName);
        }

        internal void SetTexture(string textureName, Texture2D texture)
        {
            _shader.SetTexture(_kernelIndex, textureName, texture);
        }

        internal void SetBuffer(string bufferName, ComputeBuffer buffer)
        {
            _shader.SetBuffer(_kernelIndex, bufferName, buffer);
        }

        internal void Dispatch(int threadGroupsX, int threadGroupsY, int threadGroupsZ)
        {
            _shader.Dispatch(_kernelIndex, threadGroupsX, threadGroupsY, threadGroupsZ);
        }
    }
}
