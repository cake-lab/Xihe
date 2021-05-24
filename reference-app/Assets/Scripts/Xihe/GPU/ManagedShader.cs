using System;
using System.Collections.Generic;
using UnityEngine;

namespace Xihe.GPU
{
    internal class ManagedShader : IDisposable
    {
        private readonly ComputeShader _shader;
        private readonly Dictionary<string, ComputeBuffer> _bufferRefs;
        private readonly Dictionary<string, ComputeKernel> _kernelRefs;

        public ManagedShader(string shaderName)
        {
            _bufferRefs = new Dictionary<string, ComputeBuffer>();
            _kernelRefs = new Dictionary<string, ComputeKernel>();
            _shader = Resources.Load<ComputeShader>(shaderName);
        }

        public ComputeBuffer MakeBuffer(string bufferName, int count, int stride)
        {
            var buffer = new ComputeBuffer(count, stride);
            _bufferRefs[bufferName] = buffer;
            return buffer;
        }

        public ComputeBuffer MakeBuffer(string bufferName, int count, int stride, ComputeBufferType type)
        {
            var buffer = new ComputeBuffer(count, stride, type);
            _bufferRefs[bufferName] = buffer;
            return buffer;
        }

        public ComputeKernel MakeKernel(string kernelName)
        {
            var kernel = new ComputeKernel(_shader, kernelName);
            _kernelRefs[kernelName] = kernel;
            return kernel;
        }

        public void DispatchKernel(string kernelName, int threadGroupsX, int threadGroupsY, int threadGroupsZ)
        {
            _kernelRefs[kernelName].Dispatch(threadGroupsX, threadGroupsY, threadGroupsZ);
        }

        public void BindBuffer(string kernelName, string bufferName)
        {
            _kernelRefs[kernelName].SetBuffer(bufferName, _bufferRefs[bufferName]);
        }

        public void BindExternalBuffer(string kernelName, string bufferName, ComputeBuffer buffer)
        {
            // if (_bufferRefs.ContainsKey(bufferName)) _bufferRefs[bufferName].Dispose();
            _kernelRefs[kernelName].SetBuffer(bufferName, buffer);
            // _bufferRefs[bufferName] = buffer;
        }

        public void BindTexture(string kernelName, string textureName, Texture2D texture)
        {
            _kernelRefs[kernelName].SetTexture(textureName, texture);
        }

        public void WriteBuffer(string bufferName, Array data)
        {
            _bufferRefs[bufferName].SetData(data);
        }

        public ComputeBuffer GetBuffer(string bufferName)
        {
            return _bufferRefs[bufferName];
        }

        public T[] ReadBuffer<T>(string bufferName)
        {
            var buffer = _bufferRefs[bufferName];
            var result = new T[buffer.count];

            buffer.GetData(result);

            return result;
        }

        public void SetInts(string name, params int[] data)
        {
            _shader.SetInts(name, data);
        }

        public void SetFloats(string name, params float[] data)
        {
            _shader.SetFloats(name, data);
        }

        public void Dispose()
        {
            foreach (var k in _bufferRefs.Keys)
            {
                _bufferRefs[k].Dispose();
            }
        }
    }
}