using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.XR.ARSubsystems;

namespace Xihe
{
    public class XiheLightProbe
    {
        public Vector3 Position;
        public GameObject AttachedGameObject;
        public List<int> AttachedBakedProbeIndexes = new List<int>();

        internal readonly ComputeBuffer PersistBuffer;
        internal readonly ComputeBuffer TemporaryBuffer;

        public XiheLightProbe(int numAnchors, Vector3 worldPosition)
        {
            Position = worldPosition;

            PersistBuffer = new ComputeBuffer(numAnchors, sizeof(float) * 4);
            PersistBuffer.SetData(new float[numAnchors * 4]);

            TemporaryBuffer = new ComputeBuffer(1280, sizeof(float) * 4);
            TemporaryBuffer.SetData(new float[numAnchors * 4]);
        }

        internal void Dispose()
        {
            PersistBuffer.Dispose();
            TemporaryBuffer.Dispose();
        }
    }
}
