using System.Net.Http;
using UnityEngine;
using Xihe.Network;

namespace Xihe
{
    internal class EdgeInferenceBackend
    {
        private readonly XiheHttpSession _session;

        public EdgeInferenceBackend(int numAnchors)
        {
            _session = new XiheHttpSession(numAnchors);
        }

        internal float[] Inference(byte[] anchorBuffer)
        {
            var content = new ByteArrayContent(anchorBuffer);
            var t = _session
                .PostAsync<LightingEstimationResponse>("api/v2/lighting-estimation/", content);
            return t.coefficients;
        }
    }
}
