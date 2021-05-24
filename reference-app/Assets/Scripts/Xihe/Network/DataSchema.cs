using System;

namespace Xihe.Network
{
    public class EmptyPayload
    {
    }

    public class EmptyResponse : IDisposable
    {
        public void Dispose()
        {
        }
    }

    public class SessionCreationResponse : EmptyResponse
    {
        public bool ok;
        public string sid;
    }

    public class LightingEstimationResponse : EmptyResponse
    {
        public bool ok;
        public float[] coefficients;
    }

    public class NetworkTestingPostPayload
    {
        public string Data;
    }

    public class NetworkTestingPostResponse : EmptyResponse
    {
    }


    public class ClientLogPostPostPayload
    {
        public string Data;
    }

    public class DumpResponseResponse
    {
        public bool Ok;
    }

    public class RecordingStartResponse
    {
        public bool Ok;
        public string ArchiveName;
    }
}