using System.Net.Http;
using System.Threading.Tasks;
using UnityEngine;
using Debug = System.Diagnostics.Debug;
using Xihe.GPU;

namespace Xihe.Network
{
    public enum HttpMethod
    {
        Get,
        Post,
        Put,
        Delete
    }

    public class XiheHttpSession
    {
        private const string Endpoint = "http://cake-graphics.dyn.wpi.edu:8550";
        private readonly HttpClient _client = new HttpClient();
        private readonly string _sessionId;

        public XiheHttpSession(int anchorSize)
        {
            HttpContent content = new StringContent("");
            content.Headers.Add("Anchor-Size", anchorSize.ToString());
            
            var response = _client
                .PostAsync($"{Endpoint}/api/v2/session/", content)
                .Result;

            var resContent = response.Content.ReadAsStringAsync();
            resContent.Wait();

            _sessionId = JsonUtility.FromJson<SessionCreationResponse>(resContent.Result).sid;
        }

        public T PostAsync<T>(string url, HttpContent content)
        {
            content.Headers.Add("Session-ID", _sessionId);
            var response = _client.PostAsync($"{Endpoint}/{url}", content).Result;
            response.EnsureSuccessStatusCode();

            var s = response.Content.ReadAsStringAsync().Result;
            
            return JsonUtility.FromJson<T>(s);
        }
        
        public static async Task<TR> SendJson<TP, TR>(HttpMethod method, string route, TP payload)
        {
            var client = new HttpClient();
            HttpResponseMessage response = null;

            if (method == HttpMethod.Post)
            {
                response = await client.PostAsync(
                    $"{Endpoint}/api/v2/{route}",
                    new StringContent(JsonUtility.ToJson(payload))
                );
            }

            Debug.Assert(response != null, nameof(response) + " != null");
            response.EnsureSuccessStatusCode();
            var s = await response.Content.ReadAsStringAsync();

            client.Dispose();
            return JsonUtility.FromJson<TR>(s);
        }

        public static async Task<TR> SendBytes<TR>(HttpMethod method, string route, ByteArrayContent payload)
        {
            var client = new HttpClient();
            HttpResponseMessage response = null;

            if (method == HttpMethod.Post)
            {
                response = await client.PostAsync(
                    $"{Endpoint}/api/v2/{route}",
                    payload);
            }

            Debug.Assert(response != null, nameof(response) + " != null");
            response.EnsureSuccessStatusCode();
            var s = await response.Content.ReadAsStringAsync();

            client.Dispose();
            return JsonUtility.FromJson<TR>(s);
        }

        public static async void Dump(HttpContent payload)
        {
            var client = new HttpClient();
            var response = await client.PostAsync($"{Endpoint}/api/v2/dump/", payload);
            response.EnsureSuccessStatusCode();
            client.Dispose();
        }

        // XiheHttpSession.DumpPointCloud(
        //     // "point_cloud_spherical_coordinate",
        //     // "point_cloud_spherical_coordinate",
        //     "point_cloud_spherical_ray",
        //     "point_cloud_float4_no_stripe",
        //     probe.TemporaryBuffer,
        //     1280
        // );
        public static void DumpPointCloud(string fileName, string fileType, ComputeBuffer buffer, int anchorSize=-1)
        {
            var payload = buffer.ToHttpBytesPayload();
            payload.Headers.Add("File-Name", fileName);
            payload.Headers.Add("File-Type", fileType);

            if (anchorSize > 0)
            {
                payload.Headers.Add("Anchor-Size", $"{anchorSize}");
            }
            
            Dump(payload);
        }
    }
}