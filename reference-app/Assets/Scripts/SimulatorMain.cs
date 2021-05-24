using System.Net.Http;
using System.IO;
using System.IO.Compression;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using UnityEngine.SceneManagement;
using UnityEngine.UI;
using Xihe;
using Xihe.DataProvider;
using Xihe.Network;
using Xihe.Utilities;
using Debug = UnityEngine.Debug;

public class SimulatorMain : MonoBehaviour
{
    public Camera mainCamera;
    public GameObject objectToRender;
    public RawImage replayColorImage;

    // Internal properties
    private XiheController _xiheController;

    private bool _playing;
    private bool _replayMode;
    private int _replayFrameLeft;

    private ReplayDataProvider _dataProvider;
    private FileStream _zipStream;
    private Dropdown _archiveDropdown;

    private StreamWriter _logStream;

    #region Lifecycle control

    private void Start()
    {
        #region Registring Button Events

        Utils.BindButtonClick("Play", OnPlayClick);
        Utils.BindButtonClick("NextFrame", OnNextFrameClick);
        Utils.BindButtonClick("CreateSimXihe", OnCreateSimXiheClick);
        Utils.BindButtonClick("CreateReplayXihe", OnCreateReplayXiheClick);
        Utils.BindButtonClick("OpenRecorder", OnOpenRecorderClick);

        #endregion

        // Set UI components state
        _archiveDropdown = GameObject.Find("ArchiveDropDown").GetComponent<Dropdown>();
        _archiveDropdown.ClearOptions();
        foreach (var fileName in Directory.GetFiles(Application.persistentDataPath))
        {
            _archiveDropdown.options.Add(new Dropdown.OptionData(
                Path.GetFileName(fileName)));
        }

        _archiveDropdown.RefreshShownValue();

        replayColorImage.enabled = false;

        // Keep Screen Always On
        Screen.sleepTimeout = SleepTimeout.NeverSleep;
        // Application.targetFrameRate = 20; // Set framerate
        Debug.Log("Application Started");
        // Debug.Log($"Num of light probes in lightmap: {LightmapSettings.lightProbes.count}");
        
        Debug.Log($"Please make sure you have put the recording files at {Application.persistentDataPath}");
    }

    // Update is called once per frame
    private void Update()
    {
        if (!_playing) return;

        switch (_replayMode)
        {
            case true when _replayFrameLeft <= 0:
                replayColorImage.enabled = false;
                _xiheController?.Dispose();
                _xiheController = null;
                _playing = false;
                return;
            case true:
                _replayFrameLeft--;
                replayColorImage.texture = _dataProvider.AcquireCameraColorImage();

                var m = _dataProvider
                    .FetchAdditionalData("cameraTransform.txt")
                    .Trim()
                    .Split('\n');

                var pos = m[0]
                    .Split(',')
                    .Select(float.Parse)
                    .ToArray();

                var rot = m[1]
                    .Split(',')
                    .Select(float.Parse)
                    .ToArray();

                var trans = mainCamera.transform;
                trans.position = new Vector3(pos[0], pos[1], pos[2]);
                trans.rotation = new Quaternion(rot[0], rot[1], rot[2], rot[3]);

                if (_replayFrameLeft == 0)
                {
                    Debug.Log($"All frames played, triggered {_xiheController.TriggerCounter}");
                    return;
                }

                break;
        }

        _xiheController?.Update();

        // Trigger frame by frame in debug mode
        if (_xiheController is {Configs: {Debug: true}}) _playing = false;
    }

    private void OnDestroy()
    {
        _xiheController?.Dispose();
        _logStream?.Dispose();
    }

    #endregion

    private void OnCreateSimXiheClick()
    {
        _xiheController = new XiheController(new XiheLightingEstimationConfigs
        {
            Debug = true,
            Enabled = false,
            NumAnchors = 1280,
            NumTriggerPoolingNeighbors = 4,
            DataProvider = new SimulatorDataProvider()
        });

        _xiheController.PlaceXiheLightProbe(new Vector3(-
            0.5962867736816406f, -1.6634629964828491f + 0.4f, -0.7016268372535706f));

        _replayMode = false;

        Debug.Log("XiheController sim created!");
    }

    private void OnCreateReplayXiheClick()
    {
        var archiveName = _archiveDropdown.options[_archiveDropdown.value].text;

        _zipStream?.Close();
        _zipStream = File.Open(Path.Combine(
            Application.persistentDataPath, archiveName), FileMode.Open);

        var recArchive = new ZipArchive(_zipStream);

        // Count frames
        _replayFrameLeft = recArchive
            .Entries
            .Where(v => v.Name.Contains("color.bytes"))
            .Select(v => v.Name)
            .ToArray()
            .Length;

        // Read meta info
        var infoEntry = recArchive.GetEntry("info.txt");

        using var sr = new StreamReader(infoEntry!.Open());

        // Setup object and Xihe probe
        var objPositionString = sr.ReadLine();
        var pos = objPositionString!
            .Split(',')
            .Select(v => float.Parse(v.Trim()))
            .ToArray();

        var objPos = new Vector3(pos[0], pos[1], pos[2]);

        var objRotationString = sr.ReadLine();
        var rot = objRotationString!
            .Split(',')
            .Select(v => float.Parse(v.Trim()))
            .ToArray();

        var objRot = new Quaternion(rot[0], rot[1], rot[2], rot[3]);

        Instantiate(objectToRender, objPos, objRot);

        // Intrinsics in the second line
        var intrinsics = CameraIntrinsics.CreateFromString(sr.ReadLine());
        sr.Close();

        // Data provider
        _dataProvider = new ReplayDataProvider(recArchive, intrinsics);

        _xiheController = new XiheController(new XiheLightingEstimationConfigs
        {
            Debug = true,
            Enabled = false,
            NumAnchors = 1280,
            NumTriggerPoolingNeighbors = 4,
            DataProvider = _dataProvider
        }) {OnEstimatedSHCoefficientsReceived = OnEdgeEstimationReceived};

        var probePos = objPos;
        probePos.y += 0.1f;
        _xiheController.PlaceXiheLightProbe(probePos);

        _replayMode = true;
        replayColorImage.enabled = true;

        Debug.Log($"XiheController replay created! Total frames {_replayFrameLeft}");

        //_logStream
        _logStream = new StreamWriter(File.Create(Path.Combine(
            Application.persistentDataPath, $"{archiveName}.log")));
    }

    private void OnEdgeEstimationReceived(float[] coefficients)
    {
        _logStream.WriteLine($"{_replayFrameLeft},{string.Join(",", coefficients)}");
    }

    private void OnPlayClick()
    {
        _playing = true;
        _xiheController!.Configs.Debug = false;
        _xiheController!.Configs.Enabled = true;
    }

    private void OnNextFrameClick()
    {
        _playing = true;
        _xiheController!.Configs.Enabled = true;
    }

    private void OnOpenRecorderClick()
    {
        SceneManager.LoadScene("MainScene");
    }

    private void OnProfileEncodingClick()
    {
        Debug.Log("OnProfileEncodingClick");
        XiheDebugger.ProfileEncoding(_xiheController);
    }

    private void OnProfileNetworkingClick()
    {
        var log = XiheDebugger.ProfileNetworking(_xiheController);
        var logContent = new StringContent(log);
        logContent.Headers.Add("File-Type", "client_log");
        logContent.Headers.Add("File-Name", $"client_log");
        XiheHttpSession.Dump(logContent);
    }
}