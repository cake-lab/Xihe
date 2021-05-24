using System.Collections.Generic;
using UnityEngine;
using UnityEngine.SceneManagement;
using UnityEngine.XR.ARFoundation;
using UnityEngine.XR.ARSubsystems;
using Xihe.DataProvider;
using Xihe.Utilities;

public class Main : MonoBehaviour
{
    public Camera mainCamera;
    public GameObject bunnyAsset;
    public GameObject placementIndicator;
    public ARCameraManager cameraManager;
    public ARRaycastManager raycastManager;

    // App state
    private bool _placementIndicatorEnabled = true;
    private bool _placementPostIsValid = true;
    private Pose _placementPose;

    private XiheRecorder _xiheRecorder;

    #region Lifecycle funcs

    // Start is called before the first frame update
    private void Start()
    {
        #region RegisterButtonEvents

        Utils.BindButtonClick("OpenPlayer", OnOpenPlayerClick);
        Utils.BindButtonClick("CreateRecorder", OnCreateRecorderClick);
        Utils.BindButtonClick("ToggleRecording", OnToggleRecordingClick);

        #endregion

        Screen.sleepTimeout = SleepTimeout.NeverSleep;
        Debug.Log("Application Started");
    }

    private void OnEnable()
    {
        cameraManager.frameReceived += OnCameraFrameReceived;
    }

    // Update is called once per frame
    private void Update()
    {
        UpdatePlacementPose();
        UpdatePlacementIndicator();
        _xiheRecorder?.Update();
    }

    private void OnDestroy()
    {
        // Debug.Log("Testing destroy");
        _xiheRecorder?.Dispose();
    }

    private void UpdatePlacementIndicator()
    {
        if (_placementPostIsValid && _placementIndicatorEnabled)
        {
            placementIndicator.SetActive(true);
            placementIndicator.transform.SetPositionAndRotation(
                _placementPose.position, _placementPose.rotation);
        }
        else
        {
            placementIndicator.SetActive(false);
        }
    }

    private void UpdatePlacementPose()
    {
        Vector3 screenCenter;

        try
        {
            screenCenter = Camera.current.ViewportToScreenPoint(
                new Vector3(0.5f, 0.5f));
        }
        catch
        {
            return;
        }

        var hits = new List<ARRaycastHit>();
        raycastManager.Raycast(screenCenter, hits, TrackableType.Planes);

        _placementPostIsValid = hits.Count > 0;

        if (!_placementPostIsValid) return;

        _placementPose = hits[0].pose;

        var cameraForward = Camera.current.transform.forward;
        var cameraBearing = new Vector3(cameraForward.x, 0, cameraForward.z).normalized;
        _placementPose.rotation = Quaternion.LookRotation(cameraBearing);
    }

    #endregion

    // Receiving lighting estimation data from ARKit 
    private void OnCameraFrameReceived(ARCameraFrameEventArgs args)
    {
        if (!(_xiheRecorder is {InRecording: true})) return;

        var arkitLightInfo = new[]
        {
            args.lightEstimation.averageBrightness ?? 0,
            args.lightEstimation.averageColorTemperature ?? 0
        };

        _xiheRecorder.SaveAdditionalInfo(
            "ARKitInfo",
            string.Join(",", arkitLightInfo));

        var camTrans = mainCamera.transform;
        var camPos = camTrans.position;
        var camRot = camTrans.rotation;

        _xiheRecorder.SaveAdditionalInfo(
            "cameraTransform",
            $"{camPos.x},{camPos.y},{camPos.z}\n"
            + $"{camRot.x},{camRot.y},{camRot.z},{camRot.w}");
    }

    private static void OnOpenPlayerClick()
    {
        SceneManager.LoadScene("Scenes/ARSimulation");
    }

    private void OnCreateRecorderClick()
    {
        _xiheRecorder?.Dispose();

        // Create XiheController
        var res = IOSDataProvider.CheckAvailability();

        if (res)
        {
            var dataProvider = new IOSDataProvider(cameraManager);
            _xiheRecorder = new XiheRecorder(dataProvider);
            Debug.Log("Xihe recorder created!");
        }
        else
        {
            Debug.LogError("Depth subsystem not available yet.");
        }

        // Place
        if (!_placementPostIsValid)
        {
            Debug.LogError("Not a valid position");
            return;
        }

        Instantiate(bunnyAsset, _placementPose.position, _placementPose.rotation);

        var probePos = _placementPose.position;
        probePos.y += 0.1f;
        Debug.Log("Bunny placed");
        _placementIndicatorEnabled = false;
    }

    private void OnToggleRecordingClick()
    {
        if (_xiheRecorder.InRecording)
        {
            _xiheRecorder.Stop();
            Debug.Log("Stopped recording");
        }
        else
        {
            _xiheRecorder.Start(new RecordingConfig
            {
                ObjectPose = _placementPose
            });
            Debug.Log("Started recording");
        }
    }
}