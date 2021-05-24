namespace Xihe.GPU
{
    internal struct PcpShaderNumbers
    {
        public const string IntLprNumPoints = "lpr_num_points";
        public const string Float3LpsProbePosition = "lps_probe_position";
    }

    internal struct PcpShaderBuffers
    {
        public const string PcgCameraIntrinsics = "pcg_camera_intrinsics";
        public const string PcgCameraToWorldMatrix = "pcg_camera_to_world_matrix";
        public const string PcgResultPointCloud = "pcg_result_point_cloud";

        public const string LpsAnchors = "lps_anchors";
        public const string LpnCacheGrid = "lpn_cache_grid";
        public const string LpnPointCloud = "lpn_point_cloud";
        public const string LpnResultBuffer = "lpn_result_buffer";

        public const string LprResult = "lpr_result";

        public const string MtdPoolingGrid = "mtd_pooling_grid";
        public const string MtdInputBuffer = "mtd_input_buffer";
        public const string MtdBaseBuffer = "mtd_base_buffer";
        public const string MtdDecisionBuffer = "mtd_decision_buffer";

        public const string MbInputBuffer = "mb_input_buffer";
        public const string MbBaseBuffer = "mb_base_buffer";
    }

    internal struct PcpShaderTextures
    {
        public const string PcgColorTexture = "pcg_color_texture";
        public const string PcgDepthTexture = "pcg_depth_texture";
    }

    internal struct PcpShaderKernels
    {
        public const string PointCloudGeneration = "point_cloud_generation";
        public const string LightProbeNnSearch = "light_probe_nn_search";
        public const string LightProbeNnSearchAcc = "light_probe_nn_search_acc";
        public const string LightProbeNnReduce = "light_probe_nn_reduce";
        public const string MakeTriggerDecision = "make_trigger_decision";
        public const string MergeBuffers = "merge_buffers";
    }
}
