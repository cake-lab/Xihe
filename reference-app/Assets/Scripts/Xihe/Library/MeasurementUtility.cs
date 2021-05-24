using System;
using System.Collections.Generic;
using System.Globalization;
using UnityEngine;

namespace Xihe.Library
{
    internal static class MeasurementUtility
    {
        public static readonly List<string> Logs = new List<string>{"DateTime,Mark,TimeElapsed"};

        public static void Log(string mark, long timeElapsed)
        {
            var dateTimeStr = DateTime.Now.ToString(CultureInfo.InvariantCulture);
            Logs.Add($"{dateTimeStr},{mark},{timeElapsed}");
        }

        public static void MeasureBlock(string mark, Action f)
        {
            var baseTime = DateTime.Now;
            f();
            var timeElapsed = (DateTime.Now - baseTime).TotalMilliseconds;
            var dateTimeStr = baseTime.ToString(CultureInfo.InvariantCulture);
            Logs.Add($"{dateTimeStr},{mark},{timeElapsed}");
        }
    }
}