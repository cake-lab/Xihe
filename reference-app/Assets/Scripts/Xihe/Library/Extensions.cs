using System.Collections.Generic;
using System.Linq;
using UnityEngine;

namespace Xihe.Library
{
    public static class ListExtension
    {
        private static int TotalCount<T>(this IEnumerable<T[]> list)
        {
            return list.Sum(v => v.Length);
        }

        public static T[] CastToFlatArray<T>(this List<T[]> list)
        {
            var resultBuffer = new T[list.TotalCount()];

            var k = 0;
            foreach (var t1 in list.SelectMany(t => t))
            {
                resultBuffer[k++] = t1;
            }

            return resultBuffer;
        }
    }

    public static class DebugExt
    {
        public static void Log<T>(ComputeBuffer buffer, int unitsPerElement = 1)
        {
            var data = new T[buffer.count * unitsPerElement];
            buffer.GetData(data);
            Debug.Log(string.Join(",", data));
        }
    }
}
