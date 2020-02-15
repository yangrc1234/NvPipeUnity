using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Collections;

namespace NvPipeUnity {

    public class Example01Decoder : MonoBehaviour {
        NvPipeUnity.Decoder decoder;
        Example01Recorder exampleRecorder;
        Texture2D output;
        [SerializeField]
        Material showcaseMaterial;

        System.IntPtr outputPtr;
        private void Awake() {
            exampleRecorder = GetComponent<Example01Recorder>();
            exampleRecorder.onCompressedComplete += Recorder_onCompressedComplete;
            decoder = new NvPipeUnity.Decoder(NvPipeUnity.Codec.H264, NvPipeUnity.Format.RGBA32, 500, 500);

            //Create a texture to store the decoded result.
            output = new Texture2D(500, 500, TextureFormat.RGBA32, false);
            outputPtr = output.GetNativeTexturePtr();
            showcaseMaterial.mainTexture = output;
        }

        private void Recorder_onCompressedComplete(Unity.Collections.NativeArray<byte> obj, ulong size) {
            var ot = new NativeArray<byte>(500 * 500 * 4, Allocator.Temp);
            decoder.Decode(obj, size, ot);
            output.LoadRawTextureData(ot);
            output.Apply();
        }
    }
}