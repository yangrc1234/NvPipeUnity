using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Collections;

public class Decoder : MonoBehaviour
{
    NvPipeUnity.Decoder decoder;
    Recorder recorder;
    Texture2D output;
    [SerializeField]
    Material showcaseMaterial;
    private void Awake() {
        recorder = GetComponent<Recorder>();
        recorder.onCompressedComplete += Recorder_onCompressedComplete;
        decoder = new NvPipeUnity.Decoder(NvPipeUnity.Codec.H264, NvPipeUnity.Format.RGBA32, 1024, 768);

        output = new Texture2D(1024, 768, TextureFormat.RGBA32, false);
        showcaseMaterial.mainTexture = output;
    }

    private void Recorder_onCompressedComplete(Unity.Collections.NativeArray<byte> obj, ulong size) {
        //var ot = new NativeArray<Color32>(1024 * 768 * 4, Allocator.Temp);
        //decoder.Decode(obj, size, ot);

        //output.LoadRawTextureData(ot);
        //output.Apply();
        //ot.Dispose();
    }
}
