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

    System.IntPtr outputPtr;
    private void Awake() {
        recorder = GetComponent<Recorder>();
        recorder.onCompressedComplete += Recorder_onCompressedComplete;
        decoder = new NvPipeUnity.Decoder(NvPipeUnity.Codec.H264, NvPipeUnity.Format.RGBA32, 1024, 768);

        //Create a texture to store the decoded result.
        output = new Texture2D(1024, 768, TextureFormat.RGBA32, false);
        outputPtr = output.GetNativeTexturePtr();
        showcaseMaterial.mainTexture = output;
    }

    private void Update() {
        while (tasks.Count > 0) {
            if (!tasks.Peek().isDone)
                break;
            var t = tasks.Dequeue();
            if (t.isDone) {
                if (t.isError) {
                    Debug.LogError(t.error);
                } else {
                    Debug.Log("Decode success");
                }
                t.Dispose();
            }
        }
    }

    Queue<NvPipeUnity.AsyncDecodeTask> tasks = new Queue<NvPipeUnity.AsyncDecodeTask>();

    private void Recorder_onCompressedComplete(Unity.Collections.NativeArray<byte> obj, ulong size) {
        tasks.Enqueue(decoder.DecodeAsync(obj, (uint)size, outputPtr));
        
        //decoder.Decode(obj, size, ot);
        //output.LoadRawTextureData(ot);
        //output.Apply();
        //ot.Dispose();
    }
}
