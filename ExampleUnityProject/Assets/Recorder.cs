using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Collections;
using System.IO;
using System;
using UnityEngine.Rendering;

public class Recorder : MonoBehaviour
{
    Camera camera;
    Texture2D temp;
    NvPipeUnity.Encoder encoder;
    public event System.Action<NativeArray<byte>, ulong> onCompressedComplete;
    NativeArray<byte> tmpData;
    IntPtr texID;

    private void Awake() {
        camera = GetComponent<Camera>();
        camera.targetTexture = new RenderTexture(1024, 768, 24);
        temp = new Texture2D(1024, 768, TextureFormat.RGBA32, false);
        encoder = new NvPipeUnity.Encoder(NvPipeUnity.Codec.H264, NvPipeUnity.Format.RGBA32, NvPipeUnity.Compression.LOSSY, 10.0f, 30, 1024, 768);
        texID = temp.GetNativeTexturePtr();
    }

    private void Update() {
        /*
        if (task.isDone) {
            if (!task.isError) {
                var data = task.GetData(out int encodedSize);
                onCompressedComplete?.Invoke(data, (ulong)encodedSize);
            }
            task.Dispose();
        }*/
    }

    NvPipeUnity.AsyncEncodeTask task;

    private void OnPostRender() {
        // task = encoder.EncodeOpenGLTexture(texID.ToInt32(), false);
        AsyncGPUReadback.Request(camera.targetTexture, 0, onReadback);
    }

    
    private void onReadback(AsyncGPUReadbackRequest obj) {
        var rawData = obj.GetData<byte>();

        var intermediateContainer = new NativeArray<byte>(rawData, Allocator.Temp);

        var length = encoder.Encode(rawData, intermediateContainer, false);
        onCompressedComplete(rawData, length);
        //System.Threading.Tasks.Task.Run(() => AsyncEncode(intermediateContainer));
    }

    private void AsyncEncode(NativeArray<byte> data) {

        var length = encoder.Encode(data, tmpData);
        data.Dispose();
    }
    
    private void OnDestroy() {
        encoder?.Dispose();
    }
}
