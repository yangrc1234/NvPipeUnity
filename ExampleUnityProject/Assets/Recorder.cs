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
    private void Awake() {
        camera = GetComponent<Camera>();
        camera.targetTexture = new RenderTexture(1024, 768, 24);
        temp = new Texture2D(1024, 768, TextureFormat.RGBA32, false);

        encoder = new NvPipeUnity.Encoder(NvPipeUnity.Codec.H264, NvPipeUnity.Format.RGBA32, NvPipeUnity.Compression.LOSSY, 10.0f, 30, 1024, 768);
    }

    private void OnPostRender() {
        UnityEngine.Rendering.AsyncGPUReadback.Request(camera.targetTexture, 0, onReadback);    
    }

    private void onReadback(AsyncGPUReadbackRequest obj) {
        var rawData = obj.GetData<byte>();
        var tmpOutput = new NativeArray<byte>(rawData.Length, Allocator.Temp);
        var length = encoder.Encode(rawData, tmpOutput);
        onCompressedComplete?.Invoke(tmpOutput, length);
        tmpOutput.Dispose();
    }

    private void OnDestroy() {
        encoder.Dispose();
    }
}
