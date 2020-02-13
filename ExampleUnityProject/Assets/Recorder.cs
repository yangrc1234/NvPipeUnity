using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Collections;
using System.IO;
using System;
using UnityEngine.Rendering;

public class Recorder : MonoBehaviour
{
    public class PingpongEncodeTextures {
        public PingpongEncodeTextures(RenderTextureDescriptor descriptor) {
            for (int i = 0; i < 2; i++) {
                textures[i] = new RenderTexture(descriptor);
                textures[i].depth = 0;
                textures[i].format = RenderTextureFormat.ARGB32;
                textures[i].Create();
                pointers[i] = textures[i].GetNativeTexturePtr();
            }
        }
        public int index;
        public RenderTexture[] textures = new RenderTexture[2];
        public IntPtr[] pointers = new IntPtr[2];
    }
    Camera camera;
    NvPipeUnity.Encoder encoder;
    public event System.Action<NativeArray<byte>, ulong> onCompressedComplete;
    NativeArray<byte> tmpData;
    PingpongEncodeTextures encodeTextures;
    private void Awake() {
        camera = GetComponent<Camera>();
        camera.targetTexture = new RenderTexture(1920, 1080, 24);
        encoder = new NvPipeUnity.Encoder(NvPipeUnity.Codec.H264, NvPipeUnity.Format.RGBA32, NvPipeUnity.Compression.LOSSY, 10.0f, 30, 1920, 1080);
        
        encodeTextures = new PingpongEncodeTextures(camera.targetTexture.descriptor);
    }

    private void Update() {
        while (tasks.Count > 0) {
            if (tasks.Peek().isDone) {
                var task = tasks.Dequeue();
                if (!task.isError) {
                    var data = task.GetData(out int encodedSize);
                    onCompressedComplete?.Invoke(data, (ulong)encodedSize);
                } else {
                    Debug.LogError(task.error);
                }
                task.Dispose();
            } else {
                break;
            }
        }
    }

    Queue<NvPipeUnity.AsyncEncodeTask> tasks = new Queue<NvPipeUnity.AsyncEncodeTask>();

    private void OnPostRender() {
        Graphics.Blit(camera.targetTexture, encodeTextures.textures[encodeTextures.index]);
        tasks.Enqueue(encoder.EncodeOpenGLTexture(encodeTextures.pointers[encodeTextures.index].ToInt32(), false));
        encodeTextures.index ^= 1;
        //AsyncGPUReadback.Request(camera.targetTexture, 0, onReadback);
    }

    
    private void onReadback(AsyncGPUReadbackRequest obj) {
        var rawData = obj.GetData<byte>();
        var intermediateContainer = new NativeArray<byte>(rawData, Allocator.Persistent);
        //var length = encoder.Encode(rawData, intermediateContainer, false);
        //onCompressedComplete(intermediateContainer, length);
        System.Threading.Tasks.Task.Run(() => AsyncEncode(intermediateContainer));
    }

    private void AsyncEncode(NativeArray<byte> data) {
        var length = encoder.Encode(data, tmpData);
        data.Dispose();
        
    }
    
    private void OnDestroy() {
        encoder?.Dispose();
    }
}
