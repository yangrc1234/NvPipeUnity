using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Collections;
using System.IO;
using System;
using UnityEngine.Rendering;

namespace NvPipeUnity {
    public class Example03AsyncTextureEncorder : MonoBehaviour {
        Camera camera;
        NvPipeUnity.AsyncTextureEncoder encoder;
        public event System.Action<NativeArray<byte>, ulong> onCompressedComplete;
        RenderTexture intermediateRt;
        FileStream fs;
        private void Awake() {
            camera = GetComponent<Camera>();
            encoder = new NvPipeUnity.AsyncTextureEncoder(NvPipeUnity.Codec.H264, NvPipeUnity.Format.RGBA32, NvPipeUnity.Compression.LOSSY, 10.0f, 30, 500, 500);
            fs = File.OpenWrite("ExampleRawStream.bin");
            //Camera default target texture seems can't be get. Use an intermediate rt to actually encode.
            intermediateRt = new RenderTexture(500, 500, 24);
        }

        //Stores unfinished tasks.
        Queue<NvPipeUnity.AsyncEncodeTask> tasks = new Queue<NvPipeUnity.AsyncEncodeTask>();

        private void Update() {
            while (tasks.Count > 0) {
                if (tasks.Peek().isDone) {
                    var task = tasks.Dequeue();
                    if (!task.isError) {
                        var data = task.GetData(out int encodedSize);
                        fs.Write(data.ToArray(), 0, encodedSize);
                    } else {
                        Debug.LogError("Encoder encountered error: " + task.error, this);
                    }
                    task.Dispose();
                } else {
                    break;
                }
            }
        }

        private void OnRenderImage(RenderTexture source, RenderTexture destination) {
            Graphics.Blit(source, destination);
            Graphics.Blit(source, intermediateRt);
            tasks.Enqueue(encoder.EncodeOpenGLTexture(intermediateRt, false));
        }

        private void OnDestroy() {
            encoder?.Dispose();
            fs.Close();
        }
    }
}