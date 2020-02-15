using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Collections;
using System.IO;
using System;
using UnityEngine.Rendering;

namespace NvPipeUnity {
    public class Example01Recorder : MonoBehaviour {
        Camera camera;
        NvPipeUnity.Encoder encoder;
        public event System.Action<NativeArray<byte>, ulong> onCompressedComplete;
        RenderTexture intermediateRt;

        private void Awake() {
            camera = GetComponent<Camera>();
            intermediateRt = new RenderTexture(500, 500, 24);
            encoder = new NvPipeUnity.Encoder(NvPipeUnity.Codec.H264, NvPipeUnity.Format.RGBA32, NvPipeUnity.Compression.LOSSY, 10.0f, 30, 500, 500);
        }

        private void OnRenderImage(RenderTexture source, RenderTexture destination) {
            Graphics.Blit(source, destination);
            Graphics.Blit(source, intermediateRt);
            AsyncGPUReadback.Request(intermediateRt, 0, onReadback);
        }

        private void onReadback(AsyncGPUReadbackRequest obj) {
            if (encoder != null) {
                var output = new NativeArray<byte>(500 * 500 * 4, Allocator.Temp);  //Allocate output buffer. 500 * 500 * 4 is just for safe. most time the encoded size will be much smaller.
                var encodeLength = encoder.Encode(obj.GetData<byte>(), output);
                onCompressedComplete?.Invoke(output, encodeLength);
            }
        }

        private void OnDestroy() {
            encoder?.Dispose();
            encoder = null;
        }
    }
}