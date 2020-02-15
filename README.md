# NvPipeUnity
This is a Unity3D port of the video encode/decode library NvPipe(https://github.com/NVIDIA/NvPipe)

## Features
* Normal encode/decode function using NativeArray.
* Directly encode OpenGL RenderTexture.  
    * Happens on device memory, no PCI-E bus transfer.  
    * Use seprate thread, not blocking render thread nor main thread.  
    * Implemented lock-free circular buffer for thread-safety and high performance.  

## Usage

### Encode
For normal usage, the library operates on NativeArray but not Textures. So it's users' responsibility to read frame data into NativeArray and call Encode API.  

It's recommended to use the AsyncGPUReadback API to read texture data, which returns NativeArray that could be directly encoded. Here is an example.  

```C#
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
```

### AsyncTextureEncode(OpenGL Only)
NvPipe supports directly encoding an OpenGL texture. This library supports it as well.  
The texture data is copied to dedicated cuda memory on render thread first, and be encoded on a seprate thread, so render thread will not be blocked by encoding operation.   
Also there will be much less data transfer on PCI-E than normal usage.  

```C#
    public class Example02AsyncTextureEncorder : MonoBehaviour {
        Camera camera;
        NvPipeUnity.AsyncTextureEncoder encoder;
        public event System.Action<NativeArray<byte>, ulong> onCompressedComplete;
        RenderTexture intermediateRt;

        private void Awake() {
            camera = GetComponent<Camera>();
            encoder = new NvPipeUnity.AsyncTextureEncoder(NvPipeUnity.Codec.H264, NvPipeUnity.Format.RGBA32, NvPipeUnity.Compression.LOSSY, 10.0f, 30, 500, 500);

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
                        onCompressedComplete?.Invoke(data, (ulong)encodedSize);
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
        }
    }
```

### Decode
For decoding in Unity, follow the following example.   

Decoding directly to a texture is not supported for now.  
```C#
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
```

The output is a standard h264(or hevc based on selection) stream, you could use ffmpeg to decode as well.

## Requirements  
Since it's a port of NvPipe library, it has same requirements as NvPipe, like CUDA installment, Nvidia graphics card etc.  

## Compile Native Plugin  
The plugin could be compiled using CMake, the source code is at ./Plugin folder.  