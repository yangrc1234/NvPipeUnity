# NvPipeUnity
This is Unity3D port implementation for the video encode/decode library NvPipe(https://github.com/NVIDIA/NvPipe)

The original NvPipe code is slightly modified, extra code for implementing Unity native plug-in interface is under ./Plugin/src/Unity, CMakeLists.txt is edited to include these as well.  

## Usage
Under ./UnityExampleProject is a simple demo showing how to use the encoder and decoder.  

For now the encoder and decoder both operates on unmanaged memory. NativeArray in Unity.Collections is used to pass the data for now.  

There should be no problem encoding or decoding on a non-main thread.

Following code shows how to encode camera rendertexture and decode to a texture in Unity.

### Encode

```C#
    void Awake(){
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
        encoder?.Dispose();
    }
```

### Decode
```C#
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
        var ot = new NativeArray<Color32>(1024 * 768 * 4, Allocator.Temp);
        decoder.Decode(obj, size, ot);

        output.LoadRawTextureData(ot);
        output.Apply();
        ot.Dispose();
    }
```

## Known Issues
The original NvPipe implementation supports directly encode a OpenGL buffer without copying around. It's not implemented yet.

# License
NvPipe license is under ./Plugin folder.  
Other code is under MIT license.