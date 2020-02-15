using System.Collections.Generic;
using UnityEngine;
using System.Runtime.InteropServices;
using System;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using UnityEngine.Rendering;

namespace NvPipeUnity {

    /// <summary>
    /// Remember rendertexture's native pointer.
    /// 
    /// It's cost to call GetNativeTexturePtr() for Unity, it will cause sync between render thread and main thread.
    /// </summary>
    internal class RenderTextureRegistery {
        Dictionary<RenderTexture, IntPtr> ptrs = new Dictionary<RenderTexture, IntPtr>();
        public IntPtr GetFor(RenderTexture rt) {
            if (ptrs.ContainsKey(rt)) {
                return ptrs[rt];
            } else {
                if (!rt.IsCreated()) {
                    rt.Create();
                }
                var ptr = rt.GetNativeTexturePtr();
                ptrs.Add(rt, ptr);
                return ptr;
            }
        }
    }

    /// <summary>
    /// Helps to always start a DontDestroyOnLoad manager object before scene loads.
    /// </summary>
    static class NvPipeRuntimeInitializer {
        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.BeforeSceneLoad)]
        static void Initialize() {
            if (SystemInfo.graphicsDeviceType == GraphicsDeviceType.OpenGLCore && AsyncEncodeScheduler.instance == null) {
                var go = new GameObject("__Async Encoder__");
                go.hideFlags = HideFlags.HideAndDontSave;
                GameObject.DontDestroyOnLoad(go);
                var updater = go.AddComponent<AsyncEncodeScheduler>();
            }
        }
    }

    /// <summary>
    /// Helps schedule, manage async tasks in native plugin.
    /// </summary>
    internal class AsyncEncodeScheduler : MonoBehaviour {
        public static AsyncEncodeScheduler instance {
            get {
                return _instance;
            }
        }

        private void Awake() {
            NvPipeUnityInternal.NvPipe_ResetEncodeTasks();
            _instance = this;
        }

        private void OnDestroy() {
            NvPipeUnityInternal.NvPipe_ResetEncodeTasks();
            undoneTasks.Clear();
            tasks.Clear();
        }

        private static AsyncEncodeScheduler _instance;

        public class InternalTask {
            public int internalTaskIndex;
            public bool isDone;
            public bool isError;
            public string error;
            public NativeArray<byte> encodedData;
        }

        private void Update() {
            DoUpdate();
        }

        private unsafe static void DoUpdate() {
            var t = NvPipeUnityInternal.NvPipe_GetRenderThreadPollFunc();
            GL.IssuePluginEvent(t, 0);
            while (undoneTasks.Count > 0) {
                var undoneIndex = undoneTasks.Peek();
                var task = tasks[undoneIndex];
                NvPipeUnityInternal.NvPipe_EncodeTextureAsyncQuery((uint)task.internalTaskIndex, out bool isDone, out bool isError, out IntPtr encodedData, out ulong encodeSize, out IntPtr error);
                var err = NvPipeUnityInternal.PollError(0);
                if (err != null)
                    throw new Exception(err);
                if (isDone) {
                    undoneTasks.Dequeue();
                    task.isDone = true;
                    try {
                        task.isError = isError;
                        if (isError) {
                            task.error = Marshal.PtrToStringAnsi(error);
                        } else {
                            task.encodedData = new NativeArray<byte>((int)encodeSize, Allocator.Persistent);
                            UnsafeUtility.MemCpy(task.encodedData.GetUnsafePtr(), encodedData.ToPointer(), (long)encodeSize);
                        }
                    } finally {
                        //Everything is now in managed side. free native things.
                        NvPipeUnityInternal.NvPipe_EncodeTextureAsyncClearTask((uint)task.internalTaskIndex);
                        err = NvPipeUnityInternal.PollError(0);
                        if (err != null)
                            Debug.LogError(err);
                    }
                } else {
                    break;
                }
            }
        }

        public static int taskCreationIndex;
        public static Queue<int> undoneTasks = new Queue<int>();
        public static Dictionary<int, InternalTask> tasks = new Dictionary<int, InternalTask>();
        public static int EnqueueTask(AsyncTextureEncoder encoder, uint texture, uint width, uint height, bool forceIFrame) {
            var internalTaskID = NvPipeUnityInternal.NvPipe_QueueEncodeTaskInMainThread(encoder.encoder, texture, width, height, forceIFrame);
            var error = NvPipeUnityInternal.PollError(encoder.encoder);
            if (error != null) {
                throw new Exception(error);
            }

            var taskIndex = taskCreationIndex++;
            undoneTasks.Enqueue(taskIndex);
            tasks[taskIndex] = new InternalTask() { internalTaskIndex = internalTaskID, isError = false, isDone = false };
            return taskIndex;
        }

        public static bool TaskDone(int task) {
            if (!tasks.ContainsKey(task))
                return true;
            return tasks[task].isDone;
        }

        public static bool TaskIsError(int task) {
            if (!tasks.ContainsKey(task))
                return true;

            return tasks[task].isError;
        }

        public static string TaskError(int task) {
            if (!tasks.ContainsKey(task))
                return "Task doesn't exists";

            return tasks[task].error;
        }

        public static void TaskDispose(int taskID) {
            if (!tasks.ContainsKey(taskID))
                return;
            var task = tasks[taskID];
            if (task.encodedData.IsCreated)
                task.encodedData.Dispose();
            tasks.Remove(taskID);
        }

        public static NativeArray<byte> TaskData(int taskID) {
            return tasks[taskID].encodedData;
        }
    }

    /// <summary>
    /// A handle to async encode task.
    /// </summary>
    public struct AsyncEncodeTask : IDisposable {
        public int handleID;

        /// <summary>
        /// Is task done.
        /// "done" means sucess or error, check isError for more.
        /// </summary>
        public bool isDone {
            get {
                return AsyncEncodeScheduler.TaskDone(handleID);
            }
        }

        /// <summary>
        /// Is the task errored.
        /// </summary>
        public bool isError {
            get {
                return AsyncEncodeScheduler.TaskIsError(handleID);
            }
        }

        /// <summary>
        /// The error info. This is probably exception info thrown from Native Plugin, maybe hard to understand.
        /// </summary>
        public string error {
            get {
                return AsyncEncodeScheduler.TaskError(handleID);
            }
        }

        /// <summary>
        /// Dispose the task. Always remember to call this, otherwise your memory will be consumed.
        /// </summary>
        public void Dispose() {
            AsyncEncodeScheduler.TaskDispose(handleID);
        }

        /// <summary>
        /// Get the data. The data itself is managed by plugin.
        /// As long as .Dispose() not called, the data is always valid.
        /// 
        /// Always remember to .Dispose() after data is useless!
        /// </summary>
        /// <param name="encodedSize"></param>
        /// <returns></returns>
        public NativeArray<byte> GetData(out int encodedSize) {
            var t = AsyncEncodeScheduler.TaskData(handleID);
            encodedSize = t.Length;
            return t;
        }
    }

    public class AsyncTextureEncoder : IDisposable {
        RenderTextureRegistery ptrRegistery = new RenderTextureRegistery();
        public AsyncTextureEncoder(Codec codec, Format format, Compression compression, float bitrateMbps, UInt16 targetfps, UInt16 width, UInt16 height) {
            this.width = width;
            this.height = height;
            this.codec = codec;
            this.format = format;
            this.compression = compression;
            closed = false;
            encoder = NvPipeUnityInternal.NvPipe_CreateTextureAsyncEncoder(format, codec, compression, (ulong)(bitrateMbps * 1000 * 1000), targetfps, width, height);

            var err = NvPipeUnityInternal.PollError(0);   //Null for creation error.
            if (err != null) {
                throw new NvPipeException(err);
            }

            switch (format) {
                case Format.RGBA32:
                    pitch = 4;
                    break;
                case Format.UINT4:
                case Format.UINT8:
                case Format.UINT16:
                case Format.UINT32:
                    pitch = 1;
                    break;
                default:
                    pitch = 4;
                    break;
            }
            pitch *= width;
        }
        ulong pitch;
        UInt16 width, height;
        public uint encoder { get; private set; }
        Codec codec;
        Format format;
        Compression compression;
        public bool closed { get; private set; }
        public unsafe AsyncEncodeTask EncodeOpenGLTexture(int textureID, bool forceIframe) {
            if (closed)
                throw new System.Exception("Encoder already disposed!");
            var t = AsyncEncodeScheduler.EnqueueTask(this, (uint)textureID, width, height, forceIframe);
            return new AsyncEncodeTask() { handleID = t };
        }
        public unsafe AsyncEncodeTask EncodeOpenGLTexture(RenderTexture texture, bool forceIframe) {
            if (closed)
                throw new System.Exception("Encoder already disposed!");
            if (texture.format != RenderTextureFormat.ARGB32)
                throw new System.Exception("Only ARGB32 is supported for encoding!");
            var t = AsyncEncodeScheduler.EnqueueTask(this, (uint)ptrRegistery.GetFor(texture).ToInt32(), width, height, forceIframe);
            return new AsyncEncodeTask() { handleID = t };
        }

        public void Dispose() {
            if (!closed && this.encoder != 0) {
                NvPipeUnityInternal.NvPipe_Destroy(this.encoder);
                this.encoder = 0;
                closed = true;
            }
        }
    }

    public class Encoder : IDisposable {
        /// <summary>
        /// Create a encoder.
        /// </summary>
        /// <param name="codec">Encode method</param>
        /// <param name="format">Format of the input.</param>
        /// <param name="compression">Compression method</param>
        /// <param name="bitrateMbps">bitrate in Mbps, notice it's bit not byte</param>
        /// <param name="targetfps">target fps.</param>
        /// <param name="width">input texture width</param>
        /// <param name="height">input texture height</param>
        public Encoder(Codec codec, Format format, Compression compression, float bitrateMbps, UInt16 targetfps, UInt16 width, UInt16 height) {
            this.width = width;
            this.height = height;
            this.codec = codec;
            this.format = format;
            this.compression = compression;
            closed = false;
            encoder = NvPipeUnityInternal.NvPipe_CreateEncoder(format, codec, compression, (ulong)(bitrateMbps * 1000 * 1000), targetfps, width, height);

            var err = NvPipeUnityInternal.PollError(0);
            if (err != null) {
                throw new NvPipeException(err);
            }

            switch (format) {
                case Format.RGBA32:
                    pitch = 4;
                    break;
                case Format.UINT4:
                case Format.UINT8:
                case Format.UINT16:
                case Format.UINT32:
                    pitch = 1;
                    break;
                default:
                    pitch = 4;
                    break;
            }
            pitch *= width;
        }
        ulong pitch;
        UInt16 width, height;
        uint encoder;
        Codec codec;
        Format format;
        Compression compression;
        public bool closed { get; private set; }
        /// <summary>
        /// Encode a frame
        /// </summary>
        /// <typeparam name="TIn">NativeArray data type</typeparam>
        /// <param name="uncompressedData">raw uncompressed data. Should be same as encoder's width and height</param>
        /// <param name="output">Where to put the encoded data</param>
        /// <param name="forceIframe">Force to produce an I-Frame?</param>
        /// <returns></returns>
        public unsafe ulong Encode<TIn>(NativeArray<TIn> uncompressedData, NativeArray<byte> output, bool forceIframe = false) where TIn : struct {
            if (this.encoder == 0) {
                throw new NvPipeException("The encoder is not intialized correctly!");
            }
            var result = NvPipeUnityInternal.NvPipe_Encode(encoder, new IntPtr(uncompressedData.GetUnsafePtr()), pitch, new IntPtr(output.GetUnsafePtr()), (ulong)output.Length, width, height, forceIframe);
            var err = NvPipeUnityInternal.PollError(encoder);
            if (err != null) {
                throw new NvPipeException(err);
            }
            return result;
        }

        public void Dispose() {
            if (!closed && this.encoder != 0) {
                NvPipeUnityInternal.NvPipe_Destroy(this.encoder);
                this.encoder = 0;
                closed = true;
            }
        }
    }

    public class Decoder : IDisposable {
        public Decoder(Codec codec, Format format, UInt16 width, UInt16 height) {
            this.width = width;
            this.height = height;
            this.codec = codec;
            this.format = format;
            decoder = NvPipeUnityInternal.NvPipe_CreateDecoder(format, codec, width, height);

            var err = NvPipeUnityInternal.PollError(0);   //Null for creation error.
            if (err != null) {
                throw new NvPipeException(err);
            }

            switch (format) {
                case Format.RGBA32:
                    pitch = 4;
                    break;
                case Format.UINT4:
                case Format.UINT8:
                case Format.UINT16:
                case Format.UINT32:
                    pitch = 1;
                    break;
                default:
                    pitch = 4;
                    break;
            }
            pitch *= width;
        }
        ulong pitch;
        UInt16 width, height;
        uint decoder;
        Codec codec;
        Format format;
        public bool closed { get; private set; }
        /// <summary>
        /// Decode a frame
        /// </summary>
        /// <typeparam name="TOut">What type of output is, e.g. Color32 or byte</typeparam>
        /// <param name="compressedData">compressed data</param>
        /// <param name="compressedDataSize">size of compressed data in byte</param>
        /// <param name="output">where to put decoded data</param>
        /// <returns></returns>
        public unsafe ulong Decode<TOut>(NativeArray<byte> compressedData, ulong compressedDataSize, NativeArray<TOut> output) where TOut : struct {
            if (this.decoder == 0) {
                throw new NvPipeException("The decoder is not intialized correctly!");
            }
            var result = NvPipeUnityInternal.NvPipe_Decode(decoder, new IntPtr(compressedData.GetUnsafePtr()), compressedDataSize, new IntPtr(output.GetUnsafePtr()), width, height);
            var err = NvPipeUnityInternal.PollError(decoder);
            if (err != null) {
                throw new NvPipeException(err);
            }
            return result;
        }

        public void Dispose() {
            if (this.decoder != 0) {
                closed = true;
                NvPipeUnityInternal.NvPipe_Destroy(this.decoder);
                this.decoder = 0;
            }
        }
    }

    public class NvPipeException : System.Exception {
        public NvPipeException(string msg) : base(msg) {

        }
    }

    public enum Codec {
        H264,
        HEVC,
    }

    public enum Format {
        RGBA32,
        UINT4,
        UINT8,
        UINT16,
        UINT32,
    }

    public enum Compression {
        LOSSY,
        LOSSLESS,
    }

    /// <summary>
    /// Internal library wrapper for NvPipe function.
    /// </summary>
    public class NvPipeUnityInternal {
        [DllImport("NvPipe")]
        public static extern uint NvPipe_CreateEncoder(Format format, Codec codec, Compression compression, ulong bitrate, uint targetfps, uint width, uint height);

        [DllImport("NvPipe")]
        public static extern void NvPipe_ResetEncodeTasks();

        [DllImport("NvPipe")]
        public static extern uint NvPipe_CreateTextureAsyncEncoder(Format format, Codec codec, Compression compression, ulong bitrate, uint targetfps, uint width, uint height);

        [DllImport("NvPipe")]
        public static extern void NvPipe_SetBitrate(uint pipe, ulong bitrate, uint targetFrameRate);

        [DllImport("NvPipe")]
        public static extern ulong NvPipe_Encode(uint pipe, IntPtr src, ulong srcPitch, IntPtr dst, ulong dstSize, uint width, uint height, bool forceIFrame);

        [DllImport("NvPipe")]
        public static extern ulong NvPipe_EncodeTexture(uint pipe, uint texture, uint target, IntPtr dst, ulong dstSize, uint width, uint height, bool forceIFrame);

        [DllImport("NvPipe")]
        public static extern int NvPipe_QueueEncodeTaskInMainThread(uint nvp, uint texture, uint width, uint height, bool forceIFrame);

        [DllImport("NvPipe")]
        public static extern IntPtr NvPipe_GetRenderThreadPollFunc();

        [DllImport("NvPipe")]
        public static extern void NvPipe_EncodeTextureAsyncQuery(
            uint taskIndex, out bool isDone, out bool isError, out IntPtr encodedData, out ulong encodeSize, out IntPtr error);

        [DllImport("NvPipe")]
        public static extern void NvPipe_EncodeTextureAsyncClearTask(uint taskIndex);

        [DllImport("NvPipe")]
        public static extern uint NvPipe_CreateDecoder(Format format, Codec codec, uint width, uint height);

        [DllImport("NvPipe")]
        public static extern ulong NvPipe_Decode(uint nvp, IntPtr src, ulong srcSize, IntPtr dst, uint width, uint height);

        [DllImport("NvPipe")]
        public static extern ulong NvPipe_DecodeTexture(uint nvp, IntPtr src, ulong srcSize, uint texture, uint target, uint width, uint height);

        [DllImport("NvPipe")]
        public static extern void NvPipe_Destroy(uint pipe);

        [DllImport("NvPipe")]
        public static extern IntPtr NvPipe_GetError(uint pipe);

        [DllImport("NvPipe")]
        public static extern void NvPipe_ClearError(uint pipe);

        public static string PollError(uint pipe) {
            var str = NvPipe_GetError(pipe);
            var result = Marshal.PtrToStringAnsi(str);
            if (result == "")
                return null;
            NvPipe_ClearError(pipe);
            return result;
        }

        public static string CStrToString(IntPtr cStr) {
            return Marshal.PtrToStringAnsi(cStr);
        }
    }
}
