using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Runtime.InteropServices;
using System;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;

namespace NvPipeUnity {

    public class Encoder : IDisposable {
        public Encoder(Codec codec, Format format, Compression compression, float bitrateMbps, UInt16 targetfps, UInt16 width, UInt16 height) {
            this.width = width;
            this.height = height;
            this.codec = codec;
            this.format = format;
            this.compression = compression;
            encoder = NvPipeUnityInternal.CreateNvPipeEncoder(format, codec, compression, (ulong)(bitrateMbps * 1000 * 1000), targetfps, width, height);

            var err = NvPipeUnityInternal.PollError(new IntPtr(0));   //Null for creation error.
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
        IntPtr encoder;
        Codec codec;
        Format format;
        Compression compression;

        public unsafe ulong Encode<TIn>(NativeArray<TIn> uncompressedData, NativeArray<byte> output, bool forceIframe = false) where TIn : struct {
            if (this.encoder.ToInt64() == 0) {
                throw new NvPipeException("The encoder is not intialized correctly!");
            }
            var result = NvPipeUnityInternal.Encode(encoder, new IntPtr(uncompressedData.GetUnsafePtr()), pitch, new IntPtr(output.GetUnsafePtr()), (ulong)output.Length, width, height, forceIframe);
            var err = NvPipeUnityInternal.PollError(encoder);
            if (err != null) {
                throw new NvPipeException(err);
            }
            return result;
        }

        public void Dispose() {
            if (this.encoder != null) {
                NvPipeUnityInternal.DestroyNvPipe(this.encoder);
                this.encoder = new IntPtr(0);
            }
        }
    }

    public class Decoder : IDisposable{
        public Decoder(Codec codec, Format format, UInt16 width, UInt16 height) {
            this.width = width;
            this.height = height;
            this.codec = codec;
            this.format = format;
            decoder = NvPipeUnityInternal.CreateNvPipeDecoder(format, codec, width, height);

            var err = NvPipeUnityInternal.PollError(new IntPtr(0));   //Null for creation error.
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
        IntPtr decoder;
        Codec codec;
        Format format;

        public unsafe ulong Decode<TOut>(NativeArray<byte> compressedData, ulong compressedDataSize, NativeArray<TOut> output) where TOut:struct{
            if (this.decoder.ToInt64() == 0) {
                throw new NvPipeException("The decoder is not intialized correctly!");
            }
            var result = NvPipeUnityInternal.Decode(decoder, new IntPtr(compressedData.GetUnsafePtr()), compressedDataSize, new IntPtr(output.GetUnsafePtr()), width, height);
            var err = NvPipeUnityInternal.PollError(decoder);
            if (err != null) {
                throw new NvPipeException(err);
            }
            return result;
        }

        public void Dispose() {
            if (this.decoder != null) {
                NvPipeUnityInternal.DestroyNvPipe(this.decoder);
                this.decoder = new IntPtr(0);
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

    public class NvPipeUnityInternal {

        [DllImport("NvPipe")]
        public static extern IntPtr CreateNvPipeEncoder(Format format, Codec codec, Compression compression, ulong bitrate, uint targetfps, uint width, uint height);


        [DllImport("NvPipe")]
        public static extern IntPtr CreateNvPipeDecoder(Format format, Codec codec, uint width, uint height);

        [DllImport("NvPipe")]
        public static extern IntPtr DestroyNvPipe(IntPtr pipe);

        [DllImport("NvPipe")]
        public static extern ulong Encode(IntPtr encoderPipe, IntPtr src, ulong srcPitch, IntPtr dst, ulong dstSize, uint width, uint height, bool forcelFrame);

        [DllImport("NvPipe")]
        public static extern ulong Decode(IntPtr decodePipe, IntPtr src, ulong srcSize, IntPtr dst, uint width, uint height);

        public static string PollError(IntPtr pipe) {
            var str = GetError(pipe);
            var result = Marshal.PtrToStringAnsi(str);
            ClearError(pipe);
            if (result == "")
                return null;
            return result;
        }

        [DllImport("NvPipe")]
        private static extern IntPtr GetError(IntPtr pipe);

        [DllImport("NvPipe")]
        private static extern void ClearError(IntPtr pipe);
    }
}
