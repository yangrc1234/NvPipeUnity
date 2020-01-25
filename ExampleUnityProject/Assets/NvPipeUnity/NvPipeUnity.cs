using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Runtime.InteropServices;
using System;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;

namespace NvPipeUnity {

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

        /// <summary>
        /// Encode a frame
        /// </summary>
        /// <typeparam name="TIn">NativeArray data type</typeparam>
        /// <param name="uncompressedData">raw uncompressed data. Should be same as encoder's width and height</param>
        /// <param name="output">Where to put the encoded data</param>
        /// <param name="forceIframe">Force to produce an I-Frame?</param>
        /// <returns></returns>
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

        /// <summary>
        /// Decode a frame
        /// </summary>
        /// <typeparam name="TOut">What type of output is, e.g. Color32 or byte</typeparam>
        /// <param name="compressedData">compressed data</param>
        /// <param name="compressedDataSize">size of compressed data in byte</param>
        /// <param name="output">where to put decoded data</param>
        /// <returns></returns>
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

    /// <summary>
    /// Internal library wrapper for NvPipe function.
    /// </summary>
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
