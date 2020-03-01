/* Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "NvPipe.h"

#ifdef NVPIPE_WITH_ENCODER
#include "NvCodec/NvEncoder/NvEncoderCuda.h"
#endif

#ifdef NVPIPE_WITH_DECODER
#include "NvCodec/NvDecoder/NvDecoder.h"
#endif

#include "Utils/ColorSpace.h"
#include "Utils/NvCodecUtils.h"

#include <memory>
#include <iostream>
#include <string>
#include <sstream>
#include <unordered_map>
#include <mutex>
#include <queue>
#include <thread>
#include <atomic>
#include <cuda.h>
#include <cuda_runtime_api.h>

#ifdef NVPIPE_WITH_OPENGL
#include <cuda_gl_interop.h>
#endif

#ifdef _DEBUG
#define DEBUG_LOG(fmt, ...) (::fprintf(stderr, fmt, __VA_ARGS__))
#else
#define DEBUG_LOG(fmt, ...) 
#endif

class Exception
{
public:
	Exception(std::string msg) : message(msg) {}
	std::string getErrorString() const { return message; }
public:
	std::string message;
};

template<typename T>
static bool CheckInsideQueueRange(T head, T tail, T tocheck) {	//Check when on a circular buffer, given a range[tail, head), is tocheck inside the range.
	if (head >= tail) {
		return (tocheck < head && tocheck >= tail);	//Didn't check, hope this is right.
	}
	else {
		return (tocheck < head || tocheck >= tail);
	}
}

inline void CUDA_THROW(cudaError_t code, std::string errorMessage)
{
	if (cudaSuccess != code) {
		throw Exception(errorMessage + " (Error " + std::to_string(code) + ": " + std::string(cudaGetErrorString(code)) + ")");
	}
}

inline bool isDevicePointer(const void* ptr)
{
	struct cudaPointerAttributes attr;
	const cudaError_t perr = cudaPointerGetAttributes(&attr, ptr);

#if (CUDA_VERSION >= 10000)
	return (perr == cudaSuccess) && (attr.type != cudaMemoryTypeHost);
#else
	return (perr == cudaSuccess) && (attr.memoryType != cudaMemoryTypeHost);
#endif
}

inline uint64_t getFrameSize(NvPipe_Format format, uint32_t width, uint32_t height)
{
	if (format == NVPIPE_RGBA32)
		return width * height * 4;
	else if (format == NVPIPE_UINT4)
		return width * height / 2;
	else if (format == NVPIPE_UINT8)
		return width * height;
	else if (format == NVPIPE_UINT16)
		return width * height * 2;
	else if (format == NVPIPE_UINT32)
		return width * height * 4;

	return 0;
}


__global__
void uint4_to_nv12(const uint8_t* src, uint32_t srcPitch, uint8_t* dst, uint32_t dstPitch, uint32_t width, uint32_t height)
{
	// one thread per pixel
	const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
	const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < width && y < height)
	{
		const uint32_t i = y * srcPitch + x / 2;
		const uint32_t j = y * dstPitch + x;

		// Extend 4 bit to 8 bits
		// Even thread: higher 4 bits, odd thread: lower 4 bits
		dst[j] = (x & 1 == 1) ? (src[i] & 0xF) : ((src[i] & 0xF0) >> 4);

		// Blank UV channel
		if (y < height / 2)
		{
			uint8_t* UV = dst + dstPitch * (height + y);
			UV[x] = 0;
		}
	}
}

__global__
void nv12_to_uint4(const uint8_t* src, uint32_t srcPitch, uint8_t* dst, uint32_t dstPitch, uint32_t width, uint32_t height)
{
	// one thread per TWO pixels
	const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
	const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;

	if (2 * x < width && y < height)
	{
		const uint32_t i = y * srcPitch + 2 * x;
		const uint32_t j = y * dstPitch + x;

		// Merge lower 4 bits of two Y bytes to one output byte
		uint8_t v = (src[i] & 0xF) << 4;

		if (2 * x + 1 < width)
			v = v | (src[i + 1] & 0xF);

		dst[j] = v;
	}
}

__global__
void uint8_to_nv12(const uint8_t* src, uint32_t srcPitch, uint8_t* dst, uint32_t dstPitch, uint32_t width, uint32_t height)
{
	const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
	const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < width && y < height)
	{
		const uint32_t i = y * srcPitch + x;
		const uint32_t j = y * dstPitch + x;

		// Copy grayscale image to Y channel
		dst[j] = src[i];

		// Blank UV channel
		if (y < height / 2)
		{
			uint8_t* UV = dst + dstPitch * (height + y);
			UV[x] = 0;
		}
	}
}

__global__
void nv12_to_uint8(const uint8_t* src, uint32_t srcPitch, uint8_t* dst, uint32_t dstPitch, uint32_t width, uint32_t height)
{
	const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
	const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < width && y < height)
	{
		const uint32_t i = y * srcPitch + x;
		const uint32_t j = y * dstPitch + x;

		// Copy Y channel to grayscale image
		dst[j] = src[i];

	}
}

__global__
void uint16_to_nv12(const uint8_t* src, uint32_t srcPitch, uint8_t* dst, uint32_t dstPitch, uint32_t width, uint32_t height)
{
	const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
	const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < width && y < height)
	{
		const uint32_t i = y * srcPitch + 2 * x;
		const uint32_t j = y * dstPitch + x;

		// Copy higher byte to left half of Y channel
		dst[j] = src[i];

		// Copy lower byte to right half of Y channel
		dst[j + width] = src[i + 1];

		// Blank UV channel
		if (y < height / 2)
		{
			uint8_t* UV = dst + dstPitch * (height + y);
			UV[2 * x + 0] = 0;
			UV[2 * x + 1] = 0;
		}
	}
}

__global__
void nv12_to_uint16(const uint8_t* src, uint32_t srcPitch, uint8_t* dst, uint32_t dstPitch, uint32_t width, uint32_t height)
{
	const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
	const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < width && y < height)
	{
		const uint32_t i = y * srcPitch + x;
		const uint32_t j = y * dstPitch + 2 * x;

		// Copy higher byte from left half of Y channel
		dst[j] = src[i];

		// Copy lower byte from right half of Y channel
		dst[j + 1] = src[i + width];
	}
}

__global__
void uint32_to_nv12(const uint8_t* src, uint32_t srcPitch, uint8_t* dst, uint32_t dstPitch, uint32_t width, uint32_t height)
{
	const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
	const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < width && y < height)
	{
		const uint32_t i = y * srcPitch + 4 * x;
		const uint32_t j = y * dstPitch + x;

		// Copy highest byte to left quarter of Y channel,
		// ...
		// Copy lowest byte to right quarter of Y channel
		dst[j] = src[i];
		dst[j + width] = src[i + 1];
		dst[j + 2 * width] = src[i + 2];
		dst[j + 3 * width] = src[i + 3];

		// Blank UV channel
		if (y < height / 2)
		{
			uint8_t* UV = dst + dstPitch * (height + y);
			UV[4 * x + 0] = 0;
			UV[4 * x + 1] = 0;
			UV[4 * x + 2] = 0;
			UV[4 * x + 3] = 0;
		}
	}
}

__global__
void nv12_to_uint32(const uint8_t* src, uint32_t srcPitch, uint8_t* dst, uint32_t dstPitch, uint32_t width, uint32_t height)
{
	const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
	const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < width && y < height)
	{
		const uint32_t i = y * srcPitch + x;
		const uint32_t j = y * dstPitch + 4 * x;

		// Copy highest byte from left quarter of Y channel
		// ...
		// Copy lowest byte from right quarter of Y channel
		dst[j] = src[i];
		dst[j + 1] = src[i + width];
		dst[j + 2] = src[i + 2 * width];
		dst[j + 3] = src[i + 3 * width];

	}
}

#ifdef NVPIPE_WITH_OPENGL
/**
 * @brief Utility class for managing CUDA-GL interop graphics resources.
 */
class GraphicsResourceRegistry
{
public:
	virtual ~GraphicsResourceRegistry()
	{
		// Unregister all
		for (auto& r : this->registeredPBOs)
			CUDA_THROW(cudaGraphicsUnregisterResource(r.second.graphicsResource),
				"Failed to unregister PBO graphics resource");

		for (auto& r : this->registeredTextures) {
			auto result = cudaGraphicsUnregisterResource(r.second.graphicsResource);
			//CUDA_THROW(result,"Failed to unregister texture graphics resource");  

			//This exception is suppressed in NvPipeUnity.
			//Unregister could throw due to early destroyed gl resource
			//In Unity plugin code, we can't hack into gl resource destruction. 
			//So when we press Stop button, all render textures are destroyed, and NvPipe is destructed, then an exception is thrown.
			//We can't handle the exception since it's hidden inside C interface.

			//So just disable it.
		}
	}

	cudaGraphicsResource_t getTextureGraphicsResource(uint32_t texture, uint32_t target, uint32_t width, uint32_t height, uint32_t flags)
	{
		// Check if texture needs to be (re)registered
		RegisteredTexture& reg = this->registeredTextures[texture];

		if (reg.width != width || reg.height != height || reg.target != target) {
			if (reg.graphicsResource) {
				CUDA_THROW(cudaGraphicsUnregisterResource(reg.graphicsResource),
					"Failed to unregister texture graphics resource");

				reg.graphicsResource = nullptr;
			}

			CUDA_THROW(cudaGraphicsGLRegisterImage(&reg.graphicsResource, texture, target, flags),
				"Failed to register texture as graphics resource");

			reg.width = width;
			reg.height = height;
			reg.target = target;
		}

		return reg.graphicsResource;
	}

	cudaGraphicsResource_t getPBOGraphicsResource(uint32_t pbo, uint32_t width, uint32_t height, uint32_t flags)
	{
		// Check if PBO needs to be (re)registered
		RegisteredPBO& reg = this->registeredPBOs[pbo];

		if (reg.width != width || reg.height != height) {
			if (reg.graphicsResource) {
				CUDA_THROW(cudaGraphicsUnregisterResource(reg.graphicsResource),
					"Failed to unregister PBO graphics resource");

				reg.graphicsResource = nullptr;
			}

			CUDA_THROW(cudaGraphicsGLRegisterBuffer(&reg.graphicsResource, pbo, flags),
				"Failed to register PBO as graphics resource");

			reg.width = width;
			reg.height = height;
		}

		return reg.graphicsResource;
	}

private:
	struct RegisteredTexture
	{
		cudaGraphicsResource_t graphicsResource = nullptr;
		uint32_t width = 0;
		uint32_t height = 0;
		uint32_t target = 0;
	};
	std::unordered_map<uint32_t, RegisteredTexture> registeredTextures;

	struct RegisteredPBO
	{
		cudaGraphicsResource_t graphicsResource = nullptr;
		uint32_t width = 0;
		uint32_t height = 0;
	};
	std::unordered_map<uint32_t, RegisteredPBO> registeredPBOs;
};
#endif


#ifdef NVPIPE_WITH_ENCODER

inline std::string EncErrorCodeToString(NVENCSTATUS code)
{
	std::vector<std::string> errors = {
		"NV_ENC_SUCCESS",
		"NV_ENC_ERR_NO_ENCODE_DEVICE",
		"NV_ENC_ERR_UNSUPPORTED_DEVICE",
		"NV_ENC_ERR_INVALID_ENCODERDEVICE",
		"NV_ENC_ERR_INVALID_DEVICE",
		"NV_ENC_ERR_DEVICE_NOT_EXIST",
		"NV_ENC_ERR_INVALID_PTR",
		"NV_ENC_ERR_INVALID_EVENT",
		"NV_ENC_ERR_INVALID_PARAM",
		"NV_ENC_ERR_INVALID_CALL",
		"NV_ENC_ERR_OUT_OF_MEMORY",
		"NV_ENC_ERR_ENCODER_NOT_INITIALIZED",
		"NV_ENC_ERR_UNSUPPORTED_PARAM",
		"NV_ENC_ERR_LOCK_BUSY",
		"NV_ENC_ERR_NOT_ENOUGH_BUFFER",
		"NV_ENC_ERR_INVALID_VERSION",
		"NV_ENC_ERR_MAP_FAILED",
		"NV_ENC_ERR_NEED_MORE_INPUT",
		"NV_ENC_ERR_ENCODER_BUSY",
		"NV_ENC_ERR_EVENT_NOT_REGISTERD",
		"NV_ENC_ERR_GENERIC",
		"NV_ENC_ERR_INCOMPATIBLE_CLIENT_KEY",
		"NV_ENC_ERR_UNIMPLEMENTED",
		"NV_ENC_ERR_RESOURCE_REGISTER_FAILED",
		"NV_ENC_ERR_RESOURCE_NOT_REGISTERED",
		"NV_ENC_ERR_RESOURCE_NOT_MAPPED"
	};

	if (code >= 0 && code < errors.size())
		return errors[code];

	return "Unknown error code";
}

/**
 * @brief Encoder implementation.
 */
class Encoder
{
public:
	Encoder(NvPipe_Format format, NvPipe_Codec codec, NvPipe_Compression compression, uint64_t bitrate, uint32_t targetFrameRate, uint32_t width, uint32_t height)
	{
		this->format = format;
		this->codec = codec;
		this->compression = compression;
		this->bitrate = bitrate;
		this->targetFrameRate = targetFrameRate;

		this->recreate(width, height);
	}

	virtual ~Encoder()
	{
		// Destroy encoder
		if (this->encoder)
		{
			std::vector<std::vector<uint8_t>> tmp;
			this->encoder->EndEncode(tmp);
			this->encoder->DestroyEncoder();
			this->encoder.reset();
		}

		// Free temporary device memory
		if (this->deviceBuffer)
			cudaFree(this->deviceBuffer);
	}

	void setBitrate(uint64_t bitrate, uint32_t targetFrameRate)
	{
		NV_ENC_CONFIG config;
		memset(&config, 0, sizeof(config));
		config.version = NV_ENC_CONFIG_VER;
		config.rcParams.averageBitRate = bitrate;

		NV_ENC_RECONFIGURE_PARAMS reconfigureParams;
		memset(&reconfigureParams, 0, sizeof(reconfigureParams));
		reconfigureParams.version = NV_ENC_RECONFIGURE_PARAMS_VER;
		reconfigureParams.resetEncoder = 1;
		reconfigureParams.forceIDR = 1;
		reconfigureParams.reInitEncodeParams.encodeConfig = &config;

		encoder->GetInitializeParams(&reconfigureParams.reInitEncodeParams);
		reconfigureParams.reInitEncodeParams.frameRateNum = targetFrameRate;
		reconfigureParams.reInitEncodeParams.frameRateDen = 1;

		encoder->Reconfigure(&reconfigureParams);

		this->bitrate = bitrate;
		this->targetFrameRate = targetFrameRate;
	}

	uint64_t encode(const void* src, uint64_t srcPitch, uint8_t* dst, uint64_t dstSize, uint32_t width, uint32_t height, bool forceIFrame)
	{
		// Recreate encoder if size changed
		if (this->format == NVPIPE_UINT16)
			this->recreate(width * 2, height); // split into two adjecent tiles in Y channel
		else if (this->format == NVPIPE_UINT32)
			this->recreate(width * 4, height); // split into four adjecent tiles in Y channel
		else
			this->recreate(width, height);

		// RGBA can be directly copied from host or device
		if (this->format == NVPIPE_RGBA32)
		{
			const NvEncInputFrame* f = this->encoder->GetNextInputFrame();
			CUDA_THROW(cudaMemcpy2D(f->inputPtr, f->pitch, src, srcPitch, width * 4, height, isDevicePointer(src) ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice),
				"Failed to copy input frame");
		}
		// Other formats need to be copied to the device and converted
		else
		{
			// Copy to device if necessary
			bool copyToDevice = !isDevicePointer(src);
			if (copyToDevice)
			{
				this->recreateDeviceBuffer(width, height);
				CUDA_THROW(cudaMemcpy(this->deviceBuffer, src, getFrameSize(this->format, width, height), cudaMemcpyHostToDevice),
					"Failed to copy input frame");
			}

			// Convert
			const NvEncInputFrame* f = this->encoder->GetNextInputFrame();

			if (this->format == NVPIPE_UINT4)
			{
				// one thread per pixel (extract 4 bit and copy to 8 bit)
				dim3 gridSize(width / 16 + 1, height / 2 + 1);
				dim3 blockSize(16, 2);

				uint4_to_nv12 << <gridSize, blockSize >> > ((uint8_t*)(copyToDevice ? this->deviceBuffer : src), srcPitch, (uint8_t*)f->inputPtr, f->pitch, width, height);
			}
			else if (this->format == NVPIPE_UINT8)
			{
				// one thread per pixel (copy 8 bit)
				dim3 gridSize(width / 16 + 1, height / 2 + 1);
				dim3 blockSize(16, 2);

				uint8_to_nv12 << <gridSize, blockSize >> > ((uint8_t*)(copyToDevice ? this->deviceBuffer : src), srcPitch, (uint8_t*)f->inputPtr, f->pitch, width, height);
			}
			else if (this->format == NVPIPE_UINT16)
			{
				// one thread per pixel (split 16 bit into 2x 8 bit)
				dim3 gridSize(width / 16 + 1, height / 2 + 1);
				dim3 blockSize(16, 2);

				uint16_to_nv12 << <gridSize, blockSize >> > ((uint8_t*)(copyToDevice ? this->deviceBuffer : src), srcPitch, (uint8_t*)f->inputPtr, f->pitch, width, height);
			}
			else if (this->format == NVPIPE_UINT32)
			{
				// one thread per pixel (split 32 bit into 4x 8 bit)
				dim3 gridSize(width / 16 + 1, height / 2 + 1);
				dim3 blockSize(16, 2);

				uint32_to_nv12 << <gridSize, blockSize >> > ((uint8_t*)(copyToDevice ? this->deviceBuffer : src), srcPitch, (uint8_t*)f->inputPtr, f->pitch, width, height);
			}
		}

		// Encode
		return this->encode(dst, dstSize, forceIFrame);
	}

#ifdef NVPIPE_WITH_OPENGL

	uint64_t encodeTexture(uint32_t texture, uint32_t target, uint8_t* dst, uint64_t dstSize, uint32_t width, uint32_t height, bool forceIFrame)
	{
		if (this->format != NVPIPE_RGBA32)
			throw Exception("The OpenGL interface only supports the RGBA32 format");

		// Recreate encoder if size changed
		this->recreate(width, height);

		// Map texture and copy input to encoder
		cudaGraphicsResource_t resource = this->registry.getTextureGraphicsResource(texture, target, width, height, cudaGraphicsRegisterFlagsReadOnly);
		CUDA_THROW(cudaGraphicsMapResources(1, &resource),
			"Failed to map texture graphics resource");
		cudaArray_t array;
		CUDA_THROW(cudaGraphicsSubResourceGetMappedArray(&array, resource, 0, 0),
			"Failed get texture graphics resource array");

		const NvEncInputFrame* f = this->encoder->GetNextInputFrame();
		CUDA_THROW(cudaMemcpy2DFromArray(f->inputPtr, f->pitch, array, 0, 0, width * 4, height, cudaMemcpyDeviceToDevice),
			"Failed to copy from texture array");

		// Encode
		uint64_t size = this->encode(dst, dstSize, forceIFrame);

		// Unmap texture
		CUDA_THROW(cudaGraphicsUnmapResources(1, &resource),
			"Failed to unmap texture graphics resource");

		return size;
	}

	uint64_t encodePBO(uint32_t pbo, uint8_t* dst, uint64_t dstSize, uint32_t width, uint32_t height, bool forceIFrame)
	{
		if (this->format != NVPIPE_RGBA32)
			throw Exception("The OpenGL interface only supports the RGBA32 format");

		// Map PBO and copy input to encoder
		cudaGraphicsResource_t resource = this->registry.getPBOGraphicsResource(pbo, width, height, cudaGraphicsRegisterFlagsReadOnly);
		CUDA_THROW(cudaGraphicsMapResources(1, &resource),
			"Failed to map PBO graphics resource");
		void* pboPointer;
		size_t pboSize;
		CUDA_THROW(cudaGraphicsResourceGetMappedPointer(&pboPointer, &pboSize, resource),
			"Failed to get mapped PBO pointer");

		// Encode
		uint64_t size = this->encode(pboPointer, width * 4, dst, dstSize, width, height, forceIFrame);

		// Unmap PBO
		CUDA_THROW(cudaGraphicsUnmapResources(1, &resource),
			"Failed to unmap PBO graphics resource");

		return size;
	}

#endif

protected:
	void recreate(uint32_t width, uint32_t height)
	{
		std::lock_guard<std::mutex> lock(Encoder::mutex);

		// Only recreate if necessary
		if (width == this->width && height == this->height)
			return;

		this->width = width;
		this->height = height;

		// Ensure we have a CUDA context
		CUDA_THROW(cudaDeviceSynchronize(),
			"Failed to synchronize device");
		CUcontext cudaContext;
		cuCtxGetCurrent(&cudaContext);

		// Create encoder
		try
		{
			// Destroy previous encoder
			if (this->encoder)
			{
				std::vector<std::vector<uint8_t>> tmp;
				this->encoder->EndEncode(tmp);
				this->encoder->DestroyEncoder();
				this->encoder.reset();
			}

			NV_ENC_BUFFER_FORMAT bufferFormat = (this->format == NVPIPE_RGBA32) ? NV_ENC_BUFFER_FORMAT_ABGR : NV_ENC_BUFFER_FORMAT_NV12;
			this->encoder = std::unique_ptr<NvEncoderCuda>(new NvEncoderCuda(cudaContext, width, height, bufferFormat, 0));

			NV_ENC_INITIALIZE_PARAMS initializeParams = { NV_ENC_INITIALIZE_PARAMS_VER };
			NV_ENC_CONFIG encodeConfig = { NV_ENC_CONFIG_VER };
			initializeParams.encodeConfig = &encodeConfig;

			GUID codecGUID = (this->codec == NVPIPE_HEVC) ? NV_ENC_CODEC_HEVC_GUID : NV_ENC_CODEC_H264_GUID;

			GUID presetGUID = NV_ENC_PRESET_LOW_LATENCY_HQ_GUID;
			if (this->compression == NVPIPE_LOSSLESS)
				presetGUID = NV_ENC_PRESET_LOSSLESS_DEFAULT_GUID; // NV_ENC_PRESET_LOSSLESS_HP_GUID

			encoder->CreateDefaultEncoderParams(&initializeParams, codecGUID, presetGUID);

			initializeParams.encodeWidth = width;
			initializeParams.encodeHeight = height;
			initializeParams.frameRateNum = this->targetFrameRate;
			initializeParams.frameRateDen = 1;
			initializeParams.enablePTD = 1;

			encodeConfig.gopLength = NVENC_INFINITE_GOPLENGTH; // No B-frames
			encodeConfig.frameIntervalP = 1;

			if (this->codec == NVPIPE_H264)
				encodeConfig.encodeCodecConfig.h264Config.idrPeriod = NVENC_INFINITE_GOPLENGTH;
			else if (this->codec == NVPIPE_HEVC)
				encodeConfig.encodeCodecConfig.hevcConfig.idrPeriod = NVENC_INFINITE_GOPLENGTH;

			if (this->compression == NVPIPE_LOSSY)
			{
				encodeConfig.rcParams.averageBitRate = this->bitrate;
				encodeConfig.rcParams.rateControlMode = NV_ENC_PARAMS_RC_CBR_LOWDELAY_HQ;
				encodeConfig.rcParams.vbvBufferSize = encodeConfig.rcParams.averageBitRate * initializeParams.frameRateDen / initializeParams.frameRateNum; // bitrate / framerate = one frame
				encodeConfig.rcParams.maxBitRate = encodeConfig.rcParams.averageBitRate;
				encodeConfig.rcParams.vbvInitialDelay = encodeConfig.rcParams.vbvBufferSize;
			}

			encoder->CreateEncoder(&initializeParams);
		}
		catch (NVENCException & e)
		{
			throw Exception("Failed to create encoder (" + e.getErrorString() + ", error " + std::to_string(e.getErrorCode()) + " = " + EncErrorCodeToString(e.getErrorCode()) + ")");
		}
	}

	uint64_t encode(uint8_t* dst, uint64_t dstSize, bool forceIFrame)
	{
		std::vector<std::vector<uint8_t>> packets;

		try
		{
			if (forceIFrame)
			{
				NV_ENC_PIC_PARAMS params = {};
				params.encodePicFlags = NV_ENC_PIC_FLAG_FORCEIDR | NV_ENC_PIC_FLAG_OUTPUT_SPSPPS;

				this->encoder->EncodeFrame(packets, &params);
			}
			else
			{
				this->encoder->EncodeFrame(packets);
			}
		}
		catch (NVENCException & e)
		{
			throw Exception("Encode failed (" + e.getErrorString() + ", error " + std::to_string(e.getErrorCode()) + " = " + EncErrorCodeToString(e.getErrorCode()) + ")");
		}

		// Copy output
		uint64_t size = 0;
		for (auto& p : packets)
		{
			if (size + p.size() <= dstSize)
			{
				memcpy(dst + size, p.data(), p.size());
				size += p.size();
			}
			else
			{
				throw Exception("Encode output buffer overflow");
			}
		}

		return size;
	}

	void recreateDeviceBuffer(uint32_t width, uint32_t height)
	{
		// (Re)allocate temporary device memory if necessary
		uint64_t requiredSize = getFrameSize(this->format, width, height);

		if (this->deviceBufferSize < requiredSize)
		{
			if (this->deviceBuffer)
				cudaFree(this->deviceBuffer);

			this->deviceBufferSize = requiredSize;
			CUDA_THROW(cudaMalloc(&this->deviceBuffer, this->deviceBufferSize),
				"Failed to allocate temporary device memory");
		}
	}

protected:
	NvPipe_Format format;
	NvPipe_Codec codec;
	NvPipe_Compression compression;
	uint64_t bitrate;
	uint32_t targetFrameRate;
	uint32_t width = 0;
	uint32_t height = 0;

	std::unique_ptr<NvEncoderCuda> encoder;

	void* deviceBuffer = nullptr;
	uint64_t deviceBufferSize = 0;

	static std::mutex mutex;

#ifdef NVPIPE_WITH_OPENGL
	GraphicsResourceRegistry registry;
#endif
};

std::mutex Encoder::mutex;

#endif

#ifdef NVPIPE_WITH_DECODER

inline std::string DecErrorCodeToString(CUresult code)
{
	const char* str = nullptr;
	cuGetErrorName(code, &str);

	if (str)
		return std::string(str);

	return "Unknown error code";
}

/**
 * @brief Decoder implementation.
 */
class Decoder
{
public:
	Decoder(NvPipe_Format format, NvPipe_Codec codec, uint32_t width, uint32_t height)
	{
		this->format = format;
		this->codec = codec;

		this->recreate(width, height);
	}

	~Decoder()
	{
		// Free temporary device memory
		if (this->deviceBuffer)
			cudaFree(this->deviceBuffer);
	}

	uint64_t decode(const uint8_t* src, uint64_t srcSize, void* dst, uint32_t width, uint32_t height)
	{
		// Recreate decoder if size changed
		if (this->format == NVPIPE_UINT16)
			this->recreate(width * 2, height); // split into two adjecent tiles in Y channel
		else if (this->format == NVPIPE_UINT32)
			this->recreate(width * 4, height); // split into four adjecent tiles in Y channel
		else
			this->recreate(width, height);

		// Decode
		uint8_t* decoded = this->decode(src, srcSize);

		if (nullptr != decoded)
		{
			// Allocate temporary device buffer if we need to copy to the host eventually
			bool copyToHost = !isDevicePointer(dst);
			if (copyToHost)
				this->recreateDeviceBuffer(width, height);

			// Convert to output format
			uint8_t* dstDevice = (uint8_t*)(copyToHost ? this->deviceBuffer : dst);

			if (this->format == NVPIPE_RGBA32)
			{
				Nv12ToColor32<RGBA32>(decoded, width, dstDevice, width * 4, width, height);
			}
			else if (this->format == NVPIPE_UINT4)
			{
				// one thread per TWO pixels (merge 2x4 bit to one byte per thread)
				dim3 gridSize(width / 16 / 2 + 1, height / 2 + 1);
				dim3 blockSize(16, 2);

				nv12_to_uint4 << <gridSize, blockSize >> > (decoded, this->decoder->GetDeviceFramePitch(), dstDevice, width / 2, width, height);
			}
			else if (this->format == NVPIPE_UINT8)
			{
				// one thread per pixel (copy 8 bit)
				dim3 gridSize(width / 16 + 1, height / 2 + 1);
				dim3 blockSize(16, 2);

				nv12_to_uint8 << <gridSize, blockSize >> > (decoded, this->decoder->GetDeviceFramePitch(), dstDevice, width, width, height);
			}
			else if (this->format == NVPIPE_UINT16)
			{
				// one thread per pixel (merge 2x8 bit into 16 bit pixels)
				dim3 gridSize(width / 16 + 1, height / 2 + 1);
				dim3 blockSize(16, 2);

				nv12_to_uint16 << <gridSize, blockSize >> > (decoded, this->decoder->GetDeviceFramePitch(), dstDevice, width * 2, width, height);
			}
			else if (this->format == NVPIPE_UINT32)
			{
				// one thread per pixel (merge 4x8 bit into 32 bit pixels)
				dim3 gridSize(width / 16 + 1, height / 2 + 1);
				dim3 blockSize(16, 2);

				nv12_to_uint32 << <gridSize, blockSize >> > (decoded, this->decoder->GetDeviceFramePitch(), dstDevice, width * 4, width, height);
			}

			// Copy to host if necessary
			if (copyToHost)
				CUDA_THROW(cudaMemcpy(dst, this->deviceBuffer, getFrameSize(this->format, width, height), cudaMemcpyDeviceToHost),
					"Failed to copy output to host memory");

			return getFrameSize(this->format, width, height);
		}

		return 0;
	}

#ifdef NVPIPE_WITH_OPENGL

	uint64_t decodeTexture(const uint8_t* src, uint64_t srcSize, uint32_t texture, uint32_t target, uint32_t width, uint32_t height)
	{
		if (this->format != NVPIPE_RGBA32)
			throw Exception("The OpenGL interface only supports the RGBA32 format");

		// Recreate decoder if size changed
		this->recreate(width, height);

		// Decode
		uint8_t* decoded = this->decode(src, srcSize);

		if (nullptr != decoded)
		{
			// Convert to RGBA
			this->recreateDeviceBuffer(width, height);
			Nv12ToColor32<RGBA32>(decoded, width, (uint8_t*)this->deviceBuffer, width * 4, width, height);

			// Copy output to texture
			cudaGraphicsResource_t resource = this->registry.getTextureGraphicsResource(texture, target, width, height, cudaGraphicsRegisterFlagsWriteDiscard);
			CUDA_THROW(cudaGraphicsMapResources(1, &resource),
				"Failed to map texture graphics resource");
			cudaArray_t array;
			CUDA_THROW(cudaGraphicsSubResourceGetMappedArray(&array, resource, 0, 0),
				"Failed get texture graphics resource array");
			CUDA_THROW(cudaMemcpy2DToArray(array, 0, 0, this->deviceBuffer, width * 4, width * 4, height, cudaMemcpyDeviceToDevice),
				"Failed to copy to texture array");
			CUDA_THROW(cudaGraphicsUnmapResources(1, &resource),
				"Failed to unmap texture graphics resource");

			return width * height * 4;
		}

		return 0;
	}

	uint64_t decodePBO(const uint8_t* src, uint64_t srcSize, uint32_t pbo, uint32_t width, uint32_t height)
	{
		if (this->format != NVPIPE_RGBA32)
			throw Exception("The OpenGL interface only supports the RGBA32 format");

		// Map PBO for output
		cudaGraphicsResource_t resource = this->registry.getPBOGraphicsResource(pbo, width, height, cudaGraphicsRegisterFlagsWriteDiscard);
		CUDA_THROW(cudaGraphicsMapResources(1, &resource),
			"Failed to map PBO graphics resource");
		void* pboPointer;
		size_t pboSize;
		CUDA_THROW(cudaGraphicsResourceGetMappedPointer(&pboPointer, &pboSize, resource),
			"Failed to get mapped PBO pointer");

		// Decode
		uint64_t size = this->decode(src, srcSize, pboPointer, width, height);

		// Unmap PBO
		CUDA_THROW(cudaGraphicsUnmapResources(1, &resource),
			"Failed to unmap PBO graphics resource");

		return size;
	}

#endif

private:
	void recreate(uint32_t width, uint32_t height)
	{
		std::lock_guard<std::mutex> lock(Decoder::mutex);

		// Only recreate if necessary
		if (width == this->width && height == this->height)
			return;

		this->width = width;
		this->height = height;

		// Ensure we have a CUDA context
		CUDA_THROW(cudaDeviceSynchronize(),
			"Failed to synchronize device");
		CUcontext cudaContext;
		cuCtxGetCurrent(&cudaContext);

		// Create decoder
		try
		{
			// Destroy previous decoder
			if (this->decoder)
			{
				this->decoder.reset();
			}

			this->decoder = std::unique_ptr<NvDecoder>(new NvDecoder(cudaContext, width, height, true, (this->codec == NVPIPE_HEVC) ? cudaVideoCodec_HEVC : cudaVideoCodec_H264,/* &Decoder::mutex*/ nullptr, true));
		}
		catch (NVDECException & e)
		{
			throw Exception("Failed to create decoder (" + e.getErrorString() + ", error " + std::to_string(e.getErrorCode()) + " = " + DecErrorCodeToString(e.getErrorCode()) + ")");
		}
	}

	uint8_t* decode(const uint8_t* src, uint64_t srcSize)
	{
		int numFramesDecoded = 0;
		uint8_t** decodedFrames;
		int64_t* timeStamps;

		try
		{
			// Some cuvid implementations have one frame latency. Refeed frame into pipeline in this case.
			const uint32_t DECODE_TRIES = 3;
			for (uint32_t i = 0; (i < DECODE_TRIES) && (numFramesDecoded <= 0); ++i)
				this->decoder->Decode(src, srcSize, &decodedFrames, &numFramesDecoded, CUVID_PKT_ENDOFPICTURE, &timeStamps, this->n++);
		}
		catch (NVDECException & e)
		{
			throw Exception("Decode failed (" + e.getErrorString() + ", error " + std::to_string(e.getErrorCode()) + " = " + DecErrorCodeToString(e.getErrorCode()) + ")");
		}

		if (numFramesDecoded <= 0)
		{
			throw Exception("No frame decoded (Decoder expects encoded bitstream for a single complete frame. Accumulating partial data or combining multiple frames is not supported.)");
		}

		return decodedFrames[numFramesDecoded - 1];
	}

	void recreateDeviceBuffer(uint32_t width, uint32_t height)
	{
		// (Re)allocate temporary device memory if necessary
		uint64_t requiredSize = getFrameSize(this->format, width, height);

		if (this->deviceBufferSize < requiredSize)
		{
			if (this->deviceBuffer)
				cudaFree(this->deviceBuffer);

			this->deviceBufferSize = requiredSize;
			CUDA_THROW(cudaMalloc(&this->deviceBuffer, this->deviceBufferSize),
				"Failed to allocate temporary device memory");
		}
	}

private:
	NvPipe_Format format;
	NvPipe_Codec codec;
	uint32_t width = 0;
	uint32_t height = 0;

	std::unique_ptr<NvDecoder> decoder;
	int64_t n = 0;

	void* deviceBuffer = nullptr;
	uint64_t deviceBufferSize = 0;

	static std::mutex mutex;

#ifdef NVPIPE_WITH_OPENGL
	GraphicsResourceRegistry registry;
#endif
};

std::mutex Decoder::mutex;

#endif




/*
MIT License

Copyright (c) 2020 yangrc1234

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

*/

#ifdef NVPIPE_WITH_ENCODER
#ifdef NVPIPE_WITH_OPENGL
class AsyncTextureEncoder : public Encoder
{
	struct IntermediateBuffer {
		CUdeviceptr ptr;
		size_t pitch;
		~IntermediateBuffer()
		{
			cuMemFree(ptr);
		}
	};
public:
	static constexpr int kEncodeBufferCount = 3;
	AsyncTextureEncoder(NvPipe_Format format, NvPipe_Codec codec, NvPipe_Compression compression, uint64_t bitrate, uint32_t targetFrameRate, uint32_t width, uint32_t height) :
		Encoder(format, codec, compression, bitrate, targetFrameRate, width, height),
		m_clearedPtr(0), m_encodedPtr(0), m_pendingTaskPtr(0)
	{
		m_outputBufferSize = width * height * 4;
		for (size_t i = 0; i < 3; i++)
		{
			cuMemAllocPitch(&m_intermdiateBuffer[i].ptr, &m_intermdiateBuffer[i].pitch, width * 4, height, 16);
			this->m_outputBuffer[i] = std::make_unique<uint8_t[]>(m_outputBufferSize);
		}
		m_closed = false;
		m_encodeThread = std::make_unique<std::thread>(&AsyncTextureEncoder::encodeThread, this);
	}

	virtual ~AsyncTextureEncoder()
	{
		m_closed = true;
		{
			std::lock_guard<std::mutex> lock(this->m_cvMutex);
			this->m_blockCv.notify_one();	
		}
		m_encodeThread->join();
	}

	int encodeTextureAsync(uint32_t texture, uint32_t target, uint32_t width, uint32_t height, bool forceIFrame) {
		if (this->format != NVPIPE_RGBA32)
			throw Exception("The OpenGL interface only supports the RGBA32 format");

		if ((this->m_pendingTaskPtr + 1) % kEncodeBufferCount == this->m_clearedPtr)  //Encode/Clear task is too slow.
		{
			throw Exception("Encoder is too slow or task is not cleared, failed to enqueue new encode task. \n");
		}

		auto currentTaskIndex = this->m_pendingTaskPtr.load();

		// Map texture and copy input to encoder
		cudaGraphicsResource_t resource = this->registry.getTextureGraphicsResource(texture, target, width, height, cudaGraphicsRegisterFlagsReadOnly);
		CUDA_THROW(cudaGraphicsMapResources(1, &resource),
			"Failed to map texture graphics resource");
		cudaArray_t array;
		CUDA_THROW(cudaGraphicsSubResourceGetMappedArray(&array, resource, 0, 0),
			"Failed get texture graphics resource array");

		//Copy to intermediate buffer.
		CUDA_THROW(cudaMemcpy2DFromArray(
			(void*)m_intermdiateBuffer[currentTaskIndex].ptr,
			m_intermdiateBuffer[currentTaskIndex].pitch,
			array,
			0, 0, width * 4, height, cudaMemcpyDeviceToDevice),
			"Failed to copy memory to intermediate buffer."
		);

		// Unmap texture
		CUDA_THROW(cudaGraphicsUnmapResources(1, &resource),
			"Failed to unmap texture graphics resource");

		{//Enqueue the task.

			AsyncTask task;
			task.forceIFrame = forceIFrame;
			task.height = height;
			task.width = width;
			task.isError = false;
			m_tasks[currentTaskIndex] = std::move(task);
			DEBUG_LOG("Encoder: %d task is in async queue now\n", currentTaskIndex);

			//Task is published, move pointer to next, and encode thread can operate on current task.
			//Atomic guarantees this happens after m_tasks[currentTaskIndex] = std::move(task);
			this->m_pendingTaskPtr = (this->m_pendingTaskPtr + 1) % kEncodeBufferCount;
			
			{
				//Inform encode thread.
				std::lock_guard<std::mutex> lock(this->m_cvMutex);
				this->m_blockCv.notify_one();
			}
		}
		return currentTaskIndex;    //Return task number.
	}

	void encodeThread() {
		std::unique_lock<std::mutex> lock(this->m_cvMutex);
		while (!m_closed)
		{
			if (m_encodedPtr == m_pendingTaskPtr)	//No task for now. wait.
			{
				this->m_blockCv.wait(lock);
				continue;
			}

			auto& currTask = m_tasks[m_encodedPtr];

			DEBUG_LOG("Encoder thread: Encoding task: %d\n", m_encodedPtr.load());

			try
			{
				// Encode
				const NvEncInputFrame* f = this->encoder->GetNextInputFrame();
				CUDA_THROW(cudaMemcpy2D(f->inputPtr, f->pitch,
					(void*)m_intermdiateBuffer[m_encodedPtr].ptr,
					m_intermdiateBuffer[m_encodedPtr].pitch,
					width * 4, currTask.height, cudaMemcpyDeviceToDevice),
					"Failed to copy from texture array");
				uint64_t size = this->encode(m_outputBuffer[m_encodedPtr].get(), m_outputBufferSize, currTask.forceIFrame);
				m_tasks[m_encodedPtr].isError = false;
				m_tasks[m_encodedPtr].encodedSize = size;
			}
			catch (const Exception & e)
			{
				m_tasks[m_encodedPtr].isError = true;
				m_tasks[m_encodedPtr].error = e.message;
			}

			DEBUG_LOG("Encoding finished\n");
			m_encodedPtr = (m_encodedPtr + 1) % kEncodeBufferCount;
		}
	}

	void QueryTask(int taskIndex, bool* isDone, bool* isError, std::string* error) {
		DEBUG_LOG("Encoder Query: query %d\n", taskIndex);
		if (!CheckInsideQueueRange(m_pendingTaskPtr.load(), m_clearedPtr.load(), taskIndex)) {
			//This task doesn't eixsts.
			DEBUG_LOG("Encoder Query: task doesn't exists\n");
			throw Exception("Task doesn't exists");
		}
		if (CheckInsideQueueRange(m_pendingTaskPtr.load(), m_encodedPtr.load(), taskIndex)) {
			//Not encoded yet
			DEBUG_LOG("Encoder Query: task %d not encoded yet\n", taskIndex);
			*isDone = false;
			*isError = false;
			return;
		}
		if (CheckInsideQueueRange(m_encodedPtr.load(), m_clearedPtr.load(), taskIndex)) {
			//Done
			DEBUG_LOG("Encoder Query: task %d is done.\n", taskIndex);
			auto& task = m_tasks[taskIndex];
			*isDone = true;
			*isError = task.isError;
			if (task.isError)
				*error = task.error;
			return;
		}
		DEBUG_LOG("Query Task Exception Encoutnered, Current Status: Pending %d Encoding %d Clearing %d Query %d\n", m_pendingTaskPtr.load(), m_encodedPtr.load(), m_clearedPtr.load(), taskIndex);
		throw Exception("Unknown error");
	}

	uint8_t* AcquireTaskData(int taskIndex, uint64_t* encodeSize) {
		auto& task = m_tasks[taskIndex];
		if (!CheckInsideQueueRange(m_encodedPtr.load(), m_clearedPtr.load(), taskIndex)) {
			throw Exception("The task is not done yet!");
		}
		if (!task.isError)
			*encodeSize = m_tasks[taskIndex].encodedSize;
		return m_outputBuffer[taskIndex].get();
	}

	void ClearTask(int taskIndex) {
		if (taskIndex != m_clearedPtr) {
			throw Exception("Only next task could be cleared!");
		}
		if (taskIndex == m_encodedPtr) {
			throw Exception("The task is not finished yet!");
		}
		m_tasks[m_clearedPtr] = AsyncTask();
		m_clearedPtr = (1 + m_clearedPtr) % kEncodeBufferCount;
		DEBUG_LOG("Encoder: %d task is cleared\n", m_clearedPtr.load());
	}

private:

	struct AsyncTask
	{
		AsyncTask()
		{
		}
		uint32_t width;
		uint32_t height;
		bool forceIFrame;

		//bool isDone;	//isDone could be inferred from circular buffer pointers.
		bool isError;
		uint64_t encodedSize;
		std::string error;
	};

	std::mutex m_cvMutex;	//Need lock to avoid encoder thread always running. The lock won't affect anything if encoder is busy.
	std::condition_variable m_blockCv;

	std::atomic<bool> m_closed;
	std::atomic<int> m_pendingTaskPtr;
	std::atomic<int> m_encodedPtr;
	std::atomic<int> m_clearedPtr;
	IntermediateBuffer m_intermdiateBuffer[kEncodeBufferCount];
	std::unique_ptr<uint8_t[]> m_outputBuffer[kEncodeBufferCount];
	uint64_t m_outputBufferSize;
	AsyncTask m_tasks[kEncodeBufferCount];
	std::unique_ptr<std::thread> m_encodeThread;
};
#endif
#endif

struct Instance
{
#ifdef NVPIPE_WITH_ENCODER
	std::unique_ptr<Encoder> encoder;
#ifdef NVPIPE_WITH_OPENGL
	std::unique_ptr<AsyncTextureEncoder> asyncTextureEncoder;
#endif
#endif


#ifdef NVPIPE_WITH_DECODER
	std::unique_ptr<Decoder> decoder;
#endif


	std::string error;
};

std::string sharedError; // shared error code for create functions (NOT threadsafe)
std::unordered_map<uint32_t, std::shared_ptr<Instance>> g_pipes;
std::mutex g_pipeDictMutex;
uint32_t g_pipeCreationIndex = 1;	//Start with 1, since GetError requries a special value(0 here) for global error.

static void DeletePipe(uint32_t id) {
	std::lock_guard<std::mutex> lock(g_pipeDictMutex);
	g_pipes.erase(id);
}

static std::shared_ptr<Instance> GetPipe(uint32_t id) {
	std::lock_guard<std::mutex> lock(g_pipeDictMutex);
	auto ite = g_pipes.find(id);
	if (ite == g_pipes.end()) {
		return nullptr;
	}
	return ite->second;
}

static uint32_t InsertNewPipe(std::shared_ptr<Instance> instance) {
	std::lock_guard<std::mutex> lock(g_pipeDictMutex);
	auto index = g_pipeCreationIndex++;
	g_pipes[index] = instance;
	return index;
}

#ifdef NVPIPE_WITH_ENCODER

UNITY_INTERFACE_EXPORT uint32_t UNITY_INTERFACE_API NvPipe_CreateEncoder(NvPipe_Format format, NvPipe_Codec codec, NvPipe_Compression compression, uint64_t bitrate, uint32_t targetFrameRate, uint32_t width, uint32_t height)
{
	auto instance = std::make_shared<Instance>();

	try
	{
		instance->encoder = std::unique_ptr<Encoder>(new Encoder(format, codec, compression, bitrate, targetFrameRate, width, height));
		return InsertNewPipe(instance);
	}
	catch (Exception & e)
	{
		sharedError = e.getErrorString();
		return 0;
	}

	return 0;
}

#ifdef NVPIPE_WITH_OPENGL
UNITY_INTERFACE_EXPORT uint32_t UNITY_INTERFACE_API NvPipe_CreateTextureAsyncEncoder(NvPipe_Format format, NvPipe_Codec codec, NvPipe_Compression compression, uint64_t bitrate, uint32_t targetFrameRate, uint32_t width, uint32_t height)
{
	auto instance = std::make_shared<Instance>();

	try
	{
		instance->asyncTextureEncoder = std::make_unique<AsyncTextureEncoder>(format, codec, compression, bitrate, targetFrameRate, width, height);
		return InsertNewPipe(instance);
	}
	catch (Exception & e)
	{
		sharedError = e.getErrorString();
		return 0;
	}

	return 0;
}
#endif

UNITY_INTERFACE_EXPORT void UNITY_INTERFACE_API NvPipe_SetBitrate(uint64_t pipe, uint64_t bitrate, uint32_t targetFrameRate)
{
	auto instance = GetPipe(pipe);
	if (instance == nullptr)
		return;

	if (!instance->encoder)
	{
		instance->error = "Invalid NvPipe encoder.";
		return;
	}

	try
	{
		return instance->encoder->setBitrate(bitrate, targetFrameRate);
	}
	catch (Exception & e)
	{
		instance->error = e.getErrorString();
	}
}

UNITY_INTERFACE_EXPORT uint64_t UNITY_INTERFACE_API NvPipe_Encode(uint32_t pipe, const void* src, uint64_t srcPitch, uint8_t* dst, uint64_t dstSize, uint32_t width, uint32_t height, bool forceIFrame)
{
	auto instance = GetPipe(pipe);
	if (instance == nullptr)
		return 0;
	if (!instance->encoder)
	{
		instance->error = "Invalid NvPipe encoder.";
		return 0;
	}

	try
	{
		return instance->encoder->encode(src, srcPitch, dst, dstSize, width, height, forceIFrame);
	}
	catch (Exception & e)
	{
		instance->error = e.getErrorString();
		return 0;
	}
}

#ifdef NVPIPE_WITH_OPENGL

UNITY_INTERFACE_EXPORT uint64_t UNITY_INTERFACE_API NvPipe_EncodeTexture(uint32_t pipe, uint32_t texture, uint32_t target, uint8_t* dst, uint64_t dstSize, uint32_t width, uint32_t height, bool forceIFrame)
{
	auto instance = GetPipe(pipe);

	if (instance == nullptr)
		return 0;

	if (!instance->encoder)
	{
		instance->error = "Invalid NvPipe encoder.";
		return 0;
	}

	try
	{
		return instance->encoder->encodeTexture(texture, target, dst, dstSize, width, height, forceIFrame);
	}
	catch (Exception & e)
	{
		instance->error = e.getErrorString();
		return 0;
	}
}


/*
==================================
Async OpenGL Texture Encoding.
==================================
*/
struct MainThreadPendingTask {	//Tasks from main thread.
	MainThreadPendingTask() = default;
	MainThreadPendingTask(std::shared_ptr<Instance> nvp, uint32_t texture, uint32_t width, uint32_t height, bool forceIFrame)
	{
		this->pipe = nvp;
		this->texture = texture;
		this->width = width;
		this->height = height;
		this->forceIFrame = forceIFrame;

		this->isDone = false;
		this->isError = false;
	}

	std::shared_ptr<Instance> pipe;
	uint32_t texture;
	uint32_t width;
	uint32_t height;
	bool forceIFrame;
	int mainThreadTaskIndex;

	//Available once the renderthread submitted task to encoder
	int encoderTaskIndex;

	//Available once results are polled from encoder.
	bool isDone;
	bool isError;
	uint8_t* resultBuffer;
	std::string error;
	uint64_t encodedSize;
};

static constexpr int MAX_PENDING_TASK_COUNT = 20;	//Up to 20 tasks could exist at the same time.
static MainThreadPendingTask mainThreadPendingTasks[MAX_PENDING_TASK_COUNT];	//This is a circular buffer.

static std::atomic<uint32_t> g_pendingTaskPtr(0);		//circular buffer pointer.
static std::atomic<uint32_t> g_submittedTaskPtr(0);		//circular buffer pointer.
static std::atomic<uint32_t> g_doneTaskPtr(0);
static std::atomic<uint32_t> g_cleardTaskPtr(0);

static std::mutex g_destructMutex;	/*Used when resetting encode tasks. Reset only callde by main thread, so only RenderThread calls will need this mutex.*/

/*
Called in main thread, to clear all async encoding tasks.
But actual encoders and tasks inside it won't be destructed.
*/
UNITY_INTERFACE_EXPORT void UNITY_INTERFACE_API NvPipe_ResetEncodeTasks() {
	std::lock_guard<std::mutex> lock(g_destructMutex);
	for (size_t i = 0; i < MAX_PENDING_TASK_COUNT; i++)
	{
		mainThreadPendingTasks[i] = MainThreadPendingTask();
	}
	g_pendingTaskPtr = 0;
	g_submittedTaskPtr = 0;
	g_doneTaskPtr = 0;
	g_cleardTaskPtr = 0;

	DEBUG_LOG("async encode queue reset\n");
}

/*Called in main thread, to enqueue a new task.*/
UNITY_INTERFACE_EXPORT uint32_t UNITY_INTERFACE_API NvPipe_QueueEncodeTaskInMainThread(uint32_t nvp, uint32_t texture, uint32_t width, uint32_t height, bool forceIFrame) {
	auto pipe = GetPipe(nvp);
	if (pipe == nullptr)
		return 0;
	if ((g_pendingTaskPtr + 1) % MAX_PENDING_TASK_COUNT == g_cleardTaskPtr) {	//Reached maximum submit tasks per frame, or earlier tasks are not cleared yet.
		static char msgBuffer[200];
		sprintf(msgBuffer, "Maximum task count reached. Did you forget to clear task, or submitted too many tasks(%d) at once?", MAX_PENDING_TASK_COUNT);
		pipe->error = msgBuffer;
		return 0;
	}

	if (pipe->asyncTextureEncoder == nullptr) {
		pipe->error = "Invalid async texture encoder";
		return 0;
	}

	auto ptr = g_pendingTaskPtr.load();
	mainThreadPendingTasks[g_pendingTaskPtr] = MainThreadPendingTask(pipe, texture, width, height, forceIFrame);
	g_pendingTaskPtr = (1 + g_pendingTaskPtr) % MAX_PENDING_TASK_COUNT;
	DEBUG_LOG("async encode task enqueued, task index %d\n", ptr);
	return ptr;
}

/*
Called by render thread
To move all pending tasks into corresponding encoders,
and update all task infos from encoder.
*/
UNITY_INTERFACE_EXPORT void UNITY_INTERFACE_API NvPipe_RenderThreadPoll(int)
{
	std::lock_guard<std::mutex> lock(g_destructMutex);
	DEBUG_LOG("RTP: Render thread polling(Move to encoder)\n");
	while (g_submittedTaskPtr != g_pendingTaskPtr)
	{
		DEBUG_LOG("RTP: %d, %d\n", g_submittedTaskPtr.load(), g_pendingTaskPtr.load());
		auto& task = mainThreadPendingTasks[g_submittedTaskPtr];
		//Allocate buffer

		try     //Error here is not stored in instance.error, since we're in render thread. Put error along inside SubmittedTask.
		{
			task.encoderTaskIndex = task.pipe->asyncTextureEncoder->encodeTextureAsync(task.texture, GL_TEXTURE_2D, task.width, task.height, task.forceIFrame);

			DEBUG_LOG("RTP: %d entered encoder queue\n", g_submittedTaskPtr.load());
			task.isDone = false;
			task.isError = false;
		}
		catch (const Exception & e)
		{
			task.isDone = true;
			task.isError = true;
			task.error = e.message;
			DEBUG_LOG("RTP: %d failed to enqueue to encoder, error:%s\n", g_submittedTaskPtr.load(), e.message.c_str());
		}
		g_submittedTaskPtr = (g_submittedTaskPtr + 1) % MAX_PENDING_TASK_COUNT;
	}

	DEBUG_LOG("RTP: Render thread polling(Check done)\n");
	//Query any task is done.
	while (g_doneTaskPtr != g_submittedTaskPtr)
	{
		DEBUG_LOG("RTP: %d, %d\n", g_doneTaskPtr.load(), g_submittedTaskPtr.load());
		auto& task = mainThreadPendingTasks[g_doneTaskPtr];

		if (!task.isDone)	//Try to get task done.
		{
			try
			{	//Query task status from encoder.
				task.pipe->asyncTextureEncoder->QueryTask(task.encoderTaskIndex,
					&task.isDone,
					&task.isError,
					&task.error);

				if (task.isDone)
					DEBUG_LOG("RTP: Task set to done\n");
				if (task.isError)
					DEBUG_LOG("RTP: Task done with error: %s\n", task.error.c_str());

				if (task.isDone) {
					if (!task.isError) {
						//Get buffer back.
						task.resultBuffer = task.pipe->asyncTextureEncoder->AcquireTaskData(task.encoderTaskIndex,
							&task.encodedSize);
					}
					if (task.isDone) {
						task.pipe->asyncTextureEncoder->ClearTask(task.encoderTaskIndex);
					}
				}
			}
			catch (const Exception & e)
			{
				DEBUG_LOG("RTP: Exception during query task %d status. %s\n", task.encoderTaskIndex, e.message.c_str());
				task.isDone = true; task.isError = true; task.error = e.message;
			}
		}

		if (task.isDone) {
			g_doneTaskPtr = (g_doneTaskPtr + 1) % MAX_PENDING_TASK_COUNT;
		}
		else {
			//We can't "done" next task if current task is not done yet.
			break;
		}
	}

	DEBUG_LOG("Render thread polling finished\n");
}

UNITY_INTERFACE_EXPORT UnityRenderingEvent UNITY_INTERFACE_API NvPipe_GetRenderThreadPollFunc() {
	return NvPipe_RenderThreadPoll;
}

/*
Called in main thread, query status of encode task.
Only task error will be returned in **error. Other error goes to sharedError
*/
UNITY_INTERFACE_EXPORT void UNITY_INTERFACE_API NvPipe_EncodeTextureAsyncQuery(
	uint32_t taskIndex, bool* isDone, bool* isError, uint8_t** encodedData, uint64_t* encodeSize, const char** error) {

	try
	{
		//Check taskIndex is valid.
		if (taskIndex >= MAX_PENDING_TASK_COUNT || !CheckInsideQueueRange(g_pendingTaskPtr.load(), g_cleardTaskPtr.load(), taskIndex))
		{
			throw Exception("Task is not valid!");
		}

		if (CheckInsideQueueRange(g_pendingTaskPtr.load(), g_doneTaskPtr.load(), taskIndex)) {
			//Not done yet.
			*isDone = false;
			return;
		}

		if (CheckInsideQueueRange(g_doneTaskPtr.load(), g_cleardTaskPtr.load(), taskIndex)) {
			auto& task = mainThreadPendingTasks[taskIndex];
			//Small check
			if (task.isDone != true) {
				//This should never happen
				throw Exception("Fatal error.");
			}
			*isDone = true;
			*isError = task.isError;
			if (task.isError) {
				*error = task.error.c_str();
			}
			else {
				*encodeSize = task.encodedSize;
				*encodedData = task.resultBuffer;
			}
			return;
		}

		throw Exception("Unknown error....");
	}
	catch (const Exception & e)
	{
		sharedError = e.message;
	}

}

/*Called in main thread, to notify that a task could be cleared.*/
UNITY_INTERFACE_EXPORT void UNITY_INTERFACE_API NvPipe_EncodeTextureAsyncClearTask(uint32_t taskIndex) {
	try
	{
		if (g_cleardTaskPtr != taskIndex) {
			throw Exception("Only next uncleared task could be cleared");
		}
		if (taskIndex == g_doneTaskPtr) {
			throw Exception("The task is still being executed and can't be cleared");
		}

		DEBUG_LOG("RTP: %d is cleared \n", g_cleardTaskPtr.load());
		mainThreadPendingTasks[taskIndex] = MainThreadPendingTask();	//Clear it to empty object, to release references to pipe or buffer.

		g_cleardTaskPtr = (g_cleardTaskPtr + 1) % MAX_PENDING_TASK_COUNT;
	}
	catch (Exception & e)
	{
		sharedError = e.getErrorString();
		return;
	}
}

#endif

#endif

#ifdef NVPIPE_WITH_DECODER

UNITY_INTERFACE_EXPORT uint32_t UNITY_INTERFACE_API NvPipe_CreateDecoder(NvPipe_Format format, NvPipe_Codec codec, uint32_t width, uint32_t height)
{
	auto instance = std::make_shared<Instance>();

	try
	{
		instance->decoder = std::unique_ptr<Decoder>(new Decoder(format, codec, width, height));
		return InsertNewPipe(instance);
	}
	catch (Exception & e)
	{
		sharedError = e.getErrorString();
		return 0;
	}

	return 0;
}

UNITY_INTERFACE_EXPORT uint64_t UNITY_INTERFACE_API NvPipe_Decode(uint32_t nvp, const uint8_t* src, uint64_t srcSize, void* dst, uint32_t width, uint32_t height)
{
	auto instance = GetPipe(nvp);
	if (instance == nullptr)
		return 0;
	if (!instance->decoder)
	{
		instance->error = "Invalid NvPipe decoder.";
		return 0;
	}

	try
	{
		return instance->decoder->decode(src, srcSize, dst, width, height);
	}
	catch (Exception & e)
	{
		instance->error = e.getErrorString();
		return 0;
	}
}

#ifdef NVPIPE_WITH_OPENGL

UNITY_INTERFACE_EXPORT uint32_t UNITY_INTERFACE_API NvPipe_DecodeTexture(uint32_t nvp, const uint8_t* src, uint32_t srcSize, uint32_t texture, uint32_t target, uint32_t width, uint32_t height)
{
	auto instance = GetPipe(nvp);
	if (instance == nullptr)
		return 0;
	if (!instance->decoder)
	{
		instance->error = "Invalid NvPipe decoder.";
		return 0;
	}

	try
	{
		return instance->decoder->decodeTexture(src, srcSize, texture, target, width, height);
	}
	catch (Exception & e)
	{
		instance->error = e.getErrorString();
		return 0;
	}
}

#endif

#endif

UNITY_INTERFACE_EXPORT void UNITY_INTERFACE_API NvPipe_Destroy(uint32_t pipe)
{
	DeletePipe(pipe);
}

UNITY_INTERFACE_EXPORT const char* UNITY_INTERFACE_API NvPipe_GetError(uint32_t pipe)
{
	auto p = GetPipe(pipe);
	if (p == nullptr)
		return sharedError.c_str();

	return p->error.c_str();
}

UNITY_INTERFACE_EXPORT void UNITY_INTERFACE_API NvPipe_ClearError(uint32_t pipe)
{
	auto p = GetPipe(pipe);
	if (p == nullptr) {
		sharedError = "";
		return;
	}

	p->error = "";
}