#include "IUnityInterface.h"
#include <NvPipe.h>

extern "C" {
	UNITY_INTERFACE_EXPORT NvPipe* UNITY_INTERFACE_API CreateNvPipeEncoder(NvPipe_Format format, NvPipe_Codec codec, NvPipe_Compression compression, uint64_t bitrate, uint32_t targetfps, uint32_t width, uint32_t height) {
		return NvPipe_CreateEncoder(format, codec, compression, bitrate, targetfps, width, height);
	}

	UNITY_INTERFACE_EXPORT NvPipe* UNITY_INTERFACE_API CreateNvPipeDecoder(NvPipe_Format format, NvPipe_Codec codec, uint32_t width, uint32_t height) {
		return NvPipe_CreateDecoder(format, codec, width, height);
	}

	UNITY_INTERFACE_EXPORT void UNITY_INTERFACE_API DestroyNvPipe(NvPipe* pipe) {
		NvPipe_Destroy(pipe);
	}

	UNITY_INTERFACE_EXPORT uint64_t UNITY_INTERFACE_API Encode(NvPipe* encoderPipe, const uint8_t* src, uint64_t srcPitch, uint8_t* dst, uint64_t dstSize, uint32_t width, uint32_t height, bool forcelFrame) {
		return NvPipe_Encode(encoderPipe, src, srcPitch, dst, dstSize, width, height, forcelFrame);
	}

	UNITY_INTERFACE_EXPORT uint64_t UNITY_INTERFACE_API Decode(NvPipe* decodePipe, const uint8_t* src, uint64_t srcSize, uint8_t* dst, uint32_t width, uint32_t height) {
		return NvPipe_Decode(decodePipe, src, srcSize, dst, width, height);
	}

	UNITY_INTERFACE_EXPORT const char* UNITY_INTERFACE_API GetError(NvPipe* pipe) {
		return NvPipe_GetError(pipe);
	}

	UNITY_INTERFACE_EXPORT void UNITY_INTERFACE_API ClearError(NvPipe* pipe) {
		NvPipe_ClearError(pipe);
	}
}
