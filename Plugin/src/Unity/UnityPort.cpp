#include "IUnityInterface.h"
#include <NvPipe.h>
#ifdef WIN32
#include <Windows.h>
#endif
#include <gl\GL.h>
#include <map>
#include <mutex>
#include "IUnityGraphics.h"
#include <set>
#include <fstream>

extern "C" {
#define LogToFile(...) \
	::fprintf(stderr, __VA_ARGS__)
	
	struct EncodeOpenGLTextureTask {
		NvPipe* nvp;
		uint32_t texture;
		uint32_t width;
		uint32_t height;
		bool forceIFrame;
	};

	struct TaskResult
	{
		bool success = false;
		std::string error;
		std::unique_ptr<uint8_t[]> data;	
		uint32_t encodedSize;
		uint32_t bufferSize;
	};

	enum class TaskStatus
	{
		Pending = 0,
		Success,
		Error
	};

	static std::set<NvPipe*> g_aliveEncoders;	
	//We support async encoding OpenGL Texture.
	//But it might happen that, Unity Render Thread calls Encode, after Main Thread destroys the NvPipe.
	//Leading to a dangling pointer.
	//So we record created encoders here, look it up before execute async encoding.

	static std::map<int, EncodeOpenGLTextureTask> renderThreadTasks;
	static std::map<int, TaskResult> finishedtasks;//True for success, False for error.
	static std::mutex taskMutex, resultMutex, aliveMutex;
	static int taskIndex = 0;


	UNITY_INTERFACE_EXPORT NvPipe* UNITY_INTERFACE_API CreateNvPipeEncoder(NvPipe_Format format, NvPipe_Codec codec, NvPipe_Compression compression, uint64_t bitrate, uint32_t targetfps, uint32_t width, uint32_t height) {
		auto t = NvPipe_CreateEncoder(format, codec, compression, bitrate, targetfps, width, height);
		if (t != nullptr) {
			std::lock_guard<std::mutex> lock(aliveMutex);
			g_aliveEncoders.insert(t);
			LogToFile("Encoder %p created\n", t);
		}
		return t;
	}

	UNITY_INTERFACE_EXPORT NvPipe* UNITY_INTERFACE_API CreateNvPipeDecoder(NvPipe_Format format, NvPipe_Codec codec, uint32_t width, uint32_t height) {
		auto t = NvPipe_CreateDecoder(format, codec, width, height);
		if (t != nullptr) {
			std::lock_guard<std::mutex> lock(aliveMutex);
			g_aliveEncoders.insert(t);
			LogToFile("Decoder created\n");
		}
		return t;
	}

	UNITY_INTERFACE_EXPORT void UNITY_INTERFACE_API DestroyNvPipe(NvPipe* pipe) {
		std::lock_guard<std::mutex> lock(aliveMutex);
		LogToFile("Destroying pipe, %p\n", pipe);
		auto ite = g_aliveEncoders.find(pipe);
		if (ite != g_aliveEncoders.end()) {
			g_aliveEncoders.erase(ite);
			NvPipe_Destroy(pipe);
			LogToFile("Pipe destroyed\n");
		}
		else {
			LogToFile("Destroying an already destroyed pipe\n");
		}
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

	UNITY_INTERFACE_EXPORT uint64_t UNITY_INTERFACE_API EncodeOpenGLTextureAsync(
		NvPipe* nvp, uint32_t texture,  uint32_t width, uint32_t height, bool forceIFrame
	) {
		EncodeOpenGLTextureTask task = {
			nvp,
			texture,
			width,
			height,
			forceIFrame,
		};
		
		std::lock_guard<std::mutex> lock(taskMutex);

		renderThreadTasks[taskIndex] = task;
		return taskIndex++;
	}

	UNITY_INTERFACE_EXPORT void UNITY_INTERFACE_API QueryAsyncResult(int task_id, bool detachResultData/*Acquire the result data*/, TaskStatus* status, uint8_t** data, uint32_t* bufferSize, uint32_t* encodedSize) {
		std::lock_guard<std::mutex> lock(resultMutex);
		auto ite = finishedtasks.find(task_id);
		if (ite == finishedtasks.end()) {
			*status = TaskStatus::Pending;
			return;
		}
		if (ite->second.success) {
			if (detachResultData) {
				*data = ite->second.data.release();
			}
			else
			{
				*data = ite->second.data.get();
			}
			*encodedSize = ite->second.encodedSize;
			*bufferSize = ite->second.bufferSize;
			*status = TaskStatus::Success;
		}
		else {
			*status = TaskStatus::Error;
		}
	}

	UNITY_INTERFACE_EXPORT const char* UNITY_INTERFACE_API QueryAsyncError(int task_id) {
		std::lock_guard<std::mutex> lock(resultMutex);
		auto ite = finishedtasks.find(task_id);
		if (ite == finishedtasks.end())
			return nullptr;
		return ite->second.error.c_str();
	}

	UNITY_INTERFACE_EXPORT void ClearAsyncTask(int task_id) {
		std::lock_guard<std::mutex> lock(resultMutex);
		finishedtasks.erase(task_id);
	}

	UNITY_INTERFACE_EXPORT void UNITY_INTERFACE_API KickstartRequestInRenderThread(int event_id) {
		// Get task back
		EncodeOpenGLTextureTask task;
		{
			std::lock_guard<std::mutex> lock(taskMutex);
			task = renderThreadTasks[event_id];
			renderThreadTasks.erase(event_id);
		}

		std::lock_guard<std::mutex> lock(aliveMutex);
		std::lock_guard<std::mutex> lock2(resultMutex);
		TaskResult tr;
		if (g_aliveEncoders.find(task.nvp) == g_aliveEncoders.end()) {
			LogToFile("Executing encoding on disposed encoder\n");
			tr.error = "Encoder disposed.";
			tr.success = false;
		} else {
			LogToFile("Starting encoding on OpenGL Texture.\n");
			//Execute task.
			auto tempMemSize = task.width * task.height * 4;
			tr.data = std::make_unique<uint8_t[]>(tempMemSize);
			int size = NvPipe_EncodeTexture(task.nvp, task.texture, GL_TEXTURE_2D, tr.data.get(), tempMemSize, task.width, task.height, task.forceIFrame);
			if (strcmp(NvPipe_GetError(task.nvp), "")) {
				tr.error = std::string(NvPipe_GetError(task.nvp));
				tr.success = false;
				NvPipe_ClearError(task.nvp);
				LogToFile("Encoding encountered error:\n");
				LogToFile(NvPipe_GetError(task.nvp));
				LogToFile("\n");
			}
			else {
				tr.encodedSize = size;
				tr.bufferSize = tempMemSize;
				tr.error = "";
				tr.success = true;
				LogToFile("Finished encoding without error.\n");
			}
		}
		finishedtasks[event_id] = std::move(tr);
	}

	UNITY_INTERFACE_EXPORT UnityRenderingEvent UNITY_INTERFACE_API GetKickstartFuncPtr() {
		return KickstartRequestInRenderThread;
	}
}
