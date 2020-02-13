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
	
	struct AsyncOpenGLTextureTask {
		AsyncOpenGLTextureTask() = default;
		AsyncOpenGLTextureTask(NvPipe* nvp, uint32_t texture, uint32_t width, uint32_t height, bool forceIFrame)	//Encode task
		{
			isEncodeTask = true;
			this->nvp = nvp;
			this->texture = texture;
			this->width = width;
			this->height = height;
			this->forceIFrame = forceIFrame;
		}
		AsyncOpenGLTextureTask(NvPipe* nvp, std::unique_ptr<uint8_t[]> src, uint32_t srcSize, uint32_t texture, uint32_t width, uint32_t height)	//Decode task
		{
			isEncodeTask = false;
			this->nvp = nvp;
			this->src = std::move(src);
			this->srcSize = srcSize;
			this->texture = texture;
			this->width = width;
			this->height = height;
		}
		std::unique_ptr<uint8_t[]> src;
		uint32_t srcSize;
		bool isEncodeTask;
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
		std::unique_ptr<uint8_t[]> resultBuffer;	
		uint32_t resultSize;
		uint32_t resultBufferSize;
	};

	enum class TaskStatus
	{
		Pending = 0,
		Success,
		Error
	};

	static std::set<NvPipe*> g_alivePipes;	
	//We support async encoding OpenGL Texture.
	//But it might happen that, Unity Render Thread calls Encode, after Main Thread destroys the NvPipe.
	//Leading to a dangling pointer.
	//So we record created encoders here, look it up before execute async encoding.

	static std::map<int, AsyncOpenGLTextureTask> renderThreadTasks;
	static std::map<int, TaskResult> finishedtasks;
	static std::mutex asyncMutex;
	static int taskIndex = 0;


	UNITY_INTERFACE_EXPORT NvPipe* UNITY_INTERFACE_API CreateNvPipeEncoder(NvPipe_Format format, NvPipe_Codec codec, NvPipe_Compression compression, uint64_t bitrate, uint32_t targetfps, uint32_t width, uint32_t height) {
		auto t = NvPipe_CreateEncoder(format, codec, compression, bitrate, targetfps, width, height);
		if (t != nullptr) {
			std::lock_guard<std::mutex> lock(asyncMutex);
			g_alivePipes.insert(t);
			LogToFile("Encoder %p created\n", t);
		}
		return t;
	}

	UNITY_INTERFACE_EXPORT NvPipe* UNITY_INTERFACE_API CreateNvPipeDecoder(NvPipe_Format format, NvPipe_Codec codec, uint32_t width, uint32_t height) {
		auto t = NvPipe_CreateDecoder(format, codec, width, height);
		if (t != nullptr) {
			std::lock_guard<std::mutex> lock(asyncMutex);
			g_alivePipes.insert(t);
			LogToFile("Decoder created\n");
		}
		return t;
	}

	UNITY_INTERFACE_EXPORT void UNITY_INTERFACE_API DestroyNvPipe(NvPipe* pipe) {
		std::lock_guard<std::mutex> lock(asyncMutex);
		LogToFile("Destroying pipe, %p\n", pipe);
		auto ite = g_alivePipes.find(pipe);
		if (ite != g_alivePipes.end()) {
			g_alivePipes.erase(ite);
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
		NvPipe* nvp, uint32_t texture, uint32_t width, uint32_t height, bool forceIFrame
	) {
		AsyncOpenGLTextureTask task(nvp, texture, width, height, forceIFrame);

		std::lock_guard<std::mutex> lock(asyncMutex);

		renderThreadTasks[taskIndex] = std::move(task);
		LogToFile("Current pending task count: %d", renderThreadTasks.size());
		LogToFile("Current finished task count: %d", finishedtasks.size());
		return taskIndex++;
	}

	UNITY_INTERFACE_EXPORT uint64_t UNITY_INTERFACE_API DecodeOpenGLTextureAsync(
		NvPipe* nvp, uint8_t *src, uint32_t srcSize, uint32_t texture, uint32_t width, uint32_t height
	) {
		//Copy src data to avoid src get destructed during execution.
		auto srcData = std::make_unique<uint8_t[]>(srcSize);
		memcpy(srcData.get(), src, srcSize);

		AsyncOpenGLTextureTask task(nvp, std::move(srcData), srcSize, texture, width, height);

		std::lock_guard<std::mutex> lock(asyncMutex);

		renderThreadTasks[taskIndex] = std::move(task);
		return taskIndex++;
	}

	UNITY_INTERFACE_EXPORT void UNITY_INTERFACE_API QueryAsyncResult(int task_id, bool acquireResultDataPtr/*Acquire the result data*/, TaskStatus* status, uint8_t** data, uint32_t* resultBufferSize, uint32_t* outputSize) {
		std::lock_guard<std::mutex> lock(asyncMutex);
		auto ite = finishedtasks.find(task_id);
		if (ite == finishedtasks.end()) {
			*status = TaskStatus::Pending;
			return;
		}
		if (ite->second.success) {
			*data = nullptr;
			if (ite->second.resultBuffer != nullptr) {	//async decode doesn't have a result array.
				*resultBufferSize = ite->second.resultBufferSize;
				if (acquireResultDataPtr) {
					*data = ite->second.resultBuffer.release();
				}
				else
				{	
					*data = ite->second.resultBuffer.get();
				}
			}
			*outputSize = ite->second.resultSize;
			*status = TaskStatus::Success;
		}
		else {
			*status = TaskStatus::Error;
		}
	}

	UNITY_INTERFACE_EXPORT const char* UNITY_INTERFACE_API QueryAsyncError(int task_id) {
		std::lock_guard<std::mutex> lock(asyncMutex);
		auto ite = finishedtasks.find(task_id);
		if (ite == finishedtasks.end())
			return nullptr;
		return ite->second.error.c_str();
	}

	UNITY_INTERFACE_EXPORT void ClearAsyncTask(int task_id) {
		std::lock_guard<std::mutex> lock(asyncMutex);
		finishedtasks.erase(task_id);
		renderThreadTasks.erase(task_id);
	}

	UNITY_INTERFACE_EXPORT void UNITY_INTERFACE_API KickstartRequestInRenderThread(int event_id) {
		// Get task back
		std::lock_guard<std::mutex> lock(asyncMutex);
		AsyncOpenGLTextureTask task;
		{
			auto ite = renderThreadTasks.find(event_id);
			if (ite == renderThreadTasks.end())
				return;
			task = std::move(renderThreadTasks[event_id]);
			renderThreadTasks.erase(ite);
		}

		TaskResult tr;
		if (g_alivePipes.find(task.nvp) == g_alivePipes.end()) {
			LogToFile("Executing task on disposed NvPipe\n");
			tr.error = "Pipe disposed.";
			tr.success = false;
		} else {
			if (task.isEncodeTask) {
				LogToFile("Starting encoding on OpenGL Texture.\n");
				//Execute task.
				auto resultBufferSize = task.width * task.height * 4;
				tr.resultBuffer = std::make_unique<uint8_t[]>(resultBufferSize);
				auto size = NvPipe_EncodeTexture(task.nvp, task.texture, GL_TEXTURE_2D, tr.resultBuffer.get(), resultBufferSize, task.width, task.height, task.forceIFrame);
				if (strcmp(NvPipe_GetError(task.nvp), "")) {
					tr.error = std::string(NvPipe_GetError(task.nvp));
					tr.success = false;
					NvPipe_ClearError(task.nvp);
					LogToFile("Encoding encountered error:\n");
					LogToFile(NvPipe_GetError(task.nvp));
					LogToFile("\n");
				}
				else {
					tr.resultSize = size;
					tr.resultBufferSize = resultBufferSize;
					tr.error = "";
					tr.success = true;
					LogToFile("Finished encoding without error.\n");
				}
			}
			else {
				LogToFile("Starting decode to OpenGL Texture.\n");
				NvPipe_DecodeTexture(task.nvp, task.src.get(), task.srcSize, task.texture, GL_TEXTURE_2D, task.width, task.height);
				if (strcmp(NvPipe_GetError(task.nvp), "")) {
					tr.error = std::string(NvPipe_GetError(task.nvp));
					tr.success = false;
					NvPipe_ClearError(task.nvp);
					LogToFile("Decoding encountered error:\n");
					LogToFile(NvPipe_GetError(task.nvp));
					LogToFile("\n");
				}
				else {
					tr.error = "";
					tr.success = true;
					LogToFile("Finished decoding without error.\n");
				}
			}
		}
		finishedtasks[event_id] = std::move(tr);
	}

	UNITY_INTERFACE_EXPORT UnityRenderingEvent UNITY_INTERFACE_API GetKickstartFuncPtr() {
		return KickstartRequestInRenderThread;
	}
}
