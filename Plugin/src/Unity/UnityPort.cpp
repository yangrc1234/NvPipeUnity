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
#include <thread>
#include <queue>
#include <cuda.h>
#include <cuda_gl_interop.h>
#include <unordered_map>
#include <exception>
#include <sstream>


struct PipeProxy
{
	PipeProxy(NvPipe* pipe) :pipe(pipe)
	{

	}
	NvPipe* pipe = nullptr;
	~PipeProxy()
	{
		if (pipe != nullptr) {
			NvPipe_Destroy(pipe);
		}
	}
};

std::unordered_map<uint64_t, std::shared_ptr<PipeProxy>> g_pipes;
uint64_t g_pipeCreationIndex = 1;


static std::shared_ptr<PipeProxy> GetPipe(uint64_t id) {
	auto ite = g_pipes.find(id);
	if (ite == g_pipes.end()) {
		return nullptr;
	}
	return ite->second;
}

#define LogToFile(...) \
	::fprintf(stderr, __VA_ARGS__)

extern "C" {

	UNITY_INTERFACE_EXPORT uint64_t UNITY_INTERFACE_API CreateNvPipeEncoder(NvPipe_Format format, NvPipe_Codec codec, NvPipe_Compression compression, uint64_t bitrate, uint32_t targetfps, uint32_t width, uint32_t height) {
		auto t = NvPipe_CreateEncoder(format, codec, compression, bitrate, targetfps, width, height);
		if (t != nullptr) {
			auto ptr = std::make_shared<PipeProxy>(t);
			g_pipes[g_pipeCreationIndex] = ptr;
			LogToFile("Encoder %p created\n", t);
			return g_pipeCreationIndex++;
		}
		return 0;
	}

	UNITY_INTERFACE_EXPORT uint64_t UNITY_INTERFACE_API CreateNvPipeTextureAsyncEncoder(NvPipe_Format format, NvPipe_Codec codec, NvPipe_Compression compression, uint64_t bitrate, uint32_t targetfps, uint32_t width, uint32_t height) {
		auto t = NvPipe_CreateTextureAsyncEncoder(format, codec, compression, bitrate, targetfps, width, height);
		if (t != nullptr) {
			auto ptr = std::make_shared<PipeProxy>(t);
			g_pipes[g_pipeCreationIndex] = ptr;
			LogToFile("Async texture encoder %p created\n", t);
			return g_pipeCreationIndex++;
		}
		return 0;
	}

	UNITY_INTERFACE_EXPORT uint64_t UNITY_INTERFACE_API CreateNvPipeDecoder(NvPipe_Format format, NvPipe_Codec codec, uint32_t width, uint32_t height) {
		auto t = NvPipe_CreateDecoder(format, codec, width, height);
		if (t != nullptr) {
			auto ptr = std::make_shared<PipeProxy>(t);
			g_pipes[g_pipeCreationIndex] = ptr;
			LogToFile("Decoder created\n");
			return g_pipeCreationIndex++;
		}
		return 0;
	}

	UNITY_INTERFACE_EXPORT void UNITY_INTERFACE_API DestroyNvPipe(uint64_t pipe) {
		g_pipes.erase(pipe);
	}

	UNITY_INTERFACE_EXPORT uint64_t UNITY_INTERFACE_API Encode(uint64_t encoderPipe, const uint8_t* src, uint64_t srcPitch, uint8_t* dst, uint64_t dstSize, uint32_t width, uint32_t height, bool forcelFrame) {
		auto pipe = GetPipe(encoderPipe);
		if (pipe != nullptr)
			return NvPipe_Encode(pipe->pipe, src, srcPitch, dst, dstSize, width, height, forcelFrame);
		return 0;	//This won't throw, just make sure C# code doesn't go wrong.
	}

	UNITY_INTERFACE_EXPORT uint64_t UNITY_INTERFACE_API Decode(uint64_t decodePipe, const uint8_t* src, uint64_t srcSize, uint8_t* dst, uint32_t width, uint32_t height) {
		auto pipe = GetPipe(decodePipe);
		if (pipe != nullptr)
			return NvPipe_Decode(pipe->pipe, src, srcSize, dst, width, height);
		return 0;
	}

	UNITY_INTERFACE_EXPORT const char* UNITY_INTERFACE_API GetError(uint64_t pipe) {
		if (pipe == 0)
			return NvPipe_GetError(nullptr);
		auto p = GetPipe(pipe);
		if (p != nullptr) {
			return NvPipe_GetError(p->pipe);
		}
		return "";
	}

	UNITY_INTERFACE_EXPORT void UNITY_INTERFACE_API ClearError(uint64_t pipe) {
		auto p = GetPipe(pipe);
		if (p != nullptr) {
			NvPipe_ClearError(p->pipe);
		}
	}
}
// Async tasks.
// Async task is only for texture encoding/decoding.
// Since encode/decode for host memory asyncly could be done on C# side.

struct MainThreadPendingTask {
	MainThreadPendingTask() = default;
	MainThreadPendingTask(std::shared_ptr<PipeProxy> nvp, uint32_t texture, uint32_t width, uint32_t height, bool forceIFrame)	//Encode task
	{
		this->pipe = nvp;
		this->texture = texture;
		this->width = width;
		this->height = height;
		this->forceIFrame = forceIFrame;
	}
	std::shared_ptr<PipeProxy> pipe;
	uint32_t texture;
	uint32_t width;
	uint32_t height;
	bool forceIFrame;
};

enum class TaskStatus
{
	Pending = 0,		//Not submitted to encode thread, or, encode thread is working on it.
	Success,
	Error
};

struct SubmittedTask
{
	SubmittedTask()
	{
		isDone = false;
		isError = false;
	}
	std::shared_ptr<PipeProxy> pipe;
	int taskIndex;
	std::unique_ptr<uint8_t[]> resultBuffer;
	uint32_t resultBufferSize;

	//Following are valid if encode work is finished.
	bool isDone;
	bool isError;
	std::string error;
	int encodedSize;
};

static std::map <int, MainThreadPendingTask> mainThreadPendingTasks;	//Called from Unity MainThread.
static std::map<int, SubmittedTask> submittedTasks;				//Already submitted to encode/decode thread	
static std::mutex taskMutex;
static int renderThreadPendingTaskIndex = 1;
void NvPipe_EncodeTextureAsyncQuery(NvPipe* nvp, uint64_t taskIndex, bool* isDone, bool* isError, uint64_t* encodeSize, std::string* error);

extern "C" {
	UNITY_INTERFACE_EXPORT int UNITY_INTERFACE_API EncodeOpenGLTextureAsync(
		uint64_t nvp, uint32_t texture, uint32_t width, uint32_t height, bool forceIFrame
	) {
		auto p = GetPipe(nvp);
		if (p == nullptr)
			return 0;

		MainThreadPendingTask task(p, texture, width, height, forceIFrame);
		mainThreadPendingTasks[renderThreadPendingTaskIndex] = std::move(task);
		return renderThreadPendingTaskIndex++;
	}

	UNITY_INTERFACE_EXPORT void UNITY_INTERFACE_API UpdateAsyncTasks(int ) {
		std::lock_guard<std::mutex> lock(taskMutex);
		LogToFile("Updating async tasks\n");
		for (auto& ite : submittedTasks) {

			if (ite.second.isDone)	//Already solved
			{
				continue;
			}

			bool isDone;
			bool isError;
			uint64_t encodeSize;
			std::string error;
			NvPipe_EncodeTextureAsyncQuery(ite.second.pipe->pipe, ite.second.taskIndex, &isDone, &isError, &encodeSize, &error);
			if (isDone) {
				LogToFile("Task done detected\n");
				ite.second.isDone = true;
				ite.second.isError = isError;
				if (isError) {
					LogToFile("Task error: %s\n", error.c_str());
					ite.second.error = error;
				}
				else {
					LogToFile("Task successded, encoded size %lld: \n", encodeSize);
					ite.second.encodedSize = encodeSize;
				}
				NvPipe_EncodeTextureAsyncClearTask(ite.second.pipe->pipe, ite.second.taskIndex);
			}
		}
	}

	UNITY_INTERFACE_EXPORT void UNITY_INTERFACE_API QueryAsyncResult(int task_id, TaskStatus* status, uint8_t** data, uint32_t* resultBufferSize, uint32_t* outputSize, const char **error) {

		std::lock_guard<std::mutex> lock(taskMutex);
		//Check task_id is executing or not.
		auto ite = submittedTasks.find(task_id);
		if (ite == submittedTasks.end()) {
			*status = TaskStatus::Pending;
			return;
		}

		if (ite->second.isDone)	//Already solved
		{
			if (ite->second.isError) {
				*status = TaskStatus::Error;
				*error = ite->second.error.c_str();
			}
			else {
				*status = TaskStatus::Success;
				*resultBufferSize = ite->second.resultBufferSize;
				*outputSize = ite->second.encodedSize;
				*data = ite->second.resultBuffer.get();
			}

		}
		else {
			*status = TaskStatus::Pending;
		}
	}

	UNITY_INTERFACE_EXPORT void ClearAsyncTask(int task_id) {
		std::lock_guard<std::mutex> lock(taskMutex);
		submittedTasks.erase(task_id);
	}

	UNITY_INTERFACE_EXPORT void UNITY_INTERFACE_API KickstartRequestInRenderThread(int event_id) {
		// Get task back
		MainThreadPendingTask task;
		{
			auto ite = mainThreadPendingTasks.find(event_id);
			if (ite == mainThreadPendingTasks.end())
				return;
			task = std::move(mainThreadPendingTasks[event_id]);
			mainThreadPendingTasks.erase(event_id);
		}

		SubmittedTask tr;
		LogToFile("Submit async encode task to encode thread.\n");
		//Execute task.
		auto resultBufferSize = task.width * task.height * 4;
		tr.resultBuffer = std::make_unique<uint8_t[]>(resultBufferSize);
		tr.taskIndex = NvPipe_EncodeTextureAsync(task.pipe->pipe, task.texture, GL_TEXTURE_2D, tr.resultBuffer.get(), resultBufferSize, task.width, task.height, task.forceIFrame);
		if (tr.taskIndex == 0) {
			//Too busy.
			tr.isDone = true;
			tr.isError = true;
			tr.error = "Encoder is too busy.";
		}
		tr.pipe = task.pipe;
		tr.resultBufferSize = resultBufferSize; 
		{
			std::lock_guard<std::mutex> lock(taskMutex);
			submittedTasks[event_id] = std::move(tr);
		}
	}

	UNITY_INTERFACE_EXPORT UnityRenderingEvent UNITY_INTERFACE_API GetKickstartFuncPtr() {
		return KickstartRequestInRenderThread;
	}

	UNITY_INTERFACE_EXPORT UnityRenderingEvent UNITY_INTERFACE_API GetUpdateFuncPtr() {
		return UpdateAsyncTasks;
	}
}
