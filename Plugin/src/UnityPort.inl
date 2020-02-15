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
		return currentTaskIndex;    //Return task number.
	}

	void encodeThread() {
		while (!m_closed)
		{
			if (m_encodedPtr == m_pendingTaskPtr)
			{
				std::this_thread::yield();
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
std::shared_mutex g_pipeDictMutex;
uint32_t g_pipeCreationIndex = 1;	//Start with 1, since GetError requries a special value(0 here) for global error.

static void DeletePipe(uint32_t id) {
	std::unique_lock<std::shared_mutex> lock;
	g_pipes.erase(id);
}

static std::shared_ptr<Instance> GetPipe(uint32_t id) {
	std::shared_lock<std::shared_mutex> lock;
	auto ite = g_pipes.find(id);
	if (ite == g_pipes.end()) {
		return nullptr;
	}
	return ite->second;
}

static uint32_t InsertNewPipe(std::shared_ptr<Instance> instance) {
	std::unique_lock<std::shared_mutex> lock;
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
		sprintf_s(msgBuffer, "Maximum task count reached. Did you forget to clear task, or submitted too many tasks(%d) at once?", MAX_PENDING_TASK_COUNT);
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
		uint64_t resultBufferSize = task.width * task.height * 4;

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