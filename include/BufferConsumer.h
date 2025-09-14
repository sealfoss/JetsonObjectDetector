#ifndef BUFFERCONSUMER_H
#define BUFFERCONSUMER_H

#include <mutex>
#include <vector>
#include <gstreamer-1.0/gst/app/gstappsink.h>

#define MAX_BUFFERS 10

class ObjectDetector;

class BufferConsumer
{
public:
    BufferConsumer(ObjectDetector* owner);

    ~BufferConsumer();

    bool WasBufferUpdated();

    bool AddBuffer(GstBuffer* buffer);

    GstBuffer* GetLastBuffer();

    void UnrefLastBuffer();


private:
    ObjectDetector* _owner = nullptr;
    GstBuffer** _current = nullptr;
    GstBuffer** _last = nullptr;
    GstBuffer* _buffA = nullptr;
    GstBuffer* _buffB = nullptr;
    bool _buffUpdated = false;
    std::mutex _mutex;
};

#endif