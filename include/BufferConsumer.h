#ifndef BUFFERCONSUMER_H
#define BUFFERCONSUMER_H

#include <shared_mutex>
#include <vector>
#include <gstreamer-1.0/gst/gst.h>
#include <gstreamer-1.0/gst/app/gstappsink.h>
#include "ObjectDetector.h"
#include "Logger.h"

#define MAX_BUFFERS 10

class BufferConsumer
{
public:
    BufferConsumer(ObjectDetector* owner) : _owner(owner) {}
    ~BufferConsumer() 
    {
        GetLastBuffer();
        UnrefLastBuffer();
    }

    bool HasBuffers()
    {
        bool hasBuffers;
        _mutex.lock_shared();
        hasBuffers = _buffers.size() > 0;
        _mutex.unlock_shared();
        return hasBuffers;
    }

    bool AddBuffer(GstBuffer* buffer)
    {
        GstBuffer* toRemove = nullptr;
        bool added = false;

        if(_mutex.try_lock())
        {
            if(_buffers.size() >= MAX_BUFFERS)
            {
                toRemove = _buffers[0];
                gst_buffer_unref(toRemove);
                _buffers.erase(_buffers.begin());
            }

            _buffers.push_back(buffer);
            _mutex.unlock();
            _owner->Notify();
            added = true;
        }
        return added;
    }

    inline void UnrefLastBuffer()
    {
        if(_last != nullptr)
        {
            gst_buffer_unref(_last);
            _last = nullptr;
        }
    }

    GstBuffer* GetLastBuffer()
    {
        GstBuffer* last = nullptr;
        size_t num;
        _mutex.lock();
        num = _buffers.size();
        if(num > 0)
        {
            UnrefLastBuffer();
            last = _buffers[_buffers.size()-1];
            for(GstBuffer* buff : _buffers)
                if(buff != last)
                    gst_buffer_unref(buff);
            _buffers = std::vector<GstBuffer*>();
            _last = last;
        }
        _mutex.unlock();
        return _last;
    }

private:
    ObjectDetector* _owner = nullptr;
    GstBuffer* _last = nullptr;
    std::vector<GstBuffer*> _buffers;
    std::shared_mutex _mutex;
};

#endif