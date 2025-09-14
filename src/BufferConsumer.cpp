#include "BufferConsumer.h"
#include "ObjectDetector.h"
#include "Logger.h"
#include <gstreamer-1.0/gst/gst.h>


BufferConsumer::BufferConsumer(ObjectDetector* owner) : _owner(owner) 
{
    _current = &_buffA;
    _last = &_buffB;
}

BufferConsumer::~BufferConsumer() 
{
    if(_buffA != nullptr)
    {
        gst_buffer_unref(_buffA);
        _buffA = nullptr;
    }

    if(_buffB != nullptr)
    {
        gst_buffer_unref(_buffB);
        _buffB = nullptr;
    }
}

bool BufferConsumer::WasBufferUpdated()
{
    bool updated;
    _mutex.lock();
    updated = _buffUpdated;
    _mutex.unlock();
    return updated;
}

bool BufferConsumer::AddBuffer(GstBuffer* buffer)
{
    GstBuffer* current = nullptr;
    bool added = false;

    if(_mutex.try_lock())
    {
        current = *_current;
        if(current)
            gst_buffer_unref(current);
        
        *_current = buffer;
        _buffUpdated = true;
        _mutex.unlock();
        _owner->Notify();
        added = true;
    }
    return added;
}

GstBuffer* BufferConsumer::GetLastBuffer()
{
    GstBuffer* buff;

    _mutex.lock();
    buff = *_current;
    if(_current == &_buffA)
    {
        _current = &_buffB;
        _last = &_buffA;
    }
    else
    {
        _current = &_buffA;
        _last = &_buffB;
    }
    _buffUpdated = false;
    _mutex.unlock();

    return buff;
}

void BufferConsumer::UnrefLastBuffer()
{
    GstBuffer* last;
    _mutex.lock();
    last = *_last;
    if(last)
    {
        gst_buffer_unref(last);
        *_last = nullptr;
    }
    _mutex.unlock();
}
