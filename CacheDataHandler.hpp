//
// Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <HalInterfaces.h>

#include <vector>
#include <unordered_map>

#include <NeuralNetworks.h>

namespace armnn_driver
{

using HidlToken = android::hardware::hidl_array<uint8_t, ANEURALNETWORKS_BYTE_SIZE_OF_CACHE_TOKEN>;

class CacheHandle
{
public:
    CacheHandle(const HidlToken token, const size_t cacheSize)
    : m_HidlToken(token), m_CacheSize(cacheSize) {}

    ~CacheHandle() {};

    HidlToken GetToken() const
    {
        return m_HidlToken;
    }

    size_t GetCacheSize() const
    {
        return m_CacheSize;
    }

private:
    const HidlToken m_HidlToken;
    const size_t m_CacheSize;
};

class CacheDataHandler
{
public:
    CacheDataHandler() {}
    ~CacheDataHandler() {}

    void Register(const HidlToken token, const size_t hashValue, const size_t cacheSize);

    bool Validate(const HidlToken token, const size_t hashValue, const size_t cacheSize) const;

    size_t Hash(std::vector<uint8_t>& cacheData);

    size_t GetCacheSize(HidlToken token);

    void Clear();

private:
    CacheDataHandler(const CacheDataHandler&) = delete;
    CacheDataHandler& operator=(const CacheDataHandler&) = delete;

    std::unordered_map<size_t, CacheHandle> m_CacheDataMap;
};

CacheDataHandler& CacheDataHandlerInstance();

} // armnn_driver
