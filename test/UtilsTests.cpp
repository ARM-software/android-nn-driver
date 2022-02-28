//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "DriverTestHelpers.hpp"
#include <log/log.h>

#include <armnn/src/armnn/OptimizedNetworkImpl.hpp>

#include <fstream>
#include <memory>
#include <armnn/INetwork.hpp>

#include <armnnUtils/Filesystem.hpp>

using namespace android;
using namespace android::nn;
using namespace android::hardware;
using namespace armnn_driver;

namespace armnn
{

class Graph
{
public:
    Graph(Graph&& graph) = default;
};

class MockOptimizedNetworkImpl final : public ::armnn::OptimizedNetworkImpl
{
public:
    MockOptimizedNetworkImpl(const std::string& mockSerializedContent, std::unique_ptr<armnn::Graph>)
        : ::armnn::OptimizedNetworkImpl(nullptr)
        , m_MockSerializedContent(mockSerializedContent)
    {}
    ~MockOptimizedNetworkImpl() {}

    ::armnn::Status PrintGraph() override { return ::armnn::Status::Failure; }
    ::armnn::Status SerializeToDot(std::ostream& stream) const override
    {
        stream << m_MockSerializedContent;

        return stream.good() ? ::armnn::Status::Success : ::armnn::Status::Failure;
    }

    ::arm::pipe::ProfilingGuid GetGuid() const final { return ::arm::pipe::ProfilingGuid(0); }

    void UpdateMockSerializedContent(const std::string& mockSerializedContent)
    {
        this->m_MockSerializedContent = mockSerializedContent;
    }

private:
    std::string m_MockSerializedContent;
};


} // armnn namespace


// The following are helpers for writing unit tests for the driver.
namespace
{

struct ExportNetworkGraphFixture
{
public:
    // Setup: set the output dump directory and an empty dummy model (as only its memory address is used).
    // Defaulting the output dump directory to "/data" because it should exist and be writable in all deployments.
    ExportNetworkGraphFixture()
        : ExportNetworkGraphFixture("/data")
    {}

    ExportNetworkGraphFixture(const std::string& requestInputsAndOutputsDumpDir)
        : m_RequestInputsAndOutputsDumpDir(requestInputsAndOutputsDumpDir), m_FileName(), m_FileStream()
    {
        // Set the name of the output .dot file.
        // NOTE: the export now uses a time stamp to name the file so we
        //       can't predict ahead of time what the file name will be.
        std::string timestamp = "dummy";
        m_FileName = m_RequestInputsAndOutputsDumpDir / (timestamp + "_networkgraph.dot");
    }

    // Teardown: delete the dump file regardless of the outcome of the tests.
    ~ExportNetworkGraphFixture()
    {
        // Close the file stream.
        m_FileStream.close();

        // Ignore any error (such as file not found).
        (void) remove(m_FileName.c_str());
    }

    bool FileExists()
    {
        // Close any file opened in a previous session.
        if (m_FileStream.is_open())
        {
            m_FileStream.close();
        }

        if (m_FileName.empty())
        {
            return false;
        }

        // Open the file.
        m_FileStream.open(m_FileName, std::ifstream::in);

        // Check that the file is open.
        if (!m_FileStream.is_open())
        {
            return false;
        }

        // Check that the stream is readable.
        return m_FileStream.good();
    }

    std::string GetFileContent()
    {
        // Check that the stream is readable.
        if (!m_FileStream.good())
        {
            return "";
        }

        // Get all the contents of the file.
        return std::string((std::istreambuf_iterator<char>(m_FileStream)),
                           (std::istreambuf_iterator<char>()));
    }

    fs::path m_RequestInputsAndOutputsDumpDir;
    fs::path m_FileName;

private:
    std::ifstream m_FileStream;
};


} // namespace

DOCTEST_TEST_SUITE("UtilsTests")
{

DOCTEST_TEST_CASE("ExportToEmptyDirectory")
{
    // Set the fixture for this test.
    ExportNetworkGraphFixture fixture("");

    // Set a mock content for the optimized network.
    std::string mockSerializedContent = "This is a mock serialized content.";

    // Set a mock optimized network.
    std::unique_ptr<armnn::Graph> graphPtr;

    std::unique_ptr<::armnn::OptimizedNetworkImpl> mockImpl(
        new armnn::MockOptimizedNetworkImpl(mockSerializedContent, std::move(graphPtr)));
    ::armnn::IOptimizedNetwork mockOptimizedNetwork(std::move(mockImpl));

    // Export the mock optimized network.
    fixture.m_FileName = armnn_driver::ExportNetworkGraphToDotFile(mockOptimizedNetwork,
                                                                   fixture.m_RequestInputsAndOutputsDumpDir);

    // Check that the output file does not exist.
    DOCTEST_CHECK(!fixture.FileExists());
}

DOCTEST_TEST_CASE("ExportNetwork")
{
    // Set the fixture for this test.
    ExportNetworkGraphFixture fixture;

    // Set a mock content for the optimized network.
    std::string mockSerializedContent = "This is a mock serialized content.";

    // Set a mock optimized network.
    std::unique_ptr<armnn::Graph> graphPtr;

    std::unique_ptr<::armnn::OptimizedNetworkImpl> mockImpl(
        new armnn::MockOptimizedNetworkImpl(mockSerializedContent, std::move(graphPtr)));
    ::armnn::IOptimizedNetwork mockOptimizedNetwork(std::move(mockImpl));


    // Export the mock optimized network.
    fixture.m_FileName = armnn_driver::ExportNetworkGraphToDotFile(mockOptimizedNetwork,
                                                                   fixture.m_RequestInputsAndOutputsDumpDir);

    // Check that the output file exists and that it has the correct name.
    DOCTEST_CHECK(fixture.FileExists());

    // Check that the content of the output file matches the mock content.
    DOCTEST_CHECK(fixture.GetFileContent() == mockSerializedContent);
}

DOCTEST_TEST_CASE("ExportNetworkOverwriteFile")
{
    // Set the fixture for this test.
    ExportNetworkGraphFixture fixture;

    // Set a mock content for the optimized network.
    std::string mockSerializedContent = "This is a mock serialized content.";

    // Set a mock optimized network.
    std::unique_ptr<armnn::Graph> graphPtr;

    std::unique_ptr<::armnn::OptimizedNetworkImpl> mockImpl(
        new armnn::MockOptimizedNetworkImpl(mockSerializedContent, std::move(graphPtr)));
    ::armnn::IOptimizedNetwork mockOptimizedNetwork(std::move(mockImpl));

    // Export the mock optimized network.
    fixture.m_FileName = armnn_driver::ExportNetworkGraphToDotFile(mockOptimizedNetwork,
                                                                   fixture.m_RequestInputsAndOutputsDumpDir);

    // Check that the output file exists and that it has the correct name.
    DOCTEST_CHECK(fixture.FileExists());

    // Check that the content of the output file matches the mock content.
    DOCTEST_CHECK(fixture.GetFileContent() == mockSerializedContent);

    // Update the mock serialized content of the network.
    mockSerializedContent = "This is ANOTHER mock serialized content!";
    std::unique_ptr<armnn::Graph> graphPtr2;
    std::unique_ptr<::armnn::OptimizedNetworkImpl> mockImpl2(
        new armnn::MockOptimizedNetworkImpl(mockSerializedContent, std::move(graphPtr2)));
    static_cast<armnn::MockOptimizedNetworkImpl*>(mockImpl2.get())->UpdateMockSerializedContent(mockSerializedContent);
    ::armnn::IOptimizedNetwork mockOptimizedNetwork2(std::move(mockImpl2));

    // Export the mock optimized network.
    fixture.m_FileName = armnn_driver::ExportNetworkGraphToDotFile(mockOptimizedNetwork2,
                                                                   fixture.m_RequestInputsAndOutputsDumpDir);

    // Check that the output file still exists and that it has the correct name.
    DOCTEST_CHECK(fixture.FileExists());

    // Check that the content of the output file matches the mock content.
    DOCTEST_CHECK(fixture.GetFileContent() == mockSerializedContent);
}

DOCTEST_TEST_CASE("ExportMultipleNetworks")
{
    // Set the fixtures for this test.
    ExportNetworkGraphFixture fixture1;
    ExportNetworkGraphFixture fixture2;
    ExportNetworkGraphFixture fixture3;

    // Set a mock content for the optimized network.
    std::string mockSerializedContent = "This is a mock serialized content.";

    // Set a mock optimized network.
    std::unique_ptr<armnn::Graph> graphPtr;

    std::unique_ptr<::armnn::OptimizedNetworkImpl> mockImpl(
        new armnn::MockOptimizedNetworkImpl(mockSerializedContent, std::move(graphPtr)));
    ::armnn::IOptimizedNetwork mockOptimizedNetwork(std::move(mockImpl));

    // Export the mock optimized network.
    fixture1.m_FileName = armnn_driver::ExportNetworkGraphToDotFile(mockOptimizedNetwork,
                                                                    fixture1.m_RequestInputsAndOutputsDumpDir);

    // Check that the output file exists and that it has the correct name.
    DOCTEST_CHECK(fixture1.FileExists());

    // Check that the content of the output file matches the mock content.
    DOCTEST_CHECK(fixture1.GetFileContent() == mockSerializedContent);

    // Export the mock optimized network.
    fixture2.m_FileName = armnn_driver::ExportNetworkGraphToDotFile(mockOptimizedNetwork,
                                                                    fixture2.m_RequestInputsAndOutputsDumpDir);

    // Check that the output file exists and that it has the correct name.
    DOCTEST_CHECK(fixture2.FileExists());

    // Check that the content of the output file matches the mock content.
    DOCTEST_CHECK(fixture2.GetFileContent() == mockSerializedContent);

    // Export the mock optimized network.
    fixture3.m_FileName = armnn_driver::ExportNetworkGraphToDotFile(mockOptimizedNetwork,
                                                                    fixture3.m_RequestInputsAndOutputsDumpDir);
    // Check that the output file exists and that it has the correct name.
    DOCTEST_CHECK(fixture3.FileExists());

    // Check that the content of the output file matches the mock content.
    DOCTEST_CHECK(fixture3.GetFileContent() == mockSerializedContent);
}

}
