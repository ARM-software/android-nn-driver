//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "../Dilation.hpp"

#include "../../1.2/HalPolicy.hpp"

#include <boost/test/data/test_case.hpp>

BOOST_AUTO_TEST_SUITE(DilationTests)

BOOST_AUTO_TEST_CASE(ConvolutionExplicitPaddingNoDilation)
{
    DilationTestOptions options;
    options.m_IsDepthwiseConvolution = false;
    options.m_IsPaddingExplicit      = true;
    options.m_HasDilation            = false;

    DilationTestImpl<hal_1_2::HalPolicy>(options);
}

BOOST_AUTO_TEST_CASE(ConvolutionExplicitPaddingDilation)
{
    DilationTestOptions options;
    options.m_IsDepthwiseConvolution = false;
    options.m_IsPaddingExplicit      = true;
    options.m_HasDilation            = true;

    DilationTestImpl<hal_1_2::HalPolicy>(options);
}

BOOST_AUTO_TEST_CASE(ConvolutionImplicitPaddingNoDilation)
{
    DilationTestOptions options;
    options.m_IsDepthwiseConvolution = false;
    options.m_IsPaddingExplicit      = false;
    options.m_HasDilation            = false;

    DilationTestImpl<hal_1_2::HalPolicy>(options);
}

BOOST_AUTO_TEST_CASE(ConvolutionImplicitPaddingDilation)
{
    DilationTestOptions options;
    options.m_IsDepthwiseConvolution = false;
    options.m_IsPaddingExplicit      = false;
    options.m_HasDilation            = true;

    DilationTestImpl<hal_1_2::HalPolicy>(options);
}

BOOST_AUTO_TEST_CASE(DepthwiseConvolutionExplicitPaddingNoDilation)
{
    DilationTestOptions options;
    options.m_IsDepthwiseConvolution = true;
    options.m_IsPaddingExplicit      = true;
    options.m_HasDilation            = false;

    DilationTestImpl<hal_1_2::HalPolicy>(options);
}

BOOST_AUTO_TEST_CASE(DepthwiseConvolutionExplicitPaddingDilation)
{
    DilationTestOptions options;
    options.m_IsDepthwiseConvolution = true;
    options.m_IsPaddingExplicit      = true;
    options.m_HasDilation            = true;

    DilationTestImpl<hal_1_2::HalPolicy>(options);
}

BOOST_AUTO_TEST_CASE(DepthwiseConvolutionImplicitPaddingNoDilation)
{
    DilationTestOptions options;
    options.m_IsDepthwiseConvolution = true;
    options.m_IsPaddingExplicit      = false;
    options.m_HasDilation            = false;

    DilationTestImpl<hal_1_2::HalPolicy>(options);
}

BOOST_AUTO_TEST_CASE(DepthwiseConvolutionImplicitPaddingDilation)
{
    DilationTestOptions options;
    options.m_IsDepthwiseConvolution = true;
    options.m_IsPaddingExplicit      = false;
    options.m_HasDilation            = true;

    DilationTestImpl<hal_1_2::HalPolicy>(options);
}

BOOST_AUTO_TEST_SUITE_END()