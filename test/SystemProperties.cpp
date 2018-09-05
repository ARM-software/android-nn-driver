//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include "DriverTestHelpers.hpp"
#include <boost/test/unit_test.hpp>
#include <log/log.h>
#include "../SystemPropertiesUtils.hpp"

BOOST_AUTO_TEST_SUITE(SystemProperiesTests)

BOOST_AUTO_TEST_CASE(SystemProperties)
{
    // Test default value
    {
        auto p = __system_property_find("thisDoesNotExist");
        BOOST_TEST((p == nullptr));

        int defaultValue = ParseSystemProperty("thisDoesNotExist", -4);
        BOOST_TEST((defaultValue == -4));
    }

    //  Test default value from bad data type
    {
        __system_property_set("thisIsNotFloat", "notfloat");
        float defaultValue = ParseSystemProperty("thisIsNotFloat", 0.1f);
        BOOST_TEST((defaultValue == 0.1f));
    }

    // Test fetching bool values
    {
        __system_property_set("myTestBool", "1");
        bool b = ParseSystemProperty("myTestBool", false);
        BOOST_TEST((b == true));
    }
    {
        __system_property_set("myTestBool", "0");
        bool b = ParseSystemProperty("myTestBool", true);
        BOOST_TEST((b == false));
    }

    // Test fetching int
    {
        __system_property_set("myTestInt", "567");
        int i = ParseSystemProperty("myTestInt", 890);
        BOOST_TEST((i==567));
    }

    // Test fetching float
    {
        __system_property_set("myTestFloat", "1.2f");
        float f = ParseSystemProperty("myTestFloat", 3.4f);
        BOOST_TEST((f==1.2f));
    }
}

BOOST_AUTO_TEST_SUITE_END()
