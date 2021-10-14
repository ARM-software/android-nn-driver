//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "DriverTestHelpers.hpp"
#include <log/log.h>
#include <SystemPropertiesUtils.hpp>

DOCTEST_TEST_SUITE("SystemProperiesTests")
{

DOCTEST_TEST_CASE("SystemProperties")
{
    // Test default value
    {
        auto p = __system_property_find("thisDoesNotExist");
        DOCTEST_CHECK((p == nullptr));

        int defaultValue = ParseSystemProperty("thisDoesNotExist", -4);
        DOCTEST_CHECK((defaultValue == -4));
    }

    //  Test default value from bad data type
    {
        __system_property_set("thisIsNotFloat", "notfloat");
        float defaultValue = ParseSystemProperty("thisIsNotFloat", 0.1f);
        DOCTEST_CHECK((defaultValue == 0.1f));
    }

    // Test fetching bool values
    {
        __system_property_set("myTestBool", "1");
        bool b = ParseSystemProperty("myTestBool", false);
        DOCTEST_CHECK((b == true));
    }
    {
        __system_property_set("myTestBool", "0");
        bool b = ParseSystemProperty("myTestBool", true);
        DOCTEST_CHECK((b == false));
    }

    // Test fetching int
    {
        __system_property_set("myTestInt", "567");
        int i = ParseSystemProperty("myTestInt", 890);
        DOCTEST_CHECK((i==567));
    }

    // Test fetching float
    {
        __system_property_set("myTestFloat", "1.2f");
        float f = ParseSystemProperty("myTestFloat", 3.4f);
        DOCTEST_CHECK((f==1.2f));
    }
}

}
