//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include "DriverTestHelpers.hpp"
#include <log/log.h>
#include "../SystemPropertiesUtils.hpp"

#include <doctest/doctest.h>

TEST_SUITE("SystemProperiesTests")
{
TEST_CASE("SystemProperties")
{
    // Test default value
    {
        auto p = __system_property_find("thisDoesNotExist");
        CHECK((p == nullptr));

        int defaultValue = ParseSystemProperty("thisDoesNotExist", -4);
        CHECK((defaultValue == -4));
    }

    //  Test default value from bad data type
    {
        __system_property_set("thisIsNotFloat", "notfloat");
        float defaultValue = ParseSystemProperty("thisIsNotFloat", 0.1f);
        CHECK((defaultValue == 0.1f));
    }

    // Test fetching bool values
    {
        __system_property_set("myTestBool", "1");
        bool b = ParseSystemProperty("myTestBool", false);
        CHECK((b == true));
    }
    {
        __system_property_set("myTestBool", "0");
        bool b = ParseSystemProperty("myTestBool", true);
        CHECK((b == false));
    }

    // Test fetching int
    {
        __system_property_set("myTestInt", "567");
        int i = ParseSystemProperty("myTestInt", 890);
        CHECK((i==567));
    }

    // Test fetching float
    {
        __system_property_set("myTestFloat", "1.2f");
        float f = ParseSystemProperty("myTestFloat", 3.4f);
        CHECK((f==1.2f));
    }
}

}
