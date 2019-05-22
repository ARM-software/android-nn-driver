//
// Copyright © 2017 ARM Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

package armnn_nn_driver

import (
    "android/soong/android"
    "android/soong/cc"
)

func globalFlags(ctx android.BaseContext) []string {
    var cppflags []string

    if ctx.AConfig().PlatformVersionName() == "Q"  {
        cppflags = append(cppflags, "-fno-addrsig")
    }

    return cppflags
}

func armnnNNDriverDefaults(ctx android.LoadHookContext) {
        type props struct {
                Cppflags []string
        }

        p := &props{}
        p.Cppflags = globalFlags(ctx)

        ctx.AppendProperties(p)
}

func init() {

  android.RegisterModuleType("armnn_nn_driver_defaults", armnnNNDriverDefaultsFactory)
}

func armnnNNDriverDefaultsFactory() android.Module {

   module := cc.DefaultsFactory()
   android.AddLoadHook(module, armnnNNDriverDefaults)
   return module
}
