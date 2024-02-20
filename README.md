# Arm NN Android Neural Networks driver

This directory contains the Arm NN driver for the Android Neural Networks API, implementing the HIDL based android.hardware.neuralnetworks@1.0, android.hardware.neuralnetworks@1.1, android.hardware.neuralnetworks@1.2 and android.hardware.neuralnetworks@1.3 HALs.

For Android 11 and lower, the NNAPI uses HIDL based HALs.

For Android 12 and Android 13, the NNAPI HAL revision uses AIDL instead of HIDL, and HIDL is deprecated.

For Android 14 the compatibility matrix no longer includes support for HIDL HAL revisions:
https://android.googlesource.com/platform/hardware/interfaces/+/refs/heads/android14-qpr1-release/compatibility_matrices/compatibility_matrix.8.xml

For more information about supported operations and configurations, see [NnapiSupport.txt](NnapiSupport.txt)

For documentation about integrating this driver into an Android tree, see [Integrator Guide](docs/IntegratorGuide.md)

For FAQs and troubleshooting advice, see [FAQ.md](docs/FAQ.md)

### License

The android-nn-driver is provided under the [MIT](https://spdx.org/licenses/MIT.html) license.
See [LICENSE](LICENSE) for more information. Contributions to this project are accepted under the same license.

Individual files contain the following tag instead of the full license text.

    SPDX-License-Identifier: MIT

This enables machine processing of license information based on the SPDX License Identifiers that are available here: http://spdx.org/licenses/
