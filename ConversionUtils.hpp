//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "Utils.hpp"

#include <armnn/ArmNN.hpp>
#include <armnn/ILayerSupport.hpp>
#include <armnn/BackendHelper.hpp>
#include <armnn/utility/Assert.hpp>
#include <armnn/utility/IgnoreUnused.hpp>
#include <armnn/utility/NumericCast.hpp>

#include <armnnUtils/DataLayoutIndexed.hpp>
#include <armnnUtils/Transpose.hpp>

#include "1.0/FullyConnected.hpp"

#include <ActivationFunctor.h>
#include <CpuExecutor.h>
#include <OperationsUtils.h>

#include <armnnUtils/FloatingPointComparison.hpp>

#include <log/log.h>
#include <vector>

namespace armnn_driver
{

///
/// Helper classes
///

#ifdef ARMNN_ANDROID_R
using OperandType = android::nn::hal::OperandType;
#endif

#ifdef ARMNN_ANDROID_S
#include <nnapi/Types.h>
#endif


struct ConversionData
{
    ConversionData(const std::vector<armnn::BackendId>& backends)
    : m_Backends(backends)
    , m_Network(nullptr, nullptr)
    , m_DynamicInputsEncountered(false)
    {}

    const std::vector<armnn::BackendId>       m_Backends;
    armnn::INetworkPtr                        m_Network;
    std::vector<armnn::IOutputSlot*>          m_OutputSlotForOperand;
    std::vector<android::nn::RunTimePoolInfo> m_MemPools;
    bool m_DynamicInputsEncountered;
};

class LayerInputHandle
{
public:
    LayerInputHandle();
    LayerInputHandle(bool valid, armnn::IOutputSlot* outputSlot, armnn::TensorInfo tensorInfo);

    bool IsValid() const;

    void Connect(armnn::IInputSlot& inputSlot);

    void Disconnect(armnn::IInputSlot& inputSlot);

    const armnn::TensorInfo& GetTensorInfo() const;

private:
    armnn::IOutputSlot* m_OutputSlot;
    bool                m_Valid;
    armnn::TensorInfo   m_TensorInfo;
};

class ConstTensorPin
{
public:
    // Creates an invalid tensor pin (can be used to signal errors)
    // The optional flag can be set to indicate the tensor values were missing, but it was otherwise valid
    ConstTensorPin(bool optional = false);

    // @param tensorInfo TensorInfo associated with the tensor.
    // @param valueStart Start address of tensor data. Belongs to one of the memory pools associated with
    // the model being converted.
    // @param numBytes Number of bytes for the tensor data.
    ConstTensorPin(const armnn::TensorInfo& tensorInfo, const void* valueStart, uint32_t numBytes,
                   const armnn::PermutationVector& mappings);

    ConstTensorPin(const ConstTensorPin& other) = delete;
    ConstTensorPin(ConstTensorPin&& other)      = default;

    bool IsValid() const;
    bool IsOptional() const;

    const armnn::ConstTensor& GetConstTensor() const;
    const armnn::ConstTensor* GetConstTensorPtr() const;

private:
    armnn::ConstTensor m_ConstTensor;

    // Owned memory for swizzled tensor data, only required if the tensor needed
    // swizzling. Otherwise, @ref m_ConstTensor will reference memory from one of
    // the pools associated with the model being converted.
    std::vector<uint8_t> m_SwizzledTensorData;

    // optional flag to indicate that an invalid tensor pin is not an error, but the optional values were not given
    bool m_Optional;
};

} // namespace armnn_driver

///
/// Utility functions
///

namespace
{

using namespace armnn_driver;
using namespace android::nn;

// Convenience function to log the reason for failing to convert a model.
// @return Always returns false (so that it can be used by callers as a quick way to signal an error and return)
template<class... Args>
static bool Fail(const char* formatStr, Args&&... args)
{
    ALOGD(formatStr, std::forward<Args>(args)...);
    return false;
}

// Convenience macro to call an Is*Supported function and log caller name together with reason for lack of support.
// Called as: FORWARD_LAYER_SUPPORT_FUNC(__func__, Is*Supported, backends, a, b, c, d, e)
#define FORWARD_LAYER_SUPPORT_FUNC(funcName, func, backends, supported, ...) \
try \
{ \
    for (auto&& backendId : backends) \
    { \
        auto layerSupportObject = armnn::GetILayerSupportByBackendId(backendId); \
        if (layerSupportObject.IsBackendRegistered()) \
        { \
            std::string reasonIfUnsupported; \
            supported = \
                layerSupportObject.func(__VA_ARGS__, armnn::Optional<std::string&>(reasonIfUnsupported)); \
            if (supported) \
            { \
                break; \
            } \
            else \
            { \
                if (reasonIfUnsupported.size() > 0) \
                { \
                    ALOGD("%s: not supported by armnn: %s", funcName, reasonIfUnsupported.c_str()); \
                } \
                else \
                { \
                    ALOGD("%s: not supported by armnn", funcName); \
                } \
            } \
        } \
        else \
        { \
            ALOGD("%s: backend not registered: %s", funcName, backendId.Get().c_str()); \
        } \
    } \
    if (!supported) \
    { \
        ALOGD("%s: not supported by any specified backend", funcName); \
    } \
} \
catch (const armnn::InvalidArgumentException &e) \
{ \
    throw armnn::InvalidArgumentException(e, "Failed to check layer support", CHECK_LOCATION()); \
}

template<typename HalOperand>
armnn::TensorShape GetTensorShapeForOperand(const HalOperand& operand)
{
    return armnn::TensorShape(operand.dimensions.size(), operand.dimensions.data());
}

inline bool IsOperandTypeSupportedForTensors(V1_0::OperandType type)
{
    return type == V1_0::OperandType::TENSOR_FLOAT32      ||
           type == V1_0::OperandType::TENSOR_QUANT8_ASYMM ||
           type == V1_0::OperandType::TENSOR_INT32;
}

#if defined(ARMNN_ANDROID_NN_V1_2) || defined(ARMNN_ANDROID_NN_V1_3)

// Support within the 1.2 driver for specific tensor data types
inline bool IsOperandTypeSupportedForTensors(V1_2::OperandType type)
{
    return type == V1_2::OperandType::BOOL                           ||
           type == V1_2::OperandType::TENSOR_BOOL8                   ||
           type == V1_2::OperandType::TENSOR_FLOAT16                 ||
           type == V1_2::OperandType::TENSOR_FLOAT32                 ||
           type == V1_2::OperandType::TENSOR_QUANT8_ASYMM            ||
           type == V1_2::OperandType::TENSOR_QUANT8_SYMM             ||
           type == V1_2::OperandType::TENSOR_QUANT8_SYMM_PER_CHANNEL ||
           type == V1_2::OperandType::TENSOR_QUANT16_SYMM            ||
           type == V1_2::OperandType::TENSOR_INT32;
}

#endif

#ifdef ARMNN_ANDROID_NN_V1_3

// Support within the 1.3 driver for specific tensor data types
inline bool IsOperandTypeSupportedForTensors(V1_3::OperandType type)
{
    return type == V1_3::OperandType::BOOL                           ||
           type == V1_3::OperandType::TENSOR_BOOL8                   ||
           type == V1_3::OperandType::TENSOR_FLOAT16                 ||
           type == V1_3::OperandType::TENSOR_FLOAT32                 ||
           type == V1_3::OperandType::TENSOR_QUANT8_ASYMM            ||
           type == V1_3::OperandType::TENSOR_QUANT8_ASYMM_SIGNED     ||
           type == V1_3::OperandType::TENSOR_QUANT8_SYMM             ||
           type == V1_3::OperandType::TENSOR_QUANT8_SYMM_PER_CHANNEL ||
           type == V1_3::OperandType::TENSOR_QUANT16_SYMM            ||
           type == V1_3::OperandType::TENSOR_INT32;
}

#endif

inline bool IsBool(V1_0::Operand)
{
    return false;
}

inline bool Is12OrLaterOperand(V1_0::Operand)
{
    return false;
}

#if defined(ARMNN_ANDROID_NN_V1_2) || defined(ARMNN_ANDROID_NN_V1_3)

inline bool IsBool(V1_2::Operand operand)
{
    return operand.type == V1_2::OperandType::BOOL;
}

/// Checks if a operand is 1_2 Operand
inline bool Is12OrLaterOperand(V1_2::Operand)
{
    return true;
}

#endif

#ifdef ARMNN_ANDROID_NN_V1_3

inline bool IsBool(V1_3::Operand operand)
{
    return operand.type == V1_3::OperandType::BOOL;
}

/// Checks if a operand is 1_2 Operand
inline bool Is12OrLaterOperand(V1_3::Operand)
{
    return true;
}

#endif

template<typename LayerHandleType>
armnn::IConnectableLayer& AddReshapeLayer(armnn::INetwork& network,
                                          LayerHandleType& inputLayer,
                                          armnn::TensorInfo reshapeInfo)
{
    armnn::ReshapeDescriptor reshapeDescriptor;
    reshapeDescriptor.m_TargetShape = reshapeInfo.GetShape();

    armnn::IConnectableLayer* reshapeLayer = network.AddReshapeLayer(reshapeDescriptor);
    ARMNN_ASSERT(reshapeLayer != nullptr);

    // Attach the input layer to the reshape layer
    inputLayer.Connect(reshapeLayer->GetInputSlot(0));
    reshapeLayer->GetOutputSlot(0).SetTensorInfo(reshapeInfo);

    return *reshapeLayer;
}

bool BroadcastTensor(LayerInputHandle& input0,
                     LayerInputHandle& input1,
                     armnn::IConnectableLayer* startLayer,
                     ConversionData& data)
{
    ARMNN_ASSERT(startLayer != nullptr);

    const armnn::TensorInfo& inputInfo0 = input0.GetTensorInfo();
    const armnn::TensorInfo& inputInfo1 = input1.GetTensorInfo();

    unsigned int inputDimensions0 = inputInfo0.GetNumDimensions();
    unsigned int inputDimensions1 = inputInfo1.GetNumDimensions();

    if (inputDimensions0 == inputDimensions1)
    {
        // The inputs have the same number of dimensions, simply connect them to the given layer as they are
        input0.Connect(startLayer->GetInputSlot(0));
        input1.Connect(startLayer->GetInputSlot(1));

        return true;
    }

    // Since the number of dimensions do not match then we need to add degenerate dimensions
    // to the "smaller" tensor using a reshape, while keeping the order of the inputs.

    unsigned int maxInputDimensions = std::max(inputDimensions0, inputDimensions1);
    unsigned int sizeDifference = std::abs(armnn::numeric_cast<int>(inputDimensions0) -
                                           armnn::numeric_cast<int>(inputDimensions1));

    bool input0IsSmaller = inputDimensions0 < inputDimensions1;
    LayerInputHandle& smallInputHandle = input0IsSmaller ? input0 : input1;
    const armnn::TensorInfo& smallInfo = smallInputHandle.GetTensorInfo();

    const armnn::TensorShape& smallShape = smallInfo.GetShape();
    std::vector<unsigned int> reshapedDimensions(maxInputDimensions, 1);
    for (unsigned int i = sizeDifference; i < maxInputDimensions; i++)
    {
        reshapedDimensions[i] = smallShape[i - sizeDifference];
    }

    armnn::TensorInfo reshapedInfo = smallInfo;
    reshapedInfo.SetShape(armnn::TensorShape{ armnn::numeric_cast<unsigned int>(reshapedDimensions.size()),
                                              reshapedDimensions.data() });

    // RehsapeDescriptor that is ignored in the IsReshapeSupported function
    armnn::ReshapeDescriptor reshapeDescriptor;

    bool isSupported = false;
    FORWARD_LAYER_SUPPORT_FUNC(__func__,
                               IsReshapeSupported,
                               data.m_Backends,
                               isSupported,
                               smallInfo,
                               reshapedInfo,
                               reshapeDescriptor);
    if (!isSupported)
    {
        return false;
    }

    ARMNN_ASSERT(data.m_Network != nullptr);
    armnn::IConnectableLayer& reshapeLayer = AddReshapeLayer(*data.m_Network, smallInputHandle, reshapedInfo);

    if (input0IsSmaller)
    {
        // Input0 is the "smaller" tensor, connect the reshape layer as follows:
        //
        //  Input0 Input1
        //     |     |
        //  Reshape  |
        //      \   /
        //    StartLayer

        reshapeLayer.GetOutputSlot(0).Connect(startLayer->GetInputSlot(0));
        input1.Connect(startLayer->GetInputSlot(1));
    }
    else
    {
        // Input1 is the "smaller" tensor, connect the reshape layer as follows:
        //
        //  Input0 Input1
        //     |     |
        //     |  Reshape
        //      \   /
        //    StartLayer

        input0.Connect(startLayer->GetInputSlot(0));
        reshapeLayer.GetOutputSlot(0).Connect(startLayer->GetInputSlot(1));
    }

    return true;
}

void CalcPadding(uint32_t input,
                 uint32_t kernel,
                 uint32_t stride,
                 uint32_t& outPadHead,
                 uint32_t& outPadTail,
                 android::nn::PaddingScheme scheme)
{
    int32_t padHead;
    int32_t padTail;
    calculateExplicitPadding(input, stride, kernel, scheme, &padHead, &padTail);
    outPadHead = armnn::numeric_cast<uint32_t>(padHead);
    outPadTail = armnn::numeric_cast<uint32_t>(padTail);
}

#if defined(ARMNN_ANDROID_NN_V1_2) || defined(ARMNN_ANDROID_NN_V1_3)

void CalcPadding(uint32_t input, uint32_t kernel, uint32_t stride, uint32_t dilation, uint32_t& outPadHead,
                 uint32_t& outPadTail, android::nn::PaddingScheme scheme)
{
    int32_t padHead;
    int32_t padTail;
    calculateExplicitPadding(input, stride, dilation, kernel, scheme, &padHead, &padTail);
    outPadHead = armnn::numeric_cast<uint32_t>(padHead);
    outPadTail = armnn::numeric_cast<uint32_t>(padTail);
}

void CalcPaddingTransposeConv(uint32_t output, uint32_t kernel, int32_t stride, int32_t& outPadHead,
                              int32_t& outPadTail, android::nn::PaddingScheme scheme)
{
    calculateExplicitPaddingTransposeConv(output, stride, kernel, scheme, &outPadHead, &outPadTail);
}

#endif

Shape GetOperandShape(const V1_0::Operand& operand)
{
    Shape shape;
    shape.type = OperandType(operand.type);
    shape.dimensions = operand.dimensions;
    shape.scale = operand.scale;
    shape.offset = operand.zeroPoint;
    return shape;
}

#if defined(ARMNN_ANDROID_NN_V1_2) || defined(ARMNN_ANDROID_NN_V1_3)

Shape GetOperandShape(const V1_2::Operand& operand)
{
    Shape shape;
    shape.type = OperandType(operand.type);
    shape.dimensions = operand.dimensions;
    shape.scale = operand.scale;
    shape.offset = operand.zeroPoint;
    return shape;
}

#endif

#ifdef ARMNN_ANDROID_NN_V1_3

Shape GetOperandShape(const V1_3::Operand& operand)
{
    Shape shape;
    shape.type = OperandType(operand.type);
    shape.dimensions = operand.dimensions;
    shape.scale = operand.scale;
    shape.offset = operand.zeroPoint;
    return shape;
}

#endif

// ArmNN requires the bias scale to be equal to the product of the weight and input scales, which is also
// what AndroidNN requires. However for some of the AndroidNN tests the values don't exactly match so
// we accept some tolerance. We don't want ArmNN itself to accept these inconsistencies as it is up to the
// user (us, in this case) to ensure they match.
void SanitizeBiasQuantizationScale(armnn::TensorInfo& biasInfo,
                                   const armnn::TensorInfo& weightInfo,
                                   const armnn::TensorInfo& inputInfo)
{
    if (weightInfo.HasPerAxisQuantization())
    {
        // NOTE: Bias scale is always set to 0 for per-axis quantization and
        // it needs to be calculated: scale[i] = input_scale * weight_scale[i]
        auto UpdateBiasScaleValue = [&inputInfo](float biasScale) -> float
        {
            return biasScale * inputInfo.GetQuantizationScale();
        };

        std::vector<float> biasScales(weightInfo.GetQuantizationScales());
        std::transform(biasScales.begin(), biasScales.end(), biasScales.begin(), UpdateBiasScaleValue);

        biasInfo.SetQuantizationScales(biasScales);
        biasInfo.SetQuantizationDim(weightInfo.GetQuantizationDim());

        ALOGV("Bias quantization params have been updated for per-axis quantization");
    }
    else
    {
        const float expectedBiasScale = weightInfo.GetQuantizationScale() * inputInfo.GetQuantizationScale();
        if (biasInfo.GetQuantizationScale() != expectedBiasScale)
        {
            if (armnnUtils::within_percentage_tolerance(biasInfo.GetQuantizationScale(), expectedBiasScale, 1.0f))
            {
                ALOGW("Bias quantization scale has been modified to match input * weights");
                biasInfo.SetQuantizationScale(expectedBiasScale);
            }
        }
    }
}

// 4D Tensor Permutations
const armnn::PermutationVector IdentityPermutation4D({ 0U, 1U, 2U, 3U });
const armnn::PermutationVector IdentityPermutation3D({ 0U, 1U, 2U });
const armnn::PermutationVector SwapDim1And2({ 0U, 2U, 1U, 3U });

// 3D Permutation Vectors
const armnn::PermutationVector RotateTensorLeft({ 1U, 2U, 0U });
const armnn::PermutationVector RotateTensorRight({ 2U, 0U, 1U });

template<typename OSlot>
armnn::IConnectableLayer& AddTransposeLayer(armnn::INetwork& network, OSlot& input,
                                            const armnn::PermutationVector& mappings)
{
    // Add swizzle layer
    armnn::IConnectableLayer* const layer = network.AddTransposeLayer(mappings);

    ARMNN_ASSERT(layer != nullptr);

    // Connect input to swizzle layer
    input.Connect(layer->GetInputSlot(0));

    // Setup swizzled output
    const armnn::TensorInfo outInfo = armnnUtils::TransposeTensorShape(input.GetTensorInfo(), mappings);
    layer->GetOutputSlot(0).SetTensorInfo(outInfo);

    return *layer;
}

bool ValidateConcatOutputShape(const std::vector<armnn::TensorShape> & inputShapes,
                               const armnn::TensorShape & outputShape,
                               uint32_t concatDim)
{
    // Validate the output shape is correct given the input shapes (which have just been validated)
    unsigned int numDimensions = inputShapes[0].GetNumDimensions();
    if (outputShape.GetNumDimensions() != numDimensions)
    {
        return Fail("%s: Output shape has wrong number of dimensions", __func__);
    }

    unsigned int outputSizeAlongConcatenatedDimension = 0;
    for (unsigned int i = 0; i < inputShapes.size(); i++)
    {
        outputSizeAlongConcatenatedDimension += inputShapes[i][concatDim];
    }

    for (unsigned int i = 0; i < numDimensions; ++i)
    {
        if (i == concatDim)
        {
            if (outputShape[i] != outputSizeAlongConcatenatedDimension)
            {
                return Fail(
                        "%s: Invalid output shape for dimension %d (%d != %d)",
                        __func__,
                        i,
                        outputShape[i],
                        outputSizeAlongConcatenatedDimension);
            }
        }
        else
        {
            if (outputShape[i] != inputShapes[0][i])
            {
                return Fail("%s: Invalid output shape", __func__);
            }
        }
    }

    return true;
}

bool RequiresReshape(armnn::TensorShape & inputShape)
{
    return inputShape.GetNumDimensions() < 3;
}

void SwizzleInputs(armnn::INetwork& network,
                   std::vector<LayerInputHandle>& inputs,
                   std::vector<armnn::TensorShape>& inputShapes,
                   const armnn::PermutationVector& mapping)
{
    if (!mapping.IsEqual(IdentityPermutation4D))
    {
        size_t nInputs = inputs.size();
        for (size_t i=0; i<nInputs; ++i)
        {
            // add swizzle layer
            armnn::IConnectableLayer& swizzleLayer = AddTransposeLayer(network, inputs[i], mapping);
            auto& outputSlot = swizzleLayer.GetOutputSlot(0);
            auto& outputInfo = outputSlot.GetTensorInfo();
            // replace inputs with the swizzled ones
            inputs[i] = LayerInputHandle(true, &outputSlot, outputInfo);
            inputShapes[i] = inputs[i].GetTensorInfo().GetShape();
        }
    }
}

bool TransposeInputTensors(ConversionData& data,
                          std::vector<LayerInputHandle>& inputs,
                          std::vector<armnn::TensorShape>& inputShapes,
                          const armnn::PermutationVector& mapping)
{
    // If we have a IdentityPermutation4D or IdentityPermutation3D then we are not permuting
    if (!mapping.IsEqual(IdentityPermutation4D) && !mapping.IsEqual(IdentityPermutation3D))
    {
        armnn::TensorInfo outputTransposeInfo;
        size_t nInputs = inputs.size();
        for (size_t i=0; i<nInputs; ++i)
        {
            // check permute layer
            armnn::TransposeDescriptor transposeDesc;
            transposeDesc.m_DimMappings = mapping;
            outputTransposeInfo = armnnUtils::TransposeTensorShape(inputs[i].GetTensorInfo(), mapping);

            bool isSupported = false;
            FORWARD_LAYER_SUPPORT_FUNC(__func__,
                                       IsTransposeSupported,
                                       data.m_Backends,
                                       isSupported,
                                       inputs[i].GetTensorInfo(),
                                       outputTransposeInfo,
                                       transposeDesc);
            if (!isSupported)
            {
                return false;
            }

        }
        SwizzleInputs(*data.m_Network, inputs, inputShapes, mapping);
    }
    return true;
}


bool CreateConcatPermutationParameters(const unsigned int numberOfDimensions,
                                       int32_t & concatDimension,
                                       std::pair<armnn::PermutationVector, armnn::PermutationVector> & permutationPair)
{
    bool needPermute = false;
    ARMNN_ASSERT(numberOfDimensions >= 3);

    // ArmNN uses Compute Library subtensors to perform concatenation
    // This only works when concatenating along dimension 0, 1 or 3 for a 4-D tensor,
    // or along dimension 0 or 2 for a 3-D tensor.
    if (numberOfDimensions == 4 && concatDimension == 2)
    {
        concatDimension = 1;
        permutationPair = std::make_pair(SwapDim1And2, SwapDim1And2);
        needPermute = true;
    }
    else if (numberOfDimensions == 3 && concatDimension == 1)
    {
        concatDimension = 0;
        permutationPair = std::make_pair(RotateTensorLeft, RotateTensorRight);
        needPermute = true;
    }
    // If the tensor is 3-D and the concat dimension is 2 then we don't need to permute but we do need to change the
    // permutation identity to only have 3 dimensions
    else if (numberOfDimensions == 3 && concatDimension == 2)
    {
        permutationPair = std::make_pair(IdentityPermutation3D, IdentityPermutation3D);
    }
    return needPermute;
}

} // anonymous namespace

namespace armnn_driver
{

//// Creates an ArmNN activation layer and connects it to the given layer, if the
//// passed in AndroidNN activation function requires so.
//// @return The end layer of the sequence of layers built for the given AndroidNN
//// activation function or nullptr if an error occurred (e.g. unsupported activation).
//// Note that the end layer matches the input layer if no activation is required
//// (the sequence of layers has length 1).
armnn::IConnectableLayer* ProcessActivation(const armnn::TensorInfo& tensorInfo,
                                            ActivationFn activation,
                                            armnn::IConnectableLayer* prevLayer,
                                            ConversionData& data);

} // namespace armnn_driver

///
/// Utility templates
///

namespace armnn_driver
{

using namespace android::nn;

template<typename HalPolicy,
         typename HalOperand   = typename HalPolicy::Operand,
         typename HalOperation = typename HalPolicy::Operation,
         typename HalModel     = typename HalPolicy::Model>
const HalOperand* GetInputOperand(const HalOperation& operation,
                                  uint32_t inputIndex,
                                  const HalModel& model,
                                  bool failOnIndexOutOfBounds = true)
{
    if (inputIndex >= operation.inputs.size())
    {
        if (failOnIndexOutOfBounds)
        {
            Fail("%s: invalid input index: %i out of %i", __func__, inputIndex, operation.inputs.size());
        }
        return nullptr;
    }

    // Model should have been validated beforehand
    ARMNN_ASSERT(operation.inputs[inputIndex] < getMainModel(model).operands.size());
    return &getMainModel(model).operands[operation.inputs[inputIndex]];
}

template<typename HalPolicy,
         typename HalOperand   = typename HalPolicy::Operand,
         typename HalOperation = typename HalPolicy::Operation,
         typename HalModel     = typename HalPolicy::Model>
const HalOperand* GetOutputOperand(const HalOperation& operation,
                                   uint32_t outputIndex,
                                   const HalModel& model)
{
    if (outputIndex >= operation.outputs.size())
    {
        Fail("%s: invalid output index: %i out of %i", __func__, outputIndex, operation.outputs.size());
        return nullptr;
    }

    // Model should have been validated beforehand
    ARMNN_ASSERT(operation.outputs[outputIndex] < getMainModel(model).operands.size());

    return &getMainModel(model).operands[operation.outputs[outputIndex]];
}

template<typename HalPolicy,
         typename HalOperand = typename HalPolicy::Operand,
         typename HalModel   = typename HalPolicy::Model>
const void* GetOperandValueReadOnlyAddress(const HalOperand& operand,
                                           const HalModel& model,
                                           const ConversionData& data,
                                           bool optional = false)
{
    using HalOperandLifeTime = typename HalPolicy::OperandLifeTime;

    const void* valueStart = nullptr;
    switch (operand.lifetime)
    {
        case HalOperandLifeTime::CONSTANT_COPY:
        {
            // Constant found in model.operandValues
            valueStart = &model.operandValues[operand.location.offset];
            break;
        }
        case HalOperandLifeTime::CONSTANT_REFERENCE:
        {
            // Constant specified via a Memory object
            valueStart = GetMemoryFromPool(operand.location, data.m_MemPools);
            break;
        }
        case HalOperandLifeTime::NO_VALUE:
        {
            // An optional input tensor with no values is not an error so should not register as a fail
            if (optional)
            {
                valueStart = nullptr;
                break;
            }
            [[fallthrough]];
        }
        default:
        {
            // Unsupported/invalid (e.g. can't get value of an input to the model)
            Fail("%s: unsupported/invalid operand lifetime: %s",
                 __func__, toString(operand.lifetime).c_str());
            valueStart = nullptr;
        }
    }

    return valueStart;
}

template<typename HalPolicy,
         typename HalOperation   = typename HalPolicy::Operation,
         typename HalModel       = typename HalPolicy::Model,
         typename HalOperandType = typename HalPolicy::OperandType>
bool GetOperandType(const HalOperation& operation,
                    uint32_t inputIndex,
                    const HalModel& model,
                    HalOperandType& type)
{
    using HalOperand = typename HalPolicy::Operand;

    const HalOperand* operand = GetInputOperand<HalPolicy>(operation, inputIndex, model);
    if (!operand)
    {
        return Fail("%s: invalid input operand at index %i", __func__, inputIndex);
    }

    type = operand->type;
    return true;
}

template<typename HalPolicy,
         typename HalOperand = typename HalPolicy::Operand>
bool IsOperandConstant(const HalOperand& operand)
{
    using HalOperandLifeTime = typename HalPolicy::OperandLifeTime;

    HalOperandLifeTime lifetime = operand.lifetime;

    return lifetime == HalOperandLifeTime::CONSTANT_COPY ||
           lifetime == HalOperandLifeTime::CONSTANT_REFERENCE ||
           lifetime == HalOperandLifeTime::NO_VALUE;
}

template<typename HalPolicy,
         typename HalOperand   = typename HalPolicy::Operand,
         typename HalModel     = typename HalPolicy::Model>
ConstTensorPin ConvertOperandToConstTensorPin(const HalOperand& operand,
                                              const HalModel& model,
                                              const ConversionData& data,
                                              const armnn::PermutationVector& dimensionMappings = g_DontPermute,
                                              const armnn::TensorShape* overrideTensorShape = nullptr,
                                              bool optional = false)
{
    if (!IsOperandTypeSupportedForTensors(operand.type))
    {
        Fail("%s: unsupported operand type for tensor %s", __func__, toString(operand.type).c_str());
        return ConstTensorPin();
    }

    if (!optional && !IsOperandConstant<HalPolicy>(operand))
    {
        Fail("%s: invalid operand lifetime: %s", __func__, toString(operand.lifetime).c_str());
        return ConstTensorPin();
    }

    const void* const valueStart = GetOperandValueReadOnlyAddress<HalPolicy>(operand, model, data, optional);
    if (!valueStart)
    {
        if (optional)
        {
            // optional tensor with no values is not really an error; return it as invalid, but marked as optional
            return ConstTensorPin(true);
        }
        // mandatory tensor with no values
        Fail("%s: failed to get operand address", __func__);
        return ConstTensorPin();
    }

    armnn::TensorInfo tensorInfo = GetTensorInfoForOperand(operand);
    // Android datalayout might be different than armnn datalayout, e.g. the kernel for the depthwise convolution.
    if (tensorInfo.HasPerAxisQuantization())
    {
        tensorInfo.SetQuantizationDim(dimensionMappings[tensorInfo.GetQuantizationDim().value()]);
    }

    if (overrideTensorShape != nullptr)
    {
        tensorInfo.SetShape(*overrideTensorShape);
    }
    return ConstTensorPin(tensorInfo, valueStart, operand.location.length, dimensionMappings);
}

template<typename HalPolicy,
         typename HalOperation = typename HalPolicy::Operation,
         typename HalModel     = typename HalPolicy::Model>
ConstTensorPin ConvertOperationInputToConstTensorPin(const HalOperation& operation,
                                                     uint32_t inputIndex,
                                                     const HalModel& model,
                                                     const ConversionData& data,
                                                     const armnn::PermutationVector& dimensionMappings = g_DontPermute,
                                                     const armnn::TensorShape* overrideTensorShape = nullptr,
                                                     bool optional = false)
{
    using HalOperand = typename HalPolicy::Operand;

    const HalOperand* operand = GetInputOperand<HalPolicy>(operation, inputIndex, model);
    if (!operand)
    {
        Fail("%s: failed to get input operand: index=%u", __func__, inputIndex);
        return ConstTensorPin();
    }
    return ConvertOperandToConstTensorPin<HalPolicy>(*operand,
                                                     model,
                                                     data,
                                                     dimensionMappings,
                                                     overrideTensorShape,
                                                     optional);
}

template<typename HalPolicy,
         typename OutputType,
         typename HalOperandType = typename HalPolicy::OperandType,
         typename HalOperation   = typename HalPolicy::Operation,
         typename HalModel       = typename HalPolicy::Model>
bool GetInputScalar(const HalOperation& operation,
                    uint32_t inputIndex,
                    HalOperandType type,
                    OutputType& outValue,
                    const HalModel& model,
                    const ConversionData& data,
                    bool optional = false)
{
    using HalOperand = typename HalPolicy::Operand;

    const HalOperand* operand = GetInputOperand<HalPolicy>(operation, inputIndex, model);
    if (!optional && !operand)
    {
        return Fail("%s: invalid input operand at index %i", __func__, inputIndex);
    }

    if (!optional && operand->type != type)
    {
        return Fail("%s: unexpected operand type: %s (should be %s)",
                    __func__, toString(operand->type).c_str(), toString(type).c_str());
    }

    if (!optional && operand->location.length != sizeof(OutputType))
    {
        return Fail("%s: incorrect operand location length: %i (should be %i)",
                    __func__, operand->location.length, sizeof(OutputType));
    }

    const void* valueAddress = GetOperandValueReadOnlyAddress<HalPolicy>(*operand, model, data);
    if (!optional && !valueAddress)
    {
        return Fail("%s: failed to get address for operand", __func__);
    }

    if(!optional)
    {
        outValue = *(static_cast<const OutputType*>(valueAddress));
    }

    return true;
}

template<typename HalPolicy,
         typename HalOperation = typename HalPolicy::Operation,
         typename HalModel     = typename HalPolicy::Model>
bool GetInputInt32(const HalOperation& operation,
                   uint32_t inputIndex,
                   int32_t& outValue,
                   const HalModel& model,
                   const ConversionData& data)
{
    return GetInputScalar<HalPolicy>(operation, inputIndex, HalPolicy::OperandType::INT32, outValue, model, data);
}

template<typename HalPolicy,
         typename HalOperation = typename HalPolicy::Operation,
         typename HalModel     = typename HalPolicy::Model>
bool GetInputFloat32(const HalOperation& operation,
                     uint32_t inputIndex,
                     float& outValue,
                     const HalModel& model,
                     const ConversionData& data)
{
    return GetInputScalar<HalPolicy>(operation, inputIndex, HalPolicy::OperandType::FLOAT32, outValue, model, data);
}

template<typename HalPolicy,
         typename HalOperation   = typename HalPolicy::Operation,
         typename HalOperandType = typename HalPolicy::OperandType,
         typename HalModel       = typename HalPolicy::Model>
bool GetInputActivationFunctionImpl(const HalOperation& operation,
                                    uint32_t inputIndex,
                                    HalOperandType type,
                                    ActivationFn& outActivationFunction,
                                    const HalModel& model,
                                    const ConversionData& data)
{
    if (type != HalOperandType::INT32 && type != HalOperandType::TENSOR_INT32)
    {
        return Fail("%s: unexpected operand type: %s (should be %s or %s)",
                    __func__,
                    toString(type).c_str(),
                    toString(HalOperandType::INT32).c_str(),
                    toString(HalOperandType::TENSOR_INT32).c_str());
    }

    int32_t activationFunctionAsInt;
    if (!GetInputScalar<HalPolicy>(operation, inputIndex, type, activationFunctionAsInt, model, data))
    {
        return Fail("%s: failed to get activation input value", __func__);
    }
    outActivationFunction = static_cast<ActivationFn>(activationFunctionAsInt);
    return true;
}

template<typename HalPolicy,
         typename HalOperation = typename HalPolicy::Operation,
         typename HalModel     = typename HalPolicy::Model>
bool GetInputActivationFunction(const HalOperation& operation,
                                uint32_t inputIndex,
                                ActivationFn& outActivationFunction,
                                const HalModel& model,
                                const ConversionData& data)
{
    return GetInputActivationFunctionImpl<HalPolicy>(operation,
                                                     inputIndex,
                                                     HalPolicy::OperandType::INT32,
                                                     outActivationFunction,
                                                     model,
                                                     data);
}

template<typename HalPolicy,
         typename HalOperation = typename HalPolicy::Operation,
         typename HalModel     = typename HalPolicy::Model>
bool GetInputActivationFunctionFromTensor(const HalOperation& operation,
                                          uint32_t inputIndex,
                                          ActivationFn& outActivationFunction,
                                          const HalModel& model,
                                          const ConversionData& data)
{
    // This only accepts a 1-D tensor of size 1
    return GetInputActivationFunctionImpl<HalPolicy>(operation,
                                                     inputIndex,
                                                     HalPolicy::OperandType::INT32,
                                                     outActivationFunction,
                                                     model,
                                                     data);
}


template<typename HalPolicy,
         typename HalOperation   = typename HalPolicy::Operation,
         typename HalModel       = typename HalPolicy::Model>
bool GetOptionalInputActivation(const HalOperation& operation,
                                uint32_t inputIndex,
                                ActivationFn& activationFunction,
                                const HalModel& model,
                                const ConversionData& data)
{
    if (operation.inputs.size() <= inputIndex)
    {
        activationFunction = ActivationFn::kActivationNone;
    }
    else
    {
        if (!GetInputActivationFunction<HalPolicy>(operation, inputIndex, activationFunction, model, data))
        {
            return Fail("%s: Operation has invalid inputs", __func__);
        }
    }
    return true;
}

template<typename HalPolicy,
         typename ConvolutionDescriptor,
         typename HalOperation = typename HalPolicy::Operation,
         typename HalModel     = typename HalPolicy::Model>
bool GetOptionalConvolutionDilationParams(const HalOperation& operation,
                                          uint32_t dilationXIndex,
                                          ConvolutionDescriptor& descriptor,
                                          const HalModel& model,
                                          const ConversionData& data)
{
    bool success = true;
    if (operation.inputs.size() >= dilationXIndex + 2)
    {
        success &= GetInputScalar<HalPolicy>(operation,
                                             dilationXIndex,
                                             HalPolicy::OperandType::INT32,
                                             descriptor.m_DilationX,
                                             model,
                                             data);
        success &= GetInputScalar<HalPolicy>(operation,
                                             dilationXIndex + 1,
                                             HalPolicy::OperandType::INT32,
                                             descriptor.m_DilationY,
                                             model,
                                             data);
    }

    return success;
}

template<typename HalPolicy,
         typename HalOperation   = typename HalPolicy::Operation,
         typename HalModel       = typename HalPolicy::Model>
bool GetOptionalBool(const HalOperation& operation,
                     uint32_t inputIndex,
                     const HalModel& model,
                     const ConversionData& data)
{
    using HalOperand = typename HalPolicy::Operand;

    const HalOperand* operand = GetInputOperand<HalPolicy>(operation, inputIndex, model);
    if (!operand)
    {
        return false;
    }

    if (!IsBool(*operand))
    {
        return false;
    }

    const void* valueAddress = GetOperandValueReadOnlyAddress<HalPolicy>(*operand, model, data);
    if (!valueAddress)
    {
        return false;
    }

    if (*(static_cast<const bool*>(valueAddress)))
    {
        return true;
    }
    else
    {
        return false;
    }
}

template<typename HalPolicy,
         typename HalOperand = typename HalPolicy::Operand,
         typename HalModel   = typename HalPolicy::Model>
bool GetTensorInt32Values(const HalOperand& operand,
                          std::vector<int32_t>& outValues,
                          const HalModel& model,
                          const ConversionData& data)
{
    if (operand.type != HalPolicy::OperandType::TENSOR_INT32)
    {
        return Fail("%s: invalid operand type: %s", __func__, toString(operand.type).c_str());
    }

    const void* startAddress = GetOperandValueReadOnlyAddress<HalPolicy>(operand, model, data);
    if (!startAddress)
    {
        return Fail("%s: failed to get operand address", __func__, operand.type);
    }

    // Check number of bytes is sensible
    const uint32_t numBytes = operand.location.length;
    if (numBytes % sizeof(int32_t) != 0)
    {
        return Fail("%s: invalid number of bytes: %i, expected to be a multiple of %i",
                    __func__, numBytes, sizeof(int32_t));
    }

    outValues.resize(numBytes / sizeof(int32_t));
    memcpy(outValues.data(), startAddress, numBytes);
    return true;
}

template<typename HalPolicy,
         typename HalOperation = typename HalPolicy::Operation,
         typename HalModel     = typename HalPolicy::Model>
bool GetInputPaddingScheme(const HalOperation& operation,
                           uint32_t inputIndex,
                           PaddingScheme& outPaddingScheme,
                           const HalModel& model,
                           const ConversionData& data)
{
    int32_t paddingSchemeAsInt;
    if (!GetInputInt32<HalPolicy>(operation, inputIndex, paddingSchemeAsInt, model, data))
    {
        return Fail("%s: failed to get padding scheme input value", __func__);
    }

    outPaddingScheme = static_cast<android::nn::PaddingScheme>(paddingSchemeAsInt);
    return true;
}

template<typename HalPolicy,
         typename HalOperation = typename HalPolicy::Operation,
         typename HalModel     = typename HalPolicy::Model>
LayerInputHandle ConvertToLayerInputHandle(const HalOperation& operation,
                                           uint32_t inputIndex,
                                           const HalModel& model,
                                           ConversionData& data)
{
    using HalOperand         = typename HalPolicy::Operand;
    using HalOperandType     = typename HalPolicy::OperandType;
    using HalOperandLifeTime = typename HalPolicy::OperandLifeTime;

    const HalOperand* operand = GetInputOperand<HalPolicy>(operation, inputIndex, model);
    if (!operand)
    {
        Fail("%s: failed to get input operand %i", __func__, inputIndex);
        return LayerInputHandle();
    }

    if (!IsOperandTypeSupportedForTensors(operand->type))
    {
        Fail("%s: unsupported operand type for tensor %s", __func__, toString(operand->type).c_str());
        return LayerInputHandle();
    }

    try
    {
        armnn::TensorInfo operandTensorInfo = GetTensorInfoForOperand(*operand);
        if (IsDynamicTensor(operandTensorInfo))
        {
            Fail("%s: dynamic input tensors are not supported", __func__);
            return LayerInputHandle();
        }

        switch (operand->lifetime)
        {
            case HalOperandLifeTime::MODEL_INPUT:
            {
                // NOTE: We must check whether we can support the input tensor on at least one
                // of the provided backends; otherwise we cannot convert the operation
                bool isInputSupported = false;
                FORWARD_LAYER_SUPPORT_FUNC(__func__,
                                           IsInputSupported,
                                           data.m_Backends,
                                           isInputSupported,
                                           operandTensorInfo);

                if (!isInputSupported)
                {
                    Fail("%s: unsupported input tensor", __func__);
                    return LayerInputHandle();
                }

                [[clang::fallthrough]]; // intentional fallthrough
            }
            case HalOperandLifeTime::TEMPORARY_VARIABLE: // intentional fallthrough
            case HalOperandLifeTime::MODEL_OUTPUT:
            {
                // The tensor is either an operand internal to the model, or a model input.
                // It can be associated with an ArmNN output slot for an existing layer.

                // m_OutputSlotForOperand[...] can be nullptr if the previous layer could not be converted
                const uint32_t operandIndex = operation.inputs[inputIndex];
                return LayerInputHandle(true, data.m_OutputSlotForOperand[operandIndex], operandTensorInfo);
            }
            case HalOperandLifeTime::CONSTANT_COPY: // intentional fallthrough
            case HalOperandLifeTime::CONSTANT_REFERENCE:
            {
                // The tensor has an already known constant value, and can be converted into an ArmNN Constant layer.
                ConstTensorPin tensorPin = ConvertOperandToConstTensorPin<HalPolicy>(*operand, model, data);
                if (tensorPin.IsValid())
                {
                    bool isSupported = false;
                    FORWARD_LAYER_SUPPORT_FUNC(__func__,
                                               IsConstantSupported,
                                               data.m_Backends,
                                               isSupported,
                                               tensorPin.GetConstTensor().GetInfo());
                    if (!isSupported)
                    {
                        return LayerInputHandle();
                    }

                    armnn::IConnectableLayer* constantLayer =
                                    data.m_Network->AddConstantLayer(tensorPin.GetConstTensor());
                    armnn::IOutputSlot& outputSlot = constantLayer->GetOutputSlot(0);
                    outputSlot.SetTensorInfo(tensorPin.GetConstTensor().GetInfo());

                    return LayerInputHandle(true, &outputSlot, operandTensorInfo);
                }
                else
                {
                    Fail("%s: invalid operand tensor", __func__);
                    return LayerInputHandle();
                }
                break;
            }
            default:
            {
                // Unsupported lifetime for an input tensor
                Fail("%s: unsupported lifetime for input tensor: %s",
                     __func__, toString(operand->lifetime).c_str());
                return LayerInputHandle();
            }
        }
    }
    catch (UnsupportedOperand<HalOperandType>& e)
    {
        Fail("%s: Operand type %s not supported in ArmnnDriver", __func__, toString(e.m_type).c_str());
        return LayerInputHandle();
    }
}


#ifdef ARMNN_ANDROID_NN_V1_3
template<typename HalPolicy>
LayerInputHandle ConvertToLayerInputHandle(const ::android::hardware::neuralnetworks::V1_3::Operation& operation,
                                           uint32_t inputIndex,
                                           const::android::hardware::neuralnetworks::V1_3::Model& model,
                                           ConversionData& data)
{
    using HalOperand         = typename HalPolicy::Operand;
    using HalOperandType     = typename HalPolicy::OperandType;
    using HalOperandLifeTime = typename HalPolicy::OperandLifeTime;

    const HalOperand* operand = GetInputOperand<HalPolicy>(operation, inputIndex, model);
    if (!operand)
    {
        Fail("%s: failed to get input operand %i", __func__, inputIndex);
        return LayerInputHandle();
    }

    if (!IsOperandTypeSupportedForTensors(operand->type))
    {
        Fail("%s: unsupported operand type for tensor %s", __func__, toString(operand->type).c_str());
        return LayerInputHandle();
    }

    try
    {
        armnn::TensorInfo operandTensorInfo = GetTensorInfoForOperand(*operand);

        if (IsDynamicTensor(operandTensorInfo))
        {
            data.m_DynamicInputsEncountered = true;

            const uint32_t operandIndex = operation.inputs[inputIndex];

            // Check if the dynamic input tensors have been inferred by one of the previous layers
            // If not we can't support them
            if (data.m_OutputSlotForOperand.size() >= operandIndex && data.m_OutputSlotForOperand[operandIndex])
            {
                operandTensorInfo = data.m_OutputSlotForOperand[operandIndex]->GetTensorInfo();
            }
            else
            {
                Fail("%s: Type 2 dynamic input tensors are not supported", __func__);
                return LayerInputHandle();
            }
        }

        switch (operand->lifetime)
        {
            case HalOperandLifeTime::SUBGRAPH_INPUT:
            {
                // NOTE: We must check whether we can support the input tensor on at least one
                // of the provided backends; otherwise we cannot convert the operation
                bool isInputSupported = false;
                FORWARD_LAYER_SUPPORT_FUNC(__func__,
                                           IsInputSupported,
                                           data.m_Backends,
                                           isInputSupported,
                                           operandTensorInfo);

                if (!isInputSupported)
                {
                    Fail("%s: unsupported input tensor", __func__);
                    return LayerInputHandle();
                }

                [[clang::fallthrough]]; // intentional fallthrough
            }
            case HalOperandLifeTime::TEMPORARY_VARIABLE: // intentional fallthrough
            case HalOperandLifeTime::SUBGRAPH_OUTPUT:
            {
                // The tensor is either an operand internal to the model, or a model input.
                // It can be associated with an ArmNN output slot for an existing layer.

                // m_OutputSlotForOperand[...] can be nullptr if the previous layer could not be converted
                const uint32_t operandIndex = operation.inputs[inputIndex];
                return LayerInputHandle(true, data.m_OutputSlotForOperand[operandIndex], operandTensorInfo);
            }
            case HalOperandLifeTime::CONSTANT_COPY: // intentional fallthrough
            case HalOperandLifeTime::CONSTANT_REFERENCE:
            {
                // The tensor has an already known constant value, and can be converted into an ArmNN Constant layer.
                ConstTensorPin tensorPin = ConvertOperandToConstTensorPin<HalPolicy>(*operand, model, data);
                if (tensorPin.IsValid())
                {
                    bool isSupported = false;
                    FORWARD_LAYER_SUPPORT_FUNC(__func__,
                                               IsConstantSupported,
                                               data.m_Backends,
                                               isSupported,
                                               tensorPin.GetConstTensor().GetInfo());
                    if (!isSupported)
                    {
                        return LayerInputHandle();
                    }

                    armnn::IConnectableLayer* constantLayer =
                        data.m_Network->AddConstantLayer(tensorPin.GetConstTensor());
                    armnn::IOutputSlot& outputSlot = constantLayer->GetOutputSlot(0);
                    outputSlot.SetTensorInfo(tensorPin.GetConstTensor().GetInfo());

                    return LayerInputHandle(true, &outputSlot, operandTensorInfo);
                }
                else
                {
                    Fail("%s: invalid operand tensor", __func__);
                    return LayerInputHandle();
                }
                break;
            }
            default:
            {
                // Unsupported lifetime for an input tensor
                Fail("%s: unsupported lifetime for input tensor: %s",
                     __func__, toString(operand->lifetime).c_str());
                return LayerInputHandle();
            }
        }
    }
    catch (UnsupportedOperand<HalOperandType>& e)
    {
        Fail("%s: Operand type %s not supported in ArmnnDriver", __func__, toString(e.m_type).c_str());
        return LayerInputHandle();
    }
}
#endif

template<typename HalPolicy,
         typename HalOperation = typename HalPolicy::Operation,
         typename HalModel     = typename HalPolicy::Model>
bool SetupAndTrackLayerOutputSlot(const HalOperation& operation,
                                  uint32_t operationOutputIndex,
                                  armnn::IConnectableLayer& layer,
                                  uint32_t layerOutputIndex,
                                  const HalModel& model,
                                  ConversionData& data,
                                  const armnn::TensorInfo* overrideOutputInfo = nullptr,
                                  const std::function <void (const armnn::TensorInfo&, bool&)>& validateFunc = nullptr,
                                  const ActivationFn& activationFunction = ActivationFn::kActivationNone,
                                  bool inferOutputShapes = false)
{
    using HalOperand = typename HalPolicy::Operand;

    const HalOperand* outputOperand = GetOutputOperand<HalPolicy>(operation, operationOutputIndex, model);
    if ((outputOperand == nullptr) || (operationOutputIndex >= layer.GetNumOutputSlots()))
    {
        return false;
    }

    armnn::IOutputSlot& outputSlot = layer.GetOutputSlot(layerOutputIndex);
    if (overrideOutputInfo == nullptr)
    {
        outputSlot.SetTensorInfo(GetTensorInfoForOperand(*outputOperand));
    }
    else
    {
        outputSlot.SetTensorInfo(*overrideOutputInfo);
    }

    bool isSupported = false;
    if (validateFunc && (IsDynamicTensor(outputSlot.GetTensorInfo()) || inferOutputShapes))
    {
        // Type one dynamic tensors require the previous layer's output shape for inference
        for (unsigned int inputSlotIndex = 0; inputSlotIndex < layer.GetNumInputSlots(); ++inputSlotIndex)
        {
            if(!layer.GetInputSlot(inputSlotIndex).GetConnection())
            {
                return false;
            }
        }
        // IsTensorInfoSet will infer the dynamic output shape
        outputSlot.IsTensorInfoSet();
        // Once the shape is inferred we can validate it
        validateFunc(outputSlot.GetTensorInfo(), isSupported);

        if(!isSupported)
        {
            for (unsigned int inputSlotIndex = 0; inputSlotIndex < layer.GetNumInputSlots(); ++inputSlotIndex)
            {
                layer.GetInputSlot(inputSlotIndex).GetConnection()->Disconnect(layer.GetInputSlot(inputSlotIndex));
            }
            return false;
        }
    }

    const uint32_t operandIndex = operation.outputs[operationOutputIndex];

    if (activationFunction != ActivationFn::kActivationNone)
    {
        const armnn::TensorInfo& activationOutputInfo = outputSlot.GetTensorInfo();
        armnn::IConnectableLayer* const endLayer = ProcessActivation(activationOutputInfo, activationFunction,
                                                                     &layer, data);

        if (!endLayer)
        {
            return Fail("%s: ProcessActivation failed", __func__);
        }

        armnn::IOutputSlot& activationOutputSlot = endLayer->GetOutputSlot(layerOutputIndex);
        data.m_OutputSlotForOperand[operandIndex] = &activationOutputSlot;
    }
    else
    {
        data.m_OutputSlotForOperand[operandIndex] = &outputSlot;
    }

    return true;
}

template<typename HalPolicy,
         typename HalOperation = typename HalPolicy::Operation,
         typename HalModel     = typename HalPolicy::Model>
armnn::DataLayout OptionalDataLayout(const HalOperation& operation,
                        uint32_t inputIndex,
                        const HalModel& model,
                        ConversionData& data)
{
    using HalOperand = typename HalPolicy::Operand;

    const HalOperand* operand = GetInputOperand<HalPolicy>(operation, inputIndex, model);
    if (!operand)
    {
        return armnn::DataLayout::NHWC;
    }

    if (!IsBool(*operand))
    {
        return armnn::DataLayout::NHWC;
    }

    const void* valueAddress = GetOperandValueReadOnlyAddress<HalPolicy>(*operand, model, data);
    if (!valueAddress)
    {
        return armnn::DataLayout::NHWC;
    }

    if (*(static_cast<const bool*>(valueAddress)))
    {
        return armnn::DataLayout::NCHW;
    }
    else
    {
        return armnn::DataLayout::NHWC;
    }
}

template<typename HalPolicy,
         typename HalOperation = typename HalPolicy::Operation,
         typename HalModel     = typename HalPolicy::Model>
bool SetupAndTrackLayerOutputSlot(const HalOperation& operation,
                                  uint32_t outputIndex,
                                  armnn::IConnectableLayer& layer,
                                  const HalModel& model,
                                  ConversionData& data,
                                  const armnn::TensorInfo* overrideOutputInfo = nullptr,
                                  const std::function <void (const armnn::TensorInfo&, bool&)>& validateFunc = nullptr,
                                  const ActivationFn& activationFunction = ActivationFn::kActivationNone)
{
    return SetupAndTrackLayerOutputSlot<HalPolicy>(operation,
                                                   outputIndex,
                                                   layer,
                                                   outputIndex,
                                                   model,
                                                   data,
                                                   overrideOutputInfo,
                                                   validateFunc,
                                                   activationFunction);
}

template<typename HalPolicy,
         typename HalOperation = typename HalPolicy::Operation,
         typename HalModel     = typename HalPolicy::Model>
bool ConvertToActivation(const HalOperation& operation,
                         const char* operationName,
                         const armnn::ActivationDescriptor& activationDesc,
                         const HalModel& model,
                         ConversionData& data)
{
    using HalOperand = typename HalPolicy::Operand;

    LayerInputHandle input = ConvertToLayerInputHandle<HalPolicy>(operation, 0, model, data);
    if (!input.IsValid())
    {
        return Fail("%s: Input 0 is invalid", operationName);
    }

    const HalOperand* outputOperand = GetOutputOperand<HalPolicy>(operation, 0, model);
    if (!outputOperand)
    {
        return false;
    }

    const armnn::TensorInfo& outInfo = GetTensorInfoForOperand(*outputOperand);

    bool isSupported = false;

    auto validateFunc = [&](const armnn::TensorInfo& outInfo, bool& isSupported)
    {
        FORWARD_LAYER_SUPPORT_FUNC(__func__,
                                   IsActivationSupported,
                                   data.m_Backends,
                                   isSupported,
                                   input.GetTensorInfo(),
                                   outInfo,
                                   activationDesc);
    };

    if(IsDynamicTensor(outInfo))
    {
        isSupported = AreDynamicTensorsSupported();
    }
    else
    {
        validateFunc(outInfo, isSupported);
    }

    if (!isSupported)
    {
        return false;
    }

    armnn::IConnectableLayer* layer = data.m_Network->AddActivationLayer(activationDesc);
    ARMNN_ASSERT(layer != nullptr);
    input.Connect(layer->GetInputSlot(0));

    return SetupAndTrackLayerOutputSlot<HalPolicy>(operation, 0, *layer, model, data, nullptr, validateFunc);
}

template<typename HalPolicy,
    typename HalOperation = typename HalPolicy::Operation,
    typename HalModel     = typename HalPolicy::Model>
bool ConvertReLu(const HalOperation& operation, const HalModel& model, ConversionData& data)
{
    armnn::ActivationDescriptor desc;
    desc.m_Function = armnn::ActivationFunction::ReLu;

    return ConvertToActivation<HalPolicy>(operation, __func__, desc, model, data);
}

template<typename HalPolicy,
    typename HalOperation = typename HalPolicy::Operation,
    typename HalModel     = typename HalPolicy::Model>
bool ConvertReLu1(const HalOperation& operation, const HalModel& model, ConversionData& data)
{
    armnn::ActivationDescriptor desc;
    desc.m_Function = armnn::ActivationFunction::BoundedReLu;
    desc.m_A        = 1.0f;
    desc.m_B        = -1.0f;

    return ConvertToActivation<HalPolicy>(operation, __func__, desc, model, data);
}

template<typename HalPolicy,
    typename HalOperation = typename HalPolicy::Operation,
    typename HalModel     = typename HalPolicy::Model>
bool ConvertReLu6(const HalOperation& operation, const HalModel& model, ConversionData& data)
{
    armnn::ActivationDescriptor desc;
    desc.m_Function = armnn::ActivationFunction::BoundedReLu;
    desc.m_A        = 6.0f;

    return ConvertToActivation<HalPolicy>(operation, __func__, desc, model, data);
}

template<typename HalPolicy,
    typename HalOperation = typename HalPolicy::Operation,
    typename HalModel     = typename HalPolicy::Model>
bool ConvertTanH(const HalOperation& operation, const HalModel& model, ConversionData& data)
{
    armnn::ActivationDescriptor desc;
    desc.m_Function = armnn::ActivationFunction::TanH;
    desc.m_A = 1.0f; // android nn does not support tanH parameters
    desc.m_B = 1.0f; // set to 1.0f for unity scaling

    return ConvertToActivation<HalPolicy>(operation, __func__, desc, model, data);
}

template<typename HalPolicy,
         typename HalOperation   = typename HalPolicy::Operation,
         typename HalModel       = typename HalPolicy::Model>
bool ConvertPaddings(const HalOperation& operation,
                     const HalModel& model,
                     ConversionData& data,
                     unsigned int rank,
                     armnn::PadDescriptor& padDescriptor)
{
    using HalOperand = typename HalPolicy::Operand;

    const HalOperand* paddingsOperand = GetInputOperand<HalPolicy>(operation, 1, model);
    if (!paddingsOperand)
    {
        return Fail("%s: Could not read paddings operand", __func__);
    }

    armnn::TensorShape paddingsOperandShape = GetTensorShapeForOperand(*paddingsOperand);
    if (paddingsOperandShape.GetNumDimensions() != 2 || paddingsOperandShape.GetNumElements() != rank * 2)
    {
        return Fail("%s: Operation has invalid paddings operand: expected shape [%d, 2]",  __func__, rank);
    }

    std::vector<int32_t> paddings;
    if (!GetTensorInt32Values<HalPolicy>(*paddingsOperand, paddings, model, data))
    {
        return Fail("%s: Operation has invalid or unsupported paddings operand", __func__);
    }

    // add padding for each dimension of input tensor.
    for (unsigned int i = 0; i < paddings.size() - 1; i += 2)
    {
        int paddingBeforeInput = paddings[i];
        int paddingAfterInput  = paddings[i + 1];

        if (paddingBeforeInput < 0 || paddingAfterInput < 0)
        {
            return Fail("%s: Operation has invalid paddings operand, invalid padding values.",  __func__);
        }

        padDescriptor.m_PadList.emplace_back((unsigned int) paddingBeforeInput, (unsigned int) paddingAfterInput);
    }

    return true;
}

template<typename HalPolicy,
         typename HalOperation   = typename HalPolicy::Operation,
         typename HalModel       = typename HalPolicy::Model>
bool ConvertPooling2d(const HalOperation& operation,
                      const char* operationName,
                      armnn::PoolingAlgorithm poolType,
                      const HalModel& model,
                      ConversionData& data)
{
    using HalOperand     = typename HalPolicy::Operand;
    using HalOperandType = typename HalPolicy::OperandType;

    LayerInputHandle input = ConvertToLayerInputHandle<HalPolicy>(operation, 0, model, data);
    if (!input.IsValid())
    {
        return Fail("%s: Operation Could not read input 0", operationName);
    }

    const HalOperand* output = GetOutputOperand<HalPolicy>(operation, 0, model);
    if (!output)
    {
        return Fail("%s: Could not read output 0", __func__);
    }

    const armnn::TensorInfo& inputInfo  = input.GetTensorInfo();
    const armnn::TensorInfo& outputInfo = GetTensorInfoForOperand(*output);

    armnn::Pooling2dDescriptor desc;
    desc.m_PoolType = poolType;
    desc.m_OutputShapeRounding = armnn::OutputShapeRounding::Floor;
    desc.m_DataLayout = armnn::DataLayout::NHWC;

    ActivationFn activation;

    auto inputSize = operation.inputs.size();

    if (inputSize >= 10)
    {
        // one input, 9 parameters (padding l r t b, stridex, stridey, width, height, activation type)
        if (!GetInputScalar<HalPolicy>(operation, 1, HalOperandType::INT32, desc.m_PadLeft, model, data) ||
            !GetInputScalar<HalPolicy>(operation, 2, HalOperandType::INT32, desc.m_PadRight, model, data) ||
            !GetInputScalar<HalPolicy>(operation, 3, HalOperandType::INT32, desc.m_PadTop, model, data) ||
            !GetInputScalar<HalPolicy>(operation, 4, HalOperandType::INT32, desc.m_PadBottom, model, data) ||
            !GetInputScalar<HalPolicy>(operation, 5, HalOperandType::INT32, desc.m_StrideX, model, data) ||
            !GetInputScalar<HalPolicy>(operation, 6, HalOperandType::INT32, desc.m_StrideY, model, data) ||
            !GetInputScalar<HalPolicy>(operation, 7, HalOperandType::INT32, desc.m_PoolWidth, model, data) ||
            !GetInputScalar<HalPolicy>(operation, 8, HalOperandType::INT32, desc.m_PoolHeight, model, data) ||
            !GetInputActivationFunction<HalPolicy>(operation, 9, activation, model, data))
        {
            return Fail("%s: Operation has invalid inputs", operationName);
        }

        if (Is12OrLaterOperand(*output))
        {
            desc.m_DataLayout = OptionalDataLayout<HalPolicy>(operation, 10, model, data);
        }
    }
    else
    {
        // one input, 6 parameters (padding, stridex, stridey, width, height, activation type)
        android::nn::PaddingScheme scheme;
        if (!GetInputPaddingScheme<HalPolicy>(operation, 1, scheme, model, data) ||
            !GetInputScalar<HalPolicy>(operation, 2, HalOperandType::INT32, desc.m_StrideX, model, data) ||
            !GetInputScalar<HalPolicy>(operation, 3, HalOperandType::INT32, desc.m_StrideY, model, data) ||
            !GetInputScalar<HalPolicy>(operation, 4, HalOperandType::INT32, desc.m_PoolWidth, model, data) ||
            !GetInputScalar<HalPolicy>(operation, 5, HalOperandType::INT32, desc.m_PoolHeight, model, data) ||
            !GetInputActivationFunction<HalPolicy>(operation, 6, activation, model, data))
        {
            return Fail("%s: Operation has invalid inputs", operationName);
        }

        if (Is12OrLaterOperand(*output))
        {
            desc.m_DataLayout = OptionalDataLayout<HalPolicy>(operation, 7, model, data);
        }

        const armnnUtils::DataLayoutIndexed dataLayout(desc.m_DataLayout);
        const unsigned int inputWidth  = inputInfo.GetShape()[dataLayout.GetWidthIndex()];
        const unsigned int inputHeight = inputInfo.GetShape()[dataLayout.GetHeightIndex()];

        CalcPadding(inputWidth, desc.m_PoolWidth, desc.m_StrideX, desc.m_PadLeft, desc.m_PadRight, scheme);
        CalcPadding(inputHeight, desc.m_PoolHeight, desc.m_StrideY, desc.m_PadTop, desc.m_PadBottom, scheme);
    }

    bool isSupported = false;

    auto validateFunc = [&](const armnn::TensorInfo& outputInfo, bool& isSupported)
    {
        FORWARD_LAYER_SUPPORT_FUNC(__func__,
                                   IsPooling2dSupported,
                                   data.m_Backends,
                                   isSupported,
                                   inputInfo,
                                   outputInfo,
                                   desc);

    };

    if(IsDynamicTensor(outputInfo))
    {
        isSupported = AreDynamicTensorsSupported();
    }
    else
    {
        validateFunc(outputInfo, isSupported);
    }

    if (!isSupported)
    {
        return false;
    }

    armnn::IConnectableLayer* pooling2dLayer = data.m_Network->AddPooling2dLayer(desc);
    if (!pooling2dLayer)
    {
        return Fail("%s: AddPooling2dLayer failed", __func__);
    }

    input.Connect(pooling2dLayer->GetInputSlot(0));

    if (!isSupported)
    {
        return false;
    }

    return SetupAndTrackLayerOutputSlot<HalPolicy>(operation, 0, *pooling2dLayer, model,
                                                   data, nullptr, validateFunc, activation);
}

template<typename HalPolicy,
         typename HalOperation = typename HalPolicy::Operation,
         typename HalModel     = typename HalPolicy::Model>
bool ConvertAdd(const HalOperation& operation, const HalModel& model, ConversionData& data)
{
    using HalOperand = typename HalPolicy::Operand;

    LayerInputHandle input0 = ConvertToLayerInputHandle<HalPolicy>(operation, 0, model, data);
    LayerInputHandle input1 = ConvertToLayerInputHandle<HalPolicy>(operation, 1, model, data);

    if (!input0.IsValid() || !input1.IsValid())
    {
        return Fail("%s: Operation has invalid inputs", __func__);
    }

    // The FuseActivation parameter is always the input index 2
    // and it should be optional
    ActivationFn activationFunction;
    if (!GetOptionalInputActivation<HalPolicy>(operation, 2, activationFunction, model, data))
    {
        return Fail("%s: Operation has invalid inputs", __func__);
    }

    const HalOperand* outputOperand = GetOutputOperand<HalPolicy>(operation, 0, model);
    if (!outputOperand)
    {
        return false;
    }

    const armnn::TensorInfo& inputInfo0 = input0.GetTensorInfo();
    const armnn::TensorInfo& inputInfo1 = input1.GetTensorInfo();

    const armnn::TensorInfo& outputInfo = GetTensorInfoForOperand(*outputOperand);

    bool isSupported = false;
    auto validateFunc = [&](const armnn::TensorInfo& outputInfo, bool& isSupported)
    {
        FORWARD_LAYER_SUPPORT_FUNC(__func__,
                                   IsAdditionSupported,
                                   data.m_Backends,
                                   isSupported,
                                   inputInfo0,
                                   inputInfo1,
                                   outputInfo);
    };

    if(!IsDynamicTensor(outputInfo))
    {
        validateFunc(outputInfo, isSupported);
    }
    else
    {
        isSupported = AreDynamicTensorsSupported();
    }

    if (!isSupported)
    {
        return false;
    }

    armnn::IConnectableLayer* const startLayer = data.m_Network->AddAdditionLayer();

    bool isReshapeSupported = BroadcastTensor(input0, input1, startLayer, data);
    if (!isReshapeSupported)
    {
        return false;
    }

    return SetupAndTrackLayerOutputSlot<HalPolicy>(operation, 0, *startLayer, model,
                                                   data, nullptr, validateFunc, activationFunction);

}

template<typename HalPolicy,
         typename HalOperation = typename HalPolicy::Operation,
         typename HalModel     = typename HalPolicy::Model>
bool ConvertArgMinMax(const HalOperation& operation,
                      const HalModel& model,
                      ConversionData& data,
                      armnn::ArgMinMaxFunction argMinMaxFunction)
{
    ALOGV("argMinMaxFunction = %s", GetArgMinMaxFunctionAsCString(argMinMaxFunction));

    using HalOperand     = typename HalPolicy::Operand;
    using HalOperandType = typename HalPolicy::OperandType;

    LayerInputHandle input0 = ConvertToLayerInputHandle<HalPolicy>(operation, 0, model, data);

    if (!input0.IsValid())
    {
        return Fail("%s: Operation has invalid inputs", __func__);
    }

    int32_t axis;
    if (!GetInputScalar<HalPolicy>(operation, 1, HalOperandType::INT32, axis, model, data))
    {
        return Fail("%s: Operation has invalid inputs. Failed to read axis.", __func__);
    }

    const armnn::TensorInfo& inputInfo = input0.GetTensorInfo();
    int rank = static_cast<int>(inputInfo.GetNumDimensions());

    if (((axis < -rank) && (axis < 0)) || ((axis >= rank) && (axis > 0)))
    {
        // Square bracket denotes inclusive n while parenthesis denotes exclusive n
        // E.g. Rank 4 tensor can have axis in range [-4, 3)
        // -1 == 3, -2 == 2, -3 == 1, -4 == 0
        return Fail("%s: Axis must be in range [-n, n)", __func__);
    }

    const HalOperand* output = GetOutputOperand<HalPolicy>(operation, 0, model);
    if (!output)
    {
        return Fail("%s: Could not read output 0", __func__);
    }

    const armnn::TensorInfo& inputInfo0 = input0.GetTensorInfo();

    const armnn::TensorInfo& outputInfo = GetTensorInfoForOperand(*output);

    armnn::ArgMinMaxDescriptor descriptor;
    descriptor.m_Function = argMinMaxFunction;
    descriptor.m_Axis     = axis;

    bool isSupported = false;

    auto validateFunc = [&](const armnn::TensorInfo& outputInfo, bool& isSupported)
    {
        FORWARD_LAYER_SUPPORT_FUNC(__func__,
                                   IsArgMinMaxSupported,
                                   data.m_Backends,
                                   isSupported,
                                   inputInfo0,
                                   outputInfo,
                                   descriptor);
    };

    if(IsDynamicTensor(outputInfo))
    {
        isSupported = AreDynamicTensorsSupported();
    }
    else
    {
        validateFunc(outputInfo, isSupported);
    }

    if (!isSupported)
    {
        return false;
    }

    armnn::IConnectableLayer* layer = data.m_Network->AddArgMinMaxLayer(descriptor);
    assert(layer != nullptr);

    input0.Connect(layer->GetInputSlot(0));

    return SetupAndTrackLayerOutputSlot<HalPolicy>(operation, 0, *layer, model, data, nullptr, validateFunc);
}

template<typename HalPolicy,
         typename HalOperation = typename HalPolicy::Operation,
         typename HalModel     = typename HalPolicy::Model>
bool ConvertConcatenation(const HalOperation& operation, const HalModel& model, ConversionData& data)
{
    using HalOperand = typename HalPolicy::Operand;
    using HalOperandType = typename HalPolicy::OperandType;

    // The first N (0..N-1) inputs are tensors. The Nth input is the concatenation axis.
    if (operation.inputs.size() <= 1)
    {
        return Fail("%s: Operation has insufficient arguments", __func__);
    }

    // Get inputs and outputs
    const std::size_t numInputTensors = operation.inputs.size() - 1;

    int32_t concatDim;
    if (!GetInputScalar<HalPolicy>(operation, numInputTensors, HalOperandType::INT32, concatDim, model, data))
    {
        return Fail("%s: Operation has invalid inputs", __func__);
    }

    const HalOperand* outputOperand = GetOutputOperand<HalPolicy>(operation, 0, model);
    if (!outputOperand)
    {
        return Fail("%s: Operation has no outputs", __func__);
    }

    armnn::TensorInfo  outputInfo      = GetTensorInfoForOperand(*outputOperand);
    armnn::TensorShape outputShape     = outputInfo.GetShape();
    const bool         isDynamicTensor = IsDynamicTensor(outputInfo);
    //
    // handle negative concat dims along the lines of tensorflow as described here:
    //    https://www.tensorflow.org/api_docs/python/tf/concat
    // "negative axis refers to axis + rank(values)-th dimension"
    //
    if (concatDim < 0)
    {
        concatDim += outputShape.GetNumDimensions();
    }

    if (concatDim >= static_cast<int32_t>(outputShape.GetNumDimensions()) || concatDim < 0)
    {
        return Fail("%s: Operation has invalid concat axis: %d", __func__, concatDim);
    }

    std::vector<LayerInputHandle>   inputHandles;
    std::vector<armnn::TensorShape> inputShapes;

    inputHandles.reserve(numInputTensors);
    inputShapes.reserve(numInputTensors);

    bool          inputsHaveBeenReshaped = false;
    unsigned int  tensorDimensionsAdded  = 0;
    for (uint32_t i = 0; i < numInputTensors; ++i)
    {
        const HalOperand* operand = GetInputOperand<HalPolicy>(operation, i, model);
        if (!operand)
        {
            return Fail("%s: Operation has invalid inputs", __func__);
        }

        LayerInputHandle operandInputHandle = ConvertToLayerInputHandle<HalPolicy>(operation, i, model, data);
        if (!operandInputHandle.IsValid())
        {
            return Fail("%s: Operation has invalid inputs", __func__);
        }

        armnn::TensorShape operandShape = GetTensorShapeForOperand(*operand);
        if (operandShape.GetNumDimensions() == 0)
        {
            return Fail("%s: Operands with rank 0 are not supported", __func__);
        }

        if (RequiresReshape(operandShape))
        {
            inputsHaveBeenReshaped = true;

            armnn::TensorInfo reshapeInfo = operandInputHandle.GetTensorInfo();

            // Expand the tensor to three dimensions
            if (operandShape.GetNumDimensions() == 2)
            {
                reshapeInfo.SetShape(armnn::TensorShape({1, operandShape[0], operandShape[1]}));
                tensorDimensionsAdded = 1;
            }
            else
            {
                reshapeInfo.SetShape(armnn::TensorShape({1, 1, operandShape[0]}));
                tensorDimensionsAdded = 2;
            }

            armnn::ReshapeDescriptor reshapeDescriptor;
            reshapeDescriptor.m_TargetShape = reshapeInfo.GetShape();

            bool isSupported = false;
            FORWARD_LAYER_SUPPORT_FUNC(__func__,
                                       IsReshapeSupported,
                                       data.m_Backends,
                                       isSupported,
                                       operandInputHandle.GetTensorInfo(),
                                       reshapeInfo,
                                       reshapeDescriptor);

            if (!isSupported)
            {
                return false;
            }
            armnn::IConnectableLayer& newReshape = AddReshapeLayer(*data.m_Network, operandInputHandle, reshapeInfo);

            // Point to the reshape operation rather then the input operation
            operandShape       = reshapeInfo.GetShape();
            operandInputHandle = LayerInputHandle(true, &newReshape.GetOutputSlot(0), reshapeInfo);
        }

        inputShapes.emplace_back(operandShape);
        inputHandles.emplace_back(operandInputHandle);

        if (!inputHandles.back().IsValid())
        {
            return Fail("%s: Operation has invalid inputs", __func__);
        }
    }

    ARMNN_ASSERT(inputShapes.size() == inputHandles.size());

    if (inputsHaveBeenReshaped)
    {
        // Adjust the concatenation dimension by the amount of dimensions added (if any)
        concatDim += tensorDimensionsAdded;

        // Add extra dimensions to the output shape to reflect the addition of the reshape layers
        if (tensorDimensionsAdded == 1)
        {
            if (IsDynamicTensor(outputInfo))
            {
                outputShape = armnn::TensorShape({1, 0, 0}, {true, false, false});
            }
            else
            {
                outputShape = armnn::TensorShape({1, outputShape[0], outputShape[1]});
            }
        }
        else if (tensorDimensionsAdded == 2)
        {
            if (IsDynamicTensor(outputInfo))
            {
                outputShape = armnn::TensorShape({1, 1, 0}, {true, true, false});
            }
            else
            {
                outputShape = armnn::TensorShape({1, 1, outputShape[0]});
            }
        }
    }

    // Check if permutations is required and get the pair of permutations required for the concatenation.
    // Permutation is required when the concat dimension is 2 for a 4D tensor or 1 for a 3D tensor.
    std::pair<armnn::PermutationVector, armnn::PermutationVector> permutationPair =
        std::make_pair(IdentityPermutation4D, IdentityPermutation4D);
    bool needPermute = CreateConcatPermutationParameters(inputShapes[0].GetNumDimensions(),
                                                         concatDim,
                                                         permutationPair);

    // Only relevant to static tensors as dynamic output tensors will be transposed as a result of inferring from input
    if (!isDynamicTensor)
    {
        if (needPermute)
        {
            outputShape = armnnUtils::TransposeTensorShape(outputShape, permutationPair.first);
        }

        outputInfo.SetShape(outputShape);
    }
    // this is no-op for identity swizzles, otherwise it replaces both
    // the handles and shapes with the swizzled layer output handles and shapes
    if (!TransposeInputTensors(data, inputHandles, inputShapes, permutationPair.first))
    {
        return false;
    }

    // Create an armnn concat layer descriptor - this will also perform validation on the input shapes
    armnn::OriginsDescriptor concatDescriptor;

    try
    {
        // The concat descriptor is always created across the only supported concat dimension
        // which is 0, 1 or 3 for a 4-D tensor, or 0 or 2 for a 3-D tensor.
        concatDescriptor = armnn::CreateDescriptorForConcatenation(inputShapes.begin(),
                                                                   inputShapes.end(),
                                                                   concatDim);
    } catch (std::exception& error)
    {
        return Fail("%s: Error preparing concat descriptor. %s", __func__, error.what());
    }

    // Validate the output shape is correct given the input shapes based on the
    // only valid concat dimension which is 0, 1 or 3 for a 4-D tensor, or 0 or 2 for a 3-D tensor.
    if (!isDynamicTensor)
    {
        if (!ValidateConcatOutputShape(inputShapes, outputShape, concatDim))
        {
            return Fail("%s: Error validating the output shape for concat", __func__);
        }
    }

    std::vector<const armnn::TensorInfo*> inputTensorInfos;
    std::transform(inputHandles.begin(), inputHandles.end(), std::back_inserter(inputTensorInfos),
                   [](const LayerInputHandle& h)->const armnn::TensorInfo*{ return &h.GetTensorInfo(); });

    bool isSupported  = false;
    auto validateFunc = [&](const armnn::TensorInfo& outputInfo, bool& isSupported){
        FORWARD_LAYER_SUPPORT_FUNC(__func__, IsConcatSupported, data.m_Backends, isSupported, inputTensorInfos,
                                   outputInfo, concatDescriptor);
    };

    if (!isDynamicTensor)
    {
        validateFunc(outputInfo, isSupported);
    }
    else
    {
        isSupported = AreDynamicTensorsSupported();
    }

    if (!isSupported)
    {
        return false;
    }

    armnn::IConnectableLayer* layer = data.m_Network->AddConcatLayer(concatDescriptor);
    assert(layer != nullptr);
    layer->GetOutputSlot(0).SetTensorInfo(outputInfo);
    // Connect inputs to the layer
    const int numInputSlots = layer->GetNumInputSlots();
    assert(static_cast<std::size_t>(numInputSlots) == inputHandles.size());
    for (int i = 0; i < numInputSlots; ++i)
    {
        // connect the input directly to the merge (concat) layer
        inputHandles[static_cast<unsigned int>(i)].Connect(layer->GetInputSlot(i));
    }

    // Transpose the output shape
    auto transposeOutputShape = [&](){
        armnn::TransposeDescriptor transposeDesc;
        transposeDesc.m_DimMappings = permutationPair.second;
        armnn::TensorInfo inputTransposeInfo  = layer->GetOutputSlot(0).GetTensorInfo();
        armnn::TensorInfo outputTransposeInfo = armnnUtils::TransposeTensorShape(inputTransposeInfo,
                                                                                 permutationPair.second);
        isSupported = false;
        FORWARD_LAYER_SUPPORT_FUNC(__func__,
                                   IsTransposeSupported,
                                   data.m_Backends,
                                   isSupported,
                                   inputTransposeInfo,
                                   outputTransposeInfo,
                                   transposeDesc);
        if (!isSupported)
        {
            return false;
        }
        // Add permutation layer and connect the output to it, the permutation becomes the output layer
        armnn::IConnectableLayer& deswizzleLayer = AddTransposeLayer(*data.m_Network, layer->GetOutputSlot(0),
                                                                     permutationPair.second);
        layer = &deswizzleLayer;

        return true;
    };

    if (needPermute && !isDynamicTensor)
    {
        transposeOutputShape();
    }

    if (inputsHaveBeenReshaped)
    {
        if (isDynamicTensor)
        {
            // Infer the output shapes of concat if outputs are type 1 dynamic
            ARMNN_ASSERT(layer->GetOutputSlot(0).IsTensorInfoSet());
            if (!ValidateConcatOutputShape(inputShapes,
                                           layer->GetOutputSlot(0).GetTensorInfo().GetShape(),
                                           concatDim))
            {
                return Fail("%s: Error validating the output shape for concat", __func__);
            }
            transposeOutputShape();
        }

        armnn::TensorInfo afterConcatInfo = layer->GetOutputSlot(0).GetTensorInfo();
        // Undo the reshape knowing the amount of dimensions added
        if (tensorDimensionsAdded == 1)
        {
            afterConcatInfo.SetShape(
                armnn::TensorShape({afterConcatInfo.GetShape()[1], afterConcatInfo.GetShape()[2]}));
        }
        else if (tensorDimensionsAdded == 2)
        {
            afterConcatInfo.SetShape(armnn::TensorShape({afterConcatInfo.GetShape()[2]}));
        }

        armnn::ReshapeDescriptor reshapeDescriptor;
        reshapeDescriptor.m_TargetShape = afterConcatInfo.GetShape();
        armnn::TensorInfo concatInfo = layer->GetOutputSlot(0).GetTensorInfo();

        isSupported = false;
        auto validateReshapeFunc = [&](const armnn::TensorInfo& afterConcatInfo, bool& isSupported){
            FORWARD_LAYER_SUPPORT_FUNC(__func__,
                                       IsReshapeSupported,
                                       data.m_Backends,
                                       isSupported,
                                       concatInfo,
                                       afterConcatInfo,
                                       reshapeDescriptor);
        };

        if (!IsDynamicTensor(afterConcatInfo))
        {
            validateReshapeFunc(afterConcatInfo, isSupported);
        }
        else
        {
            isSupported = AreDynamicTensorsSupported();
        }

        if (!isSupported)
        {
            return false;
        }
        layer = &AddReshapeLayer(*data.m_Network, layer->GetOutputSlot(0), afterConcatInfo);
        return SetupAndTrackLayerOutputSlot<HalPolicy>(operation,
                                                       0,
                                                       *layer,
                                                       model,
                                                       data,
                                                       nullptr,
                                                       validateReshapeFunc);
    }

    return SetupAndTrackLayerOutputSlot<HalPolicy>(operation, 0, *layer, model, data, nullptr, validateFunc);
}

template<typename HalPolicy,
         typename HalOperation   = typename HalPolicy::Operation,
         typename HalModel       = typename HalPolicy::Model>
bool ConvertConv2d(const HalOperation& operation, const HalModel& model, ConversionData& data)
{
    using HalOperand     = typename HalPolicy::Operand;
    using HalOperandType = typename HalPolicy::OperandType;

    LayerInputHandle input = ConvertToLayerInputHandle<HalPolicy>(operation, 0, model, data);
    if (!input.IsValid())
    {
        return Fail("%s: Operation has invalid inputs", __func__);
    }

    const HalOperand* output = GetOutputOperand<HalPolicy>(operation, 0, model);
    if (!output)
    {
        return Fail("%s: Could not read output 0", __func__);
    }

    const armnn::TensorInfo& inputInfo  = input.GetTensorInfo();
    const armnn::TensorInfo& outputInfo = GetTensorInfoForOperand(*output);

    // ArmNN does not currently support non-fixed weights or bias
    const ConstTensorPin weightsPin = ConvertOperationInputToConstTensorPin<HalPolicy>(operation, 1, model, data);
    const ConstTensorPin biasPin    = ConvertOperationInputToConstTensorPin<HalPolicy>(operation, 2, model, data);

    if (!weightsPin.IsValid() || !biasPin.IsValid())
    {
        return Fail("%s: Operation has invalid inputs", __func__);
    }

    armnn::ConstTensor weights = weightsPin.GetConstTensor();
    armnn::ConstTensor bias    = biasPin.GetConstTensor();
    SanitizeBiasQuantizationScale(bias.GetInfo(), weights.GetInfo(), inputInfo);

    armnn::Convolution2dDescriptor desc;
    desc.m_DataLayout = armnn::DataLayout::NHWC;
    ActivationFn activation;

    if (operation.inputs.size() == 10)
    {
        if (!GetInputScalar<HalPolicy>(operation, 3, HalOperandType::INT32, desc.m_PadLeft, model, data) ||
            !GetInputScalar<HalPolicy>(operation, 4, HalOperandType::INT32, desc.m_PadRight, model, data) ||
            !GetInputScalar<HalPolicy>(operation, 5, HalOperandType::INT32, desc.m_PadTop, model, data) ||
            !GetInputScalar<HalPolicy>(operation, 6, HalOperandType::INT32, desc.m_PadBottom, model, data) ||
            !GetInputScalar<HalPolicy>(operation, 7, HalOperandType::INT32, desc.m_StrideX, model, data) ||
            !GetInputScalar<HalPolicy>(operation, 8, HalOperandType::INT32, desc.m_StrideY, model, data) ||
            !GetInputActivationFunction<HalPolicy>(operation, 9, activation, model, data))
        {
            return Fail("%s: Operation has invalid inputs", __func__);
        }
    }
    else if (operation.inputs.size() == 7)
    {
        android::nn::PaddingScheme paddingScheme;
        if (!GetInputPaddingScheme<HalPolicy>(operation, 3, paddingScheme, model, data) ||
            !GetInputScalar<HalPolicy>(operation, 4, HalOperandType::INT32, desc.m_StrideX, model, data) ||
            !GetInputScalar<HalPolicy>(operation, 5, HalOperandType::INT32, desc.m_StrideY, model, data) ||
            !GetInputActivationFunction<HalPolicy>(operation, 6, activation, model, data))
        {
            return Fail("%s: Operation has invalid inputs", __func__);
        }

        const uint32_t kernelX = weights.GetShape()[2];
        const uint32_t kernelY = weights.GetShape()[1];
        const uint32_t inputX  = inputInfo.GetShape()[2];
        const uint32_t inputY  = inputInfo.GetShape()[1];

        CalcPadding(inputX, kernelX, desc.m_StrideX, desc.m_PadLeft, desc.m_PadRight, paddingScheme);
        CalcPadding(inputY, kernelY, desc.m_StrideY, desc.m_PadTop, desc.m_PadBottom, paddingScheme);
    }
    else
    {
        return Fail("%s: Unsupported number of operation inputs", __func__);
    }

    desc.m_BiasEnabled = true;
    armnn::Optional<armnn::TensorInfo> biases(bias.GetInfo());

    bool isSupported = false;
    auto validateFunc = [&](const armnn::TensorInfo& outputInfo, bool& isSupported)
    {
        FORWARD_LAYER_SUPPORT_FUNC(__func__,
                                   IsConvolution2dSupported,
                                   data.m_Backends,
                                   isSupported,
                                   inputInfo,
                                   outputInfo,
                                   desc,
                                   weights.GetInfo(),
                                   biases);
    };

    if(!IsDynamicTensor(outputInfo))
    {
        validateFunc(outputInfo, isSupported);
    }
    else
    {
        isSupported = AreDynamicTensorsSupported();
    }

    if (!isSupported)
    {
        return false;
    }

    armnn::IConnectableLayer* startLayer =
            data.m_Network->AddConvolution2dLayer(desc, weights, armnn::Optional<armnn::ConstTensor>(bias));

    if (!startLayer)
    {
        return Fail("%s: AddConvolution2dLayer failed", __func__);
    }

    input.Connect(startLayer->GetInputSlot(0));

    return SetupAndTrackLayerOutputSlot<HalPolicy>(operation, 0, *startLayer, model,
                                                   data, nullptr, validateFunc, activation);
}

template<typename HalPolicy,
         typename HalOperation   = typename HalPolicy::Operation,
         typename HalModel       = typename HalPolicy::Model>
bool ConvertDepthToSpace(const HalOperation& operation, const HalModel& model, ConversionData& data)
{
    using HalOperand     = typename HalPolicy::Operand;
    using HalOperandType = typename HalPolicy::OperandType;

    LayerInputHandle input = ConvertToLayerInputHandle<HalPolicy>(operation, 0, model, data);
    if (!input.IsValid() )
    {
        return Fail("%s: Operation has invalid inputs", __func__);
    }

    const armnn::TensorInfo& inputInfo = input.GetTensorInfo();
    unsigned int rank = inputInfo.GetNumDimensions();
    if (rank != 4)
    {
        return Fail("%s: Only inputs with rank 4 are supported", __func__);
    }

    const HalOperand* output = GetOutputOperand<HalPolicy>(operation, 0, model);
    if (!output)
    {
        return Fail("%s: Could not read output 0", __func__);
    }

    const armnn::TensorInfo& outputInfo = GetTensorInfoForOperand(*output);

    armnn::DepthToSpaceDescriptor descriptor;

    GetInputScalar<HalPolicy>(operation, 1, HalOperandType::INT32, descriptor.m_BlockSize, model, data);
    if (descriptor.m_BlockSize <= 1)
    {
        return Fail("%s: Block size must be at least 1 in all dimensions");
    }

    descriptor.m_DataLayout = armnn::DataLayout::NHWC;
    if (Is12OrLaterOperand(*output))
    {
        descriptor.m_DataLayout = OptionalDataLayout<HalPolicy>(operation, 2, model, data);
    }

    bool isSupported = false;
    auto validateFunc = [&](const armnn::TensorInfo& outputInfo, bool& isSupported)
    {
        FORWARD_LAYER_SUPPORT_FUNC(__func__,
                                   IsDepthToSpaceSupported,
                                   data.m_Backends,
                                   isSupported,
                                   inputInfo,
                                   outputInfo,
                                   descriptor);
    };

    if(!IsDynamicTensor(outputInfo))
    {
        validateFunc(outputInfo, isSupported);
    }
    else
    {
        isSupported = AreDynamicTensorsSupported();
    }

    if (!isSupported)
    {
        return false;
    }

    armnn::IConnectableLayer* const layer = data.m_Network->AddDepthToSpaceLayer(descriptor);
    assert(layer != nullptr);
    input.Connect(layer->GetInputSlot(0));

    return SetupAndTrackLayerOutputSlot<HalPolicy>(operation, 0, *layer, model, data, nullptr, validateFunc);
}

template<typename HalPolicy,
         typename HalOperation   = typename HalPolicy::Operation,
         typename HalModel       = typename HalPolicy::Model>
bool ConvertDepthwiseConv2d(const HalOperation& operation, const HalModel& model, ConversionData& data)
{
    using HalOperand     = typename HalPolicy::Operand;
    using HalOperandType = typename HalPolicy::OperandType;

    LayerInputHandle input = ConvertToLayerInputHandle<HalPolicy>(operation, 0, model, data);

    if (!input.IsValid())
    {
        return Fail("%s: Operation has invalid inputs", __func__);
    }

    const HalOperand* output = GetOutputOperand<HalPolicy>(operation, 0, model);

    if (!output)
    {
        return Fail("%s: Could not read output 0", __func__);
    }

    const armnn::TensorInfo& inputInfo  = input.GetTensorInfo();
    const armnn::TensorInfo& outputInfo = GetTensorInfoForOperand(*output);

    // ArmNN does not currently support non-fixed weights or bias
    // Find the shape of the weights tensor. In AndroidNN this will be [ 1, H, W, I * M ]
    const HalOperand* weightsOperand = GetInputOperand<HalPolicy>(operation, 1, model);

    if (weightsOperand == nullptr)
    {
        return Fail("%s: Operand is invalid", __func__);
    }
    armnn::DepthwiseConvolution2dDescriptor desc;
    desc.m_DataLayout = armnn::DataLayout::NHWC;

    // Reinterpret weight data as [ H, W, I, M ]
    armnn::TensorShape weightsShape({ weightsOperand->dimensions[1],
                                      weightsOperand->dimensions[2],
                                      inputInfo.GetShape()[3],
                                      weightsOperand->dimensions[3] / inputInfo.GetShape()[3] });

    // Swizzle weight data [ H, W, I, M ] -> [ M, I, H, W ]
    const armnn::PermutationVector HWIMToMIHW = { 2U, 3U, 1U, 0U };

    const ConstTensorPin weightsPin =
        ConvertOperationInputToConstTensorPin<HalPolicy>(operation,
                                                         1,
                                                         model,
                                                         data,
                                                         HWIMToMIHW,
                                                         &weightsShape);

    // Bias is a 1D tensor
    const ConstTensorPin biasPin = ConvertOperationInputToConstTensorPin<HalPolicy>(operation, 2, model, data);

    if (!weightsPin.IsValid() || !biasPin.IsValid())
    {
        return Fail("%s: Operation has invalid inputs", __func__);
    }

    armnn::ConstTensor weights = weightsPin.GetConstTensor();
    armnn::ConstTensor bias = biasPin.GetConstTensor();
    SanitizeBiasQuantizationScale(bias.GetInfo(), weights.GetInfo(), inputInfo);

    ActivationFn activation;

    if (operation.inputs.size() == 11)
    {
        if (!GetInputScalar<HalPolicy>(operation, 3, HalOperandType::INT32, desc.m_PadLeft, model, data) ||
            !GetInputScalar<HalPolicy>(operation, 4, HalOperandType::INT32, desc.m_PadRight, model, data) ||
            !GetInputScalar<HalPolicy>(operation, 5, HalOperandType::INT32, desc.m_PadTop, model, data) ||
            !GetInputScalar<HalPolicy>(operation, 6, HalOperandType::INT32, desc.m_PadBottom, model, data) ||
            !GetInputScalar<HalPolicy>(operation, 7, HalOperandType::INT32, desc.m_StrideX, model, data) ||
            !GetInputScalar<HalPolicy>(operation, 8, HalOperandType::INT32, desc.m_StrideY, model, data) ||
            !GetInputActivationFunction<HalPolicy>(operation,  10, activation, model, data))
        {
            return Fail("%s: Operation has invalid inputs", __func__);
        }
    }
    else if (operation.inputs.size() == 8)
    {
        android::nn::PaddingScheme paddingScheme;
        if (!GetInputPaddingScheme<HalPolicy>(operation, 3, paddingScheme, model, data) ||
            !GetInputScalar<HalPolicy>(operation, 4, HalOperandType::INT32, desc.m_StrideX, model, data) ||
            !GetInputScalar<HalPolicy>(operation, 5, HalOperandType::INT32, desc.m_StrideY, model, data) ||
            !GetInputActivationFunction<HalPolicy>(operation, 7, activation, model, data))
        {
            return Fail("%s: Operation has invalid inputs", __func__);
        }

        const uint32_t kernelX = weights.GetShape()[3];
        const uint32_t kernelY = weights.GetShape()[2];
        const uint32_t inputX  = inputInfo.GetShape()[2];
        const uint32_t inputY  = inputInfo.GetShape()[1];

        CalcPadding(inputX, kernelX, desc.m_StrideX, desc.m_PadLeft, desc.m_PadRight, paddingScheme);
        CalcPadding(inputY, kernelY, desc.m_StrideY, desc.m_PadTop, desc.m_PadBottom, paddingScheme);
    }
    else
    {
        return Fail("%s: Unsupported number of operation inputs", __func__);
    }

    desc.m_BiasEnabled = true;
    armnn::Optional<armnn::TensorInfo> biases(bias.GetInfo());

    bool isSupported = false;
    auto validateFunc = [&](const armnn::TensorInfo& outputInfo, bool& isSupported)
    {
        FORWARD_LAYER_SUPPORT_FUNC(__func__,
                                   IsDepthwiseConvolutionSupported,
                                   data.m_Backends,
                                   isSupported,
                                   inputInfo,
                                   outputInfo,
                                   desc,
                                   weights.GetInfo(),
                                   biases);
    };

    if(!IsDynamicTensor(outputInfo))
    {
        validateFunc(outputInfo, isSupported);
    }
    else
    {
        isSupported = AreDynamicTensorsSupported();
    }


    if (!isSupported)
    {
        return false;
    }

    armnn::IConnectableLayer* startLayer =
            data.m_Network->AddDepthwiseConvolution2dLayer(desc, weights, armnn::Optional<armnn::ConstTensor>(bias));
    if (!startLayer)
    {
        return Fail("%s: AddDepthwiseConvolution2dLayer failed", __func__);
    }

    input.Connect(startLayer->GetInputSlot(0));

    return SetupAndTrackLayerOutputSlot<HalPolicy>(operation, 0, *startLayer, model,
                                                   data, nullptr, validateFunc, activation);
}

template<typename HalPolicy,
         typename HalOperation = typename HalPolicy::Operation,
         typename HalModel     = typename HalPolicy::Model>
bool ConvertDequantize(const HalOperation& operation, const HalModel& model, ConversionData& data)
{
    using HalOperand = typename HalPolicy::Operand;

    LayerInputHandle input = ConvertToLayerInputHandle<HalPolicy>(operation, 0, model, data);
    if (!input.IsValid())
    {
        return Fail("%s: Operation has invalid input", __func__);
    }

    const armnn::TensorInfo& inputInfo  = input.GetTensorInfo();
    const armnn::Optional<unsigned int>& quantizationDim = inputInfo.GetQuantizationDim();
    if (quantizationDim.has_value() && quantizationDim.value() != 0)
    {
        return Fail("%s: Operation has quantization dimension different than 0", __func__);
    }

    const HalOperand* const outputOperand = GetOutputOperand<HalPolicy>(operation, 0, model);
    if (!outputOperand)
    {
        return Fail("%s: Operation has invalid outputs", __func__);
    }

    const armnn::TensorInfo& outputInfo = GetTensorInfoForOperand(*outputOperand);

    bool isSupported = false;
    auto validateFunc = [&](const armnn::TensorInfo& outputInfo, bool& isSupported)
    {
        FORWARD_LAYER_SUPPORT_FUNC(__func__,
                                   IsDequantizeSupported,
                                   data.m_Backends,
                                   isSupported,
                                   inputInfo,
                                   outputInfo);
    };

    if(IsDynamicTensor(outputInfo))
    {
        isSupported = AreDynamicTensorsSupported();
    }
    else
    {
        validateFunc(outputInfo, isSupported);
    }

    if (!isSupported)
    {
        return false;
    }

    armnn::IConnectableLayer* const layer = data.m_Network->AddDequantizeLayer();
    assert(layer != nullptr);
    input.Connect(layer->GetInputSlot(0));

    return SetupAndTrackLayerOutputSlot<HalPolicy>(operation, 0, *layer, model, data, nullptr, validateFunc);
}

template<typename HalPolicy,
         typename HalOperation = typename HalPolicy::Operation,
         typename HalModel     = typename HalPolicy::Model>
bool ConvertDiv(const HalOperation& operation, const HalModel& model, ConversionData& data)
{
    using HalOperand = typename HalPolicy::Operand;

    LayerInputHandle input0 = ConvertToLayerInputHandle<HalPolicy>(operation, 0, model, data);
    LayerInputHandle input1 = ConvertToLayerInputHandle<HalPolicy>(operation, 1, model, data);

    if (!input0.IsValid() || !input1.IsValid())
    {
        return Fail("%s: Operation has invalid inputs", __func__);
    }

    // The FuseActivation parameter is always the input index 2
    // and it should be optional
    ActivationFn activationFunction;
    if (!GetOptionalInputActivation<HalPolicy>(operation, 2, activationFunction, model, data))
    {
        return Fail("%s: Operation has invalid inputs", __func__);
    }

    const HalOperand* output = GetOutputOperand<HalPolicy>(operation, 0, model);
    if (!output)
    {
        return Fail("%s: Could not read output 0", __func__);
    }

    const armnn::TensorInfo& outputInfo = GetTensorInfoForOperand(*output);

    bool isSupported = false;
    auto validateFunc = [&](const armnn::TensorInfo& outputInfo, bool& isSupported)
    {
        FORWARD_LAYER_SUPPORT_FUNC(__func__,
                                   IsDivisionSupported,
                                   data.m_Backends,
                                   isSupported,
                                   input0.GetTensorInfo(),
                                   input1.GetTensorInfo(),
                                   outputInfo);
    };

    if(!IsDynamicTensor(outputInfo))
    {
        validateFunc(outputInfo, isSupported);
    }
    else
    {
        isSupported = AreDynamicTensorsSupported();
    }

    if (!isSupported)
    {
        return false;
    }

    armnn::IConnectableLayer* const startLayer = data.m_Network->AddDivisionLayer();

    bool isReshapeSupported = BroadcastTensor(input0, input1, startLayer, data);
    if (!isReshapeSupported)
    {
        return false;
    }

    return SetupAndTrackLayerOutputSlot<HalPolicy>(operation, 0, *startLayer, model,
                                                   data, nullptr, validateFunc, activationFunction);

}

template<typename HalPolicy,
         typename HalOperation = typename HalPolicy::Operation,
         typename HalModel     = typename HalPolicy::Model>
bool ConvertFloor(const HalOperation& operation, const HalModel& model, ConversionData& data)
{
    using HalOperand = typename HalPolicy::Operand;

    LayerInputHandle input = ConvertToLayerInputHandle<HalPolicy>(operation, 0, model, data);
    if (!input.IsValid())
    {
        return Fail("%s: Operation has invalid inputs", __func__);
    }

    const HalOperand* const outputOperand = GetOutputOperand<HalPolicy>(operation, 0, model);
    if (!outputOperand)
    {
        return Fail("%s: Operation has invalid outputs", __func__);
    }

    const armnn::TensorInfo& outputInfo = GetTensorInfoForOperand(*outputOperand);

    bool isSupported = false;
    auto validateFunc = [&](const armnn::TensorInfo& outputInfo, bool& isSupported)
    {
        FORWARD_LAYER_SUPPORT_FUNC(__func__,
                                   IsFloorSupported,
                                   data.m_Backends,
                                   isSupported,
                                   input.GetTensorInfo(),
                                   outputInfo);
    };

    if(!IsDynamicTensor(outputInfo))
    {
        validateFunc(outputInfo, isSupported);
    }
    else
    {
        isSupported = AreDynamicTensorsSupported();
    }

    if (!isSupported)
    {
        return false;
    }

    armnn::IConnectableLayer* layer = data.m_Network->AddFloorLayer();
    assert(layer != nullptr);
    input.Connect(layer->GetInputSlot(0));

    return SetupAndTrackLayerOutputSlot<HalPolicy>(operation, 0, *layer, model, data, nullptr, validateFunc);
}

inline bool IsQSymm8(const V1_0::Operand&)
{
    return false;
}

#if defined(ARMNN_ANDROID_NN_V1_2) || defined(ARMNN_ANDROID_NN_V1_3)

inline bool IsQSymm8(const V1_2::Operand& operand)
{
    return operand.type == V1_2::OperandType::TENSOR_QUANT8_SYMM;
}

#endif

#ifdef ARMNN_ANDROID_NN_V1_3

inline bool IsQSymm8(const V1_3::Operand& operand)
{
    return operand.type == V1_3::OperandType::TENSOR_QUANT8_SYMM;
}

#endif

enum class DequantizeStatus
{
    SUCCESS,
    NOT_REQUIRED,
    INVALID_OPERAND
};

using DequantizeResult = std::tuple<std::unique_ptr<float[]>, size_t, armnn::TensorInfo, DequantizeStatus>;

template<typename HalPolicy,
         typename HalOperation = typename HalPolicy::Operation,
         typename HalModel     = typename HalPolicy::Model>
DequantizeResult DequantizeIfRequired(size_t operand_index,
                                      const HalOperation& operation,
                                      const HalModel& model,
                                      const ConversionData& data)
{
    using HalOperand = typename HalPolicy::Operand;

    const HalOperand* weightsOperand = GetInputOperand<HalPolicy>(operation, operand_index, model);
    if (!weightsOperand)
    {
        return { nullptr, 0, armnn::TensorInfo(), DequantizeStatus::INVALID_OPERAND };
    }

    if (IsOperandConstant<HalPolicy>(*weightsOperand))
    {
        // Weights are already constant
        return { nullptr, 0, armnn::TensorInfo(), DequantizeStatus::NOT_REQUIRED };
    }

    const size_t weightsInputIndex = operation.inputs[operand_index];

    // The weights are a non const tensor, this indicates they might be the output of a dequantize op.
    // Iterate over the nodes and find the previous operation which should be DEQUANTIZE
    for (uint32_t operationIdx = 0; operationIdx < getMainModel(model).operations.size(); ++operationIdx)
    {
        // Search for the DEQUANTIZE op which has the operand with index equal to operandIndex
        const auto& operationIt = getMainModel(model).operations[operationIdx];
        if (operationIt.type != HalPolicy::OperationType::DEQUANTIZE)
        {
            continue;
        }

        size_t outOpIndex = weightsInputIndex + 1;
        for (size_t i = 0; outOpIndex != weightsInputIndex && i < operationIt.outputs.size(); ++i)
        {
            outOpIndex = operationIt.outputs[i];
        }

        if (outOpIndex != weightsInputIndex)
        {
            continue;
        }

        const HalOperand* operand = GetInputOperand<HalPolicy>(operationIt, 0, model);
        ARMNN_ASSERT(operand);

        if (!IsQSymm8(*operand))
        {
            // Only supporting dequantize from QSYMM8 to FLOAT
            break;
        }

        // Allocate a new buffer for the dequantized data and manually dequantize
        const void* startValue = GetOperandValueReadOnlyAddress<HalPolicy>(*operand, model, data);
        if (!startValue)
        {
            // Failed to get the operand address
            break;
        }

        const uint8_t* quantizedBuffer = reinterpret_cast<const uint8_t*>(startValue);
        size_t dequantizedBufferLength = operand->location.length;
        const float quantizationScale  = operand->scale;

        auto dequantizedBuffer = std::make_unique<float[]>(dequantizedBufferLength + 1);
        for (size_t i = 0; i < dequantizedBufferLength; ++i)
        {
            float* dstPtr = dequantizedBuffer.get();
            ARMNN_ASSERT(dstPtr);
            *dstPtr++ = quantizedBuffer[i] * quantizationScale;
        }

        // Construct tensor info for dequantized ConstTensor
        armnn::TensorInfo tensorInfo(operand->dimensions.size(),
                                     operand->dimensions.data(),
                                     armnn::DataType::Float32);

        return { std::move(dequantizedBuffer), dequantizedBufferLength * sizeof(float),
                 std::move(tensorInfo),
                 DequantizeStatus::SUCCESS };
    }

    return { nullptr, 0, armnn::TensorInfo() , DequantizeStatus::NOT_REQUIRED};
}

template<typename HalPolicy,
         typename HalOperation = typename HalPolicy::Operation,
         typename HalModel     = typename HalPolicy::Model>
ConstTensorPin DequantizeAndMakeConstTensorPin(const HalOperation& operation,
                                               const HalModel& model,
                                               const ConversionData& data,
                                               size_t operandIndex,
                                               bool optional = false)
{
    DequantizeResult dequantized = DequantizeIfRequired<HalPolicy>(operandIndex,operation, model, data);

    DequantizeStatus status = std::get<3>(dequantized);
    switch (status)
    {
        case DequantizeStatus::INVALID_OPERAND:
        {
            // return invalid const tensor pin
            return ConstTensorPin();
        }
        case DequantizeStatus::NOT_REQUIRED:
        {
            return ConvertOperationInputToConstTensorPin<HalPolicy>(
                operation, operandIndex, model, data, g_DontPermute, nullptr, optional);
        }
        case DequantizeStatus::SUCCESS:
        default:
        {
            return ConstTensorPin(
                std::get<2>(dequantized), std::get<0>(dequantized).get(), std::get<1>(dequantized), g_DontPermute);
        }
    }
}


template<typename HalPolicy,
         typename HalOperation = typename HalPolicy::Operation,
         typename HalModel     = typename HalPolicy::Model>
bool ConvertFullyConnected(const HalOperation& operation, const HalModel& model, ConversionData& data)
{
    using HalOperand = typename HalPolicy::Operand;

    LayerInputHandle input = ConvertToLayerInputHandle<HalPolicy>(operation, 0, model, data);
    if (!input.IsValid())
    {
        return Fail("%s: Operation has invalid inputs", __func__);
    }

    const HalOperand* output = GetOutputOperand<HalPolicy>(operation, 0, model);
    if (!output)
    {
        return Fail("%s: Could not read output 0", __func__);
    }

    const armnn::TensorInfo& inputInfo  = input.GetTensorInfo();
    const armnn::TensorInfo& outputInfo = GetTensorInfoForOperand(*output);

    ConstTensorPin weightsPin = DequantizeAndMakeConstTensorPin<HalPolicy>(operation, model, data, 1);
    ConstTensorPin biasPin    = ConvertOperationInputToConstTensorPin<HalPolicy>(operation, 2, model, data); // 1D

    if (!weightsPin.IsValid())
    {
        return Fail("%s: Operation has invalid weights", __func__);
    }

    if (!biasPin.IsValid())
    {
        return Fail("%s: Operation has invalid bias", __func__);
    }

    armnn::ConstTensor weights = weightsPin.GetConstTensor();
    armnn::ConstTensor bias    = biasPin.GetConstTensor();
    armnn::TensorInfo reshapedInfo = inputInfo;

    try
    {
        reshapedInfo.SetShape(FlattenFullyConnectedInput(inputInfo.GetShape(), weights.GetInfo().GetShape()));
    }
    catch (const std::exception& e)
    {
        return Fail("%s: %s", __func__, e.what());
    }

    // ensuring that the bias value is within 1% of the weights input (small float differences can exist)
    SanitizeBiasQuantizationScale(bias.GetInfo(), weights.GetInfo(), reshapedInfo);

    ActivationFn activationFunction;
    if (!GetInputActivationFunction<HalPolicy>(operation, 3, activationFunction, model, data))
    {
        return Fail("%s: Operation has invalid inputs", __func__);
    }

    armnn::FullyConnectedDescriptor desc;
    desc.m_TransposeWeightMatrix = true;
    desc.m_BiasEnabled           = true;

    bool isSupported = false;
    auto validateFunc = [&](const armnn::TensorInfo& outputInfo, bool& isSupported)
    {
        if (!VerifyFullyConnectedShapes(reshapedInfo.GetShape(),
                                        weights.GetInfo().GetShape(),
                                        outputInfo.GetShape(),
                                        desc.m_TransposeWeightMatrix))
        {
            isSupported = false;
            Fail("%s: Expected outputShape does not match actual outputShape", __func__);
            return;
        }

        FORWARD_LAYER_SUPPORT_FUNC(__func__,
                               IsFullyConnectedSupported,
                               data.m_Backends,
                               isSupported,
                               reshapedInfo,
                               outputInfo,
                               weights.GetInfo(),
                               bias.GetInfo(),
                               desc);
    };

    if(!IsDynamicTensor(outputInfo))
    {
        validateFunc(outputInfo, isSupported);
    }
    else
    {
        isSupported = AreDynamicTensorsSupported();
    }

    if (!isSupported)
    {
        return false;
    }

    armnn::IConnectableLayer* startLayer =
            data.m_Network->AddFullyConnectedLayer(desc, weights, armnn::Optional<armnn::ConstTensor>(bias));

    if (inputInfo.GetNumDimensions() > 2U)
    {
        armnn::ReshapeDescriptor reshapeDescriptor;
        reshapeDescriptor.m_TargetShape = reshapedInfo.GetShape();

        armnn::IConnectableLayer* reshapeLayer = data.m_Network->AddReshapeLayer(reshapeDescriptor);
        assert(reshapeLayer != nullptr);
        input.Connect(reshapeLayer->GetInputSlot(0));
        reshapeLayer->GetOutputSlot(0).SetTensorInfo(reshapedInfo);
        reshapeLayer->GetOutputSlot(0).Connect(startLayer->GetInputSlot(0));
    }
    else
    {
        input.Connect(startLayer->GetInputSlot(0));
    }

    return SetupAndTrackLayerOutputSlot<HalPolicy>(operation, 0, *startLayer, model,
                                                   data, nullptr, validateFunc, activationFunction);
}

template<typename HalPolicy,
         typename HalOperation = typename HalPolicy::Operation,
         typename HalModel     = typename HalPolicy::Model>
bool ConvertL2Normalization(const HalOperation& operation, const HalModel& model, ConversionData& data)
{
    using HalOperand = typename HalPolicy::Operand;

    if (operation.inputs.size() != 1)
    {
        return Fail("%s: Optional inputs are not supported", __func__);
    }

    LayerInputHandle input = ConvertToLayerInputHandle<HalPolicy>(operation, 0, model, data);
    if (!input.IsValid())
    {
        return Fail("%s: Operation has invalid inputs", __func__);
    }

    const HalOperand* output = GetOutputOperand<HalPolicy>(operation, 0, model);
    if (!output)
    {
        return Fail("%s: Could not read output 0", __func__);
    }

    const armnn::TensorInfo& inputInfo  = input.GetTensorInfo();
    const armnn::TensorInfo& outputInfo = GetTensorInfoForOperand(*output);

    if (outputInfo.GetNumDimensions() != 4u)
    {
        return Fail("%s: Tensor Rank other than 4 is not supported", __func__);
    }

    armnn::L2NormalizationDescriptor desc;
    desc.m_DataLayout = armnn::DataLayout::NHWC;

    bool isSupported = false;
    auto validateFunc = [&](const armnn::TensorInfo& outputInfo, bool& isSupported)
    {
        FORWARD_LAYER_SUPPORT_FUNC(__func__,
                                   IsL2NormalizationSupported,
                                   data.m_Backends,
                                   isSupported,
                                   inputInfo,
                                   outputInfo,
                                   desc);
    };

    if(!IsDynamicTensor(outputInfo))
    {
        validateFunc(outputInfo, isSupported);
    }
    else
    {
        isSupported = AreDynamicTensorsSupported();
    }

    if (!isSupported)
    {
        return false;
    }

    armnn::IConnectableLayer* layer = data.m_Network->AddL2NormalizationLayer(desc);
    assert(layer != nullptr);
    input.Connect(layer->GetInputSlot(0));

    return SetupAndTrackLayerOutputSlot<HalPolicy>(operation, 0, *layer, model, data, nullptr, validateFunc);
}

template<typename HalPolicy,
         typename HalOperation = typename HalPolicy::Operation,
         typename HalModel     = typename HalPolicy::Model>
bool ConvertLocalResponseNormalization(const HalOperation& operation,
                                       const HalModel& model,
                                       ConversionData& data)
{
    if (operation.inputs.size() != 5)
    {
        return Fail("%s: Optional inputs are not supported", __func__);
    }

    using HalOperand     = typename HalPolicy::Operand;
    using HalOperandType = typename HalPolicy::OperandType;

    LayerInputHandle input = ConvertToLayerInputHandle<HalPolicy>(operation, 0, model, data);
    if (!input.IsValid())
    {
        return Fail("%s: Operation has invalid inputs", __func__);
    }

    const HalOperand* output = GetOutputOperand<HalPolicy>(operation, 0, model);
    if (!output)
    {
        return Fail("%s: Could not read output 0", __func__);
    }

    const armnn::TensorInfo& inputInfo  = input.GetTensorInfo();
    const armnn::TensorInfo& outputInfo = GetTensorInfoForOperand(*output);

    if (outputInfo.GetNumDimensions() != 4u)
    {
        return Fail("%s: Tensor Rank other than 4 is not supported", __func__);
    }

    armnn::NormalizationDescriptor descriptor;
    descriptor.m_DataLayout      = armnn::DataLayout::NHWC;
    descriptor.m_NormChannelType = armnn::NormalizationAlgorithmChannel::Across;
    descriptor.m_NormMethodType  = armnn::NormalizationAlgorithmMethod::LocalBrightness;

    if (!input.IsValid() ||
        !GetInputScalar<HalPolicy>(operation, 1, HalOperandType::INT32, descriptor.m_NormSize, model, data) ||
        !GetInputFloat32<HalPolicy>(operation, 2, descriptor.m_K, model, data) ||
        !GetInputFloat32<HalPolicy>(operation, 3, descriptor.m_Alpha, model, data) ||
        !GetInputFloat32<HalPolicy>(operation, 4, descriptor.m_Beta, model, data))
    {
        return Fail("%s: Operation has invalid inputs", __func__);
    }

    // ArmNN expects normSize to be the full size of the normalization
    // window rather than the radius as in AndroidNN.
    descriptor.m_NormSize = 1 + (2 * descriptor.m_NormSize);

    bool isSupported = false;
    auto validateFunc = [&](const armnn::TensorInfo& outputInfo, bool& isSupported)
    {
        FORWARD_LAYER_SUPPORT_FUNC(__func__,
                                   IsNormalizationSupported,
                                   data.m_Backends,
                                   isSupported,
                                   inputInfo,
                                   outputInfo,
                                   descriptor);
    };

    if(!IsDynamicTensor(outputInfo))
    {
        validateFunc(outputInfo, isSupported);
    }
    else
    {
        isSupported = AreDynamicTensorsSupported();
    }

    if (!isSupported)
    {
        return false;
    }


    armnn::IConnectableLayer* layer = data.m_Network->AddNormalizationLayer(descriptor);
    assert(layer != nullptr);
    input.Connect(layer->GetInputSlot(0));

    return SetupAndTrackLayerOutputSlot<HalPolicy>(operation, 0, *layer, model, data, nullptr, validateFunc);
}

template<typename HalPolicy,
         typename HalOperation = typename HalPolicy::Operation,
         typename HalModel     = typename HalPolicy::Model>
bool ConvertLogistic(const HalOperation& operation, const HalModel& model, ConversionData& data)
{
    armnn::ActivationDescriptor desc;
    desc.m_Function = armnn::ActivationFunction::Sigmoid;

    return ConvertToActivation<HalPolicy>(operation, __func__, desc, model, data);
}

template<typename HalPolicy,
         typename HalOperation = typename HalPolicy::Operation,
         typename HalModel     = typename HalPolicy::Model>
bool ConvertMean(const HalOperation& operation, const HalModel& model, ConversionData& data)
{
    using HalOperand = typename HalPolicy::Operand;

    LayerInputHandle input = ConvertToLayerInputHandle<HalPolicy>(operation, 0, model, data);
    if (!input.IsValid())
    {
        return Fail("%s: Operation has invalid inputs", __func__);
    }

    const HalOperand* output = GetOutputOperand<HalPolicy>(operation, 0, model);
    if (!output)
    {
        return Fail("%s: Could not read output 0", __func__);
    }

    const armnn::TensorInfo& outputInfo = GetTensorInfoForOperand(*output);

    const HalOperand* axisOperand = GetInputOperand<HalPolicy>(operation, 1, model);
    if (!axisOperand)
    {
        return Fail("%s: Could not read input 1", __func__);
    }

    std::vector<int32_t> axis;
    if (!GetTensorInt32Values<HalPolicy>(*axisOperand, axis, model, data))
    {
        return Fail("%s: Input 1 has invalid values", __func__);
    }

    const armnn::TensorInfo& inputInfo = input.GetTensorInfo();

    // Convert the axis to unsigned int and remove duplicates.
    unsigned int rank = inputInfo.GetNumDimensions();
    std::set<unsigned int> uniqueAxis;
    std::transform(axis.begin(), axis.end(),
                   std::inserter(uniqueAxis, uniqueAxis.begin()),
                   [rank](int i) -> unsigned int { return (i + rank) % rank; });

    // Get the "keep dims" flag.
    int32_t keepDims = 0;
    if (!GetInputInt32<HalPolicy>(operation, 2, keepDims, model, data))
    {
        return Fail("%s: Could not read input 2", __func__);
    }

    armnn::MeanDescriptor descriptor;
    descriptor.m_Axis.assign(uniqueAxis.begin(), uniqueAxis.end());
    descriptor.m_KeepDims = keepDims > 0;

    bool isSupported = false;
    auto validateFunc = [&](const armnn::TensorInfo& outputInfo, bool& isSupported)
    {
        FORWARD_LAYER_SUPPORT_FUNC(__func__,
                                   IsMeanSupported,
                                   data.m_Backends,
                                   isSupported,
                                   inputInfo,
                                   outputInfo,
                                   descriptor);
    };

    if(!IsDynamicTensor(outputInfo))
    {
        validateFunc(outputInfo, isSupported);
    }
    else
    {
        isSupported = AreDynamicTensorsSupported();
    }

    if (!isSupported)
    {
        return false;
    }

    armnn::IConnectableLayer* const layer = data.m_Network->AddMeanLayer(descriptor);
    assert(layer != nullptr);
    input.Connect(layer->GetInputSlot(0));

    return SetupAndTrackLayerOutputSlot<HalPolicy>(operation, 0, *layer, model, data, nullptr, validateFunc);
}

template<typename HalPolicy,
         typename HalOperation = typename HalPolicy::Operation,
         typename HalModel     = typename HalPolicy::Model>
bool ConvertMul(const HalOperation& operation, const HalModel& model, ConversionData& data)
{
    using HalOperand = typename HalPolicy::Operand;

    LayerInputHandle input0 = ConvertToLayerInputHandle<HalPolicy>(operation, 0, model, data);
    LayerInputHandle input1 = ConvertToLayerInputHandle<HalPolicy>(operation, 1, model, data);

    if (!input0.IsValid() || !input1.IsValid())
    {
        return Fail("%s: Operation has invalid inputs", __func__);
    }

    // The FuseActivation parameter is always the input index 2
    // and it should be optional
    ActivationFn activationFunction;
    if (!GetOptionalInputActivation<HalPolicy>(operation, 2, activationFunction, model, data))
    {
        return Fail("%s: Operation has invalid inputs", __func__);
    }

    const HalOperand* outputOperand = GetOutputOperand<HalPolicy>(operation, 0, model);

    if (outputOperand == nullptr)
    {
        return false;
    }

    const armnn::TensorInfo& outputInfo = GetTensorInfoForOperand(*outputOperand);

    bool isSupported = false;
    auto validateFunc = [&](const armnn::TensorInfo& outputInfo, bool& isSupported)
    {
        FORWARD_LAYER_SUPPORT_FUNC(__func__,
                                   IsMultiplicationSupported,
                                   data.m_Backends,
                                   isSupported,
                                   input0.GetTensorInfo(),
                                   input1.GetTensorInfo(),
                                   outputInfo);
    };

    if(!IsDynamicTensor(outputInfo))
    {
        validateFunc(outputInfo, isSupported);
    }
    else
    {
        isSupported = AreDynamicTensorsSupported();
    }

    if (!isSupported)
    {
        return false;
    }

    armnn::IConnectableLayer* const startLayer = data.m_Network->AddMultiplicationLayer();

    bool isReshapeSupported = BroadcastTensor(input0, input1, startLayer, data);
    if (!isReshapeSupported)
    {
        return false;
    }

    return SetupAndTrackLayerOutputSlot<HalPolicy>(operation, 0, *startLayer, model,
                                                   data, nullptr, validateFunc, activationFunction);
}

template<typename HalPolicy,
         typename HalOperation = typename HalPolicy::Operation,
         typename HalModel     = typename HalPolicy::Model>
bool ConvertPad(HalOperation& operation, const HalModel& model, ConversionData& data)
{
    using HalOperand = typename HalPolicy::Operand;

    LayerInputHandle input = ConvertToLayerInputHandle<HalPolicy>(operation, 0, model, data);
    if (!input.IsValid())
    {
        return Fail("%s: Operation has invalid inputs", __func__);
    }

    const armnn::TensorInfo& inputInfo = input.GetTensorInfo();
    unsigned int rank = inputInfo.GetNumDimensions();

    armnn::PadDescriptor descriptor;
    if (!ConvertPaddings<HalPolicy>(operation, model, data, rank, descriptor))
    {
        return Fail("%s: Could not convert paddings", __func__);
    }

    // For a ANEURALNETWORKS_TENSOR_QUANT8_ASYMM and ANEURALNETWORKS_TENSOR_QUANT8_ASYMM_SIGNED tensor,
    // the scale and zeroPoint must be the same as input0
    // Before Android Q, the pad value for ANEURALNETWORKS_TENSOR_QUANT8_ASYMM was undefined. Since Android Q the pad
    // value must be "logical zero" we set it to be equal to the QuantizationOffset so effectively it ends up as
    // (QuantizationOffset - QuantizationOffset) * scale = 0.
    if (inputInfo.GetDataType() == armnn::DataType::QAsymmU8 || inputInfo.GetDataType() == armnn::DataType::QAsymmS8)
    {
        descriptor.m_PadValue = inputInfo.GetQuantizationOffset();
    }

    const HalOperand* output = GetOutputOperand<HalPolicy>(operation, 0, model);
    if (!output)
    {
        return Fail("%s: Could not read output", __func__);
    }

    const armnn::TensorInfo& outputInfo = GetTensorInfoForOperand(*output);

    bool isSupported = false;
    auto validateFunc = [&](const armnn::TensorInfo& outputInfo, bool& isSupported)
    {
        FORWARD_LAYER_SUPPORT_FUNC(__func__,
                                   IsPadSupported,
                                   data.m_Backends,
                                   isSupported,
                                   inputInfo,
                                   outputInfo,
                                   descriptor);
    };

    if(!IsDynamicTensor(outputInfo))
    {
        validateFunc(outputInfo, isSupported);
    }
    else
    {
        isSupported = AreDynamicTensorsSupported();
    }

    if (!isSupported)
    {
        return false;
    }

    armnn::IConnectableLayer* const layer = data.m_Network->AddPadLayer(descriptor);
    assert(layer != nullptr);
    input.Connect(layer->GetInputSlot(0));

    return SetupAndTrackLayerOutputSlot<HalPolicy>(operation, 0, *layer, model, data, nullptr, validateFunc);
}

template<typename HalPolicy,
         typename HalOperation = typename HalPolicy::Operation,
         typename HalModel     = typename HalPolicy::Model>
bool ConvertReshape(const HalOperation& operation, const HalModel& model, ConversionData& data)
{
    using HalOperand = typename HalPolicy::Operand;

    const HalOperand* inputOperand = GetInputOperand<HalPolicy>(operation, 0, model);
    const HalOperand* requestedShapeOperand = GetInputOperand<HalPolicy>(operation, 1, model);
    const HalOperand* outputOperand = GetOutputOperand<HalPolicy>(operation, 0, model);

    if (inputOperand == nullptr
        || requestedShapeOperand == nullptr
        || outputOperand == nullptr)
    {
        return Fail("%s: Operation has invalid inputs", __func__);
    }

    if (requestedShapeOperand->dimensions.size() != 1)
    {
        return Fail("%s: Input 1 expected to be one-dimensional (found %i dimensions)",
                    __func__, requestedShapeOperand->dimensions.size());
    }

    std::vector<int32_t> targetDimensions;
    if (!GetTensorInt32Values<HalPolicy>(*requestedShapeOperand, targetDimensions, model, data))
    {
        return Fail("%s: Could not read values of input 1", __func__);
    }

    const Shape inputOperandShape = GetOperandShape(*inputOperand);

    Shape requestedShape;
    // targetDimensions may contain special values (e.g. -1). reshapePrepare() is an AndroidNN provided utility
    // function that resolves these values into a fully specified tensor shape.
    if (!reshapePrepare(inputOperandShape, targetDimensions.data(), targetDimensions.size(), &requestedShape))
    {
        return Fail("%s: Failed to resolve the requested shape", __func__);
    }

    LayerInputHandle input = ConvertToLayerInputHandle<HalPolicy>(operation, 0, model, data);
    if (!input.IsValid())
    {
        return Fail("%s: Could not read input 0", __func__);
    }

    armnn::ReshapeDescriptor reshapeDescriptor;
    reshapeDescriptor.m_TargetShape = armnn::TensorShape(requestedShape.dimensions.size(),
                                                         requestedShape.dimensions.data());

    const armnn::TensorInfo& outputInfo = GetTensorInfoForOperand(*outputOperand);

    bool isSupported = false;
    auto validateFunc = [&](const armnn::TensorInfo& outputInfo, bool& isSupported)
    {
        FORWARD_LAYER_SUPPORT_FUNC(__func__,
                                   IsReshapeSupported,
                                   data.m_Backends,
                                   isSupported,
                                   input.GetTensorInfo(),
                                   outputInfo,
                                   reshapeDescriptor);
    };

    if(!IsDynamicTensor(outputInfo))
    {
        validateFunc(outputInfo, isSupported);
    }
    else
    {
        isSupported = AreDynamicTensorsSupported();
    }

    if (!isSupported)
    {
        return false;
    }

    armnn::IConnectableLayer* layer = data.m_Network->AddReshapeLayer(reshapeDescriptor);
    assert(layer != nullptr);
    input.Connect(layer->GetInputSlot(0));

    return SetupAndTrackLayerOutputSlot<HalPolicy>(operation, 0, *layer, model, data, nullptr, validateFunc);
}

template<typename HalPolicy,
         typename HalOperation = typename HalPolicy::Operation,
         typename HalModel     = typename HalPolicy::Model>
bool ConvertSub(const HalOperation& operation, const HalModel& model, ConversionData& data)
{
    using HalOperand = typename HalPolicy::Operand;

    LayerInputHandle input0 = ConvertToLayerInputHandle<HalPolicy>(operation, 0, model, data);
    LayerInputHandle input1 = ConvertToLayerInputHandle<HalPolicy>(operation, 1, model, data);

    if (!input0.IsValid() || !input1.IsValid())
    {
        return Fail("%s: Operation has invalid inputs", __func__);
    }

    // The FuseActivation parameter is always the input index 2
    // and it should be optional
    ActivationFn activationFunction;
    if (!GetOptionalInputActivation<HalPolicy>(operation, 2, activationFunction, model, data))
    {
        return Fail("%s: Operation has invalid inputs", __func__);
    }

    const HalOperand* output = GetOutputOperand<HalPolicy>(operation, 0, model);
    if (!output)
    {
        return Fail("%s: Could not read output 0", __func__);
    }

    const armnn::TensorInfo& outputInfo = GetTensorInfoForOperand(*output);

    bool isSupported = false;
    auto validateFunc = [&](const armnn::TensorInfo& outputInfo, bool& isSupported)
    {
        FORWARD_LAYER_SUPPORT_FUNC(__func__,
                                   IsSubtractionSupported,
                                   data.m_Backends,
                                   isSupported,
                                   input0.GetTensorInfo(),
                                   input1.GetTensorInfo(),
                                   outputInfo);
    };

    if(IsDynamicTensor(outputInfo))
    {
        isSupported = AreDynamicTensorsSupported();
    }
    else
    {
        validateFunc(outputInfo, isSupported);
    }

    if (!isSupported)
    {
        return false;
    }

    armnn::IConnectableLayer* const startLayer = data.m_Network->AddSubtractionLayer();

    bool isReshapeSupported = BroadcastTensor(input0, input1, startLayer, data);
    if (!isReshapeSupported)
    {
        return false;
    }
    return SetupAndTrackLayerOutputSlot<HalPolicy>(operation, 0, *startLayer, model,
                                                   data, nullptr, validateFunc, activationFunction);
}

template<typename HalPolicy,
         typename HalOperation = typename HalPolicy::Operation,
         typename HalModel     = typename HalPolicy::Model>
bool ConvertSqueeze(const HalOperation& operation, const HalModel& model, ConversionData& data)
{
    using HalOperand = typename HalPolicy::Operand;

    LayerInputHandle input = ConvertToLayerInputHandle<HalPolicy>(operation, 0, model, data);
    if (!input.IsValid())
    {
        return Fail("%s: Operation has invalid inputs", __func__);
    }

    const armnn::TensorInfo& inputInfo  = input.GetTensorInfo();
    unsigned int rank = inputInfo.GetNumDimensions();
    if (rank > 4)
    {
        Fail("%s: Inputs with rank greater than 4 are not supported", __func__);
    }

    const HalOperand* output = GetOutputOperand<HalPolicy>(operation, 0, model);
    if (!output)
    {
        return Fail("%s: Could not read output 0", __func__);
    }

    if (IsDynamicTensor(GetTensorInfoForOperand(*output)) && !(AreDynamicTensorsSupported()))
    {
        return Fail("%s: Dynamic output tensors are not supported", __func__);
    }

    // NOTE: Axis is an optional parameter to SQUEEZE, therefore we do not want to generate a failure
    // if the operand index is out of bounds.
    const HalOperand* axisOperand = GetInputOperand<HalPolicy>(operation, 1, model, false);

    const uint32_t dimensionSequence[] = { 0, 1, 2, 3 };

    std::vector<int32_t> axis;
    if (!axisOperand)
    {
        axis.assign(dimensionSequence,
                    dimensionSequence + rank);
    }
    else if (!GetTensorInt32Values<HalPolicy>(*axisOperand, axis, model, data))
    {
        return Fail("%s: Operation has an invalid or unsupported axis operand", __func__);
    }

    std::vector<uint32_t> outputDims;
    for (unsigned int i = 0; i < rank; i++)
    {
        bool skipSqueeze = (std::find(axis.begin(), axis.end(), i) == axis.end());
        auto currentDimension = inputInfo.GetShape()[i];
        if (skipSqueeze || currentDimension != 1)
        {
            outputDims.push_back(currentDimension);
        }
    }

    armnn::TensorShape outShape = armnn::TensorShape(outputDims.size(), outputDims.data());

    armnn::TensorInfo outputInfo = inputInfo;
    outputInfo.SetShape(outShape);

    armnn::ReshapeDescriptor reshapeDesc;
    reshapeDesc.m_TargetShape = outputInfo.GetShape();

    bool isSupported = false;
    FORWARD_LAYER_SUPPORT_FUNC(__func__,
                               IsReshapeSupported,
                               data.m_Backends,
                               isSupported,
                               inputInfo,
                               outputInfo,
                               reshapeDesc);

    if (!isSupported)
    {
        return false;
    }

    armnn::IConnectableLayer* const layer = data.m_Network->AddReshapeLayer(reshapeDesc);
    assert(layer != nullptr);
    input.Connect(layer->GetInputSlot(0));

    return SetupAndTrackLayerOutputSlot<HalPolicy>(operation, 0, *layer, model, data);
}

template<typename HalPolicy,
         typename HalOperation = typename HalPolicy::Operation,
         typename HalModel     = typename HalPolicy::Model>
bool ConvertStridedSlice(const HalOperation& operation, const HalModel& model, ConversionData& data)
{
    using HalOperand = typename HalPolicy::Operand;

    LayerInputHandle input = ConvertToLayerInputHandle<HalPolicy>(operation, 0, model, data);
    if (!input.IsValid())
    {
        return Fail("%s: Operation has invalid inputs", __func__);
    }

    const armnn::TensorInfo& inputInfo = input.GetTensorInfo();
    unsigned int rank = inputInfo.GetNumDimensions();
    if (rank > 4)
    {
        Fail("%s: Inputs with rank greater than 4 are not supported", __func__);
    }

    const HalOperand* output = GetOutputOperand<HalPolicy>(operation, 0, model);
    if (!output)
    {
        return Fail("%s: Could not read output 0", __func__);
    }

    const armnn::TensorInfo& outputInfo = GetTensorInfoForOperand(*output);

    const HalOperand* beginOperand   = GetInputOperand<HalPolicy>(operation, 1, model);
    const HalOperand* endOperand     = GetInputOperand<HalPolicy>(operation, 2, model);
    const HalOperand* stridesOperand = GetInputOperand<HalPolicy>(operation, 3, model);

    std::vector<int32_t> beginValues;
    std::vector<int32_t> endValues;
    std::vector<int32_t> stridesValues;

    // The length of the beginOperand, endOperand and stridesOperand must be of a rank(input)
    auto ValidateInputOperands = [&] (const HalOperand& operand, std::vector<int32_t>& operandValues)
    {
        if (!GetTensorInt32Values<HalPolicy>(operand, operandValues, model, data))
        {
            return false;
        }

        if (operandValues.size() != rank)
        {
            return false;
        }

        return true;
    };

    if (!ValidateInputOperands(*beginOperand, beginValues)
        || !ValidateInputOperands(*endOperand, endValues)
        || !ValidateInputOperands(*stridesOperand, stridesValues))
    {
        return Fail("%s: Operation has invalid input operand", __func__);
    }

    // Stride cannot have value '0'
    if (std::any_of(stridesValues.cbegin(), stridesValues.cend(), [](int32_t i){ return i == 0; }))
    {
        return Fail("%s: Stride must be non-zero value.", __func__);
    }

    armnn::StridedSliceDescriptor descriptor;
    descriptor.m_Begin.assign(beginValues.cbegin(), beginValues.cend());
    descriptor.m_End.assign(endValues.cbegin(), endValues.cend());
    descriptor.m_Stride.assign(stridesValues.cbegin(), stridesValues.cend());
    descriptor.m_DataLayout = armnn::DataLayout::NHWC;

    // Get the "begin_mask", "end_mask", and "shrink_axis_mask" flags
    if (!GetInputInt32<HalPolicy>(operation, 4, descriptor.m_BeginMask, model, data) ||
        !GetInputInt32<HalPolicy>(operation, 5, descriptor.m_EndMask, model, data) ||
        !GetInputInt32<HalPolicy>(operation, 6, descriptor.m_ShrinkAxisMask, model, data))
    {
        return Fail("%s: Operation has invalid inputs", __func__);
    }

    bool isSupported = false;
    auto validateFunc = [&](const armnn::TensorInfo& outputInfo, bool& isSupported)
    {
        FORWARD_LAYER_SUPPORT_FUNC(__func__,
                                   IsStridedSliceSupported,
                                   data.m_Backends,
                                   isSupported,
                                   inputInfo,
                                   outputInfo,
                                   descriptor);
    };

    if(IsDynamicTensor(outputInfo))
    {
        isSupported = AreDynamicTensorsSupported();
    }
    else
    {
        validateFunc(outputInfo, isSupported);
    }

    if (!isSupported)
    {
        return false;
    }

    // Check if slice can fit in a inferred output
    armnn::TensorShape inputShape = inputInfo.GetShape();
    for (unsigned int i = 0; i < inputShape.GetNumDimensions(); i++)
    {
        int stride = descriptor.m_Stride[i];

        if (descriptor.m_ShrinkAxisMask & (1 << i))
        {
            // If the difference between the start point and the end point of the slice on an axis being shrunk
            // is greater than 1 then throw an error as the output will not be large enough to hold the slice
            if (((descriptor.m_Begin[i] - descriptor.m_End[i]) > 1)
                               || ((descriptor.m_Begin[i] - descriptor.m_End[i]) < -1))
            {
                return Fail("%s: StridedSlice: Output will not be large enough to hold the slice", __func__);
            }

            if(stride < 0)
            {
                return Fail("%s: StridedSlice: Stride can not be negative while ShrinkAxisMask is set.", __func__);
            }
        }
    }

    armnn::IConnectableLayer* const layer = data.m_Network->AddStridedSliceLayer(descriptor);
    assert(layer != nullptr);
    input.Connect(layer->GetInputSlot(0));

    return SetupAndTrackLayerOutputSlot<HalPolicy>(operation, 0, *layer, model, data, nullptr, validateFunc);
}

template<typename HalPolicy,
         typename HalOperation = typename HalPolicy::Operation,
         typename HalModel     = typename HalPolicy::Model>
bool ConvertTranspose(const HalOperation& operation, const HalModel& model, ConversionData& data)
{
    using HalOperand = typename HalPolicy::Operand;
    using HalOperandLifeTime = typename HalPolicy::OperandLifeTime;

    LayerInputHandle input = ConvertToLayerInputHandle<HalPolicy>(operation, 0, model, data);
    if (!input.IsValid())
    {
        return Fail("%s: Operation has invalid inputs", __func__);
    }

    const armnn::TensorInfo& inputInfo = input.GetTensorInfo();
    unsigned int rank = inputInfo.GetNumDimensions();
    if (rank > 4)
    {
        Fail("%s: Inputs with rank greater than 4 are not supported", __func__);
    }

    // NOTE: Axis is an optional parameter to TRANSPOSE, therefore we do not want to generate a failure
    // if the operand index is out of bounds.
    const HalOperand* permOperand = GetInputOperand<HalPolicy>(operation, 1, model, false);

    std::vector<int32_t> perm(rank);
    if (!permOperand || (permOperand->lifetime == HalOperandLifeTime::NO_VALUE))
    {
        for (unsigned int i = rank; i > 0; i--)
        {
            perm[rank - i] = armnn::numeric_cast<int> (i - 1);
        }
    }
    else if (!GetTensorInt32Values<HalPolicy>(*permOperand, perm, model, data))
    {
        return Fail("%s: Operation has an invalid or unsupported permutation operand", __func__);
    }

    std::vector<uint32_t> outputDims(perm.begin(), perm.begin() + rank);

    armnn::TransposeDescriptor transposeDesc;
    transposeDesc.m_DimMappings = armnn::PermutationVector(outputDims.data(), outputDims.size());

    const HalOperand* output = GetOutputOperand<HalPolicy>(operation, 0, model);
    if (!output)
    {
        return Fail("%s: Could not read output 0", __func__);
    }

    const armnn::TensorInfo& outputInfo = GetTensorInfoForOperand(*output);

    bool isSupported = false;
    auto validateFunc = [&](const armnn::TensorInfo& outputInfo, bool& isSupported)
    {
        FORWARD_LAYER_SUPPORT_FUNC(__func__,
                                   IsTransposeSupported,
                                   data.m_Backends,
                                   isSupported,
                                   inputInfo,
                                   outputInfo,
                                   transposeDesc);
        };

    if(IsDynamicTensor(outputInfo))
    {
        isSupported = AreDynamicTensorsSupported();
    }
    else
    {
        validateFunc(outputInfo, isSupported);
    }

    if (!isSupported)
    {
        return false;
    }

    armnn::IConnectableLayer* const layer = data.m_Network->AddTransposeLayer(transposeDesc);
    assert(layer != nullptr);
    input.Connect(layer->GetInputSlot(0));

    return SetupAndTrackLayerOutputSlot<HalPolicy>(operation, 0, *layer, model, data, nullptr, validateFunc);
}

template<typename HalPolicy,
         typename HalOperation   = typename HalPolicy::Operation,
         typename HalOperand     = typename HalPolicy::Operand,
         typename HalModel       = typename HalPolicy::Model>
bool ConvertBatchToSpaceNd(const HalOperation& operation,
                           const HalModel& model,
                           ConversionData& data)
{

    LayerInputHandle input = ConvertToLayerInputHandle<HalPolicy>(operation, 0, model, data);
    if (!input.IsValid())
    {
        return Fail("%s: Operation has invalid inputs", __func__);
    }

    const HalOperand* output = GetOutputOperand<HalPolicy>(operation, 0, model);
    if (!output)
    {
        return Fail("%s: Could not read output 0", __func__);
    }

    const armnn::TensorInfo& outputInfo = GetTensorInfoForOperand(*output);

    const HalOperand* blockOperand = GetInputOperand<HalPolicy>(operation, 1, model);
    if (!blockOperand)
    {
        return Fail("%s: Could not read input 1", __func__);
    }

    // Convert the block operand to int32
    std::vector<int32_t> block;
    if (!GetTensorInt32Values<HalPolicy>(*blockOperand, block, model, data))
    {
        return Fail("%s: Input 1 has invalid values", __func__);
    }

    const armnn::TensorInfo& inputInfo = input.GetTensorInfo();

    unsigned int rank = inputInfo.GetNumDimensions();
    if (rank != 4)
    {
        Fail("%s: Only inputs with rank equal to 4 are supported", __func__);
    }

    if (std::any_of(block.cbegin(), block.cend(), [](int32_t i){ return i < 1; }))
    {
        return Fail("%s: Block sizes for each spatial dimension of the input tensor must be"
                    " greater than or equal to 1", __func__);
    }

    armnn::BatchToSpaceNdDescriptor batchToSpaceNdDesc;
    batchToSpaceNdDesc.m_BlockShape.assign(block.cbegin(), block.cend());
    batchToSpaceNdDesc.m_DataLayout = armnn::DataLayout::NHWC;

    if (Is12OrLaterOperand(*output))
    {
        batchToSpaceNdDesc.m_DataLayout = OptionalDataLayout<HalPolicy>(operation, 2, model, data);
    }
    // Setting crops to 0,0 0,0 as it is not supported in Android NN API
    batchToSpaceNdDesc.m_Crops = {{0, 0}, {0, 0}};

    bool isSupported = false;
    auto validateFunc = [&](const armnn::TensorInfo& outputInfo, bool& isSupported)
    {
        FORWARD_LAYER_SUPPORT_FUNC(__func__,
                                   IsBatchToSpaceNdSupported,
                                   data.m_Backends,
                                   isSupported,
                                   inputInfo,
                                   outputInfo,
                                   batchToSpaceNdDesc);
    };

    if(!IsDynamicTensor(outputInfo))
    {
        validateFunc(outputInfo, isSupported);
    }
    else
    {
        isSupported = AreDynamicTensorsSupported();
    }


    if (!isSupported)
    {
        return false;
    }

    armnn::IConnectableLayer* const layer = data.m_Network->AddBatchToSpaceNdLayer(batchToSpaceNdDesc);
    assert(layer != nullptr);
    input.Connect(layer->GetInputSlot(0));

    return SetupAndTrackLayerOutputSlot<HalPolicy>(operation, 0, *layer, model, data, nullptr, validateFunc);
}

template<typename HalPolicy,
         typename HalOperation = typename HalPolicy::Operation,
         typename HalOperand   = typename HalPolicy::Operand,
         typename HalModel     = typename HalPolicy::Model>
bool ConvertSpaceToBatchNd(const HalOperation& operation, const HalModel& model, ConversionData& data)
{
    LayerInputHandle input = ConvertToLayerInputHandle<HalPolicy>(operation, 0, model, data);
    if (!input.IsValid())
    {
        return Fail("%s: Operation has invalid inputs", __func__);
    }

    const armnn::TensorInfo& inputInfo = input.GetTensorInfo();
    unsigned int rank = inputInfo.GetNumDimensions();
    unsigned int spatialDim = rank - 2;

    if (rank != 4)
    {
        Fail("%s: Only inputs with rank 4 are supported", __func__);
    }

    const HalOperand* output = GetOutputOperand<HalPolicy>(operation, 0, model);
    if (!output)
    {
        return Fail("%s: Could not read output 0", __func__);
    }

    const armnn::TensorInfo& outputInfo = GetTensorInfoForOperand(*output);

    const HalOperand* blockShapeOperand = GetInputOperand<HalPolicy>(operation, 1, model);
    const HalOperand* paddingsOperand   = GetInputOperand<HalPolicy>(operation, 2, model);

    armnn::TensorShape blockShapeOperandShape = GetTensorShapeForOperand(*blockShapeOperand);
    if (blockShapeOperandShape.GetNumDimensions() != 1 || blockShapeOperandShape.GetNumElements() != spatialDim)
    {
        return Fail("%s: Operation has invalid block shape operand: expected shape [%d]", __func__, spatialDim);
    }

    std::vector<int32_t> blockShape;
    if (!GetTensorInt32Values<HalPolicy>(*blockShapeOperand, blockShape, model, data))
    {
        return Fail("%s: Operation has an invalid or unsupported block size operand", __func__);
    }
    if (std::any_of(blockShape.cbegin(), blockShape.cend(), [](int32_t i){ return i < 1; }))
    {
        return Fail("%s: Block shape must be at least 1 in all dimensions.", __func__);
    }

    armnn::TensorShape paddingsOperandShape = GetTensorShapeForOperand(*paddingsOperand);
    if (paddingsOperandShape.GetNumDimensions() != 2 || paddingsOperandShape.GetNumElements() != 2 * spatialDim)
    {
        return Fail("%s: Operation has invalid paddings operand: expected shape [%d, 2]", __func__, spatialDim);
    }

    std::vector<std::pair<unsigned int, unsigned int>> paddingList;
    std::vector<int32_t> paddings;
    if (!GetTensorInt32Values<HalPolicy>(*paddingsOperand, paddings, model, data))
    {
        return Fail("%s: Operation has an invalid or unsupported paddings operand", __func__);
    }
    for (unsigned int i = 0; i < paddings.size() - 1; i += 2)
    {
        int paddingBeforeInput = paddings[i];
        int paddingAfterInput = paddings[i + 1];
        if (paddingBeforeInput < 0 || paddingAfterInput < 0)
        {
            return Fail("%s: Operation has invalid paddings operand, invalid padding values.", __func__);
        }

        paddingList.emplace_back((unsigned int) paddingBeforeInput, (unsigned int) paddingAfterInput);
    }

    armnn::SpaceToBatchNdDescriptor descriptor;
    descriptor.m_DataLayout = armnn::DataLayout::NHWC;
    descriptor.m_BlockShape.assign(blockShape.cbegin(), blockShape.cend());
    descriptor.m_PadList.assign(paddingList.cbegin(), paddingList.cend());

    if (Is12OrLaterOperand(*output))
    {
        descriptor.m_DataLayout = OptionalDataLayout<HalPolicy>(operation, 3, model, data);
    }

    bool isSupported = false;
    auto validateFunc = [&](const armnn::TensorInfo& outputInfo, bool& isSupported)
    {
        FORWARD_LAYER_SUPPORT_FUNC(__func__,
                                   IsSpaceToBatchNdSupported,
                                   data.m_Backends,
                                   isSupported,
                                   inputInfo,
                                   outputInfo,
                                   descriptor);
    };

    if(IsDynamicTensor(outputInfo))
    {
        isSupported = AreDynamicTensorsSupported();
    }
    else
    {
        validateFunc(outputInfo, isSupported);
    }

    if (!isSupported)
    {
        return false;
    }

    armnn::IConnectableLayer* const layer = data.m_Network->AddSpaceToBatchNdLayer(descriptor);
    assert(layer != nullptr);
    input.Connect(layer->GetInputSlot(0));

    return SetupAndTrackLayerOutputSlot<HalPolicy>(operation, 0, *layer, model, data, nullptr, validateFunc);
}

} // namespace armnn_driver
