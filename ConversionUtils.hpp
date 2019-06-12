//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <armnn/ArmNN.hpp>

#include "armnn/src/armnnUtils/DataLayoutIndexed.hpp"
#include "armnn/src/armnnUtils/Permute.hpp"
#include "Utils.hpp"

#include <ActivationFunctor.h>
#include <CpuExecutor.h>
#include <OperationsUtils.h>

#include <boost/assert.hpp>
#include <boost/core/ignore_unused.hpp>
#include <boost/numeric/conversion/cast.hpp>
#include <boost/test/tools/floating_point_comparison.hpp>

#include <log/log.h>
#include <vector>

namespace armnn_driver
{

///
/// Helper classes
///

struct ConversionData
{
    ConversionData(const std::vector<armnn::BackendId>& backends)
    : m_Backends(backends)
    , m_Network(nullptr, nullptr)
    {}

    const std::vector<armnn::BackendId>       m_Backends;
    armnn::INetworkPtr                        m_Network;
    std::vector<armnn::IOutputSlot*>          m_OutputSlotForOperand;
    std::vector<android::nn::RunTimePoolInfo> m_MemPools;
};

class LayerInputHandle
{
public:
    LayerInputHandle();
    LayerInputHandle(bool valid, armnn::IOutputSlot* outputSlot, armnn::TensorInfo tensorInfo);

    bool IsValid() const;

    void Connect(armnn::IInputSlot& inputSlot);

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

// Convenience function to call an Is*Supported function and log caller name together with reason for lack of support.
// Called as: IsLayerSupported(__func__, Is*Supported, a, b, c, d, e)
template<typename IsLayerSupportedFunc, typename ... Args>
bool IsLayerSupported(const char* funcName, IsLayerSupportedFunc f, Args&&... args)
{
    std::vector<char> unsupportedReason(1024+1);
    bool isSupported = f(std::forward<Args>(args)..., unsupportedReason.data(), unsupportedReason.size()-1);
    if(isSupported)
    {
        return true;
    }
    else
    {
        std::string sUnsupportedReason(unsupportedReason.data());
        if (sUnsupportedReason.size() > 0)
        {
            ALOGD("%s: not supported by armnn: %s", funcName, sUnsupportedReason.c_str());
        } else
        {
            ALOGD("%s: not supported by armnn", funcName);
        }
        return false;
    }
}

template<typename IsLayerSupportedFunc, typename ... Args>
bool IsLayerSupportedForAnyBackend(const char* funcName,
                                   IsLayerSupportedFunc f,
                                   const std::vector<armnn::BackendId>& backends,
                                   Args&&... args)
{
    for (auto&& backend : backends)
    {
        if (IsLayerSupported(funcName, f, backend, std::forward<Args>(args)...))
        {
            return true;
        }
    }

    ALOGD("%s: not supported by any specified backend", funcName);
    return false;
}

template<typename Operand>
armnn::TensorShape GetTensorShapeForOperand(const Operand& operand)
{
    return armnn::TensorShape(operand.dimensions.size(), operand.dimensions.data());
}

inline bool IsOperandTypeSupportedForTensors(V1_0::OperandType type)
{
    return type == V1_0::OperandType::TENSOR_FLOAT32      ||
           type == V1_0::OperandType::TENSOR_QUANT8_ASYMM ||
           type == V1_0::OperandType::TENSOR_INT32;
}

#ifdef ARMNN_ANDROID_NN_V1_2

inline bool IsOperandTypeSupportedForTensors(V1_2::OperandType type)
{
    return type == V1_2::OperandType::BOOL                ||
           type == V1_2::OperandType::TENSOR_FLOAT16      ||
           type == V1_2::OperandType::TENSOR_FLOAT32      ||
           type == V1_2::OperandType::TENSOR_QUANT8_ASYMM ||
           type == V1_2::OperandType::TENSOR_QUANT16_SYMM ||
           type == V1_2::OperandType::TENSOR_INT32;
}

#endif

inline bool IsBool(V1_0::Operand)
{
    return false;
}

#ifdef ARMNN_ANDROID_NN_V1_2

inline bool IsBool(V1_2::Operand operand)
{
    return operand.type == V1_2::OperandType::BOOL;
}

#endif

void BroadcastTensor(LayerInputHandle& input0, LayerInputHandle& input1, armnn::IConnectableLayer* startLayer,
                     armnn::INetwork& network)
{
    BOOST_ASSERT(startLayer != nullptr);
    const armnn::TensorInfo& inputTensorInfo0 = input0.GetTensorInfo();
    const armnn::TensorInfo& inputTensorInfo1 = input1.GetTensorInfo();

    if (inputTensorInfo0.GetNumDimensions() != inputTensorInfo1.GetNumDimensions())
    {
        // If the number of dimensions do not match then we need to add degenerate dimensions
        // to the "smaller" tensor using a reshape:
        //   Small  Big
        //     |     |
        //  Reshape  |
        //      \   /
        //       Add
        bool input0IsBigger = inputTensorInfo0.GetNumDimensions() > inputTensorInfo1.GetNumDimensions();

        LayerInputHandle& smallTensorHandle = input0IsBigger ? input1 : input0;
        const armnn::TensorInfo& smallTensorDims = smallTensorHandle.GetTensorInfo();

        LayerInputHandle& bigTensorHandle =  input0IsBigger ? input0 : input1;
        const armnn::TensorInfo& bigTensorDims = bigTensorHandle.GetTensorInfo();

        const unsigned int bigTensorDimsNumber = bigTensorDims.GetNumDimensions();
        std::vector<unsigned int> reshapedDims(bigTensorDimsNumber, 1);
        unsigned int sizeDifference = bigTensorDimsNumber - smallTensorDims.GetNumDimensions();
        for (unsigned i = sizeDifference; i < bigTensorDimsNumber; ++i)
        {
            reshapedDims[i] = smallTensorDims.GetShape()[i-sizeDifference];
        }
        armnn::TensorInfo reshapedInfo = smallTensorDims;
        reshapedInfo.SetShape(armnn::TensorShape{ static_cast<unsigned int>(reshapedDims.size()),
                                                  reshapedDims.data() });

        armnn::ReshapeDescriptor reshapeDesc;
        reshapeDesc.m_TargetShape = reshapedInfo.GetShape();
        armnn::IConnectableLayer* const reshapeLayer = network.AddReshapeLayer(reshapeDesc);
        smallTensorHandle.Connect(reshapeLayer->GetInputSlot(0));
        reshapeLayer->GetOutputSlot(0).SetTensorInfo(reshapedInfo);

        // Connect the outputs from new reshape and original input layer
        reshapeLayer->GetOutputSlot(0).Connect(startLayer->GetInputSlot(0));
        bigTensorHandle.Connect(startLayer->GetInputSlot(1));
    }
    else
    {
        input0.Connect(startLayer->GetInputSlot(0));
        input1.Connect(startLayer->GetInputSlot(1));
    }
}

void CalcPadding(uint32_t input, uint32_t kernel, uint32_t stride, uint32_t& outPadHead, uint32_t& outPadTail,
                 android::nn::PaddingScheme scheme)
{
    int32_t padHead;
    int32_t padTail;
    calculateExplicitPadding(input, stride, kernel, scheme, &padHead, &padTail);
    outPadHead = boost::numeric_cast<uint32_t>(padHead);
    outPadTail = boost::numeric_cast<uint32_t>(padTail);
}

Shape GetOperandShape(const V1_0::Operand& operand)
{
    Shape shape;
    shape.type = OperandType(operand.type);
    shape.dimensions = operand.dimensions;
    shape.scale = operand.scale;
    shape.offset = operand.zeroPoint;
    return shape;
}

// ArmNN requires the bias scale to be equal to the product of the weight and input scales, which is also
// what AndroidNN requires. However for some of the AndroidNN tests the values don't exactly match so
// we accept some tolerance. We don't want to ArmNN itself to accept these inconsistencies as it is up to the user
// (us, in this case) to ensure they match.
void SanitizeBiasQuantizationScale(armnn::TensorInfo& biasInfo,
                                   const armnn::TensorInfo& weightInfo, const armnn::TensorInfo& inputInfo)
{
    const float expectedBiasScale = weightInfo.GetQuantizationScale() * inputInfo.GetQuantizationScale();
    if (biasInfo.GetQuantizationScale() != expectedBiasScale)
    {
        boost::math::fpc::close_at_tolerance<float> comparer(boost::math::fpc::percent_tolerance(1.0f));
        if (comparer(biasInfo.GetQuantizationScale(), expectedBiasScale))
        {
            ALOGW("Bias quantization scale has been modified to match input*weights");
            biasInfo.SetQuantizationScale(expectedBiasScale);
        }
    }
}

// 4D Tensor Permutations
const armnn::PermutationVector IdentityPermutation4D({ 0U, 1U, 2U, 3U });
const armnn::PermutationVector NHWCToArmNN({ 0U, 2U, 3U, 1U });
const armnn::PermutationVector ArmNNToNHWC({ 0U, 3U, 1U, 2U });
const armnn::PermutationVector SwapDim1And2({ 0U, 2U, 1U, 3U });

// 3D Permutation Vectors
const armnn::PermutationVector IdentityPermutation3D({ 0U, 1U, 2U });
const armnn::PermutationVector RotateTensorLeft({ 2U, 0U, 1U });
const armnn::PermutationVector RotateTensorRight({ 1U, 2U, 0U });

template<typename OSlot>
armnn::IConnectableLayer& AddPermuteLayer(armnn::INetwork& network, OSlot& input,
                                          const armnn::PermutationVector& mappings)
{
    // Add swizzle layer
    armnn::IConnectableLayer* const layer = network.AddPermuteLayer(mappings);

    BOOST_ASSERT(layer != nullptr);

    // Connect input to swizzle layer
    input.Connect(layer->GetInputSlot(0));

    // Setup swizzled output
    const armnn::TensorInfo outInfo = armnnUtils::Permuted(input.GetTensorInfo(), mappings);
    layer->GetOutputSlot(0).SetTensorInfo(outInfo);

    return *layer;
}

void SwizzleIn(armnn::INetwork& network, LayerInputHandle& input, armnn::IConnectableLayer& layer, unsigned int index)
{
    // Add swizzle layer
    armnn::IConnectableLayer& swizzleLayer = AddPermuteLayer(network, input, NHWCToArmNN);
    // Connect swizzled input to layer
    swizzleLayer.GetOutputSlot(0).Connect(layer.GetInputSlot(index));
}

armnn::IConnectableLayer& DeswizzleOut(armnn::INetwork& network, armnn::IConnectableLayer& layer, unsigned int index)
{
    // Add deswizzle layer
    armnn::IConnectableLayer& deswizzleLayer = AddPermuteLayer(network, layer.GetOutputSlot(index), ArmNNToNHWC);
    return deswizzleLayer;
}

// only suitable for input/output slot index 0, for other slots, use SwizzleIn and DeswizzleOut directly
armnn::IConnectableLayer& SwizzleInDeswizzleOut(armnn::INetwork& network,
                                                LayerInputHandle& input,
                                                armnn::IConnectableLayer& firstLayer,
                                                armnn::IConnectableLayer& lastLayer)
{
    SwizzleIn(network, input, firstLayer, 0);
    return DeswizzleOut(network, lastLayer, 0);
}

// only suitable for input/output slot index 0, for other slots, use SwizzleIn and DeswizzleOut directly
armnn::IConnectableLayer& SwizzleInDeswizzleOut(armnn::INetwork& network, LayerInputHandle& input,
                                                armnn::IConnectableLayer& layer)
{
    return SwizzleInDeswizzleOut(network, input, layer, layer);
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

template<typename OSlot>
armnn::IConnectableLayer& AddReshapeLayer(armnn::INetwork& network, OSlot& inputLayer,
                                          armnn::TensorInfo reshapeInfo)
{
    armnn::ReshapeDescriptor reshapeDescriptor;
    reshapeDescriptor.m_TargetShape = reshapeInfo.GetShape();

    armnn::IConnectableLayer* reshapeLayer = network.AddReshapeLayer(reshapeDescriptor);
    BOOST_ASSERT(reshapeLayer != nullptr);

    // Attach the input layer to the reshape layer
    inputLayer.Connect(reshapeLayer->GetInputSlot(0));
    reshapeLayer->GetOutputSlot(0).SetTensorInfo(reshapeInfo);

    return *reshapeLayer;
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
            armnn::IConnectableLayer& swizzleLayer = AddPermuteLayer(network, inputs[i], mapping);
            auto& outputSlot = swizzleLayer.GetOutputSlot(0);
            auto& outputInfo = outputSlot.GetTensorInfo();
            // replace inputs with the swizzled ones
            inputs[i] = LayerInputHandle(true, &outputSlot, outputInfo);
            inputShapes[i] = inputs[i].GetTensorInfo().GetShape();
        }
    }
}

bool CreateConcatPermutationParameters(const unsigned int numberOfDimensions,
                                       int32_t & concatDimension,
                                       std::pair<armnn::PermutationVector, armnn::PermutationVector> & permutationPair)
{
    bool needPermute = false;
    BOOST_ASSERT(numberOfDimensions >= 3);

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

template<typename HalOperand, typename HalOperation, typename HalModel>
const HalOperand* GetInputOperand(const HalOperation& operation, uint32_t inputIndex, const HalModel& model,
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

    BOOST_ASSERT(operation.inputs[inputIndex] < model.operands.size()); // Model should have been validated beforehand
    return &model.operands[operation.inputs[inputIndex]];
}

template<typename HalOperand, typename HalOperation, typename HalModel>
const HalOperand* GetOutputOperand(const HalOperation& operation, uint32_t outputIndex, const HalModel& model)
{
    if (outputIndex >= operation.outputs.size())
    {
        Fail("%s: invalid output index: %i out of %i", __func__, outputIndex, operation.outputs.size());
        return nullptr;
    }

    // Model should have been validated beforehand
    BOOST_ASSERT(operation.outputs[outputIndex] < model.operands.size());

    return &model.operands[operation.outputs[outputIndex]];
}

template<typename HalOperand, typename HalModel>
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

    if (!optional &&
        operand.lifetime !=V1_0::OperandLifeTime::CONSTANT_COPY &&
        operand.lifetime !=V1_0::OperandLifeTime::CONSTANT_REFERENCE &&
        operand.lifetime !=V1_0::OperandLifeTime::NO_VALUE)
    {
        Fail("%s: invalid operand lifetime: %s", __func__, toString(operand.lifetime).c_str());
        return ConstTensorPin();
    }

    const void* const valueStart = GetOperandValueReadOnlyAddress(operand, model, data, optional);
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
    if (overrideTensorShape != nullptr)
    {
        tensorInfo.SetShape(*overrideTensorShape);
    }
    return ConstTensorPin(tensorInfo, valueStart, operand.location.length, dimensionMappings);
}

template<typename HalOperand, typename HalOperation, typename HalModel>
ConstTensorPin ConvertOperationInputToConstTensorPin(const HalOperation& operation,
                                                     uint32_t inputIndex,
                                                     const HalModel& model,
                                                     const ConversionData& data,
                                                     const armnn::PermutationVector& dimensionMappings = g_DontPermute,
                                                     const armnn::TensorShape* overrideTensorShape = nullptr,
                                                     bool optional = false)
{
    const HalOperand* operand = GetInputOperand<HalOperand>(operation, inputIndex, model);
    if (!operand)
    {
        Fail("%s: failed to get input operand: index=%u", __func__, inputIndex);
        return ConstTensorPin();
    }
    return ConvertOperandToConstTensorPin(*operand,
                                          model,
                                          data,
                                          dimensionMappings,
                                          overrideTensorShape,
                                          optional);
}

template<typename HalOperand, typename HalModel>
const void* GetOperandValueReadOnlyAddress(const HalOperand& operand,
                                           const HalModel& model,
                                           const ConversionData& data,
                                           bool optional = false)
{
    const void* valueStart = nullptr;

    switch (operand.lifetime)
    {
        case V1_0::OperandLifeTime::CONSTANT_COPY:
        {
            // Constant found in model.operandValues
            valueStart = &model.operandValues[operand.location.offset];
            break;
        }
        case V1_0::OperandLifeTime::CONSTANT_REFERENCE:
        {
            // Constant specified via a Memory object
            valueStart = GetMemoryFromPool(operand.location, data.m_MemPools);
            break;
        }
        case V1_0::OperandLifeTime::NO_VALUE:
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

template<typename HalOperand, typename HalOperandType, typename HalOperation, typename HalModel, typename OutputType>
bool GetInputScalar(const HalOperation& operation,
                    uint32_t inputIndex,
                    HalOperandType type,
                    OutputType& outValue,
                    const HalModel& model,
                    const ConversionData& data)
{
    const HalOperand* operand = GetInputOperand<HalOperand>(operation, inputIndex, model);
    if (!operand)
    {
        return Fail("%s: invalid input operand at index %i", __func__, inputIndex);
    }

    if (operand->type != type)
    {
        return Fail("%s: unexpected operand type: %s (should be %s)",
                    __func__, toString(operand->type).c_str(), toString(type).c_str());
    }

    if (operand->location.length != sizeof(OutputType))
    {
        return Fail("%s: incorrect operand location length: %i (should be %i)",
                    __func__, operand->location.length, sizeof(OutputType));
    }

    const void* valueAddress = GetOperandValueReadOnlyAddress(*operand, model, data);
    if (!valueAddress)
    {
        return Fail("%s: failed to get address for operand", __func__);
    }

    outValue = *(static_cast<const OutputType*>(valueAddress));
    return true;
}

template<typename HalOperand, typename HalOperandType, typename HalOperation, typename HalModel>
bool GetInputInt32(const HalOperation& operation,
                   uint32_t inputIndex,
                   int32_t& outValue,
                   const HalModel& model,
                   const ConversionData& data)
{
    return GetInputScalar<HalOperand, HalOperandType>(operation, inputIndex, HalOperandType::INT32, outValue, model,
            data);
}

template<typename HalOperand, typename HalOperandType, typename HalOperation, typename HalModel>
bool GetInputFloat32(const HalOperation& operation,
                     uint32_t inputIndex,
                     float& outValue,
                     const HalModel& model,
                     const ConversionData& data)
{
    return GetInputScalar<HalOperand, HalOperandType>(operation, inputIndex, HalOperandType::FLOAT32, outValue, model,
            data);
}

template<typename HalOperand, typename HalOperandType, typename HalOperation, typename HalModel>
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
                    toString(OperandType::INT32).c_str(),
                    toString(OperandType::TENSOR_INT32).c_str());
    }

    int32_t activationFunctionAsInt;
    if (!GetInputScalar<HalOperand, HalOperandType>(operation, inputIndex, type, activationFunctionAsInt, model, data))
    {
        return Fail("%s: failed to get activation input value", __func__);
    }
    outActivationFunction = static_cast<ActivationFn>(activationFunctionAsInt);
    return true;
}

template<typename HalOperand, typename HalOperandType, typename HalOperation, typename HalModel>
bool GetInputActivationFunction(const HalOperation& operation,
                                uint32_t inputIndex,
                                ActivationFn& outActivationFunction,
                                const HalModel& model,
                                const ConversionData& data)
{
    return GetInputActivationFunctionImpl<HalOperand, HalOperandType>(operation,
                                                                      inputIndex,
                                                                      HalOperandType::INT32,
                                                                      outActivationFunction,
                                                                      model,
                                                                      data);
}

template<typename HalOperand, typename HalOperandType, typename HalOperation, typename HalModel>
bool GetInputActivationFunctionFromTensor(const HalOperation& operation,
                                          uint32_t inputIndex,
                                          ActivationFn& outActivationFunction,
                                          const HalModel& model,
                                          const ConversionData& data)
{
    // This only accepts a 1-D tensor of size 1
    return GetInputActivationFunctionImpl<HalOperand, HalOperandType>(operation,
                                                                      inputIndex,
                                                                      HalOperandType::INT32,
                                                                      outActivationFunction,
                                                                      model,
                                                                      data);
}


template<typename HalOperand, typename HalOperandType, typename HalOperation, typename HalModel>
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
        if (!GetInputActivationFunction<HalOperand, HalOperandType>(operation, inputIndex, activationFunction, model,
                data))
        {
            return Fail("%s: Operation has invalid inputs", __func__);
        }
    }
    return true;
}

template <typename HalOperand,
          typename HalOperandType,
          typename HalOperation,
          typename HalModel,
          typename ConvolutionDescriptor>
bool GetOptionalConvolutionDilationParams(const HalOperation& operation,
                                          uint32_t dilationXIndex,
                                          ConvolutionDescriptor& descriptor,
                                          const HalModel& model,
                                          const ConversionData& data)
{
    bool success = true;
    if (operation.inputs.size() >= dilationXIndex + 2)
    {
        success &= GetInputScalar<HalOperand, HalOperandType>(operation,
                                                              dilationXIndex,
                                                              HalOperandType::INT32,
                                                              descriptor.m_DilationX,
                                                              model,
                                                              data);
        success &= GetInputScalar<HalOperand, HalOperandType>(operation,
                                                              dilationXIndex + 1,
                                                              HalOperandType::INT32,
                                                              descriptor.m_DilationY,
                                                              model,
                                                              data);
    }

    return success;
}

template<typename HalOperand, typename HalOperandType, typename HalModel>
bool GetTensorInt32Values(const HalOperand& operand,
                          std::vector<int32_t>& outValues,
                          const HalModel& model,
                          const ConversionData& data)
{
    if (operand.type != HalOperandType::TENSOR_INT32)
    {
        return Fail("%s: invalid operand type: %s", __func__, toString(operand.type).c_str());
    }

    const void* startAddress = GetOperandValueReadOnlyAddress(operand, model, data);
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

template<typename HalOperand, typename HalOperandType, typename HalOperation, typename HalModel>
bool GetInputPaddingScheme(const HalOperation& operation,
                           uint32_t inputIndex,
                           PaddingScheme& outPaddingScheme,
                           const HalModel& model,
                           const ConversionData& data)
{
    int32_t paddingSchemeAsInt;
    if (!GetInputInt32<HalOperand, HalOperandType>(operation, inputIndex, paddingSchemeAsInt, model, data))
    {
        return Fail("%s: failed to get padding scheme input value", __func__);
    }

    outPaddingScheme = static_cast<android::nn::PaddingScheme>(paddingSchemeAsInt);
    return true;
}

template<typename HalOperand, typename HalOperation, typename HalModel>
LayerInputHandle ConvertToLayerInputHandle(const HalOperation& operation,
                                           uint32_t inputIndex,
                                           const HalModel& model,
                                           ConversionData& data)
{
    const HalOperand* operand = GetInputOperand<HalOperand>(operation, inputIndex, model);
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

    armnn::TensorInfo operandTensorInfo = GetTensorInfoForOperand(*operand);

    switch (operand->lifetime)
    {
        case V1_0::OperandLifeTime::TEMPORARY_VARIABLE: // intentional fallthrough
        case V1_0::OperandLifeTime::MODEL_INPUT:
        case V1_0::OperandLifeTime::MODEL_OUTPUT:
        {
            // The tensor is either an operand internal to the model, or a model input.
            // It can be associated with an ArmNN output slot for an existing layer.

            // m_OutputSlotForOperand[...] can be nullptr if the previous layer could not be converted
            const uint32_t operandIndex = operation.inputs[inputIndex];
            return LayerInputHandle(true, data.m_OutputSlotForOperand[operandIndex], operandTensorInfo);
            break;
        }
        case V1_0::OperandLifeTime::CONSTANT_COPY:
        case V1_0::OperandLifeTime::CONSTANT_REFERENCE:
        {
            // The tensor has an already known constant value, and can be converted into an ArmNN Constant layer.
            ConstTensorPin tensorPin = ConvertOperandToConstTensorPin(*operand, model, data);
            if (tensorPin.IsValid())
            {
                if (!IsLayerSupportedForAnyBackend(__func__,
                                                   armnn::IsConstantSupported,
                                                   data.m_Backends,
                                                   tensorPin.GetConstTensor().GetInfo()))
                {
                    return LayerInputHandle();
                }

                armnn::IConnectableLayer* constantLayer = data.m_Network->AddConstantLayer(tensorPin.GetConstTensor());
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

template<typename HalOperand, typename HalOperation, typename HalModel>
bool SetupAndTrackLayerOutputSlot(const HalOperation& operation,
                                  uint32_t operationOutputIndex,
                                  armnn::IConnectableLayer& layer,
                                  uint32_t layerOutputIndex,
                                  const HalModel& model,
                                  ConversionData& data)
{
    const HalOperand* outputOperand = GetOutputOperand<HalOperand>(operation, operationOutputIndex, model);
    if ((outputOperand == nullptr) || (operationOutputIndex >= layer.GetNumOutputSlots()))
    {
        return false;
    }

    armnn::IOutputSlot& outputSlot = layer.GetOutputSlot(layerOutputIndex);

    const uint32_t operandIndex = operation.outputs[operationOutputIndex];
    data.m_OutputSlotForOperand[operandIndex] = &outputSlot;

    outputSlot.SetTensorInfo(GetTensorInfoForOperand(*outputOperand));

    return true;
}

template<typename HalOperand, typename HalOperation, typename HalModel>
armnn::DataLayout OptionalDataLayout(const HalOperation& operation,
                        uint32_t inputIndex,
                        const HalModel& model,
                        ConversionData& data)
{
    const HalOperand* operand = GetInputOperand<HalOperand>(operation, inputIndex, model);
    if (!operand)
    {
        return armnn::DataLayout::NHWC;
    }

    if (!IsBool(*operand))
    {
        return armnn::DataLayout::NHWC;
    }

    const void* valueAddress = GetOperandValueReadOnlyAddress(*operand, model, data);
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

template<typename HalOperand, typename HalOperation, typename HalModel>
bool SetupAndTrackLayerOutputSlot(const HalOperation& operation,
                                  uint32_t outputIndex,
                                  armnn::IConnectableLayer& layer,
                                  const HalModel& model,
                                  ConversionData& data)
{
    return SetupAndTrackLayerOutputSlot<HalOperand>(operation, outputIndex, layer, outputIndex, model, data);
}

template<typename HalOperand, typename HalOperation, typename HalModel>
bool ConvertToActivation(const HalOperation& operation,
                         const char* operationName,
                         const armnn::ActivationDescriptor& activationDesc,
                         const HalModel& model,
                         ConversionData& data)
{
    LayerInputHandle input = ConvertToLayerInputHandle<HalOperand>(operation, 0, model, data);
    if (!input.IsValid())
    {
        return Fail("%s: Input 0 is invalid", operationName);
    }

    const HalOperand* outputOperand = GetOutputOperand<HalOperand>(operation, 0, model);
    if (!outputOperand)
    {
        return false;
    }
    const armnn::TensorInfo outInfo = GetTensorInfoForOperand(*outputOperand);
    if (!IsLayerSupportedForAnyBackend(__func__,
                                       armnn::IsActivationSupported,
                                       data.m_Backends,
                                       input.GetTensorInfo(),
                                       outInfo,
                                       activationDesc))
    {
        return false;
    }

    armnn::IConnectableLayer* layer = data.m_Network->AddActivationLayer(activationDesc);
    BOOST_ASSERT(layer != nullptr);
    input.Connect(layer->GetInputSlot(0));

    return SetupAndTrackLayerOutputSlot<HalOperand>(operation, 0, *layer, model, data);
}

template<typename HalOperand, typename HalOperandType, typename HalOperation, typename HalModel>
bool ConvertPooling2d(const HalOperation& operation,
                      const char* operationName,
                      armnn::PoolingAlgorithm poolType,
                      const HalModel& model,
                      ConversionData& data)
{
    LayerInputHandle input = ConvertToLayerInputHandle<HalOperand>(operation, 0, model, data);
    if (!input.IsValid())
    {
        return Fail("%s: Could not read input 0", operationName);
    }

    const HalOperand* output = GetOutputOperand<HalOperand>(operation, 0, model);
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

    if (operation.inputs.size() == 7)
    {
        // one input, 6 parameters (padding, stridex, stridey, width, height, activation type)
        android::nn::PaddingScheme scheme;
        if (!GetInputPaddingScheme<HalOperand, HalOperandType>(operation, 1, scheme, model, data)
            || !GetInputScalar<HalOperand, HalOperandType>(operation, 2, HalOperandType::INT32, desc.m_StrideX, model,
                    data)
            || !GetInputScalar<HalOperand, HalOperandType>(operation, 3, HalOperandType::INT32, desc.m_StrideY, model,
                    data)
            || !GetInputScalar<HalOperand, HalOperandType>(operation, 4, HalOperandType::INT32, desc.m_PoolWidth, model,
                    data)
            || !GetInputScalar<HalOperand, HalOperandType>(operation, 5, HalOperandType::INT32, desc.m_PoolHeight,
                    model, data)
            || !GetInputActivationFunction<HalOperand, HalOperandType>(operation, 6, activation, model, data))
        {
            return Fail("%s: Operation has invalid inputs", operationName);
        }

        const unsigned int inputWidth  = inputInfo.GetShape()[2];
        const unsigned int inputHeight = inputInfo.GetShape()[1];

        CalcPadding(inputWidth, desc.m_PoolWidth, desc.m_StrideX, desc.m_PadLeft, desc.m_PadRight, scheme);
        CalcPadding(inputHeight, desc.m_PoolHeight, desc.m_StrideY, desc.m_PadTop, desc.m_PadBottom, scheme);
    }
    else
    {
        // one input, 9 parameters (padding l r t b, stridex, stridey, width, height, activation type)
        if (!GetInputScalar<HalOperand, HalOperandType>(operation, 1, HalOperandType::INT32, desc.m_PadLeft, model,
                data)
            || !GetInputScalar<HalOperand, HalOperandType>(operation, 2, HalOperandType::INT32, desc.m_PadRight, model,
                    data)
            || !GetInputScalar<HalOperand, HalOperandType>(operation, 3, HalOperandType::INT32, desc.m_PadTop, model,
                    data)
            || !GetInputScalar<HalOperand, HalOperandType>(operation, 4, HalOperandType::INT32, desc.m_PadBottom, model,
                    data)
            || !GetInputScalar<HalOperand, HalOperandType>(operation, 5, HalOperandType::INT32, desc.m_StrideX, model,
                    data)
            || !GetInputScalar<HalOperand, HalOperandType>(operation, 6, HalOperandType::INT32, desc.m_StrideY, model,
                    data)
            || !GetInputScalar<HalOperand, HalOperandType>(operation, 7, HalOperandType::INT32, desc.m_PoolWidth, model,
                    data)
            || !GetInputScalar<HalOperand, HalOperandType>(operation, 8, HalOperandType::INT32, desc.m_PoolHeight,
                    model, data)
            || !GetInputActivationFunction<HalOperand, HalOperandType>(operation, 9, activation, model, data))
        {
            return Fail("%s: Operation has invalid inputs", operationName);
        }
    }

    if (!IsLayerSupportedForAnyBackend(__func__,
                                       armnn::IsPooling2dSupported,
                                       data.m_Backends,
                                       inputInfo,
                                       outputInfo,
                                       desc))
    {
        return false;
    }

    armnn::IConnectableLayer* pooling2dLayer = data.m_Network->AddPooling2dLayer(desc);
    if (!pooling2dLayer)
    {
        return Fail("%s: AddPooling2dLayer failed", __func__);
    }

    armnn::IConnectableLayer* endLayer = ProcessActivation(outputInfo, activation, pooling2dLayer, data);
    if (!endLayer)
    {
        return Fail("%s: ProcessActivation failed", __func__);
    }

    input.Connect(pooling2dLayer->GetInputSlot(0));

    return SetupAndTrackLayerOutputSlot<HalOperand>(operation, 0, *endLayer, model, data);
}

template<typename HalOperand, typename HalOperandType, typename HalOperation, typename HalModel>
bool ConvertConv2d(const HalOperation& operation, const HalModel& model, ConversionData& data)
{
    LayerInputHandle input = ConvertToLayerInputHandle<HalOperand>(operation, 0, model, data);
    if (!input.IsValid())
    {
        return Fail("%s: Operation has invalid inputs", __func__);
    }

    const HalOperand* output = GetOutputOperand<HalOperand>(operation, 0, model);
    if (!output)
    {
        return Fail("%s: Could not read output 0", __func__);
    }

    const armnn::TensorInfo& inputInfo  = input.GetTensorInfo();
    const armnn::TensorInfo& outputInfo = GetTensorInfoForOperand(*output);

    // ArmNN does not currently support non-fixed weights or bias
    const ConstTensorPin weightsPin = ConvertOperationInputToConstTensorPin<HalOperand>(operation, 1, model, data);
    const ConstTensorPin biasPin    = ConvertOperationInputToConstTensorPin<HalOperand>(operation, 2, model, data);

    if (!weightsPin.IsValid() || !biasPin.IsValid())
    {
        return Fail("%s: Operation has invalid inputs", __func__);
    }

    armnn::ConstTensor weights = weightsPin.GetConstTensor();
    armnn::ConstTensor bias = biasPin.GetConstTensor();
    SanitizeBiasQuantizationScale(bias.GetInfo(), weights.GetInfo(), inputInfo);

    armnn::Convolution2dDescriptor desc;
    desc.m_DataLayout = armnn::DataLayout::NHWC;
    ActivationFn activation;

    if (operation.inputs.size() >= 10)
    {
        if (!GetInputScalar<HalOperand, HalOperandType>(operation, 3, HalOperandType::INT32, desc.m_PadLeft, model,
                data)
            || !GetInputScalar<HalOperand, HalOperandType>(operation, 4, HalOperandType::INT32, desc.m_PadRight, model,
                    data)
            || !GetInputScalar<HalOperand, HalOperandType>(operation, 5, HalOperandType::INT32, desc.m_PadTop, model,
                    data)
            || !GetInputScalar<HalOperand, HalOperandType>(operation, 6, HalOperandType::INT32, desc.m_PadBottom, model,
                    data)
            || !GetInputScalar<HalOperand, HalOperandType>(operation, 7, HalOperandType::INT32, desc.m_StrideX, model,
                    data)
            || !GetInputScalar<HalOperand, HalOperandType>(operation, 8, HalOperandType::INT32, desc.m_StrideY, model,
                    data)
            || !GetInputActivationFunction<HalOperand, HalOperandType>(operation, 9, activation, model, data)
            || !GetOptionalConvolutionDilationParams<HalOperand, HalOperandType>(operation, 11, desc, model, data))
        {
            return Fail("%s: Operation has invalid inputs", __func__);
        }
        desc.m_DataLayout = OptionalDataLayout<HalOperand>(operation, 10, model, data);
    }
    else if (operation.inputs.size() >= 7)
    {
        android::nn::PaddingScheme paddingScheme;
        if (!GetInputPaddingScheme<HalOperand, HalOperandType>(operation, 3, paddingScheme, model, data)
            || !GetInputScalar<HalOperand, HalOperandType>(operation, 4, HalOperandType::INT32, desc.m_StrideX,
                    model, data)
            || !GetInputScalar<HalOperand, HalOperandType>(operation, 5, HalOperandType::INT32, desc.m_StrideY, model,
                    data)
            || !GetInputActivationFunction<HalOperand, HalOperandType>(operation, 6, activation, model, data)
            || !GetOptionalConvolutionDilationParams<HalOperand, HalOperandType>(operation, 8, desc, model, data))
        {
            return Fail("%s: Operation has invalid inputs", __func__);
        }

        const uint32_t kernelX = weights.GetShape()[2];
        const uint32_t kernelY = weights.GetShape()[1];
        const uint32_t inputX  = inputInfo.GetShape()[2];
        const uint32_t inputY  = inputInfo.GetShape()[1];

        CalcPadding(inputX, kernelX, desc.m_StrideX, desc.m_PadLeft, desc.m_PadRight, paddingScheme);
        CalcPadding(inputY, kernelY, desc.m_StrideY, desc.m_PadTop, desc.m_PadBottom, paddingScheme);

        desc.m_DataLayout = OptionalDataLayout<HalOperand>(operation, 7, model, data);
    }
    else
    {
        return Fail("%s: Unsupported number of operation inputs", __func__);
    }

    desc.m_BiasEnabled = true;
    armnn::Optional<armnn::TensorInfo> biases(bias.GetInfo());

    if (!IsLayerSupportedForAnyBackend(__func__,
                                       armnn::IsConvolution2dSupported,
                                       data.m_Backends,
                                       inputInfo,
                                       outputInfo,
                                       desc,
                                       weights.GetInfo(),
                                       biases))
    {
        return false;
    }

    armnn::IConnectableLayer* startLayer =
            data.m_Network->AddConvolution2dLayer(desc, weights, armnn::Optional<armnn::ConstTensor>(bias));

    if (!startLayer)
    {
        return Fail("%s: AddConvolution2dLayer failed", __func__);
    }

    armnn::IConnectableLayer* endLayer = ProcessActivation(outputInfo, activation, startLayer, data);

    if (!endLayer)
    {
        return Fail("%s: ProcessActivation failed", __func__);
    }

    input.Connect(startLayer->GetInputSlot(0));

    return SetupAndTrackLayerOutputSlot<HalOperand>(operation, 0, *endLayer, model, data);
}

template<typename HalOperand, typename HalOperandType, typename HalOperation, typename HalModel>
bool ConvertDepthwiseConv2d(const HalOperation& operation, const HalModel& model, ConversionData& data)
{
    LayerInputHandle input = ConvertToLayerInputHandle<HalOperand>(operation, 0, model, data);

    if (!input.IsValid())
    {
        return Fail("%s: Operation has invalid inputs", __func__);
    }

    const HalOperand* output = GetOutputOperand<HalOperand>(operation, 0, model);

    if (!output)
    {
        return Fail("%s: Could not read output 0", __func__);
    }

    const armnn::TensorInfo& inputInfo  = input.GetTensorInfo();
    const armnn::TensorInfo& outputInfo = GetTensorInfoForOperand(*output);

    // ArmNN does not currently support non-fixed weights or bias

    // Find the shape of the weights tensor. In AndroidNN this will be [ 1, H, W, I * M ]
    const HalOperand* weightsOperand = GetInputOperand<HalOperand>(operation, 1, model);

    if (weightsOperand == nullptr)
    {
        return Fail("%s: Operand is invalid", __func__);
    }
    armnn::DepthwiseConvolution2dDescriptor desc;
    desc.m_DataLayout = armnn::DataLayout::NHWC;

    // Look ahead to find the optional DataLayout, if present
    if (operation.inputs.size() >= 12)
    {
        desc.m_DataLayout = OptionalDataLayout<HalOperand>(operation, 11, model, data);
    }
    else if (operation.inputs.size() >= 9)
    {
        desc.m_DataLayout = OptionalDataLayout<HalOperand>(operation, 8, model, data);
    }

    armnnUtils::DataLayoutIndexed dataLayoutIndexed(desc.m_DataLayout);
    unsigned int channelsIndex = dataLayoutIndexed.GetChannelsIndex();
    unsigned int widthIndex = dataLayoutIndexed.GetWidthIndex();
    unsigned int heightIndex = dataLayoutIndexed.GetHeightIndex();

    // Reinterpret weight data as [ H, W, I, M ]
    armnn::TensorShape weightsShape({ weightsOperand->dimensions[1],
                                      weightsOperand->dimensions[2],
                                      inputInfo.GetShape()[channelsIndex],
                                      weightsOperand->dimensions[3] / inputInfo.GetShape()[channelsIndex] });

    // Swizzle weight data [ H, W, I, M ] -> [ M, I, H, W ]
    const armnn::PermutationVector HWIMToMIHW = { 2U, 3U, 1U, 0U };

    const ConstTensorPin weightsPin = ConvertOperationInputToConstTensorPin<HalOperand>(operation, 1, model, data,
                                                                                        HWIMToMIHW, &weightsShape);

    // Bias is a 1D tensor
    const ConstTensorPin biasPin = ConvertOperationInputToConstTensorPin<HalOperand>(operation, 2, model, data);

    if (!weightsPin.IsValid() || !biasPin.IsValid())
    {
        return Fail("%s: Operation has invalid inputs", __func__);
    }

    armnn::ConstTensor weights = weightsPin.GetConstTensor();
    armnn::ConstTensor bias = biasPin.GetConstTensor();
    SanitizeBiasQuantizationScale(bias.GetInfo(), weights.GetInfo(), inputInfo);

    ActivationFn activation;

    if (operation.inputs.size() >= 11)
    {
        if (!GetInputScalar<HalOperand, HalOperandType>(operation, 3, HalOperandType::INT32, desc.m_PadLeft, model,
                data)
            || !GetInputScalar<HalOperand, HalOperandType>(operation, 4, HalOperandType::INT32, desc.m_PadRight, model,
                    data)
            || !GetInputScalar<HalOperand, HalOperandType>(operation, 5, HalOperandType::INT32, desc.m_PadTop, model,
                    data)
            || !GetInputScalar<HalOperand, HalOperandType>(operation, 6, HalOperandType::INT32, desc.m_PadBottom, model,
                    data)
            || !GetInputScalar<HalOperand, HalOperandType>(operation, 7, HalOperandType::INT32, desc.m_StrideX, model,
                    data)
            || !GetInputScalar<HalOperand, HalOperandType>(operation, 8, HalOperandType::INT32, desc.m_StrideY, model,
                    data)
            || !GetInputActivationFunction<HalOperand, HalOperandType>(operation,  10, activation, model, data)
            || !GetOptionalConvolutionDilationParams<HalOperand, HalOperandType>(operation, 12, desc, model, data))
        {
            return Fail("%s: Operation has invalid inputs", __func__);
        }
    }
    else if (operation.inputs.size() >= 8)
    {
        android::nn::PaddingScheme paddingScheme;
        if (!GetInputPaddingScheme<HalOperand, HalOperandType>(operation, 3, paddingScheme, model, data)
            || !GetInputScalar<HalOperand, HalOperandType>(operation, 4, HalOperandType::INT32, desc.m_StrideX, model,
                    data)
            || !GetInputScalar<HalOperand, HalOperandType>(operation, 5, HalOperandType::INT32, desc.m_StrideY, model,
                    data)
            || !GetInputActivationFunction<HalOperand, HalOperandType>(operation, 7, activation, model, data)
            || !GetOptionalConvolutionDilationParams<HalOperand, HalOperandType>(operation, 9, desc, model, data))
        {
            return Fail("%s: Operation has invalid inputs", __func__);
        }

        const uint32_t kernelX = weights.GetShape()[3];
        const uint32_t kernelY = weights.GetShape()[2];
        const uint32_t inputX  = inputInfo.GetShape()[widthIndex];
        const uint32_t inputY  = inputInfo.GetShape()[heightIndex];

        CalcPadding(inputX, kernelX, desc.m_StrideX, desc.m_PadLeft, desc.m_PadRight, paddingScheme);
        CalcPadding(inputY, kernelY, desc.m_StrideY, desc.m_PadTop, desc.m_PadBottom, paddingScheme);
    }
    else
    {
        return Fail("%s: Unsupported number of operation inputs", __func__);
    }

    desc.m_BiasEnabled = true;
    armnn::Optional<armnn::TensorInfo> biases(bias.GetInfo());

    if (!IsLayerSupportedForAnyBackend(__func__,
                                       armnn::IsDepthwiseConvolutionSupported,
                                       data.m_Backends,
                                       inputInfo,
                                       outputInfo,
                                       desc,
                                       weights.GetInfo(),
                                       biases))
    {
        return false;
    }

    armnn::IConnectableLayer* startLayer =
            data.m_Network->AddDepthwiseConvolution2dLayer(desc, weights, armnn::Optional<armnn::ConstTensor>(bias));
    if (!startLayer)
    {
        return Fail("%s: AddDepthwiseConvolution2dLayer failed", __func__);
    }

    armnn::IConnectableLayer* endLayer = ProcessActivation(outputInfo, activation, startLayer, data);
    if (!endLayer)
    {
        return Fail("%s: ProcessActivation failed", __func__);
    }

    input.Connect(startLayer->GetInputSlot(0));

    return SetupAndTrackLayerOutputSlot<HalOperand>(operation, 0, *endLayer, model, data);
}

} // namespace armnn_driver
