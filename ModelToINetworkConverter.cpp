//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//

#define LOG_TAG "ArmnnDriver"

#include "ModelToINetworkConverter.hpp"
#include <OperationsUtils.h>

#include <armnn/LayerSupport.hpp>
#include <Permute.hpp>

#include <log/log.h>
#include <cassert>

#include <boost/format.hpp>
#include <boost/core/ignore_unused.hpp>
#include <boost/test/tools/floating_point_comparison.hpp>
#include <boost/cast.hpp>
#include <boost/optional.hpp>

using namespace android::hardware;

namespace armnn_driver
{

class LayerInputHandle
{
public:
    LayerInputHandle()
        : m_OutputSlot(nullptr)
        , m_Valid(false)
    {}

    LayerInputHandle(bool valid, armnn::IOutputSlot* outputSlot, armnn::TensorInfo tensorInfo)
        : m_OutputSlot(outputSlot)
        , m_Valid(valid)
        , m_TensorInfo(tensorInfo)
    {}

    bool IsValid() const { return m_Valid; }
    void Connect(armnn::IInputSlot& inputSlot)
    {
        assert(IsValid());

        if (m_OutputSlot)
        {
            m_OutputSlot->Connect(inputSlot);
        }
    }
    const armnn::TensorInfo& GetTensorInfo() const { return m_TensorInfo; }

private:
    armnn::IOutputSlot* m_OutputSlot;
    bool m_Valid;
    armnn::TensorInfo m_TensorInfo;
};

} // namespace armnn_driver

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

armnn::TensorShape GetTensorShapeForOperand(const Operand& operand)
{
    return armnn::TensorShape(operand.dimensions.size(), operand.dimensions.data());
}

inline bool IsOperandTypeSupportedForTensors(OperandType type)
{
    return type == OperandType::TENSOR_FLOAT32      ||
           type == OperandType::TENSOR_QUANT8_ASYMM ||
           type == OperandType::TENSOR_INT32;
}

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

Shape GetOperandShape(const Operand& operand)
{
    Shape shape;
    shape.type = operand.type;
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

    assert(layer != nullptr);

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
    assert(reshapeLayer != nullptr);

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

void CreatePermutationParameters(const unsigned int numberOfDimensions,
                       int32_t & concatDimension,
                       std::pair<armnn::PermutationVector, armnn::PermutationVector> & permutationPair)
{
    assert(numberOfDimensions >= 3);

    // ArmNN uses Compute Library subtensors to perform concatenation
    // This only works when concatenating along dimension 0 or 1 for a 4-D tensor,
    // or along dimension 0 for a 3-D tensor.
    if (numberOfDimensions == 4)
    {
        if (concatDimension == 3)
        {
            concatDimension = 1;
            permutationPair = std::make_pair(NHWCToArmNN, ArmNNToNHWC);
        }
        else if (concatDimension == 2)
        {
            concatDimension = 1;
            permutationPair = std::make_pair(SwapDim1And2, SwapDim1And2);
        }
        else
        {
            permutationPair = std::make_pair(IdentityPermutation4D, IdentityPermutation4D);
        }

    }
    else if (numberOfDimensions == 3)
    {
        if (concatDimension == 2)
        {
            concatDimension = 0;
            permutationPair = std::make_pair(RotateTensorRight, RotateTensorLeft);
        }
        else if (concatDimension == 1)
        {
            concatDimension = 0;
            permutationPair = std::make_pair(RotateTensorLeft, RotateTensorRight);
        }
        else
        {
            permutationPair = std::make_pair(IdentityPermutation3D, IdentityPermutation3D);
        }
    }
}

} // anonymous namespace

namespace armnn_driver
{

class ConstTensorPin
{
public:
    // Creates an invalid tensor pin (can be used to signal errors)
    // The optional flag can be set to indicate the tensor values were missing, but it was otherwise valid
    ConstTensorPin(bool optional = false) : m_Optional(optional) {}

    // @param tensorInfo TensorInfo associated with the tensor.
    // @param valueStart Start address of tensor data. Belongs to one of the memory pools associated with
    // the model being converted.
    // @param numBytes Number of bytes for the tensor data.
    ConstTensorPin(const armnn::TensorInfo& tensorInfo, const void* valueStart, uint32_t numBytes,
                   const armnn::PermutationVector& mappings)
    {
        boost::ignore_unused(numBytes);
        assert(tensorInfo.GetNumBytes() == numBytes);

        const bool needsSwizzling = (mappings.GetSize() > 0);
        if (needsSwizzling)
        {
            m_SwizzledTensorData.resize(tensorInfo.GetNumBytes());
            SwizzleAndroidNn4dTensorToArmNn(tensorInfo, valueStart, m_SwizzledTensorData.data(), mappings);

            m_ConstTensor = armnn::ConstTensor(armnnUtils::Permuted(tensorInfo, mappings), m_SwizzledTensorData.data());
        }
        else
        {
            m_ConstTensor = armnn::ConstTensor(tensorInfo, valueStart);
        }
    }

    ConstTensorPin(const ConstTensorPin& other) = delete;
    ConstTensorPin(ConstTensorPin&& other) = default;

    bool IsValid() const { return m_ConstTensor.GetMemoryArea() != nullptr; }
    bool IsOptional() const { return m_Optional; }
    const armnn::ConstTensor& GetConstTensor() const { return m_ConstTensor; }
    const armnn::ConstTensor* GetConstTensorPtr() const
    {
        if (IsValid() && m_ConstTensor.GetNumElements() > 0)
        {
            return &m_ConstTensor;
        }
        // tensor is either invalid, or has no elements (indicating an optional tensor that was not provided)
        return nullptr;
    }

private:
    armnn::ConstTensor m_ConstTensor;
    // Owned memory for swizzled tensor data, only required if the tensor needed
    // swizzling. Otherwise, @ref m_ConstTensor will reference memory from one of
    // the pools associated with the model being converted.
    std::vector<uint8_t> m_SwizzledTensorData;
    // optional flag to indicate that an invalid tensor pin is not an error, but the optional values were not given
    bool m_Optional;
};

template<typename HalVersion>
ModelToINetworkConverter<HalVersion>::ModelToINetworkConverter(armnn::Compute compute,
    const HalModel& model,
    const std::set<unsigned int>& forcedUnsupportedOperations)
    : m_Compute(compute)
    , m_Model(model)
    , m_ForcedUnsupportedOperations(forcedUnsupportedOperations)
    , m_Network(nullptr, nullptr)
    , m_ConversionResult(ConversionResult::Success)
{
    try
    {
        Convert();
    }
    catch (armnn::Exception& e)
    {
        m_ConversionResult = ConversionResult::UnsupportedFeature;
        ALOGE("%s: Unexpected exception: %s", __func__, e.what());
        assert(false);
    }
}

template<typename HalVersion>
void ModelToINetworkConverter<HalVersion>::Convert()
{
    using HalModel = typename HalVersion::Model;
    ALOGV("ModelToINetworkConverter::Convert(): %s", GetModelSummary<HalModel>(m_Model).c_str());

    // map the memory pool into shared pointers
    m_MemPools.clear();
    if (!setRunTimePoolInfosFromHidlMemories(&m_MemPools, m_Model.pools))
    {
        Fail("%s: Setting of run time pool infos from Hidl Memories has failed.", __func__);
        m_ConversionResult = ConversionResult::ErrorMappingPools;
        return;
    }

    uint32_t totalPoolSize = 0;
    for (auto&& pool : m_Model.pools)
    {
        totalPoolSize += pool.size();
    }

    // Create armnn::INetwork
    m_Network = armnn::INetwork::Create();

    // add operations to it
    // track which layer outputs each operand
    m_OutputSlotForOperand = std::vector<armnn::IOutputSlot*>(m_Model.operands.size(), nullptr);

    try
    {
        for (uint32_t i = 0; i < m_Model.inputIndexes.size(); i++)
        {
            // inputs in android nn are represented by operands
            uint32_t inputIndex = m_Model.inputIndexes[i];
            const Operand& operand = m_Model.operands[inputIndex];
            const armnn::TensorInfo& tensor = GetTensorInfoForOperand(operand);
            armnn::IConnectableLayer* layer = m_Network->AddInputLayer(i);

            armnn::IOutputSlot& outputSlot = layer->GetOutputSlot(0);
            outputSlot.SetTensorInfo(GetTensorInfoForOperand(operand));

            // store for later layers
            m_OutputSlotForOperand[inputIndex] = &outputSlot;
        }
    }
    catch (UnsupportedOperand& e)
    {
        Fail("%s: Operand type %s not supported in ArmnnDriver", __func__, toString(e.m_type).c_str());
        m_ConversionResult = ConversionResult::UnsupportedFeature;
    }
    catch (const armnn::InvalidArgumentException& e)
    {
        Fail("%s: Failed to convert input operand to TensorShape: %s", __func__, e.what());
        m_ConversionResult = ConversionResult::UnsupportedFeature;
    }

    for (uint32_t operationIdx = 0; operationIdx < m_Model.operations.size(); operationIdx++)
    {
        const auto& operation = m_Model.operations[operationIdx];

        bool ok = true;
        if (m_ForcedUnsupportedOperations.find(operationIdx) != m_ForcedUnsupportedOperations.end())
        {
            Fail("%s: Operation at index %i has been forced to be unsupported.", __func__, operationIdx);
            ok = false;
        }

        if (ok)
        {
            try
            {
                ok = ConvertOperation(operation);
            }
            catch (UnsupportedOperand& e)
            {
                Fail("%s: Operand type %s not supported in ArmnnDriver", __func__, toString(e.m_type).c_str());
                ok = false;
            }
            catch (const armnn::InvalidArgumentException& e)
            {
                Fail("%s: Failed to convert operation in %s", __func__, e.what());
                ok = false;
            }
        }

        // Store whether this operation was successfully converted.
        m_OperationSupported.emplace(operationIdx, ok);

        // Any single operation failing will fail the entire conversion.
        // We still need to continue and check the other ones.
        if (!ok)
        {
            m_ConversionResult = ConversionResult::UnsupportedFeature;
        }
    }
    try
    {
        if (m_ConversionResult == ConversionResult::Success)
        {
            for (uint32_t i = 0; i < m_Model.outputIndexes.size(); i++)
            {
                // outputs in android nn are represented by operands
                uint32_t outputIndex = m_Model.outputIndexes[i];
                const Operand& operand = m_Model.operands[outputIndex];
                const armnn::TensorInfo& tensor = GetTensorInfoForOperand(operand);
                armnn::IConnectableLayer* layer = m_Network->AddOutputLayer(i);

                assert(m_OutputSlotForOperand[outputIndex]);
                m_OutputSlotForOperand[outputIndex]->Connect(layer->GetInputSlot(0));
            }
        }
    }
    catch (const armnn::InvalidArgumentException& e)
    {
        Fail("%s: Failed to convert output operand to TensorShape: %s", __func__, e.what());
        m_ConversionResult = ConversionResult::UnsupportedFeature;
    }
}

template<typename HalVersion>
bool ModelToINetworkConverter<HalVersion>::ConvertOperation(const neuralnetworks::V1_0::Operation& operation)
{
    switch (operation.type)
    {
        case neuralnetworks::V1_0::OperationType::ADD:
            return ConvertAdd(operation);
        case neuralnetworks::V1_0::OperationType::AVERAGE_POOL_2D:
            return ConvertAveragePool2d(operation);
        case neuralnetworks::V1_0::OperationType::CONCATENATION:
            return ConvertConcatenation(operation);
        case neuralnetworks::V1_0::OperationType::CONV_2D:
            return ConvertConv2d(operation);
        case neuralnetworks::V1_0::OperationType::DEPTHWISE_CONV_2D:
            return ConvertDepthwiseConv2d(operation);
        case neuralnetworks::V1_0::OperationType::FLOOR:
            return ConvertFloor(operation);
        case neuralnetworks::V1_0::OperationType::FULLY_CONNECTED:
            return ConvertFullyConnected(operation);
        case neuralnetworks::V1_0::OperationType::LOCAL_RESPONSE_NORMALIZATION:
            return ConvertLocalResponseNormalization(operation);
        case neuralnetworks::V1_0::OperationType::LOGISTIC:
            return ConvertLogistic(operation);
        case neuralnetworks::V1_0::OperationType::LSTM:
            return ConvertLstm(operation);
        case neuralnetworks::V1_0::OperationType::L2_NORMALIZATION:
            return ConvertL2Normalization(operation);
        case neuralnetworks::V1_0::OperationType::L2_POOL_2D:
            return ConvertL2Pool2d(operation);
        case neuralnetworks::V1_0::OperationType::MAX_POOL_2D:
            return ConvertMaxPool2d(operation);
        case neuralnetworks::V1_0::OperationType::MUL:
            return ConvertMul(operation);
        case neuralnetworks::V1_0::OperationType::RELU:
            return ConvertReLu(operation);
        case neuralnetworks::V1_0::OperationType::RELU1:
            return ConvertReLu1(operation);
        case neuralnetworks::V1_0::OperationType::RELU6:
            return ConvertReLu6(operation);
        case neuralnetworks::V1_0::OperationType::SOFTMAX:
            return ConvertSoftmax(operation);
        case neuralnetworks::V1_0::OperationType::TANH:
            return ConvertTanH(operation);
        case neuralnetworks::V1_0::OperationType::RESHAPE:
            return ConvertReshape(operation);
        case neuralnetworks::V1_0::OperationType::RESIZE_BILINEAR:
            return ConvertResizeBilinear(operation);
        default:
            return Fail("%s: Operation type %s not supported in ArmnnDriver",
                        __func__, toString(operation.type).c_str());
    }
}

#if defined(ARMNN_ANDROID_NN_V1_1)
template<typename HalVersion>
bool ModelToINetworkConverter<HalVersion>::ConvertOperation(const neuralnetworks::V1_1::Operation& operation)
{
    if (compliantWithV1_0(operation))
    {
        neuralnetworks::V1_0::Operation v1Operation = convertToV1_0(operation);
        return ConvertOperation(v1Operation);
    }
    else
    {
        switch (operation.type)
        {
            case neuralnetworks::V1_1::OperationType::DIV:
                return ConvertDiv(operation);
            default:
                return Fail("%s: Operation type %s not supported in ArmnnDriver",
                            __func__, toString(operation.type).c_str());
        }
    }
}

template<typename HalVersion>
bool ModelToINetworkConverter<HalVersion>::ConvertDiv(const neuralnetworks::V1_1::Operation& operation)
{
    LayerInputHandle input0 = ConvertToLayerInputHandle(operation, 0);
    LayerInputHandle input1 = ConvertToLayerInputHandle(operation, 1);

    if (!input0.IsValid() || !input1.IsValid())
    {
        return Fail("%s: Operation has invalid inputs", __func__);
    }

    // The FuseActivation parameter is always the input index 2
    // and it should be optional
    ActivationFn activationFunction;
    if (!GetOptionalInputActivation(operation, 2, activationFunction))
    {
        return Fail("%s: Operation has invalid inputs", __func__);
    }

    const Operand* outputOperand = GetOutputOperand(operation, 0);
    if (!outputOperand)
    {
        return false;
    }

    const armnn::TensorInfo& outInfo = GetTensorInfoForOperand(*outputOperand);

    if (!IsLayerSupported(__func__,
                          armnn::IsDivisionSupported,
                          m_Compute,
                          input0.GetTensorInfo(),
                          input1.GetTensorInfo(),
                          outInfo))
    {
        return false;
    }

    armnn::IConnectableLayer* const startLayer = m_Network->AddDivisionLayer();
    armnn::IConnectableLayer* const endLayer = ProcessActivation(outInfo, activationFunction, startLayer);

    const armnn::TensorInfo& inputTensorInfo0 = input0.GetTensorInfo();
    const armnn::TensorInfo& inputTensorInfo1 = input1.GetTensorInfo();

    if (endLayer)
    {
        BroadcastTensor(input0, input1, startLayer, *m_Network);
        return SetupAndTrackLayerOutputSlot(operation, 0, *endLayer);
    }

    return Fail("%s: ProcessActivation failed", __func__);
}
#endif

template<typename HalVersion>
bool ModelToINetworkConverter<HalVersion>::ConvertAdd(const neuralnetworks::V1_0::Operation& operation)
{
    LayerInputHandle input0 = ConvertToLayerInputHandle(operation, 0);
    LayerInputHandle input1 = ConvertToLayerInputHandle(operation, 1);

    if (!input0.IsValid() || !input1.IsValid())
    {
        return Fail("%s: Operation has invalid inputs", __func__);
    }

    // The FuseActivation parameter is always the input index 2
    // and it should be optional
    ActivationFn activationFunction;
    if (!GetOptionalInputActivation(operation, 2, activationFunction))
    {
        return Fail("%s: Operation has invalid inputs", __func__);
    }

    const Operand* outputOperand = GetOutputOperand(operation, 0);
    if (!outputOperand)
    {
        return false;
    }

    const armnn::TensorInfo outInfo = GetTensorInfoForOperand(*outputOperand);

    if (!IsLayerSupported(__func__,
                          armnn::IsAdditionSupported,
                          m_Compute,
                          input0.GetTensorInfo(),
                          input1.GetTensorInfo(),
                          outInfo))
    {
        return false;
    }

    armnn::IConnectableLayer* const startLayer = m_Network->AddAdditionLayer();
    armnn::IConnectableLayer* const endLayer = ProcessActivation(outInfo, activationFunction, startLayer);

    const armnn::TensorInfo& inputTensorInfo0 = input0.GetTensorInfo();
    const armnn::TensorInfo& inputTensorInfo1 = input1.GetTensorInfo();

    if (endLayer != nullptr)
    {
        BroadcastTensor(input0, input1, startLayer, *m_Network);
        return SetupAndTrackLayerOutputSlot(operation, 0, *endLayer);
    }
    else
    {
        return Fail("%s: ProcessActivation failed", __func__);
    }
}

template<typename HalVersion>
bool ModelToINetworkConverter<HalVersion>::ConvertAveragePool2d(const neuralnetworks::V1_0::Operation& operation)
{
    return ConvertPooling2d(operation, __func__, armnn::PoolingAlgorithm::Average);
}

template<typename HalVersion>
bool ModelToINetworkConverter<HalVersion>::ConvertConcatenation(const neuralnetworks::V1_0::Operation& operation)
{
    // The first N (0..N-1) inputs are tensors. The Nth input is the concatenation axis.
    if (operation.inputs.size() <= 1)
    {
        return Fail("%s: Operation has insufficient arguments", __func__);
    }

    // Get inputs and outputs
    const std::size_t numInputTensors = operation.inputs.size() - 1;

    int32_t concatDim;
    if (!GetInputScalar(operation, numInputTensors, OperandType::INT32, concatDim))
    {
        return Fail("%s: Operation has invalid inputs", __func__);
    }

    const Operand* const outputOperand = GetOutputOperand(operation, 0);
    if (!outputOperand)
    {
        return Fail("%s: Operation has no outputs", __func__);
    }


    armnn::TensorInfo  outputInfo  = GetTensorInfoForOperand(*outputOperand);
    armnn::TensorShape outputShape = outputInfo.GetShape();

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

    std::vector<LayerInputHandle> inputHandles;
    std::vector<armnn::TensorShape> inputShapes;

    inputHandles.reserve(numInputTensors);
    inputShapes.reserve(numInputTensors);

    bool inputsHaveBeenReshaped = false;
    unsigned int tensorDimensionsAdded = 0;

    for (uint32_t i = 0; i < numInputTensors; ++i)
    {
        const Operand* const operand = GetInputOperand(operation, i);
        if (!operand)
        {
            return Fail("%s: Operation has invalid inputs", __func__);
        }

        armnn::TensorShape operandShape = GetTensorShapeForOperand(*operand);
        LayerInputHandle operandInputHandle = ConvertToLayerInputHandle(operation, i);

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

            armnn::IConnectableLayer& newReshape = AddReshapeLayer(
                    *m_Network,
                    operandInputHandle,
                    reshapeInfo
            );

            // Point to the reshape operation rather then the input operation
            operandShape = reshapeInfo.GetShape();
            operandInputHandle = LayerInputHandle(true, &newReshape.GetOutputSlot(0), reshapeInfo);
        }

        inputShapes.emplace_back(operandShape);
        inputHandles.emplace_back(operandInputHandle);

        if (!inputHandles.back().IsValid())
        {
            return Fail("%s: Operation has invalid inputs", __func__);
        }
    }

    assert(inputShapes.size() == inputHandles.size());

    if (inputsHaveBeenReshaped)
    {
        // Adjust the concatenation dimension by the amount of dimensions added (if any)
        concatDim += tensorDimensionsAdded;

        // Add extra dimensions to the output shape to reflect the addition of the reshape layers
        if (tensorDimensionsAdded == 1)
        {
            outputShape = armnn::TensorShape({1, outputShape[0], outputShape[1]});
        }
        else if (tensorDimensionsAdded == 2)
        {
            outputShape = armnn::TensorShape({1, 1, outputShape[0], outputShape[1]});
        }
    }

    // Get the pair of permutations required for the concatenation
    std::pair<armnn::PermutationVector, armnn::PermutationVector> permutationPair =
            std::make_pair(IdentityPermutation4D, IdentityPermutation4D);

    CreatePermutationParameters(inputShapes[0].GetNumDimensions(), concatDim, permutationPair);

    outputShape = armnnUtils::Permuted(outputShape, permutationPair.first);
    outputInfo.SetShape(outputShape);

    // this is no-op for identity swizzles, otherwise it replaces both
    // the handles and shapes with the swizzled layer output handles and shapes
    SwizzleInputs(*m_Network, inputHandles, inputShapes, permutationPair.first);

    // Create an armnn merger layer descriptor - this will also perform validation on the input shapes
    armnn::OriginsDescriptor mergerDescriptor;
    try
    {
        // The merger descriptor is always created across the only supported concat
        // dimension, which is 0 or 1
        mergerDescriptor =
            armnn::CreateMergerDescriptorForConcatenation(
                inputShapes.begin(), inputShapes.end(), concatDim);
    }
    catch (const armnn::Exception& error)
    {
        return Fail("%s: Error preparing merger descriptor. %s", __func__, error.what());
    }

    // Validate the output shape is correct given the input shapes based on the
    // only valid concat dimension which is 0 or 1
    if (!ValidateConcatOutputShape(inputShapes, outputShape, concatDim))
    {
        return Fail("%s: Error validating the output shape for concat", __func__);
    }

    std::vector<const armnn::TensorInfo*> inputTensorInfos;
    std::transform(inputHandles.begin(), inputHandles.end(), std::back_inserter(inputTensorInfos),
        [](const LayerInputHandle& h) -> const armnn::TensorInfo*{ return &h.GetTensorInfo(); });
    if (!IsLayerSupported(__func__,
                          armnn::IsMergerSupported,
                          m_Compute,
                          inputTensorInfos,
                          mergerDescriptor))
    {
        return false;
    }

    armnn::IConnectableLayer* layer = m_Network->AddMergerLayer(mergerDescriptor);
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

    // Add permutation layer and connect the output to it, the permutation becomes the output layer
    armnn::IConnectableLayer& deswizzleLayer = AddPermuteLayer(*m_Network,
                                                               layer->GetOutputSlot(0),
                                                               permutationPair.second);
    layer = &deswizzleLayer;

    if (inputsHaveBeenReshaped)
    {
        armnn::TensorInfo afterConcatInfo = layer->GetOutputSlot(0).GetTensorInfo();

        // Undo the reshape knowing the amount of dimensions added
        if (tensorDimensionsAdded == 1)
        {
            afterConcatInfo.SetShape(armnn::TensorShape({ afterConcatInfo.GetShape()[1],
                                                          afterConcatInfo.GetShape()[2] }));
        }
        else if (tensorDimensionsAdded == 2)
        {
            afterConcatInfo.SetShape(armnn::TensorShape({ afterConcatInfo.GetShape()[2],
                                                          afterConcatInfo.GetShape()[3] }));
        }

        layer = &AddReshapeLayer(
                *m_Network,
                layer->GetOutputSlot(0),
                afterConcatInfo
        );
    }

    return SetupAndTrackLayerOutputSlot(operation, 0, *layer);
}

template<typename HalVersion>
bool ModelToINetworkConverter<HalVersion>::ConvertConv2d(const neuralnetworks::V1_0::Operation& operation)
{
    LayerInputHandle input = ConvertToLayerInputHandle(operation, 0);
    if (!input.IsValid())
    {
        return Fail("%s: Operation has invalid inputs", __func__);
    }

    const Operand* output = GetOutputOperand(operation, 0);
    if (!output)
    {
        return Fail("%s: Could not read output 0", __func__);
    }

    const armnn::TensorInfo& inputInfo = input.GetTensorInfo();
    const armnn::TensorInfo& outputInfo = GetTensorInfoForOperand(*output);

    const armnn::TensorInfo swizzledInputInfo = armnnUtils::Permuted(inputInfo, NHWCToArmNN);
    const armnn::TensorInfo swizzledOutputInfo = armnnUtils::Permuted(outputInfo, NHWCToArmNN);

    // ArmNN does not currently support non-fixed weights or bias
    const ConstTensorPin weightsPin = ConvertOperationInputToConstTensorPin(operation, 1, NHWCToArmNN);
    const ConstTensorPin biasPin = ConvertOperationInputToConstTensorPin(operation, 2);

    if (!weightsPin.IsValid() || !biasPin.IsValid())
    {
        return Fail("%s: Operation has invalid inputs", __func__);
    }

    armnn::ConstTensor weights = weightsPin.GetConstTensor();
    armnn::ConstTensor bias = biasPin.GetConstTensor();
    SanitizeBiasQuantizationScale(bias.GetInfo(), weights.GetInfo(), swizzledInputInfo);

    armnn::Convolution2dDescriptor desc;
    ActivationFn activation;

    if (operation.inputs.size() == 10)
    {
        if (!GetInputScalar(operation, 3, OperandType::INT32, desc.m_PadLeft)   ||
            !GetInputScalar(operation, 4, OperandType::INT32, desc.m_PadRight)  ||
            !GetInputScalar(operation, 5, OperandType::INT32, desc.m_PadTop)    ||
            !GetInputScalar(operation, 6, OperandType::INT32, desc.m_PadBottom) ||
            !GetInputScalar(operation, 7, OperandType::INT32, desc.m_StrideX)   ||
            !GetInputScalar(operation, 8, OperandType::INT32, desc.m_StrideY)   ||
            !GetInputActivationFunction(operation, 9, activation))
        {
            return Fail("%s: Operation has invalid inputs", __func__);
        }
    }
    else if (operation.inputs.size() == 7)
    {
        android::nn::PaddingScheme paddingScheme;

        if (!GetInputPaddingScheme(operation, 3, paddingScheme)               ||
            !GetInputScalar(operation, 4, OperandType::INT32, desc.m_StrideX) ||
            !GetInputScalar(operation, 5, OperandType::INT32, desc.m_StrideY) ||
            !GetInputActivationFunction(operation, 6, activation))
        {
            return Fail("%s: Operation has invalid inputs", __func__);
        }

        const uint32_t kernelX = weights.GetShape()[3];
        const uint32_t kernelY = weights.GetShape()[2];
        const uint32_t inputX  = swizzledInputInfo.GetShape()[3];
        const uint32_t inputY  = swizzledInputInfo.GetShape()[2];

        CalcPadding(inputX, kernelX, desc.m_StrideX, desc.m_PadLeft, desc.m_PadRight, paddingScheme);
        CalcPadding(inputY, kernelY, desc.m_StrideY, desc.m_PadTop, desc.m_PadBottom, paddingScheme);
    }
    else
    {
        return Fail("%s: Unsupported number of operation inputs", __func__);
    }

    desc.m_BiasEnabled = true;
    auto biases = boost::make_optional(bias.GetInfo());

    if (!IsLayerSupported(__func__,
                          armnn::IsConvolution2dSupported,
                          m_Compute,
                          swizzledInputInfo,
                          swizzledOutputInfo,
                          desc,
                          weights.GetInfo(),
                          biases))
    {
        return false;
    }

    armnn::IConnectableLayer* startLayer = m_Network->AddConvolution2dLayer(desc, weights, bias);
    armnn::IConnectableLayer* endLayer = ProcessActivation(swizzledOutputInfo, activation, startLayer);

    if (endLayer != nullptr)
    {
        armnn::IConnectableLayer& outSwizzleLayer = SwizzleInDeswizzleOut(*m_Network, input, *startLayer, *endLayer);
        return SetupAndTrackLayerOutputSlot(operation, 0, outSwizzleLayer);
    }
    else
    {
        return Fail("%s: ProcessActivation failed", __func__);
    }
}

template<typename HalVersion>
bool ModelToINetworkConverter<HalVersion>::ConvertDepthwiseConv2d(const neuralnetworks::V1_0::Operation& operation)
{
    LayerInputHandle input = ConvertToLayerInputHandle(operation, 0);
    if (!input.IsValid())
    {
        return Fail("%s: Operation has invalid inputs", __func__);
    }

    const Operand* output = GetOutputOperand(operation, 0);
    if (!output)
    {
        return Fail("%s: Could not read output 0", __func__);
    }

    const armnn::TensorInfo& inputInfo = input.GetTensorInfo();
    const armnn::TensorInfo& outputInfo = GetTensorInfoForOperand(*output);

    const armnn::TensorInfo swizzledInputInfo = armnnUtils::Permuted(inputInfo, NHWCToArmNN);
    const armnn::TensorInfo swizzledOutputInfo = armnnUtils::Permuted(outputInfo, NHWCToArmNN);

    // ArmNN does not currently support non-fixed weights or bias

    // Find the shape of the weights tensor. In AndroidNN this will be [ 1, H, W, I * M ]
    // but in ArmNN it needs to be [ M, I, H, W ]
    const Operand* weightsOperand = GetInputOperand(operation, 1);

    if (weightsOperand == nullptr)
    {
        return Fail("%s: Operand is invalid", __func__);
    }

    // Reinterpret weight data as [ H, W, I, M ]
    armnn::TensorShape weightsShape({ weightsOperand->dimensions[1], weightsOperand->dimensions[2],
                                      inputInfo.GetShape()[3],
                                      weightsOperand->dimensions[3] / inputInfo.GetShape()[3] });

    // Swizzle weight data [ H, W, I, M ] -> [ M, I, H, W ]
    const armnn::PermutationVector HWIMToMIHW = { 2U, 3U, 1U, 0U };
    ConstTensorPin weightsPin = ConvertOperationInputToConstTensorPin(operation, 1, HWIMToMIHW, &weightsShape);

    // Bias is a 1D tensor
    ConstTensorPin biasPin = ConvertOperationInputToConstTensorPin(operation, 2);

    if (!weightsPin.IsValid() || !biasPin.IsValid())
    {
        return Fail("%s: Operation has invalid inputs", __func__);
    }

    armnn::ConstTensor weights = weightsPin.GetConstTensor();
    armnn::ConstTensor bias = biasPin.GetConstTensor();
    SanitizeBiasQuantizationScale(bias.GetInfo(), weights.GetInfo(), swizzledInputInfo);

    armnn::DepthwiseConvolution2dDescriptor desc;
    ActivationFn activation;

    if (operation.inputs.size() == 11)
    {
        if (!GetInputScalar(operation, 3, OperandType::INT32, desc.m_PadLeft)         ||
            !GetInputScalar(operation, 4, OperandType::INT32, desc.m_PadRight)        ||
            !GetInputScalar(operation, 5, OperandType::INT32, desc.m_PadTop)          ||
            !GetInputScalar(operation, 6, OperandType::INT32, desc.m_PadBottom)       ||
            !GetInputScalar(operation, 7, OperandType::INT32, desc.m_StrideX)         ||
            !GetInputScalar(operation, 8, OperandType::INT32, desc.m_StrideY)         ||
            !GetInputActivationFunction(operation,  10, activation))
        {
            return Fail("%s: Operation has invalid inputs", __func__);
        }
    }
    else if (operation.inputs.size() == 8)
    {
        android::nn::PaddingScheme paddingScheme;

        if (!GetInputPaddingScheme(operation, 3, paddingScheme)                       ||
            !GetInputScalar(operation, 4, OperandType::INT32, desc.m_StrideX)         ||
            !GetInputScalar(operation, 5, OperandType::INT32, desc.m_StrideY)         ||
            !GetInputActivationFunction(operation, 7, activation))
        {
            return Fail("%s: Operation has invalid inputs", __func__);
        }

        const uint32_t kernelX = weights.GetShape()[3];
        const uint32_t kernelY = weights.GetShape()[2];
        const uint32_t inputX  = swizzledInputInfo.GetShape()[3];
        const uint32_t inputY  = swizzledInputInfo.GetShape()[2];

        CalcPadding(inputX, kernelX, desc.m_StrideX, desc.m_PadLeft, desc.m_PadRight, paddingScheme);
        CalcPadding(inputY, kernelY, desc.m_StrideY, desc.m_PadTop, desc.m_PadBottom, paddingScheme);
    }
    else
    {
        return Fail("%s: Unsupported number of operation inputs", __func__);
    }

    desc.m_BiasEnabled = true;
    auto biases = boost::make_optional(bias.GetInfo());

    if (!IsLayerSupported(__func__,
                          armnn::IsDepthwiseConvolutionSupported,
                          m_Compute,
                          swizzledInputInfo,
                          swizzledOutputInfo,
                          desc,
                          weights.GetInfo(),
                          biases))
    {
        return false;
    }

    armnn::IConnectableLayer* startLayer = m_Network->AddDepthwiseConvolution2dLayer(desc, weights, bias);
    armnn::IConnectableLayer* endLayer = ProcessActivation(swizzledOutputInfo, activation, startLayer);

    if (endLayer != nullptr)
    {
        armnn::IConnectableLayer& outSwizzleLayer = SwizzleInDeswizzleOut(*m_Network, input, *startLayer, *endLayer);
        return SetupAndTrackLayerOutputSlot(operation, 0, outSwizzleLayer);
    }
    else
    {
        return Fail("%s: ProcessActivation failed", __func__);
    }
}

template<typename HalVersion>
bool ModelToINetworkConverter<HalVersion>::ConvertFloor(const neuralnetworks::V1_0::Operation& operation)
{
    LayerInputHandle input = ConvertToLayerInputHandle(operation, 0);
    if (!input.IsValid())
    {
        return Fail("%s: Operation has invalid inputs", __func__);
    }

    const Operand* const outputOperand = GetOutputOperand(operation, 0);
    if (!outputOperand)
    {
        return Fail("%s: Operation has invalid outputs", __func__);
    }

    if (!IsLayerSupported(__func__,
                          armnn::IsFloorSupported,
                          m_Compute,
                          input.GetTensorInfo(),
                          GetTensorInfoForOperand(*outputOperand)))
    {
        return false;
    }

    armnn::IConnectableLayer* layer = m_Network->AddFloorLayer();
    assert(layer != nullptr);
    input.Connect(layer->GetInputSlot(0));

    return SetupAndTrackLayerOutputSlot(operation, 0, *layer);
}

template<typename HalVersion>
bool ModelToINetworkConverter<HalVersion>::ConvertFullyConnected(const neuralnetworks::V1_0::Operation& operation)
{
    LayerInputHandle input = ConvertToLayerInputHandle(operation, 0);
    if (!input.IsValid())
    {
        return Fail("%s: Operation has invalid inputs", __func__);
    }

    const Operand* output = GetOutputOperand(operation, 0);
    if (!output)
    {
        return Fail("%s: Could not read output 0", __func__);
    }

    const armnn::TensorInfo& inputInfo = input.GetTensorInfo();
    const armnn::TensorInfo& outputInfo = GetTensorInfoForOperand(*output);

    // ArmNN does not currently support non-fixed weights or bias
    ConstTensorPin weightsPin = ConvertOperationInputToConstTensorPin(operation, 1); // 2D
    ConstTensorPin biasPin = ConvertOperationInputToConstTensorPin(operation, 2);    // 1D

    if (!weightsPin.IsValid() || !biasPin.IsValid())
    {
        return Fail("%s: Operation has invalid inputs", __func__);
    }

    armnn::ConstTensor weights = weightsPin.GetConstTensor();
    armnn::ConstTensor bias = biasPin.GetConstTensor();

    armnn::TensorInfo reshapedInfo = inputInfo;
    if (inputInfo.GetNumDimensions() > 2U)
    {
        unsigned int dim0 = inputInfo.GetShape()[0];
        unsigned int dim1 = inputInfo.GetShape()[1];

        for (unsigned int i = 2U; i < inputInfo.GetNumDimensions(); ++i)
        {
            dim1 *= inputInfo.GetShape()[i];
        }

        unsigned int divisor = weights.GetInfo().GetShape()[1] / dim1;
        if(dim0 % divisor != 0)
        {
            return Fail("%s: Failed to deduce tensor shape", __func__);
        }

        reshapedInfo.SetShape(armnn::TensorShape({dim0 / divisor, dim1 * divisor}));
    }

    // ensuring that the bias value is within 1% of the weights input (small float differences can exist)
    SanitizeBiasQuantizationScale(bias.GetInfo(), weights.GetInfo(), reshapedInfo);

    ActivationFn activationFunction;
    if (!GetInputActivationFunction(operation, 3, activationFunction))
    {
        return Fail("%s: Operation has invalid inputs", __func__);
    }

    armnn::FullyConnectedDescriptor desc;
    desc.m_TransposeWeightMatrix = true;
    desc.m_BiasEnabled           = true;

    if (!IsLayerSupported(__func__,
                          armnn::IsFullyConnectedSupported,
                          m_Compute,
                          inputInfo,
                          outputInfo,
                          weights.GetInfo(),
                          bias.GetInfo(),
                          desc))
    {
        return false;
    }

    armnn::IConnectableLayer* startLayer = m_Network->AddFullyConnectedLayer(desc, weights, bias);
    armnn::IConnectableLayer* endLayer = ProcessActivation(outputInfo, activationFunction, startLayer);

    if (endLayer != nullptr)
    {
        if (inputInfo.GetNumDimensions() > 2U)
        {
            armnn::ReshapeDescriptor reshapeDescriptor;
            reshapeDescriptor.m_TargetShape = reshapedInfo.GetShape();

            armnn::IConnectableLayer* reshapeLayer = m_Network->AddReshapeLayer(reshapeDescriptor);
            assert(reshapeLayer != nullptr);
            input.Connect(reshapeLayer->GetInputSlot(0));
            reshapeLayer->GetOutputSlot(0).SetTensorInfo(reshapedInfo);
            reshapeLayer->GetOutputSlot(0).Connect(startLayer->GetInputSlot(0));
        }
        else
        {
            input.Connect(startLayer->GetInputSlot(0));
        }

        return SetupAndTrackLayerOutputSlot(operation, 0, *endLayer);
    }
    else
    {
        return Fail("%s: ProcessActivation failed", __func__);
    }
}

template<typename HalVersion>
bool ModelToINetworkConverter<HalVersion>::ConvertLocalResponseNormalization(
    const neuralnetworks::V1_0::Operation& operation)
{
    LayerInputHandle input = ConvertToLayerInputHandle(operation, 0);
    if (!input.IsValid())
    {
        return Fail("%s: Operation has invalid inputs", __func__);
    }

    const Operand* output = GetOutputOperand(operation, 0);
    if (!output)
    {
        return Fail("%s: Could not read output 0", __func__);
    }

    const armnn::TensorInfo& inputInfo = input.GetTensorInfo();
    const armnn::TensorInfo& outputInfo = GetTensorInfoForOperand(*output);

    const armnn::TensorInfo swizzledInputInfo = armnnUtils::Permuted(inputInfo, NHWCToArmNN);
    const armnn::TensorInfo swizzledOutputInfo = armnnUtils::Permuted(outputInfo, NHWCToArmNN);

    armnn::NormalizationDescriptor descriptor;

    descriptor.m_NormChannelType = armnn::NormalizationAlgorithmChannel::Across;
    descriptor.m_NormMethodType = armnn::NormalizationAlgorithmMethod::LocalBrightness;

    if (!input.IsValid() ||
        !GetInputScalar(operation, 1, OperandType::INT32, descriptor.m_NormSize) ||
        !GetInputFloat32(operation, 2, descriptor.m_K) ||
        !GetInputFloat32(operation, 3, descriptor.m_Alpha) ||
        !GetInputFloat32(operation, 4, descriptor.m_Beta))
    {
        return Fail("%s: Operation has invalid inputs", __func__);
    }

    // ArmNN expects normSize to be the full size of the normalization
    // window rather than the radius as in AndroidNN.
    descriptor.m_NormSize = 1 + (2 * descriptor.m_NormSize);

    if (!IsLayerSupported(__func__,
                        armnn::IsNormalizationSupported,
                        m_Compute,
                        swizzledInputInfo,
                        swizzledOutputInfo,
                        descriptor))
    {
        return false;
    }


    armnn::IConnectableLayer* layer = m_Network->AddNormalizationLayer(descriptor);
    assert(layer != nullptr);
    layer->GetOutputSlot(0).SetTensorInfo(swizzledOutputInfo);

    armnn::IConnectableLayer& outSwizzleLayer = SwizzleInDeswizzleOut(*m_Network, input, *layer);

    return SetupAndTrackLayerOutputSlot(operation, 0, outSwizzleLayer);
}

template<typename HalVersion>
bool ModelToINetworkConverter<HalVersion>::ConvertLogistic(const neuralnetworks::V1_0::Operation& operation)
{
    armnn::ActivationDescriptor desc;
    desc.m_Function = armnn::ActivationFunction::Sigmoid;

    return ConvertToActivation(operation, __func__, desc);
}

template<typename HalVersion>
bool ModelToINetworkConverter<HalVersion>::ConvertL2Normalization(const neuralnetworks::V1_0::Operation& operation)
{
    LayerInputHandle input = ConvertToLayerInputHandle(operation, 0);
    if (!input.IsValid())
    {
        return Fail("%s: Operation has invalid inputs", __func__);
    }

    const Operand* output = GetOutputOperand(operation, 0);
    if (!output)
    {
        return Fail("%s: Could not read output 0", __func__);
    }

    const armnn::TensorInfo& inputInfo = input.GetTensorInfo();
    const armnn::TensorInfo& outputInfo = GetTensorInfoForOperand(*output);

    const armnn::TensorInfo swizzledInputInfo = armnnUtils::Permuted(inputInfo, NHWCToArmNN);
    const armnn::TensorInfo swizzledOutputInfo = armnnUtils::Permuted(outputInfo, NHWCToArmNN);

    if (!IsLayerSupported(__func__,
                          armnn::IsL2NormalizationSupported,
                          m_Compute,
                          swizzledInputInfo,
                          swizzledOutputInfo))
    {
        return false;
    }

    armnn::IConnectableLayer* layer = m_Network->AddL2NormalizationLayer();
    assert(layer != nullptr);
    layer->GetOutputSlot(0).SetTensorInfo(swizzledOutputInfo);

    armnn::IConnectableLayer& outSwizzleLayer = SwizzleInDeswizzleOut(*m_Network, input, *layer);

    return SetupAndTrackLayerOutputSlot(operation, 0, outSwizzleLayer);
}

template<typename HalVersion>
bool ModelToINetworkConverter<HalVersion>::ConvertL2Pool2d(const neuralnetworks::V1_0::Operation& operation)
{
    return ConvertPooling2d(operation, __func__, armnn::PoolingAlgorithm::L2);
}

template<typename HalVersion>
bool ModelToINetworkConverter<HalVersion>::ConvertMaxPool2d(const neuralnetworks::V1_0::Operation& operation)
{
    return ConvertPooling2d(operation, __func__, armnn::PoolingAlgorithm::Max);
}

template<typename HalVersion>
bool ModelToINetworkConverter<HalVersion>::ConvertMul(const neuralnetworks::V1_0::Operation& operation)
{
    LayerInputHandle input0 = ConvertToLayerInputHandle(operation, 0);
    LayerInputHandle input1 = ConvertToLayerInputHandle(operation, 1);

    if (!input0.IsValid() || !input1.IsValid())
    {
        return Fail("%s: Operation has invalid inputs", __func__);
    }

    // The FuseActivation parameter is always the input index 2
    // and it should be optional
    ActivationFn activationFunction;
    if (!GetOptionalInputActivation(operation, 2, activationFunction))
    {
        return Fail("%s: Operation has invalid inputs", __func__);
    }

    const Operand* outputOperand = GetOutputOperand(operation, 0);

    if (outputOperand == nullptr)
    {
        return false;
    }

    const armnn::TensorInfo& outInfo = GetTensorInfoForOperand(*outputOperand);

    if (!IsLayerSupported(__func__,
                          armnn::IsMultiplicationSupported,
                          m_Compute,
                          input0.GetTensorInfo(),
                          input1.GetTensorInfo(),
                          outInfo))
    {
        return false;
    }

    armnn::IConnectableLayer* const startLayer = m_Network->AddMultiplicationLayer();
    armnn::IConnectableLayer* const endLayer = ProcessActivation(outInfo, activationFunction, startLayer);

    const armnn::TensorInfo& inputTensorInfo0 = input0.GetTensorInfo();
    const armnn::TensorInfo& inputTensorInfo1 = input1.GetTensorInfo();

    if (endLayer != nullptr)
    {
        BroadcastTensor(input0, input1, startLayer, *m_Network);
        return SetupAndTrackLayerOutputSlot(operation, 0, *endLayer);
    }
    else
    {
        return Fail("%s: ProcessActivation failed", __func__);
    }
}

template<typename HalVersion>
bool ModelToINetworkConverter<HalVersion>::ConvertReLu(const neuralnetworks::V1_0::Operation& operation)
{
    armnn::ActivationDescriptor desc;
    desc.m_Function = armnn::ActivationFunction::ReLu;

    return ConvertToActivation(operation, __func__, desc);
}

template<typename HalVersion>
bool ModelToINetworkConverter<HalVersion>::ConvertReLu1(const neuralnetworks::V1_0::Operation& operation)
{
    armnn::ActivationDescriptor desc;
    desc.m_Function = armnn::ActivationFunction::BoundedReLu;
    desc.m_A        = 1.0f;
    desc.m_B        = -1.0f;

    return ConvertToActivation(operation, __func__, desc);
}

template<typename HalVersion>
bool ModelToINetworkConverter<HalVersion>::ConvertReLu6(const neuralnetworks::V1_0::Operation& operation)
{
    armnn::ActivationDescriptor desc;
    desc.m_Function = armnn::ActivationFunction::BoundedReLu;
    desc.m_A        = 6.0f;

    return ConvertToActivation(operation, __func__, desc);
}

template<typename HalVersion>
bool ModelToINetworkConverter<HalVersion>::ConvertSoftmax(const neuralnetworks::V1_0::Operation& operation)
{
    LayerInputHandle input = ConvertToLayerInputHandle(operation, 0);
    if (!input.IsValid())
    {
        return Fail("%s: Operation has invalid inputs", __func__);
    }

    const Operand* outputOperand = GetOutputOperand(operation, 0);
    if (!outputOperand)
    {
        return Fail("%s: Operation has no outputs", __func__);
    }

    const armnn::TensorInfo outInfo = GetTensorInfoForOperand(*outputOperand);

    armnn::SoftmaxDescriptor desc;
    if (!GetInputFloat32(operation, 1, desc.m_Beta))
    {
        return Fail("%s: Operation has invalid inputs", __func__);
    }

    if (!IsLayerSupported(__func__,
                          armnn::IsSoftmaxSupported,
                          m_Compute,
                          input.GetTensorInfo(),
                          outInfo,
                          desc))
    {
        return false;
    }

    armnn::IConnectableLayer* layer = m_Network->AddSoftmaxLayer(desc);
    assert(layer != nullptr);
    input.Connect(layer->GetInputSlot(0));

    return SetupAndTrackLayerOutputSlot(operation, 0, *layer);
}

template<typename HalVersion>
bool ModelToINetworkConverter<HalVersion>::ConvertTanH(const neuralnetworks::V1_0::Operation& operation)
{
    armnn::ActivationDescriptor desc;
    desc.m_Function = armnn::ActivationFunction::TanH;
    desc.m_A = 1.0f; // android nn does not support tanH parameters
    desc.m_B = 1.0f; // set to 1.0f for unity scaling

    return ConvertToActivation(operation, __func__, desc);
}

template<typename HalVersion>
bool ModelToINetworkConverter<HalVersion>::ConvertReshape(const neuralnetworks::V1_0::Operation& operation)
{
    const Operand* inputOperand = GetInputOperand(operation, 0);
    const Operand* requestedShapeOperand = GetInputOperand(operation, 1);
    const Operand* outputOperand = GetOutputOperand(operation, 0);

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
    if (!GetTensorInt32Values(*requestedShapeOperand, targetDimensions))
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

    const Shape outputOperandShape = GetOperandShape(*outputOperand);
    if (!SameShape(requestedShape, outputOperandShape))
    {
        return Fail("%s: Shape of output operand does not match resolved requested shape", __func__);
    }

    LayerInputHandle input = ConvertToLayerInputHandle(operation, 0);
    if (!input.IsValid())
    {
        return Fail("%s: Could not read input 0", __func__);
    }

    if (!IsLayerSupported(__func__,
                          armnn::IsReshapeSupported,
                          m_Compute,
                          input.GetTensorInfo()))
    {
        return false;
    }


    armnn::ReshapeDescriptor reshapeDescriptor;
    reshapeDescriptor.m_TargetShape = armnn::TensorShape(requestedShape.dimensions.size(),
                                                         requestedShape.dimensions.data());

    armnn::IConnectableLayer* layer = m_Network->AddReshapeLayer(reshapeDescriptor);
    assert(layer != nullptr);
    input.Connect(layer->GetInputSlot(0));

    return SetupAndTrackLayerOutputSlot(operation, 0, *layer);
}

template<typename HalVersion>
bool ModelToINetworkConverter<HalVersion>::ConvertResizeBilinear(const neuralnetworks::V1_0::Operation& operation)
{
    LayerInputHandle input = ConvertToLayerInputHandle(operation, 0);
    if (!input.IsValid())
    {
        return Fail("%s: Could not read input 0", __func__);
    }

    const Operand* output = GetOutputOperand(operation, 0);
    if (!output)
    {
        return Fail("%s: Could not read output 0", __func__);
    }

    const armnn::TensorInfo& inputInfo = input.GetTensorInfo();
    const armnn::TensorInfo& outputInfo = GetTensorInfoForOperand(*output);

    const armnn::TensorInfo swizzledInputInfo = armnnUtils::Permuted(inputInfo, NHWCToArmNN);
    const armnn::TensorInfo swizzledOutputInfo = armnnUtils::Permuted(outputInfo, NHWCToArmNN);

    if (!IsLayerSupported(__func__,
                          armnn::IsResizeBilinearSupported,
                          m_Compute,
                          swizzledInputInfo))
    {
        return false;
    }

    armnn::ResizeBilinearDescriptor desc;

    if (   !GetInputScalar(operation, 1, OperandType::INT32, desc.m_TargetHeight)
        || !GetInputScalar(operation, 2, OperandType::INT32, desc.m_TargetWidth))
    {
        return Fail("%s: Operation has invalid inputs", __func__);
    }

    armnn::IConnectableLayer* layer = m_Network->AddResizeBilinearLayer(desc);
    assert(layer != nullptr);
    layer->GetOutputSlot(0).SetTensorInfo(swizzledOutputInfo);

    armnn::IConnectableLayer& outSwizzleLayer = SwizzleInDeswizzleOut(*m_Network, input, *layer);

    return SetupAndTrackLayerOutputSlot(operation, 0, outSwizzleLayer);

}

template<typename HalVersion>
bool ModelToINetworkConverter<HalVersion>::ConvertLstm(const neuralnetworks::V1_0::Operation& operation)
{
    // Inputs:
    // 00: The input: A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [batch_size, input_size], where
    //      “batch_size” corresponds to the batching dimension, and “input_size” is the size of the input.
    LayerInputHandle input = ConvertToLayerInputHandle(operation, 0);
    if (!input.IsValid())
    {
        return Fail("%s: Could not read input 0: input", __func__);
    }
    // 18: The output state: A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [batch_size, output_size].
    LayerInputHandle outputStateIn = ConvertToLayerInputHandle(operation, 18);
    if (!outputStateIn.IsValid())
    {
        return Fail("%s: Could not read input 18: outputStateIn", __func__);
    }
    // 19: The cell state: A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [batch_size, num_units].
    LayerInputHandle cellStateIn = ConvertToLayerInputHandle(operation, 19);
    if (!cellStateIn.IsValid())
    {
        return Fail("%s: Could not read input 19: cellStateIn", __func__);
    }

    // Get the mandatory input tensors:
    // 02: The input-to-forget weights: A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape
    //     [num_units, input_size].
    const ConstTensorPin inputToForgetWeightsPin = ConvertOperationInputToConstTensorPin(operation, 2);
    // 03: The input-to-cell weights: A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [num_units, input_size].
    const ConstTensorPin inputToCellWeightsPin = ConvertOperationInputToConstTensorPin(operation, 3);
    // 04: The input-to-output weights: A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape
    //     [num_units, input_size].
    const ConstTensorPin inputToOutputWeightsPin = ConvertOperationInputToConstTensorPin(operation, 4);
    // 06: The recurrent-to-forget weights: A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape
    //     [num_units, output_size].
    const ConstTensorPin recurrentToForgetWeightsPin = ConvertOperationInputToConstTensorPin(operation, 6);
    // 07: The recurrent-to-cell weights: A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape
    //     [num_units, output_size].
    const ConstTensorPin recurrentToCellWeightsPin = ConvertOperationInputToConstTensorPin(operation, 7);
    // 08: The recurrent-to-output weights: A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape
    //     [num_units, output_size].
    const ConstTensorPin recurrentToOutputWeightsPin = ConvertOperationInputToConstTensorPin(operation, 8);
    // 13: The forget gate bias: A 1-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [num_units].
    const ConstTensorPin forgetGateBiasPin = ConvertOperationInputToConstTensorPin(operation, 13);
    // 14: The cell bias: A 1-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [num_units].
    const ConstTensorPin cellBiasPin = ConvertOperationInputToConstTensorPin(operation, 14);
    // 15: The output gate bias: A 1-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [num_units].
    const ConstTensorPin outputGateBiasPin = ConvertOperationInputToConstTensorPin(operation, 15);

    if (!inputToForgetWeightsPin.IsValid() ||
        !inputToCellWeightsPin.IsValid() ||
        !inputToOutputWeightsPin.IsValid() ||
        !recurrentToForgetWeightsPin.IsValid() ||
        !recurrentToCellWeightsPin.IsValid() ||
        !recurrentToOutputWeightsPin.IsValid() ||
        !forgetGateBiasPin.IsValid() ||
        !cellBiasPin.IsValid() ||
        !outputGateBiasPin.IsValid())
    {
        return Fail("%s: Operation has invalid tensor inputs", __func__);
    }

    // Get the optional input tensors:
    // 01: The input-to-input weights: Optional. A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape
    //     [num_units, input_size], where “num_units” corresponds to the number of cell units.
    const ConstTensorPin inputToInputWeightsPin = ConvertOperationInputToConstTensorPin(operation, 1);
    // 05: The recurrent-to-input weights: Optional. A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape
    //     [num_units, output_size], where “output_size” corresponds to either the number of cell units (i.e.,
    //     “num_units”), or the second dimension of the “projection_weights”, if defined.
    const ConstTensorPin recurrentToInputWeightsPin = ConvertOperationInputToConstTensorPin(operation, 5);
    // 09: The cell-to-input weights: Optional. A 1-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [num_units].
    const ConstTensorPin cellToInputWeightsPin = ConvertOperationInputToConstTensorPin(operation, 9);
    // 10: The cell-to-forget weights: Optional. A 1-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [num_units].
    const ConstTensorPin cellToForgetWeightsPin = ConvertOperationInputToConstTensorPin(operation, 10);
    // 11: The cell-to-output weights: Optional. A 1-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [num_units].
    const ConstTensorPin cellToOutputWeightsPin = ConvertOperationInputToConstTensorPin(operation, 11);
    // 12: The input gate bias: Optional. A 1-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [num_units].
    const ConstTensorPin inputGateBiasPin = ConvertOperationInputToConstTensorPin(operation, 12);
    // 16: The projection weights: Optional. A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape
    //     [output_size, num_units].
    const ConstTensorPin projectionWeightsPin = ConvertOperationInputToConstTensorPin(operation, 16);
    // 17: The projection bias: Optional. A 1-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [output_size].
    const ConstTensorPin projectionBiasPin = ConvertOperationInputToConstTensorPin(operation, 17);

    if ((!inputToInputWeightsPin.IsValid() && !inputToInputWeightsPin.IsOptional()) ||
        (!recurrentToInputWeightsPin.IsValid() && !recurrentToInputWeightsPin.IsOptional()) ||
        (!cellToInputWeightsPin.IsValid() && !cellToInputWeightsPin.IsOptional()) ||
        (!cellToForgetWeightsPin.IsValid() && !cellToForgetWeightsPin.IsOptional()) ||
        (!cellToOutputWeightsPin.IsValid() && !cellToOutputWeightsPin.IsOptional()) ||
        (!inputGateBiasPin.IsValid() && !inputGateBiasPin.IsOptional()) ||
        (!projectionWeightsPin.IsValid() && !projectionWeightsPin.IsOptional()) ||
        (!projectionBiasPin.IsValid() && !projectionBiasPin.IsOptional()))
    {
        return Fail("%s: Operation has invalid tensor inputs", __func__);
    }

    // Get the mandatory input scalars (actually 1-D tensors of size 1):
    // 20: The activation function: A value indicating the activation function:
    //     0: None; 1: Relu; 3: Relu6; 4: Tanh; 6: Sigmoid.
    // 21: The clipping threshold: for the cell state, such that values are bound within [-cell_clip, cell_clip].
    //     If set to 0.0 then clipping is disabled.
    // 22: The clipping threshold: for the output from the projection layer, such that values are bound within
    //     [-proj_clip, proj_clip]. If set to 0.0 then clipping is disabled.
    ActivationFn activation;
    float cellClip;
    float projClip;
    if (!GetInputActivationFunctionFromTensor(operation, 20, activation) ||
        !GetInputScalar(operation, 21, OperandType::FLOAT32, cellClip) ||
        !GetInputScalar(operation, 22, OperandType::FLOAT32, projClip))
    {
        return Fail("%s: Operation has invalid scalar inputs", __func__);
    }

    // Outputs:
    // 00: The scratch buffer: A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [batch_size, num_units * 4] with
    //     CIFG, or [batch_size, num_units * 3] without CIFG.
    const Operand* scratchBuffer = GetOutputOperand(operation, 0);
    if (!scratchBuffer)
    {
        return Fail("%s: Could not read output 0: scratchBuffer", __func__);
    }
    // 01: The output state (out): A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [batch_size, output_size].
    const Operand* outputStateOut = GetOutputOperand(operation, 1);
    if (!outputStateOut)
    {
        return Fail("%s: Could not read output 1: outputStateOut", __func__);
    }
    // 02: The cell state (out): A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [batch_size, num_units].
    const Operand* cellStateOut = GetOutputOperand(operation, 2);
    if (!cellStateOut)
    {
        return Fail("%s: Could not read output 2: cellStateOut", __func__);
    }
    // 03: The output: A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [batch_size, output_size]. This is
    //     effectively the same as the current “output state (out)” value.
    const Operand* output = GetOutputOperand(operation, 3);
    if (!output)
    {
        return Fail("%s: Could not read output 3: output", __func__);
    }

    // set the params structure for the AddLstmLayer call
    armnn::LstmInputParams params;
    params.m_InputToInputWeights = inputToInputWeightsPin.GetConstTensorPtr();
    params.m_InputToForgetWeights = inputToForgetWeightsPin.GetConstTensorPtr();
    params.m_InputToCellWeights = inputToCellWeightsPin.GetConstTensorPtr();
    params.m_InputToOutputWeights = inputToOutputWeightsPin.GetConstTensorPtr();
    params.m_RecurrentToInputWeights = recurrentToInputWeightsPin.GetConstTensorPtr();
    params.m_RecurrentToForgetWeights = recurrentToForgetWeightsPin.GetConstTensorPtr();
    params.m_RecurrentToCellWeights = recurrentToCellWeightsPin.GetConstTensorPtr();
    params.m_RecurrentToOutputWeights = recurrentToOutputWeightsPin.GetConstTensorPtr();
    params.m_CellToInputWeights = cellToInputWeightsPin.GetConstTensorPtr();
    params.m_CellToForgetWeights = cellToForgetWeightsPin.GetConstTensorPtr();
    params.m_CellToOutputWeights = cellToOutputWeightsPin.GetConstTensorPtr();
    params.m_InputGateBias = inputGateBiasPin.GetConstTensorPtr();
    params.m_ForgetGateBias = forgetGateBiasPin.GetConstTensorPtr();
    params.m_CellBias = cellBiasPin.GetConstTensorPtr();
    params.m_OutputGateBias = outputGateBiasPin.GetConstTensorPtr();
    params.m_ProjectionWeights = projectionWeightsPin.GetConstTensorPtr();
    params.m_ProjectionBias = projectionBiasPin.GetConstTensorPtr();

    // set the layer descriptor
    armnn::LstmDescriptor desc;
    desc.m_ActivationFunc = activation;
    desc.m_ClippingThresCell = cellClip;
    desc.m_ClippingThresProj = projClip;
    desc.m_CifgEnabled = (params.m_InputToInputWeights == nullptr ||
                          params.m_RecurrentToInputWeights == nullptr ||
                          params.m_InputGateBias == nullptr);
    desc.m_PeepholeEnabled = (params.m_CellToForgetWeights != nullptr ||
                              params.m_CellToOutputWeights != nullptr);
    desc.m_ProjectionEnabled = (params.m_ProjectionWeights != nullptr);

    // validate the optional input groups
    if (desc.m_CifgEnabled &&
        (params.m_InputToInputWeights != nullptr ||
         params.m_RecurrentToInputWeights != nullptr ||
         params.m_InputGateBias != nullptr))
    {
        return Fail("%s: All, or none, of input-to-input weights, recurrent-to-input weights,"
                    " and input gate bias must be provided", __func__);
    }

    if (!desc.m_ProjectionEnabled && params.m_ProjectionBias != nullptr)
    {
        return Fail("%s: projection bias should not be provided without projection weights", __func__);
    }

    if (desc.m_PeepholeEnabled &&
        (params.m_CellToForgetWeights == nullptr ||
         params.m_CellToOutputWeights == nullptr ||
         (!desc.m_CifgEnabled && params.m_CellToInputWeights == nullptr)))
    {
        return Fail("%s: All, or none, of cell-to-forget weights and cell-to-output weights must be provided"
                    " and, if CIFG is not enabled, cell-to-input weights must also be provided", __func__);
    }

    // Check if the layer is supported
    // Inputs
    const armnn::TensorInfo& inputInfo         = input.GetTensorInfo();
    const armnn::TensorInfo& outputStateInInfo = outputStateIn.GetTensorInfo();
    const armnn::TensorInfo& cellStateInInfo   = cellStateIn.GetTensorInfo();

    // Outputs
    const armnn::TensorInfo& scratchBufferInfo  = GetTensorInfoForOperand(*scratchBuffer);
    const armnn::TensorInfo& outputStateOutInfo = GetTensorInfoForOperand(*outputStateOut);
    const armnn::TensorInfo& cellStateOutInfo   = GetTensorInfoForOperand(*cellStateOut);
    const armnn::TensorInfo& outputInfo         = GetTensorInfoForOperand(*output);

    // Basic parameters
    const armnn::TensorInfo& inputToForgetWeights = params.m_InputToForgetWeights->GetInfo();
    const armnn::TensorInfo& inputToCellWeights   = params.m_InputToCellWeights->GetInfo();
    const armnn::TensorInfo& inputToOutputWeights = params.m_InputToOutputWeights->GetInfo();
    const armnn::TensorInfo& recurrentToForgetWeights = params.m_RecurrentToForgetWeights->GetInfo();
    const armnn::TensorInfo& recurrentToCellWeights = params.m_RecurrentToCellWeights->GetInfo();
    const armnn::TensorInfo& recurrentToOutputWeights = params.m_RecurrentToOutputWeights->GetInfo();
    const armnn::TensorInfo& forgetGateBias = params.m_ForgetGateBias->GetInfo();
    const armnn::TensorInfo& cellBias = params.m_CellBias->GetInfo();
    const armnn::TensorInfo& outputGateBias = params.m_OutputGateBias->GetInfo();

    //Optional parameters
    const armnn::TensorInfo* inputToInputWeights = nullptr;
    const armnn::TensorInfo* recurrentToInputWeights = nullptr;
    const armnn::TensorInfo* cellToInputWeights = nullptr;
    const armnn::TensorInfo* inputGateBias = nullptr;
    const armnn::TensorInfo* projectionWeights = nullptr;
    const armnn::TensorInfo* projectionBias    = nullptr;
    const armnn::TensorInfo* cellToForgetWeights = nullptr;
    const armnn::TensorInfo* cellToOutputWeights = nullptr;

    if(!desc.m_CifgEnabled)
    {
        inputToInputWeights = &(params.m_InputToInputWeights->GetInfo());
        recurrentToInputWeights = &(params.m_RecurrentToInputWeights->GetInfo());
        if (params.m_CellToInputWeights != nullptr)
        {
            cellToInputWeights = &(params.m_CellToInputWeights->GetInfo());
        }
        inputGateBias = &(params.m_InputGateBias->GetInfo());
    }

    if(desc.m_ProjectionEnabled)
    {
        projectionWeights = &(params.m_ProjectionWeights->GetInfo());
        if (params.m_ProjectionBias != nullptr)
        {
            projectionBias = &(params.m_ProjectionBias->GetInfo());
        }
    }

    if(desc.m_PeepholeEnabled)
    {
        cellToForgetWeights = &(params.m_CellToForgetWeights->GetInfo());
        cellToOutputWeights = &(params.m_CellToOutputWeights->GetInfo());
    }

    if (!IsLayerSupported(__func__,
                          armnn::IsLstmSupported,
                          m_Compute,
                          inputInfo,
                          outputStateInInfo,
                          cellStateInInfo,
                          scratchBufferInfo,
                          outputStateOutInfo,
                          cellStateOutInfo,
                          outputInfo,
                          desc,
                          inputToForgetWeights,
                          inputToCellWeights,
                          inputToOutputWeights,
                          recurrentToForgetWeights,
                          recurrentToCellWeights,
                          recurrentToOutputWeights,
                          forgetGateBias,
                          cellBias,
                          outputGateBias,
                          inputToInputWeights,
                          recurrentToInputWeights,
                          cellToInputWeights,
                          inputGateBias,
                          projectionWeights,
                          projectionBias,
                          cellToForgetWeights,
                          cellToOutputWeights))
    {
        return false;
    }

    // Add the layer
    armnn::IConnectableLayer* layer = m_Network->AddLstmLayer(desc, params, "Lstm");

    input.Connect(layer->GetInputSlot(0));
    outputStateIn.Connect(layer->GetInputSlot(1));
    cellStateIn.Connect(layer->GetInputSlot(2));

    return (SetupAndTrackLayerOutputSlot(operation, 0, *layer, 0) &&
            SetupAndTrackLayerOutputSlot(operation, 1, *layer, 1) &&
            SetupAndTrackLayerOutputSlot(operation, 2, *layer, 2) &&
            SetupAndTrackLayerOutputSlot(operation, 3, *layer, 3));
}

template<typename HalVersion>
bool ModelToINetworkConverter<HalVersion>::ConvertToActivation(const neuralnetworks::V1_0::Operation& operation,
    const char* operationName,
    const armnn::ActivationDescriptor& activationDesc)
{
    LayerInputHandle input = ConvertToLayerInputHandle(operation, 0);
    if (!input.IsValid())
    {
        return Fail("%s: Input 0 is invalid", operationName);
    }

    const Operand* outputOperand = GetOutputOperand(operation, 0);
    if (!outputOperand)
    {
        return false;
    }
    const armnn::TensorInfo outInfo = GetTensorInfoForOperand(*outputOperand);
    if (!IsLayerSupported(__func__,
                          armnn::IsActivationSupported,
                          m_Compute,
                          input.GetTensorInfo(),
                          outInfo,
                          activationDesc))
    {
        return false;
    }

    armnn::IConnectableLayer* layer = m_Network->AddActivationLayer(activationDesc);
    assert(layer != nullptr);
    input.Connect(layer->GetInputSlot(0));

    return SetupAndTrackLayerOutputSlot(operation, 0, *layer);
}

template<typename HalVersion>
bool ModelToINetworkConverter<HalVersion>::ConvertPooling2d(const neuralnetworks::V1_0::Operation& operation,
    const char* operationName,
    armnn::PoolingAlgorithm poolType)
{
    LayerInputHandle input = ConvertToLayerInputHandle(operation, 0);
    if (!input.IsValid())
    {
        return Fail("%s: Could not read input 0", operationName);
    }

    const Operand* output = GetOutputOperand(operation, 0);
    if (!output)
    {
        return Fail("%s: Could not read output 0", __func__);
    }

    const armnn::TensorInfo& inputInfo = input.GetTensorInfo();
    const armnn::TensorInfo& outputInfo = GetTensorInfoForOperand(*output);

    const armnn::TensorInfo swizzledInputInfo = armnnUtils::Permuted(inputInfo, NHWCToArmNN);
    const armnn::TensorInfo swizzledOutputInfo = armnnUtils::Permuted(outputInfo, NHWCToArmNN);

    armnn::Pooling2dDescriptor desc;
    desc.m_PoolType = poolType;
    desc.m_OutputShapeRounding = armnn::OutputShapeRounding::Floor;

    ActivationFn activation;

    if (operation.inputs.size() == 7)
    {
        // one input, 6 parameters (padding, stridex, stridey, width, height, activation type)
        android::nn::PaddingScheme scheme;

        if (   !GetInputPaddingScheme(operation, 1, scheme)
            || !GetInputScalar(operation, 2, OperandType::INT32, desc.m_StrideX)
            || !GetInputScalar(operation, 3, OperandType::INT32, desc.m_StrideY)
            || !GetInputScalar(operation, 4, OperandType::INT32, desc.m_PoolWidth)
            || !GetInputScalar(operation, 5, OperandType::INT32, desc.m_PoolHeight)
            || !GetInputActivationFunction(operation, 6, activation))
        {
            return Fail("%s: Operation has invalid inputs", operationName);
        }

        const unsigned int inputWidth = swizzledInputInfo.GetShape()[3];
        const unsigned int inputHeight = swizzledInputInfo.GetShape()[2];

        CalcPadding(inputWidth, desc.m_PoolWidth, desc.m_StrideX, desc.m_PadLeft, desc.m_PadRight, scheme);
        CalcPadding(inputHeight, desc.m_PoolHeight, desc.m_StrideY, desc.m_PadTop, desc.m_PadBottom, scheme);
    }
    else
    {
        // one input, 9 parameters (padding l r t b, stridex, stridey, width, height, activation type)
        if (   !GetInputScalar(operation, 1, OperandType::INT32, desc.m_PadLeft)
            || !GetInputScalar(operation, 2, OperandType::INT32, desc.m_PadRight)
            || !GetInputScalar(operation, 3, OperandType::INT32, desc.m_PadTop)
            || !GetInputScalar(operation, 4, OperandType::INT32, desc.m_PadBottom)
            || !GetInputScalar(operation, 5, OperandType::INT32, desc.m_StrideX)
            || !GetInputScalar(operation, 6, OperandType::INT32, desc.m_StrideY)
            || !GetInputScalar(operation, 7, OperandType::INT32, desc.m_PoolWidth)
            || !GetInputScalar(operation, 8, OperandType::INT32, desc.m_PoolHeight)
            || !GetInputActivationFunction(operation, 9, activation))
        {
            return Fail("%s: Operation has invalid inputs", operationName);
        }
    }

    // ArmNN does not accept a pool size of 1, but the ArmNN driver is expected to cope.
    // This is mapped to a trivial splitter instead.
    armnn::IConnectableLayer* startLayer = nullptr;
    if (desc.m_PoolWidth != 1 || desc.m_PoolHeight != 1)
    {
        if (!IsLayerSupported(__func__,
                              armnn::IsPooling2dSupported,
                              m_Compute,
                              swizzledInputInfo,
                              swizzledOutputInfo,
                              desc))
        {
            return false;
        }

        startLayer = m_Network->AddPooling2dLayer(desc);
    }
    else
    {
        const unsigned int numDims = swizzledOutputInfo.GetNumDimensions();

        armnn::ViewsDescriptor viewsDesc(1, numDims);

        for (unsigned int i = 0; i < numDims; ++i)
        {
            viewsDesc.SetViewOriginCoord(0, i, 0);
            viewsDesc.SetViewSize(0, i, swizzledOutputInfo.GetShape()[i]);
        }

        if (!IsLayerSupported(__func__,
                              armnn::IsSplitterSupported,
                              m_Compute,
                              swizzledInputInfo,
                              viewsDesc))
        {
            return false;
        }

        startLayer = m_Network->AddSplitterLayer(viewsDesc);
    }

    armnn::IConnectableLayer* endLayer = ProcessActivation(swizzledOutputInfo, activation, startLayer);

    if (endLayer != nullptr)
    {
        armnn::IConnectableLayer& outSwizzleLayer = SwizzleInDeswizzleOut(*m_Network, input, *startLayer, *endLayer);
        return SetupAndTrackLayerOutputSlot(operation, 0, outSwizzleLayer);
    }
    else
    {
        return Fail("%s: ProcessActivation failed", operationName);
    }
}

template<typename HalVersion>
const void* ModelToINetworkConverter<HalVersion>::GetOperandValueReadOnlyAddress(const Operand& operand) const
{
    const void* valueStart = nullptr;

    switch (operand.lifetime)
    {
        case OperandLifeTime::CONSTANT_COPY:
        {
            // Constant found in model.operandValues
            valueStart = &m_Model.operandValues[operand.location.offset];
            break;
        }
        case OperandLifeTime::CONSTANT_REFERENCE:
        {
            // Constant specified via a Memory object
            valueStart = GetMemoryFromPool(operand.location, m_MemPools);
            break;
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

template<typename HalVersion>
template<typename HalOperation>
const Operand* ModelToINetworkConverter<HalVersion>::GetInputOperand(const HalOperation& operation,
                                                                     uint32_t inputIndex) const
{
    if (inputIndex >= operation.inputs.size())
    {
        Fail("%s: invalid input index: %i out of %i", __func__, inputIndex, operation.inputs.size());
        return nullptr;
    }

    assert(operation.inputs[inputIndex] < m_Model.operands.size()); // Model should have been validated beforehand
    return &m_Model.operands[operation.inputs[inputIndex]];
}

template<typename HalVersion>
template<typename HalOperation>
const Operand* ModelToINetworkConverter<HalVersion>::GetOutputOperand(const HalOperation& operation,
                                                                      uint32_t outputIndex) const
{
    if (outputIndex >= operation.outputs.size())
    {
        Fail("%s: invalid output index: %i out of %i", __func__, outputIndex, operation.outputs.size());
        return nullptr;
    }

    assert(operation.outputs[outputIndex] < m_Model.operands.size()); // Model should have been validated beforehand
    return &m_Model.operands[operation.outputs[outputIndex]];
}

template<typename HalVersion>
template<typename HalOperation, typename T>
bool ModelToINetworkConverter<HalVersion>::GetInputScalar(const HalOperation& operation,
                                                          uint32_t inputIndex,
                                                          OperandType type,
                                                          T& outValue) const
{
    const Operand* operand = GetInputOperand(operation, inputIndex);
    if (!operand)
    {
        return Fail("%s: invalid input operand at index %i", __func__, inputIndex);
    }

    if (operand->type != type)
    {
        return Fail("%s: unexpected operand type: %s (should be %s)",
            __func__, toString(operand->type).c_str(), toString(type).c_str());
    }

    if (operand->location.length != sizeof(T))
    {
        return Fail("%s: incorrect operand location length: %i (should be %i)",
            __func__, operand->location.length, sizeof(T));
    }

    const void* valueAddress = GetOperandValueReadOnlyAddress(*operand);
    if (!valueAddress)
    {
        return Fail("%s: failed to get address for operand", __func__);
    }

    outValue = *(static_cast<const T*>(valueAddress));
    return true;
}

template<typename HalVersion>
template<typename HalOperation>
bool ModelToINetworkConverter<HalVersion>::GetInputInt32(const HalOperation& operation,
                                                         uint32_t inputIndex,
                                                         int32_t& outValue) const
{
    return GetInputScalar(operation, inputIndex, OperandType::INT32, outValue);
}

template<typename HalVersion>
template<typename HalOperation>
bool ModelToINetworkConverter<HalVersion>::GetInputFloat32(const HalOperation& operation,
                                                           uint32_t inputIndex,
                                                           float& outValue) const
{
    return GetInputScalar(operation, inputIndex, OperandType::FLOAT32, outValue);
}

template<typename HalVersion>
template<typename HalOperation>
bool ModelToINetworkConverter<HalVersion>::GetInputActivationFunctionImpl(const HalOperation& operation,
                                                                          uint32_t inputIndex,
                                                                          OperandType type,
                                                                          ActivationFn& outActivationFunction) const
{
    if (type != OperandType::INT32 && type != OperandType::TENSOR_INT32)
    {
        return Fail("%s: unexpected operand type: %s (should be %s or %s)",
                    __func__,
                    toString(type).c_str(),
                    toString(OperandType::INT32).c_str(),
                    toString(OperandType::TENSOR_INT32).c_str());
    }

    int32_t activationFunctionAsInt;
    if (!GetInputScalar(operation, inputIndex, type, activationFunctionAsInt))
    {
        return Fail("%s: failed to get activation input value", __func__);
    }
    outActivationFunction = static_cast<ActivationFn>(activationFunctionAsInt);
    return true;
}

template<typename HalVersion>
template<typename HalOperation>
bool ModelToINetworkConverter<HalVersion>::GetInputActivationFunction(const HalOperation& operation,
                                                                      uint32_t inputIndex,
                                                                      ActivationFn& outActivationFunction) const
{
    return GetInputActivationFunctionImpl(operation, inputIndex, OperandType::INT32, outActivationFunction);
}

template<typename HalVersion>
template<typename HalOperation>
bool ModelToINetworkConverter<HalVersion>::GetInputActivationFunctionFromTensor(
    const HalOperation& operation,
    uint32_t inputIndex,
    ActivationFn& outActivationFunction) const
{
    // This only accepts a 1-D tensor of size 1
    return GetInputActivationFunctionImpl(operation, inputIndex, OperandType::INT32, outActivationFunction);
}

template<typename HalVersion>
template<typename HalOperation>
bool ModelToINetworkConverter<HalVersion>::GetOptionalInputActivation(const HalOperation& operation,
                                                                      uint32_t inputIndex,
                                                                      ActivationFn& activationFunction) const
{
    if (operation.inputs.size() <= inputIndex)
    {
        activationFunction = ActivationFn::kActivationNone;
    }
    else
    {
        if (!GetInputActivationFunction(operation, inputIndex, activationFunction))
        {
            return Fail("%s: Operation has invalid inputs", __func__);
        }
    }
    return true;
}

template<typename HalVersion>
template<typename HalOperation>
bool ModelToINetworkConverter<HalVersion>::GetInputPaddingScheme(const HalOperation& operation,
                                                                 uint32_t inputIndex,
                                                                 PaddingScheme& outPaddingScheme) const
{
    int32_t paddingSchemeAsInt;
    if (!GetInputInt32(operation, inputIndex, paddingSchemeAsInt))
    {
        return Fail("%s: failed to get padding scheme input value", __func__);
    }

    outPaddingScheme = static_cast<android::nn::PaddingScheme>(paddingSchemeAsInt);
    return true;
}

template<typename HalVersion>
template<typename HalOperation>
LayerInputHandle ModelToINetworkConverter<HalVersion>::ConvertToLayerInputHandle(const HalOperation& operation,
                                                                                 uint32_t inputIndex)
{
    const Operand* operand = GetInputOperand(operation, inputIndex);
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
        case OperandLifeTime::TEMPORARY_VARIABLE: // intentional fallthrough
        case OperandLifeTime::MODEL_INPUT:
        {
            // The tensor is either an operand internal to the model, or a model input.
            // It can be associated with an ArmNN output slot for an existing layer.

            // m_OutputSlotForOperand[...] can be nullptr if the previous layer could not be converted
            const uint32_t operandIndex = operation.inputs[inputIndex];
            return LayerInputHandle(true, m_OutputSlotForOperand[operandIndex], operandTensorInfo);
            break;
        }
        case OperandLifeTime::CONSTANT_COPY:
        case OperandLifeTime::CONSTANT_REFERENCE:
        {
            // The tensor has an already known constant value, and can be converted into an ArmNN Constant layer.
            ConstTensorPin tensorPin = ConvertOperandToConstTensorPin(*operand);
            if (tensorPin.IsValid())
            {
                if (!IsLayerSupported(__func__,
                                      armnn::IsConstantSupported,
                                      m_Compute,
                                      tensorPin.GetConstTensor().GetInfo()))
                {
                    return LayerInputHandle();
                }

                armnn::IConnectableLayer* constantLayer = m_Network->AddConstantLayer(tensorPin.GetConstTensor());
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

template<typename HalVersion>
template<typename HalOperation>
ConstTensorPin ModelToINetworkConverter<HalVersion>::ConvertOperationInputToConstTensorPin(
    const HalOperation& operation,
    uint32_t inputIndex,
    const armnn::PermutationVector& dimensionMappings,
    const armnn::TensorShape* overrideTensorShape,
    bool optional)
{
    const Operand* operand = GetInputOperand(operation, inputIndex);
    if (!operand)
    {
        Fail("%s: failed to get input operand: index=%u", __func__, inputIndex);
        return ConstTensorPin();
    }
    return ConvertOperandToConstTensorPin(*operand, dimensionMappings, overrideTensorShape, optional);
}

template<typename HalVersion>
ConstTensorPin ModelToINetworkConverter<HalVersion>::ConvertOperandToConstTensorPin(const Operand& operand,
    const armnn::PermutationVector& dimensionMappings, const armnn::TensorShape* overrideTensorShape, bool optional)
{
    if (!IsOperandTypeSupportedForTensors(operand.type))
    {
        Fail("%s: unsupported operand type for tensor %s", __func__, toString(operand.type).c_str());
        return ConstTensorPin();
    }

    if (operand.lifetime != OperandLifeTime::CONSTANT_COPY && operand.lifetime != OperandLifeTime::CONSTANT_REFERENCE)
    {
        Fail("%s: invalid operand lifetime: %s", __func__, toString(operand.lifetime).c_str());
        return ConstTensorPin();
    }

    const void* const valueStart = GetOperandValueReadOnlyAddress(operand);
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

template<typename HalVersion>
bool ModelToINetworkConverter<HalVersion>::GetTensorInt32Values(const Operand& operand,
    std::vector<int32_t>& outValues) const
{
    if (operand.type != OperandType::TENSOR_INT32)
    {
        return Fail("%s: invalid operand type: %s", __func__, toString(operand.type).c_str());
    }

    const void* startAddress = GetOperandValueReadOnlyAddress(operand);
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

// Creates an ArmNN activation layer and connects it to the given layer, if the
// passed in AndroidNN activation function requires so.
// @return The end layer of the sequence of layers built for the given AndroidNN
// activation function or nullptr if an error occurred (e.g. unsupported activation).
// Note that the end layer matches the input layer if no activation is required
// (the sequence of layers has length 1).
template<typename HalVersion>
armnn::IConnectableLayer* ModelToINetworkConverter<HalVersion>::ProcessActivation(const armnn::TensorInfo& tensorInfo,
    ActivationFn activation, armnn::IConnectableLayer* prevLayer)
{
    assert(prevLayer->GetNumOutputSlots() == 1);

    prevLayer->GetOutputSlot(0).SetTensorInfo(tensorInfo);

    armnn::IConnectableLayer* activationLayer = prevLayer;

    if (activation != ActivationFn::kActivationNone)
    {
        armnn::ActivationDescriptor activationDesc;
        switch (activation)
        {
            case ActivationFn::kActivationRelu:
            {
                activationDesc.m_Function = armnn::ActivationFunction::ReLu;
                break;
            }
            case ActivationFn::kActivationRelu1:
            {
                activationDesc.m_Function = armnn::ActivationFunction::BoundedReLu;
                activationDesc.m_A = 1.0f;
                activationDesc.m_B = -1.0f;
                break;
            }
            case ActivationFn::kActivationRelu6:
            {
                activationDesc.m_Function = armnn::ActivationFunction::BoundedReLu;
                activationDesc.m_A = 6.0f;
                break;
            }
            case ActivationFn::kActivationSigmoid:
            {
                activationDesc.m_Function = armnn::ActivationFunction::Sigmoid;
                break;
            }
            case ActivationFn::kActivationTanh:
            {
                activationDesc.m_Function = armnn::ActivationFunction::TanH;
                activationDesc.m_A = 1.0f;
                activationDesc.m_B = 1.0f;
                break;
            }
            default:
            {
                Fail("%s: Invalid activation enum value %i", __func__, activation);
                return nullptr;
            }
        }

        if (!IsLayerSupported(__func__, armnn::IsActivationSupported, m_Compute,
                              prevLayer->GetOutputSlot(0).GetTensorInfo(), tensorInfo, activationDesc))
        {
            return nullptr;
        }

        activationLayer = m_Network->AddActivationLayer(activationDesc);

        prevLayer->GetOutputSlot(0).Connect(activationLayer->GetInputSlot(0));
        activationLayer->GetOutputSlot(0).SetTensorInfo(tensorInfo);
    }

    return activationLayer;
}

template<typename HalVersion>
template<typename HalOperation>
bool ModelToINetworkConverter<HalVersion>::SetupAndTrackLayerOutputSlot(const HalOperation& operation,
                                                                        uint32_t operationOutputIndex,
                                                                        armnn::IConnectableLayer& layer,
                                                                        uint32_t layerOutputIndex)
{
    const Operand* outputOperand = GetOutputOperand(operation, operationOutputIndex);

    if ((outputOperand == nullptr) || (operationOutputIndex >= layer.GetNumOutputSlots()))
    {
        return false;
    }

    armnn::IOutputSlot& outputSlot = layer.GetOutputSlot(layerOutputIndex);

    const uint32_t operandIndex = operation.outputs[operationOutputIndex];
    m_OutputSlotForOperand[operandIndex] = &outputSlot;

    outputSlot.SetTensorInfo(GetTensorInfoForOperand(*outputOperand));

    return true;
}

template<typename HalVersion>
template<typename HalOperation>
bool ModelToINetworkConverter<HalVersion>::SetupAndTrackLayerOutputSlot(const HalOperation& operation,
                                                                        uint32_t outputIndex,
                                                                        armnn::IConnectableLayer& layer)
{
    return SetupAndTrackLayerOutputSlot(operation, outputIndex, layer, outputIndex);
}

template<typename HalVersion>
bool ModelToINetworkConverter<HalVersion>::IsOperationSupported(uint32_t operationIndex) const
{
    std::map<uint32_t, bool>::const_iterator it = m_OperationSupported.find(operationIndex);
    assert(it != m_OperationSupported.end());
    return it->second;
}

template class ModelToINetworkConverter<HalVersion_1_0>;

#if defined(ARMNN_ANDROID_NN_V1_1)
template class ModelToINetworkConverter<HalVersion_1_1>;
#endif

} // armnn_driver