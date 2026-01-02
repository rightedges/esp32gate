
import os

pooling_path = "/Users/william/Documents/Arduino/libraries/EloquentTinyML/src/eloquent_tinyml/tensorflow/esp32/tensorflow/lite/experimental/micro/kernels/pooling.cpp"
softmax_path = "/Users/william/Documents/Arduino/libraries/EloquentTinyML/src/eloquent_tinyml/tensorflow/esp32/tensorflow/lite/experimental/micro/kernels/softmax.cpp"

# 1. Patch pooling.cpp
if os.path.exists(pooling_path):
    with open(pooling_path, 'r') as f:
        content = f.read()
    
    # Fix previously flawed patch if exists
    content = content.replace(
        "MaxEvalInt8(context, node, params, data, input, output);",
        "MaxEvalInt8(context, node, params, \&data, input, output);"
    ).replace("\\&", "&") # safeguard against double escaping in some environments
    
    # Add MaxEvalInt8 if not exists
    if "void MaxEvalInt8" not in content:
        max_eval_int8 = """
void MaxEvalInt8(TfLiteContext* context, TfLiteNode* node,
                 TfLitePoolParams* params, OpData* data,
                 const TfLiteTensor* input, TfLiteTensor* output) {
  int32_t activation_min, activation_max;
  CalculateActivationRangeInt8(params->activation, output, &activation_min,
                               &activation_max);

  tflite::PoolParams op_params;
  op_params.stride_height = params->stride_height;
  op_params.stride_width = params->stride_width;
  op_params.filter_height = params->filter_height;
  op_params.filter_width = params->filter_width;
  op_params.padding_values.height = data->padding.height;
  op_params.padding_values.width = data->padding.width;
  op_params.quantized_activation_min = activation_min;
  op_params.quantized_activation_max = activation_max;
  reference_integer_ops::MaxPool(op_params, GetTensorShape(input),
                                 GetTensorData<int8_t>(input),
                                 GetTensorShape(output),
                                 GetTensorData<int8_t>(output));
}
"""
        insertion_point = content.find("void MaxEvalQuantizedUInt8")
        if insertion_point != -1:
            # Find the end of MaxEvalQuantizedUInt8
            end_of_func = content.find("}", insertion_point) + 1
            content = content[:end_of_func] + max_eval_int8 + content[end_of_func:]
            
            # Add to switch (with &data fix)
            content = content.replace(
                '    default:\n      context->ReportError(context, "Type %s not currently supported."',
                '    case kTfLiteInt8:\n      MaxEvalInt8(context, node, params, &data, input, output);\n      break;\n    default:\n      context->ReportError(context, "Type %s not currently supported."'
            )
            
            with open(pooling_path, 'w') as f:
                f.write(content)
            print("Patched pooling.cpp")
        else:
            # Maybe it was already partially patched, let's fix the call
            if "MaxEvalInt8(context, node, params, data, input, output);" in content:
                 content = content.replace(
                    "MaxEvalInt8(context, node, params, data, input, output);",
                    "MaxEvalInt8(context, node, params, &data, input, output);"
                 )
                 with open(pooling_path, 'w') as f:
                    f.write(content)
                 print("Fixed MaxEvalInt8 call in pooling.cpp")

# 2. Patch softmax.cpp
if os.path.exists(softmax_path):
    with open(softmax_path, 'r') as f:
        content = f.read()
    
    if "input->type == kTfLiteInt8" not in content:
        # Update CalculateSoftmaxOpData
        content = content.replace(
            "  if (input->type == kTfLiteUInt8) {",
            "  if (input->type == kTfLiteUInt8 || input->type == kTfLiteInt8) {"
        )
        
        # Update SoftmaxEval
        content = content.replace(
            "    case kTfLiteUInt8: {",
            "    case kTfLiteInt8:\n    case kTfLiteUInt8: {"
        )
        
        with open(softmax_path, 'w') as f:
            f.write(content)
        print("Patched softmax.cpp")
    else:
        print("softmax.cpp already patched")
