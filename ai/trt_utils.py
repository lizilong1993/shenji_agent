import tensorrt as trt
import os

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def build_int8_engine(onnx_file_path, engine_file_path, calib=None):
    """
    Builds a TensorRT engine with INT8 quantization from an ONNX model.
    """
    if os.path.exists(engine_file_path):
        print(f"Engine found at {engine_file_path}, skipping build.")
        return

    builder = trt.Builder(TRT_LOGGER)
    config = builder.create_builder_config()
    
    # Enable INT8 mode if supported
    if builder.platform_has_fast_int8:
        config.set_flag(trt.BuilderFlag.INT8)
        if calib:
            config.int8_calibrator = calib
    else:
        print("Platform does not support INT8, falling back to FP16/FP32")
    
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)
    
    if not os.path.exists(onnx_file_path):
        print(f"ONNX file not found: {onnx_file_path}")
        return None

    with open(onnx_file_path, 'rb') as model:
        if not parser.parse(model.read()):
            print("Failed to parse ONNX file:")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None
            
    # Build engine
    try:
        plan = builder.build_serialized_network(network, config)
        if plan:
            with open(engine_file_path, "wb") as f:
                f.write(plan)
            print(f"Engine built and saved to {engine_file_path}")
            return plan
    except AttributeError:
        # Fallback for older TRT versions
        engine = builder.build_engine(network, config)
        if engine:
            with open(engine_file_path, "wb") as f:
                f.write(engine.serialize())
            print(f"Engine built and saved to {engine_file_path}")
            return engine
    return None
