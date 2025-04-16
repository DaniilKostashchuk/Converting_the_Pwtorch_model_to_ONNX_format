torch.onnx.export(
    model,
    dummy_input,
    'efficientnet-b3.onnx',
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
    opset_version=12
)

print("Конвертация завершена!")
