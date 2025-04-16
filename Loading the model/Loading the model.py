model = EfficientNet.from_pretrained('efficientnet-b3', num_classes=2)
model.load_state_dict(torch.load('best_model.pth'))
model.eval()
