# Converting_the_Pwtorch_model_to_ONNX_format
Этот код демонстрирует процесс преобразования модели EfficientNet из формата PyTorch в ONNX для последующего использования в Telegram-боте.
## Подробное описание кода
### 1. Загрузка предобученной модели
```python
import torch
from efficientnet_pytorch import EfficientNet

# Инициализация модели EfficientNet-b3 с 2 выходными классами
model = EfficientNet.from_pretrained('efficientnet-b3', num_classes=2)

# Загрузка весов из файла
model.load_state_dict(torch.load('best_model.pth'))

# Перевод модели в режим оценки (отключение слоев типа Dropout и BatchNorm)
model.eval()
```
#### Пояснения:
* `EfficientNet.from_pretrained()` загружает архитектуру EfficientNet-b3 с предобученными весами
* `num_classes=2` указывает, что модель имеет 2 выходных класса
* `load_state_dict()` загружает специфичные веса модели из файла .pth
* `eval()` переводит модель в режим инференса

### 2. Создание фиктивного входного тензора
```python
# Создание тестового входного тензора размерности [1, 3, 224, 224]
dummy_input = torch.randn(1, 3, 224, 224)
```
#### Пояснения:
* Размерности тензора соответствуют:
  - `1` - размер батча

  - `3` - количество каналов (RGB)

  - `224, 224` - высота и ширина изображения

* `randn()` создает тензор с нормально распределенными случайными значениями

### 3. Экспорт модели в ONNX формат
```python
torch.onnx.export(
    model,                   # Модель PyTorch
    dummy_input,            # Пример входных данных
    'efficientnet-b3.onnx', # Имя выходного файла
    input_names=['input'],   # Имя входного тензора
    output_names=['output'], # Имя выходного тензора
    dynamic_axes={
        'input': {0: 'batch_size'},  # Динамическая размерность батча
        'output': {0: 'batch_size'}  # Для входа и выхода
    },
    opset_version=12        # Версия ONNX операторов
)
```
#### Ключевые параметры экспорта:
* `dynamic_axes` позволяет обрабатывать входы с переменным размером батча
* `opset_version=12` обеспечивает совместимость с современными runtime

### 4. Проверка успешности конвертации
```python
print("Конвертация завершена!")
```

## Требования для выполнения конвертации
1. Установленные библиотеки:
```python
pip install torch efficientnet-pytorch onnx
```
2. Наличие файла с весами модели:
   - `best_model.pth`
3. Достаточно места на диске для сохранения ONNX модели

## Особенности работы
1. __Поддержка динамических размеров:__ модель сможет обрабатывать входы с разным размером батча
2. __Оптимизация для инференса:__ экспортированная модель готова для быстрого выполнения предсказаний
3. __Совместимость:__ ONNX модель может быть запущена на различных платформах и устройствах

Полученный файл efficientnet-b3.onnx можно использовать в Telegram-боте для классификации изображений с помощью ONNX Runtime.

## Лицензия
Этот проект распространяется под лицензией MIT. Подробности см. в файле `LICENSE`.

## Ссылки
Ссылка на нейронную сеть: https://github.com/DaniilKostashchuk/Classification_of_images_of_minerals_and_rock_cuts<p>
Сылка на TG-бота: https://github.com/DaniilKostashchuk/Telegram_Bot_for_Mineral_Classification
