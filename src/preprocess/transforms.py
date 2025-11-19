import torch
from torchvision import transforms as T
from typing import Dict, Any, List

# Импортируем Реестр из нашего utils
# Предполагаем, что у нас есть центральный трансформ-реестр
from ..utils.registry import Registry

# --- 1. РЕЕСТР ТРАНСФОРМАЦИЙ ---
# Создаем отдельный реестр для всех классов трансформаций
TRANSFORM_REGISTRY = Registry("Transforms")

# --- 2. РЕГИСТРАЦИЯ СТАНДАРТНЫХ ТРАНСФОРМАЦИЙ ---
# Регистрируем наиболее часто используемые классы из torchvision.transforms
# Это позволяет нам обращаться к ним по строковому имени в конфиге.

TRANSFORM_REGISTRY.register(name="RandomCrop")(T.RandomCrop)
TRANSFORM_REGISTRY.register(name="RandomHorizontalFlip")(T.RandomHorizontalFlip)
TRANSFORM_REGISTRY.register(name="ToTensor")(T.ToTensor)
TRANSFORM_REGISTRY.register(name="Normalize")(T.Normalize)
# Добавляем другие по мере необходимости: Resize, ColorJitter, RandomRotation, etc.


# --- 3. ФУНКЦИЯ ДЛЯ СБОРКИ ЦЕПОЧКИ (COMPOSE) ---

def get_transforms_chain(config_list: List[Dict[str, Any]]) -> T.Compose:
    """
    Создает объект T.Compose (цепочку преобразований) из списка конфигураций.

    Args:
        config_list: Список словарей. Каждый словарь должен содержать:
                     {'name': 'TransformName', 'params': {...}}

    Returns:
        Объект torchvision.transforms.Compose.
    """
    
    transforms_list = []
    
    for item in config_list:
        try:
            name = item['name']
            params = item.get('params', {})
            
            # 1. Получаем класс трансформации из Реестра по имени
            transform_class = TRANSFORM_REGISTRY.get(name)
            
            # 2. Создаем экземпляр класса, передавая параметры
            transform_instance = transform_class(**params)
            
            transforms_list.append(transform_instance)
            
        except KeyError as e:
            raise ValueError(
                f"Ошибка в конфигурации трансформаций. Не найден ключ: {e}. "
                f"Убедитесь, что 'name' и 'params' указаны корректно."
            )
        except Exception as e:
            raise TypeError(
                f"Не удалось инициализировать трансформацию '{name}' с параметрами {params}: {e}"
            )
            
    # 3. Собираем все экземпляры в T.Compose
    return T.Compose(transforms_list)


# --- 4. ПРИМЕР КАСТОМНОЙ ТРАНСФОРМАЦИИ (для демонстрации) ---

@TRANSFORM_REGISTRY.register(name="CustomGeometricTransform")
class CustomGeometricTransform(object):
    """
    Пример кастомной трансформации, привязанной к вашей логике (например, 
    добавление шума или аугментации, специфичной для геометрической регуляризации).
    """
    def __init__(self, magnitude: float = 0.1):
        self.magnitude = magnitude

    def __call__(self, img):
        # Здесь была бы ваша кастомная логика преобразования PyTorch Tensor или PIL Image
        # Например, добавление Gaussian Noise
        return img