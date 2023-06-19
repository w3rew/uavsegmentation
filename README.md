# Семантическая сегментация изображений с дрона

В этом репозитории находится модель для семантической сегментации изображений с дрона с целью помощи наземному роботу.

## Датасет
UAVid --- датасет для семантической сегментации, снятый с беспилотного летательного аппарата.

![UAVid example](https://uavid.nl/UAVid_files/imgs/UAVid_example.png)

В нём 8 классов:

1. Building
2. Road
3. Static car
4. Tree
5. Low vegetation
6. Human
7. Moving car
8. Background clutter

Поскольку он представляет интерес и отдельно от задачи помощи роботу, обучение производилось для классов из датасета.
Во время инференса есть возможность преобразовать в другой формат классов.

## Requirements
Модель была обучена на Nvidia A100 с использованием 77Gb GPU. Возможно обойтись меньшими ресурсами, уменьшив batch size и размер изображений --- настраивается в `config.yml`.
Inference практически на любом компьютере.

Работа проверена только под ОС Linux.
Необходим Python>=3.10 и библиотеки к нему, они перечислены в `requirements.txt`.
Рекомендуется воспользоваться `virtualenv`: для создания нового окружения выполните `python -m venv venv`, а для его активации `source venv/bin/activate`. После этого можно установить зависимости: `pip install -r requirements.txt`.

## Структура проекта

- `train.py` --- скрипт для обучения и валидации модели.
- `inference.py` --- скрипт для инференса модели на произвольных изображениях
- `config.yml` --- файл конфигурации в формате YAML. Настройка модели осуществляется с использованием этого файла.
- `UAVidToolKit` --- git submodule SDK датасета
- `models/*` --- веса обученных моделей. Хранятся в Git LFS.
- `crop_imgs.py` --- скрипт для обрезания изображений датасета. В новой версии модель работает с полноразмерными изображениеями, поэтому не используется.
- `architecture.py`, `auxiliary.py`, `datasets.py` --- вспомогательные файлы, непосредственная работа с которорыми не предполагается.

## Использование

После клонирования репозитория не забудте инициализировать подмодуль Git.
Сделать это можно, например, командой `git clone --recurse-submodules`.

### Обучение

Для обучение используется фреймворк PyTorch и CUDA.

Перед началом работы необходимо загрузить датасет.
Это можно сделать со страницы в [Kaggle](https://www.kaggle.com/datasets/dasmehdixtr/uavid-v1), либо с [сайта](https://uavid.nl/) датасета.
Датасет поставляется в виде цветных семантических карт. После загрузки сгенерируйте одноканальные семантические карты при помощи
скрипта из SDK:

```shell
    python prepareTrainIdFiles.py -s $UAVID_DIR/uavid_train/ -t $UAVID_DIR/uavid_train
    python prepareTrainIdFiles.py -s $UAVID_DIR/uavid_val/ -t $UAVID_DIR/uavid_val
```

Параметры обучения:

- `decoder`:  используется архитектура SegNet с декодером и энкодером. Есть возможность выбрать как декодер, так и энкодер. По умолчанию `Unet++`. Также доступен `Unet`.
- `resolution_k`: параметр задаёт размер квадратного блока, которому должно быть кратно изображение на входе модели.
- `encoder_name`: название энкодера. По умолчанию `resnet34`. Возможен любой энкодер из библиотеки [Segmentation models PyTorch](https://github.com/qubvel/segmentation_models.pytorch)
- `in_channels`: количество каналов на входе
- `classes`: количество классов для сегментации. Это значение используется только для обучения модели, поэтому его имеет смысл оставить равным 8.
- `shape`: размер изображения на входе сети при обучении в формате HxW. По умолчанию 1056x1920. Для уменьшения потребления памяти можно уменьшить, однако это 
- `dataloader`: стандартные параметры для даталоадера. Применяются только к train даталоадеру; для validation используется такое же значение `num_workers`, `batch_size=1`, `shuffle=False`.
- `dataset`: на настоящий момент только `uavid`.
    - `mean`, `std`: поканальные средние и стандартные отклонения в датасете для нормализации. При отсутствии будут вычислены.
- `epochs`: количество эпох обучения. Реализован early stopping с валидацией, поэтому можно выбирать большое число.
- `optim`: оптимизатор. Adam либо AdamW.
- `lr`: learning rate. Кроме того используется ReduceLrOnPlateau.
- `loss`: `jaccard`, `dice`, `cross_entropy`

Пример запуска скрипта обучения:

```shell
    CUDA_VISIBLE_DEVICES=1 python train.py -i models/resnet_crop_crossentropy.pth -c config.yml -o large --model_name resnet34_finetune_jaccard --dataset_path ../uavid --dataset uavid
```

Здесь

- `CUDA_VISIBLE_DEVICES=1` --- обучение на втором ускорителе
- `-i` --- модель для fine-tuning. Если не указать аргумент `-i`, будет взят предобученный энкодер и декодер без обучения.
- `-c` --- путь до конфига
- `-o` --- путь для сохранения результатов запуска, в данном случае `./large`.
- `--model_name` --- название модели (произвольное)
- `--dataset_path` --- путь к датасету
- `--dataset` --- тип датасета, в настоящее время только uavid

Данный скрипт дообучит модель на всем датасете uavid, используя предобученную модель `models/resnet_crop_crossentropy.pth`.

Также возможно совершить прогон модели на валидационном датасете. Пример такого запуска:

```shell
    python train.py -c config.yml -i models/resnet_crop_crossentropy.pth --dataset_path /media/Temp/uavid/ --dataset uavid --validate_only
```

Параметры остаются прежними, только добавляется флаг `--validate_only`, а `-o` и `--model_name` теперь не обязательны.

### Предобученные модели
В `models` находятся предобученные модели, которые можно использовать для предсказаний.
Для этого предназначен скрипт `inference.py`.

Список моделей:

|                                | Jaccard index on validation | Comment                                           |
|--------------------------------|-----------------------------|---------------------------------------------------|
| `resnet_crop_crossentropy.pth` |                       0.603 | Обучена на обрезанных изображениях с CrossEntropy |
| `resnet_full_crossentropy.pth` |                       0.633 |     Обучена на полных изображениях с CrossEntropy |
| `resnet_full_jaccard.pth`      |                       0.642 |  Обучена на обрезанных изображениях с JaccardLoss |

### Инференс
Для инференса служит скрипт `inference.py`.
Пример запуска:

```shell
    python inference.py -o $UAVID_DIR/uavid_test/seq21/Outputs -m models/resnet_crop_crossentropy.pth -c config.yml $UAVID_DIR/uavid_test/seq21/Images/000000.png
```

Этот скрипт совершит запустит модель на одном изображении и сохранит результат в каталог, указанный в `-o`.
Аналогично можно запускать инференс для целого каталога:

```shell
    python inference.py -o $UAVID_DIR/uavid_test/seq21/Outputs -m models/resnet_crop_crossentropy.pth -c config.yml $UAVID_DIR/uavid_test/seq21/Images
```

В данном случае процедура будет произведена для всех изображений в этом каталоге.

Есть возможность на лету маппить классы датасета в другие классы. Для робота можно предложить 3 класса: дорога, движущееся препятствие и неподвижное препятствие.
Тогда

- Дорога
    1. Road
- Движущееся препятствие
    1. Human
    2. Moving car
- Неподвижное препятствие
    1. Building
    2. Static Car
    3. Tree
    4. Low vegetation
    5. Background clutter

### Примеры работы

![Example 1](examples/seq21/Labels/000000.png)

![Example 2](examples/seq25/Labels/000000.png)

![Example 3](examples/seq30/Labels/000700.png)

![Example 4](examples/seq40/Labels/000600.png)
