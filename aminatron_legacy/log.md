Last login: Fri May 22 13:07:58 on ttys001
/Users/aminmammadov/.zshenv:.:1: no such file or directory: /Users/aminmammadov/.cargo/env
aminmammadov@Mac ~ % cd aiwork/models/cat_dog_al
cd: no such file or directory: aiwork/models/cat_dog_al
aminmammadov@Mac ~ % cd aiwork/models/cat_dog_ai
cd: no such file or directory: aiwork/models/cat_dog_ai
aminmammadov@Mac ~ % cd aiwork/models/catddog-ai
cd: no such file or directory: aiwork/models/catddog-ai
aminmammadov@Mac ~ % cd aiwork/models/cat-ddog-ai
cd: no such file or directory: aiwork/models/cat-ddog-ai
aminmammadov@Mac ~ % cd aiwork/models/cat-dog-ai
aminmammadov@Mac cat-dog-ai % source .venv/bin/activate
(.venv) aminmammadov@Mac cat-dog-ai % python main.py
Загружаю готовую модель: /Users/aminmammadov/aiwork/models/cat-dog-ai/models/cat_dog.keras
bothanim.png: Собака (100.00% dog)
bothcatdog.jpg: Собака (100.00% dog)
cat.png: Кот (0.00% dog)
dog.png: Собака (100.00% dog)
/Users/aminmammadov/aiwork/models/cat-dog-ai/.venv/lib/python3.12/site-packages/PIL/Image.py:1137: UserWarning: Palette images with Transparency expressed in bytes should be converted to RGBA images
  warnings.warn(
dogart.png: Собака (99.58% dog)
dogrealy.jpg: Собака (100.00% dog)
doog.jpg: Собака (99.97% dog)
maybedog.jpg: Собака (94.84% dog)
twocat.jpg: Кот (0.00% dog)
(.venv) aminmammadov@Mac cat-dog-ai % python main.py --train-more --epochs 50 --extra-data extra-data
Found 8566 files belonging to 2 classes.
Using 6853 files for training.
Found 8566 files belonging to 2 classes.
Using 1713 files for validation.
Продолжаю обучение модели: /Users/aminmammadov/aiwork/models/cat-dog-ai/models/cat_dog.keras
Model: "sequential"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ mobilenetv2_1.00_224 (Functional)    │ (None, 7, 7, 1280)          │       2,257,984 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ global_average_pooling2d             │ (None, 1280)                │               0 │
│ (GlobalAveragePooling2D)             │                             │                 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dropout (Dropout)                    │ (None, 1280)                │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense (Dense)                        │ (None, 1)                   │           1,281 │
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
 Total params: 2,261,829 (8.63 MB)
 Trainable params: 1,281 (5.00 KB)
 Non-trainable params: 2,257,984 (8.61 MB)
 Optimizer params: 2,564 (10.02 KB)
Epoch 1/50
1592/1592 ━━━━━━━━━━━━━━━━━━━━ 131s 82ms/step - accuracy: 0.9919 - loss: 0.0227 - val_accuracy: 0.9857 - val_loss: 0.0411
Epoch 2/50
1592/1592 ━━━━━━━━━━━━━━━━━━━━ 138s 86ms/step - accuracy: 0.9918 - loss: 0.0244 - val_accuracy: 0.9925 - val_loss: 0.0274
Epoch 3/50
1592/1592 ━━━━━━━━━━━━━━━━━━━━ 136s 85ms/step - accuracy: 0.9921 - loss: 0.0234 - val_accuracy: 0.9914 - val_loss: 0.0279
Epoch 4/50
1592/1592 ━━━━━━━━━━━━━━━━━━━━ 131s 82ms/step - accuracy: 0.9921 - loss: 0.0231 - val_accuracy: 0.9918 - val_loss: 0.0274
Epoch 5/50
1592/1592 ━━━━━━━━━━━━━━━━━━━━ 129s 81ms/step - accuracy: 0.9916 - loss: 0.0240 - val_accuracy: 0.9925 - val_loss: 0.0268
Epoch 6/50
1592/1592 ━━━━━━━━━━━━━━━━━━━━ 129s 81ms/step - accuracy: 0.9922 - loss: 0.0223 - val_accuracy: 0.9928 - val_loss: 0.0277
Epoch 7/50
1592/1592 ━━━━━━━━━━━━━━━━━━━━ 133s 83ms/step - accuracy: 0.9931 - loss: 0.0195 - val_accuracy: 0.9931 - val_loss: 0.0266
Epoch 8/50
1592/1592 ━━━━━━━━━━━━━━━━━━━━ 135s 85ms/step - accuracy: 0.9934 - loss: 0.0196 - val_accuracy: 0.9931 - val_loss: 0.0269
Epoch 9/50
1592/1592 ━━━━━━━━━━━━━━━━━━━━ 132s 83ms/step - accuracy: 0.9925 - loss: 0.0212 - val_accuracy: 0.9923 - val_loss: 0.0264
Epoch 10/50
1592/1592 ━━━━━━━━━━━━━━━━━━━━ 133s 84ms/step - accuracy: 0.9930 - loss: 0.0207 - val_accuracy: 0.9912 - val_loss: 0.0269
Epoch 11/50
1592/1592 ━━━━━━━━━━━━━━━━━━━━ 134s 84ms/step - accuracy: 0.9927 - loss: 0.0219 - val_accuracy: 0.9920 - val_loss: 0.0277
Epoch 12/50
1592/1592 ━━━━━━━━━━━━━━━━━━━━ 135s 85ms/step - accuracy: 0.9924 - loss: 0.0216 - val_accuracy: 0.9925 - val_loss: 0.0289
Epoch 13/50
1592/1592 ━━━━━━━━━━━━━━━━━━━━ 135s 85ms/step - accuracy: 0.9926 - loss: 0.0217 - val_accuracy: 0.9920 - val_loss: 0.0320
Epoch 14/50
1592/1592 ━━━━━━━━━━━━━━━━━━━━ 134s 84ms/step - accuracy: 0.9920 - loss: 0.0214 - val_accuracy: 0.9910 - val_loss: 0.0292
Epoch 15/50
1592/1592 ━━━━━━━━━━━━━━━━━━━━ 136s 85ms/step - accuracy: 0.9918 - loss: 0.0230 - val_accuracy: 0.9925 - val_loss: 0.0272
Epoch 16/50
1592/1592 ━━━━━━━━━━━━━━━━━━━━ 134s 84ms/step - accuracy: 0.9925 - loss: 0.0219 - val_accuracy: 0.9928 - val_loss: 0.0289
Epoch 17/50
1592/1592 ━━━━━━━━━━━━━━━━━━━━ 135s 85ms/step - accuracy: 0.9921 - loss: 0.0224 - val_accuracy: 0.9912 - val_loss: 0.0345
Epoch 18/50
1592/1592 ━━━━━━━━━━━━━━━━━━━━ 134s 84ms/step - accuracy: 0.9919 - loss: 0.0217 - val_accuracy: 0.9928 - val_loss: 0.0276
Epoch 19/50
1592/1592 ━━━━━━━━━━━━━━━━━━━━ 136s 86ms/step - accuracy: 0.9929 - loss: 0.0206 - val_accuracy: 0.9932 - val_loss: 0.0281
Epoch 20/50
1592/1592 ━━━━━━━━━━━━━━━━━━━━ 137s 86ms/step - accuracy: 0.9923 - loss: 0.0229 - val_accuracy: 0.9925 - val_loss: 0.0286
Epoch 21/50
1592/1592 ━━━━━━━━━━━━━━━━━━━━ 137s 86ms/step - accuracy: 0.9932 - loss: 0.0195 - val_accuracy: 0.9918 - val_loss: 0.0290
Epoch 22/50
1592/1592 ━━━━━━━━━━━━━━━━━━━━ 135s 85ms/step - accuracy: 0.9924 - loss: 0.0232 - val_accuracy: 0.9931 - val_loss: 0.0307
Epoch 23/50
1592/1592 ━━━━━━━━━━━━━━━━━━━━ 137s 86ms/step - accuracy: 0.9922 - loss: 0.0211 - val_accuracy: 0.9920 - val_loss: 0.0288
Epoch 24/50
1592/1592 ━━━━━━━━━━━━━━━━━━━━ 135s 85ms/step - accuracy: 0.9925 - loss: 0.0214 - val_accuracy: 0.9923 - val_loss: 0.0301
Epoch 25/50
1592/1592 ━━━━━━━━━━━━━━━━━━━━ 136s 85ms/step - accuracy: 0.9924 - loss: 0.0233 - val_accuracy: 0.9929 - val_loss: 0.0290
Epoch 26/50
1592/1592 ━━━━━━━━━━━━━━━━━━━━ 139s 87ms/step - accuracy: 0.9925 - loss: 0.0220 - val_accuracy: 0.9892 - val_loss: 0.0342
Epoch 27/50
1592/1592 ━━━━━━━━━━━━━━━━━━━━ 136s 85ms/step - accuracy: 0.9922 - loss: 0.0220 - val_accuracy: 0.9917 - val_loss: 0.0294
Epoch 28/50
1592/1592 ━━━━━━━━━━━━━━━━━━━━ 134s 84ms/step - accuracy: 0.9913 - loss: 0.0240 - val_accuracy: 0.9926 - val_loss: 0.0319
Epoch 29/50
1592/1592 ━━━━━━━━━━━━━━━━━━━━ 135s 85ms/step - accuracy: 0.9927 - loss: 0.0216 - val_accuracy: 0.9921 - val_loss: 0.0297
Epoch 30/50
1592/1592 ━━━━━━━━━━━━━━━━━━━━ 134s 84ms/step - accuracy: 0.9927 - loss: 0.0198 - val_accuracy: 0.9915 - val_loss: 0.0302
Epoch 31/50
1592/1592 ━━━━━━━━━━━━━━━━━━━━ 139s 87ms/step - accuracy: 0.9927 - loss: 0.0203 - val_accuracy: 0.9917 - val_loss: 0.0309
Epoch 32/50
1592/1592 ━━━━━━━━━━━━━━━━━━━━ 137s 86ms/step - accuracy: 0.9921 - loss: 0.0223 - val_accuracy: 0.9926 - val_loss: 0.0305
Epoch 33/50
1592/1592 ━━━━━━━━━━━━━━━━━━━━ 135s 85ms/step - accuracy: 0.9920 - loss: 0.0214 - val_accuracy: 0.9929 - val_loss: 0.0305
Epoch 34/50
1592/1592 ━━━━━━━━━━━━━━━━━━━━ 133s 84ms/step - accuracy: 0.9929 - loss: 0.0197 - val_accuracy: 0.9921 - val_loss: 0.0292
Epoch 35/50
1592/1592 ━━━━━━━━━━━━━━━━━━━━ 134s 84ms/step - accuracy: 0.9924 - loss: 0.0212 - val_accuracy: 0.9901 - val_loss: 0.0313
Epoch 36/50
1592/1592 ━━━━━━━━━━━━━━━━━━━━ 139s 87ms/step - accuracy: 0.9923 - loss: 0.0223 - val_accuracy: 0.9920 - val_loss: 0.0292
Epoch 37/50
1592/1592 ━━━━━━━━━━━━━━━━━━━━ 136s 85ms/step - accuracy: 0.9919 - loss: 0.0231 - val_accuracy: 0.9912 - val_loss: 0.0333
Epoch 38/50
1592/1592 ━━━━━━━━━━━━━━━━━━━━ 134s 84ms/step - accuracy: 0.9923 - loss: 0.0222 - val_accuracy: 0.9918 - val_loss: 0.0317
Epoch 39/50
1592/1592 ━━━━━━━━━━━━━━━━━━━━ 136s 85ms/step - accuracy: 0.9923 - loss: 0.0222 - val_accuracy: 0.9907 - val_loss: 0.0331
Epoch 40/50
1592/1592 ━━━━━━━━━━━━━━━━━━━━ 137s 86ms/step - accuracy: 0.9927 - loss: 0.0210 - val_accuracy: 0.9910 - val_loss: 0.0313
Epoch 41/50
1592/1592 ━━━━━━━━━━━━━━━━━━━━ 140s 88ms/step - accuracy: 0.9924 - loss: 0.0220 - val_accuracy: 0.9928 - val_loss: 0.0305
Epoch 42/50
1592/1592 ━━━━━━━━━━━━━━━━━━━━ 134s 84ms/step - accuracy: 0.9924 - loss: 0.0196 - val_accuracy: 0.9921 - val_loss: 0.0345
Epoch 43/50
1592/1592 ━━━━━━━━━━━━━━━━━━━━ 136s 85ms/step - accuracy: 0.9925 - loss: 0.0213 - val_accuracy: 0.9910 - val_loss: 0.0307
Epoch 44/50
1592/1592 ━━━━━━━━━━━━━━━━━━━━ 137s 86ms/step - accuracy: 0.9919 - loss: 0.0197 - val_accuracy: 0.9921 - val_loss: 0.0292
Epoch 45/50
1592/1592 ━━━━━━━━━━━━━━━━━━━━ 135s 85ms/step - accuracy: 0.9921 - loss: 0.0206 - val_accuracy: 0.9931 - val_loss: 0.0315
Epoch 46/50
1592/1592 ━━━━━━━━━━━━━━━━━━━━ 135s 85ms/step - accuracy: 0.9926 - loss: 0.0209 - val_accuracy: 0.9925 - val_loss: 0.0307
Epoch 47/50
1592/1592 ━━━━━━━━━━━━━━━━━━━━ 139s 87ms/step - accuracy: 0.9930 - loss: 0.0216 - val_accuracy: 0.9926 - val_loss: 0.0296
Epoch 48/50
1592/1592 ━━━━━━━━━━━━━━━━━━━━ 135s 85ms/step - accuracy: 0.9926 - loss: 0.0225 - val_accuracy: 0.9928 - val_loss: 0.0291
Epoch 49/50
1592/1592 ━━━━━━━━━━━━━━━━━━━━ 133s 84ms/step - accuracy: 0.9918 - loss: 0.0207 - val_accuracy: 0.9928 - val_loss: 0.0284
Epoch 50/50
1592/1592 ━━━━━━━━━━━━━━━━━━━━ 141s 89ms/step - accuracy: 0.9928 - loss: 0.0215 - val_accuracy: 0.9923 - val_loss: 0.0304
Модель сохранена: /Users/aminmammadov/aiwork/models/cat-dog-ai/models/cat_dog.keras
bothanim.png: Собака (100.00% dog)
bothcatdog.jpg: Собака (100.00% dog)
cat.png: Кот (0.00% dog)
dog.png: Собака (100.00% dog)
/Users/aminmammadov/aiwork/models/cat-dog-ai/.venv/lib/python3.12/site-packages/PIL/Image.py:1137: UserWarning: Palette images with Transparency expressed in bytes should be converted to RGBA images
  warnings.warn(
dogart.png: Собака (97.79% dog)
dogrealy.jpg: Собака (100.00% dog)
doog.jpg: Собака (100.00% dog)
maybedog.jpg: Собака (99.23% dog)
twocat.jpg: Кот (0.00% dog)
(.venv) aminmammadov@Mac cat-dog-ai %
