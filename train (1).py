import os
import sys

# Ensure TensorFlow can find CUDA libraries by restarting the interpreter
# with the required LD_LIBRARY_PATH set before TensorFlow imports.
env_lib = os.path.join(sys.prefix, 'lib')
system_libs = ['/usr/lib/x86_64-linux-gnu', '/usr/lib']
required_paths = [env_lib] + system_libs
current_ld = os.environ.get('LD_LIBRARY_PATH', '')
current_paths = [p for p in current_ld.split(os.pathsep) if p]

print('train.py: start', flush=True)
print('train.py: sys.prefix=', sys.prefix, flush=True)
print('train.py: LD_LIBRARY_PATH before=', repr(current_ld), flush=True)

cuda_path = sys.prefix
os.environ.setdefault('CUDA_PATH', cuda_path)
os.environ.setdefault('CUDA_HOME', cuda_path)

xla_flags = os.environ.get('XLA_FLAGS', '')
if '--xla_gpu_cuda_data_dir=' not in xla_flags:
    xla_flags = (xla_flags + ' ' if xla_flags else '') + f'--xla_gpu_cuda_data_dir={env_lib}'
    os.environ['XLA_FLAGS'] = xla_flags

if os.environ.get('TF_CUDA_REEXEC') != '1':
    new_paths = []
    for p in required_paths + current_paths:
        if p and p not in new_paths:
            new_paths.append(p)
    if new_paths != current_paths or 'TF_CUDA_REEXEC' not in os.environ:
        print('train.py: restarting with LD_LIBRARY_PATH=', os.pathsep.join(new_paths), flush=True)
        os.environ['LD_LIBRARY_PATH'] = os.pathsep.join(new_paths)
        os.environ['TF_CUDA_REEXEC'] = '1'
        os.execvpe(sys.executable, [sys.executable] + sys.argv, os.environ)

print('train.py: after reexec check, TF_CUDA_REEXEC=', os.environ.get('TF_CUDA_REEXEC'), flush=True)
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# 1. Cấu hình để tối ưu GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# 2. Tiền xử lý dữ liệu (Densenet yêu cầu input 224x224)
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_val_datagen = ImageDataGenerator(rescale=1./255)

# Kiểm tra đường dẫn dataset, hỗ trợ cả hai cấu trúc thư mục phổ biến
base_dir = 'dataset'
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')
if not os.path.isdir(train_dir) or not os.path.isdir(val_dir):
    alt_base = os.path.join(base_dir, 'chest_xray')
    alt_train = os.path.join(alt_base, 'train')
    alt_val = os.path.join(alt_base, 'val')
    if os.path.isdir(alt_train) and os.path.isdir(alt_val):
        train_dir = alt_train
        val_dir = alt_val
    else:
        raise FileNotFoundError(
            f"Không tìm thấy thư mục dữ liệu: {train_dir} và {val_dir}. "
            f"Hãy kiểm tra lại cấu trúc thư mục dataset."
        )

train_generator = train_datagen.flow_from_directory(
    train_dir, target_size=(224, 224), batch_size=32, class_mode='binary'
)

val_generator = test_val_datagen.flow_from_directory(
    val_dir, target_size=(224, 224), batch_size=32, class_mode='binary'
)

# 3. Xây dựng mô hình Fine-tune
# Tải base model với trọng số ImageNet
base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Giai đoạn 1: Đóng băng base model để train các lớp mới thêm vào
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# 4. Compile và Huấn luyện
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
              loss='binary_crossentropy', 
              metrics=['accuracy'])

# Callbacks để lưu model tốt nhất và dừng sớm nếu không cải thiện
callbacks = [
    ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_loss'),
    EarlyStopping(patience=5, monitor='val_loss')
]

print("Bắt đầu huấn luyện giai đoạn 1...")
model.fit(train_generator, validation_data=val_generator, epochs=10, callbacks=callbacks)

# Giai đoạn 2: Fine-tune (Mở khóa một số lớp cuối của DenseNet)
print("Bắt đầu Fine-tuning giai đoạn 2...")
base_model.trainable = True
# Chỉ mở khóa khoảng 20 lớp cuối cùng để tránh làm hỏng các đặc trưng đã học
for layer in base_model.layers[:-20]:
    layer.trainable = False

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), # Giảm lr khi fine-tune
              loss='binary_crossentropy', 
              metrics=['accuracy'])

model.fit(train_generator, validation_data=val_generator, epochs=10, callbacks=callbacks)