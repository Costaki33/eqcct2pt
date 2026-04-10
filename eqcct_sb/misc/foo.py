# CPU RAM delta
import os, psutil
from eqcct_sb.reference.predictor_tf import load_eqcct_model

p = psutil.Process(os.getpid())
ram_before = p.memory_info().rss / 1e6

# (Optional) GPU snapshot with TensorFlow
try:
    import tensorflow as tf
    vram_before = tf.config.experimental.get_memory_info('GPU:0')['current'] / 1e6
except Exception:
    vram_before = None

model = load_eqcct_model('ModelPS/test_trainer_024.h5', 'ModelPS/test_trainer_021.h5')  # <— measure JUST the load

ram_after = p.memory_info().rss / 1e6
try:
    vram_after = tf.config.experimental.get_memory_info('GPU:0')['current'] / 1e6
except Exception:
    vram_after = None

print(f"Model-load ΔRAM: {ram_after - ram_before:.2f} MB")
if vram_before is not None and vram_after is not None:
    print(f"Model-load ΔVRAM: {vram_after - vram_before:.2f} MB")
