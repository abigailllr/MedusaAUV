import os
import sys
import yaml
import numpy as np
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../ros2/medusa_auv/medusa_auv"))
from algorithms import gray_world, confusion_counts, precision_recall_f1

try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow.lite as tflite


CONFIG_PATH = os.path.join(os.path.dirname(__file__), "../../config.yaml")

with open(CONFIG_PATH) as f:
    CFG = yaml.safe_load(f)

IMG_SIZE = CFG["img_size"]
THRESHOLD = CFG["inference_threshold"]
TEST_DIR = os.path.join(os.path.dirname(__file__), "../../", CFG.get("eval_test_dir", "data/test"))
MODEL_PATH = os.path.join(os.path.dirname(__file__), "../../", CFG["model_path"])


def load_model():
    interpreter = tflite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    return interpreter, interpreter.get_input_details(), interpreter.get_output_details()


def predict(interpreter, inp, out, image, balance):
    img = image.astype(np.float32)
    if balance:
        img = gray_world(img)
    resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    tensor = np.expand_dims(resized / 255.0, axis=0)
    interpreter.set_tensor(inp[0]["index"], tensor)
    interpreter.invoke()
    return float(interpreter.get_tensor(out[0]["index"])[0][0])


def run(balance):
    interpreter, inp, out = load_model()
    y_true, y_pred = [], []
    for label, cls in [(1, "jellyfish"), (0, "no_jellyfish")]:
        folder = os.path.join(TEST_DIR, cls)
        if not os.path.isdir(folder):
            continue
        for name in os.listdir(folder):
            image = cv2.imread(os.path.join(folder, name))
            if image is None:
                continue
            confidence = predict(interpreter, inp, out, image, balance)
            y_true.append(label)
            y_pred.append(1 if confidence >= THRESHOLD else 0)
    return y_true, y_pred


def report(name, y_true, y_pred):
    tp, tn, fp, fn = confusion_counts(y_true, y_pred)
    m = precision_recall_f1(y_true, y_pred)
    print(f"\n{name}  (n={len(y_true)})")
    print(f"  confusion: TP={tp} TN={tn} FP={fp} FN={fn}")
    print(f"  accuracy={m['accuracy']:.3f} precision={m['precision']:.3f} recall={m['recall']:.3f} f1={m['f1']:.3f}")
    return m


def main():
    if not os.path.isdir(TEST_DIR):
        print(f"no test set at {TEST_DIR}")
        return
    print("Ablation: effect of underwater white balance on freshwater jellyfish detection")
    raw = report("model only", *run(balance=False))
    balanced = report("model + white balance", *run(balance=True))
    print(f"\nF1 change from white balance: {balanced['f1'] - raw['f1']:+.3f}")


if __name__ == "__main__":
    main()
