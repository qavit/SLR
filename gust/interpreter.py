import tensorflow as tf
import json

interpreter = tf.lite.Interpreter("model.tflite")

# REQUIRED_SIGNATURE = "serving_default"
# REQUIRED_OUTPUT = "outputs"

# with open ("/kaggle/input/asl-fingerspelling/character_to_prediction_index.json", "r") as f:
#     character_map = json.load(f)
# rev_character_map = {j:i for i,j in character_map.items()}

# found_signatures = list(interpreter.get_signature_list().keys())

# if REQUIRED_SIGNATURE not in found_signatures:
#     raise KernelEvalException('Required input signature not found.')

# prediction_fn = interpreter.get_signature_runner("serving_default")
# output = prediction_fn(inputs=batch[0][0])
# prediction_str = "".join([rev_character_map.get(s, "") for s in np.argmax(output[REQUIRED_OUTPUT], axis=1)])
# print(prediction_str)