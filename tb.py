from tensorflow.keras.callbacks import TensorBoard
import os
import datetime

def make_TensorBoard(dir_name):
    root_dir = os.path.join(os.curdir, dir_name)
    sub_dir_name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    TB_log_dir = os.path.join(root_dir, sub_dir_name)
    return TensorBoard(log_dir=TB_log_dir)