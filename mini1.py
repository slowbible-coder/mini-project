import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Input, Dense, Dropout, GlobalAveragePooling2D
from PIL import Image
import numpy as np
from io import StringIO
import sys
import pandas as pd
import os

# --- ������ ���� ---
st.set_page_config(
    page_title="�ѽ� �̹��� ���� ��",
    page_icon="?",
    layout="centered",
)


# --- �� ���� ���� �� ����ġ �ε� ---

# ������ �н� �ڵ�(ipynb)�� �ִ� �� ���� �Լ��� �� �ڵ忡 ���Խ�ŵ�ϴ�.
def create_kfood_model(input_shape, num_classes):
    """
    EfficientNetB4 ����� ���� �н� �� ������ �����մϴ�.
    """
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.2),
        tf.keras.layers.RandomZoom(0.2),
    ], name="data_augmentation")

    base_model = tf.keras.applications.EfficientNetB4(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape,
    )
    base_model.trainable = False

    inputs = Input(shape=input_shape)
    x = data_augmentation(inputs)
    x = base_model(x, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation="softmax", dtype='float32')(x)
    model = Model(inputs, outputs)
    return model


# �� ������ ����� �̸� �н��� ����ġ�� �ε��ϴ� �Լ�
@st.cache_resource
def load_pretrained_model(weights_path):
    """
    �� ���� �� �� ������ ����� ����� ����ġ�� �ҷ��ɴϴ�.
    �� �Լ��� ĳ�õǾ� �� ���� �� �� ���� ����˴ϴ�.
    """
    try:
        # Ŭ���� 8��, �Է� shape (224, 224, 3)���� �� ������ ����ϴ�.
        model = create_kfood_model(input_shape=(224, 224, 3), num_classes=8)
        # ����� ����ġ ������ �𵨿� �ε��մϴ�.
        model.load_weights(weights_path)
        return model
    except Exception as e:
        st.error(f"�� ����ġ �ε� �� ���� �߻�: {e}")
        st.error(f"'{weights_path}' ������ app.py�� ���� ������ �ִ��� Ȯ���ϼ���.")
        return None


# --- ���� ���ø����̼� ---

st.title("? �ѽ� �̹��� ���� ����")
st.markdown("---")

# �� ����ġ ���� ���
WEIGHTS_FILE = "best_kfood_model.keras"

# �� �ε�
model = load_pretrained_model(WEIGHTS_FILE)

# ���̵��
with st.sidebar:
    st.header("?? �̹��� ���ε�")
    image_file = st.file_uploader("������ �ѽ� �̹����� ���ε��ϼ���.", type=["jpg", "jpeg", "png"])

    st.header("? Ŭ���� �̸�")
    st.write("�� ���� ������ �� �ִ� ���� �����Դϴ�.")
    # Ŭ���� �̸��� ���� �ڵ忡 ����մϴ�.
    class_names_list = [
        "����", "��", "��ġ", "����", "��",
        "��ħ", "��", "����"
    ]
    st.markdown(f"**- {'- '.join(class_names_list)}**")


def preprocess_image(image, target_size):
    """
    ������ ���� �̹����� ��ó���մϴ�.
    """
    if image.mode != "RGB":
        image = image.convert("RGB")
... (59�� ����)