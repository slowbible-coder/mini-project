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

# --- 페이지 설정 ---
st.set_page_config(
    page_title="한식 이미지 예측 앱",
    page_icon="?",
    layout="centered",
)


# --- 모델 구조 정의 및 가중치 로드 ---

# 이전에 학습 코드(ipynb)에 있던 모델 생성 함수를 앱 코드에 포함시킵니다.
def create_kfood_model(input_shape, num_classes):
    """
    EfficientNetB4 기반의 전이 학습 모델 구조를 생성합니다.
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


# 모델 구조를 만들고 미리 학습된 가중치를 로드하는 함수
@st.cache_resource
def load_pretrained_model(weights_path):
    """
    앱 실행 시 모델 구조를 만들고 저장된 가중치를 불러옵니다.
    이 함수는 캐시되어 앱 실행 중 한 번만 실행됩니다.
    """
    try:
        # 클래스 8개, 입력 shape (224, 224, 3)으로 모델 구조를 만듭니다.
        model = create_kfood_model(input_shape=(224, 224, 3), num_classes=8)
        # 저장된 가중치 파일을 모델에 로드합니다.
        model.load_weights(weights_path)
        return model
    except Exception as e:
        st.error(f"모델 가중치 로딩 중 오류 발생: {e}")
        st.error(f"'{weights_path}' 파일이 app.py와 같은 폴더에 있는지 확인하세요.")
        return None


# --- 메인 애플리케이션 ---

st.title("? 한식 이미지 예측 서비스")
st.markdown("---")

# 모델 가중치 파일 경로
WEIGHTS_FILE = "best_kfood_model.keras"

# 모델 로드
model = load_pretrained_model(WEIGHTS_FILE)

# 사이드바
with st.sidebar:
    st.header("?? 이미지 업로드")
    image_file = st.file_uploader("예측할 한식 이미지를 업로드하세요.", type=["jpg", "jpeg", "png"])

    st.header("? 클래스 이름")
    st.write("이 모델이 예측할 수 있는 음식 종류입니다.")
    # 클래스 이름을 직접 코드에 명시합니다.
    class_names_list = [
        "구이", "국", "김치", "나물", "면",
        "무침", "밥", "볶음"
    ]
    st.markdown(f"**- {'- '.join(class_names_list)}**")


def preprocess_image(image, target_size):
    """
    예측을 위해 이미지를 전처리합니다.
    """
    if image.mode != "RGB":
        image = image.convert("RGB")
... (59줄 남음)