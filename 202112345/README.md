# 설명서

- 201701981 김기환


### 환경설정
- inference library
    > environments.txt 참고

### 소스코드 실행
- 상대 경로로 되어 있기 때문에 파이썬 파일만 실행하면 됩니다.

- image to shape
    1. clip, clipcap 을 설치
        > clip : env.txt 에서 참고해 pip 설치
        > clipcap : https://github.com/rmokady/CLIP_prefix_caption 참고
        ```yaml
            - pip: # 아래 라이브러리 설치
                - ftfy
                - regex
                - tqdm
                - git+https://github.com/openai/CLIP.git
        ```
    2. 실행
        ```python
        01_image_to_shape.py
        01_image_to_shape_evaluation.py

        ... # 동일하게 실행
        ```
    - test dataset 의 경우 validation 코드의 파일 주소 변수의 경로만 변경하면 됩니다.
        ```python
            to_fn = f"./release/shape/{student_id}.{task}.valid.txt"
            description = f"Generating Validation {task} {student_id}"
            excel = pd.read_excel("./dataset/valid/scene.all.xlsx", engine='openpyxl', index_col="id")

        ```
        excel 변수에서 valid -> test

# 코드 설명

### image_to_shape

1. 사용 모델
    - CLIPCAP [참고 link](https://github.com/rmokady/CLIP_prefix_caption/tree/main)
    - encoder : CLIP 모델에서 image 를 embedding 하는 부분만 사용
    - decoder : mapping network (Transformer) + gpt2
        clip 모델이 image 에서 추출한 embedding vector 를 추출하면 decoder 에서 텍스트(토큰) 생성
2. generation
    - `image2shape_utils.py` 참고
    - beamsearch 사용

### text_to_color
1. hugging face gpt2 사용 [참고 link](https://github.com/SKRohit/Generating_Text_Summary_With_GPT2/tree/master)
2. generatoin
    - top-k filtering 사용



# Optimization

### image_to_shape

1. Post-training dynamic quantization 적용
    - float32 -> qint8
    - conv1d, conv2d 에 대해서만 적용
    - 적용 전 1500개 실행시간 35분 4초
    - 적용 후 1500개 실행시간 15분 49초

2. Decoder 모델 크기 수정
    - mapping network, clipcap-gpt2
    - layer num, head num hyper parameter 수정


### text_to_color
1. gpt2 layer num hyper parameter 수정