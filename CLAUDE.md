# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 프로젝트 개요

이 저장소는 "밑바닥부터 만들면서 배우는 LLM" (Build a Large Language Model From Scratch) 책의 공식 코드 저장소입니다. GPT와 유사한 대규모 언어 모델(LLM)을 PyTorch로 밑바닥부터 구현하고, 사전 훈련(pretraining) 및 미세 튜닝(fine-tuning)하는 방법을 다룹니다.

---

## ⚠️ 시작하기 전 필수 사항

### 환경 활성화 (모든 작업의 첫 번째 단계)

```bash
# 항상 먼저 실행!
conda activate llm-scratch
```

**환경 활성화 없이 Python 스크립트나 노트북을 실행하면 올바른 라이브러리를 찾지 못합니다!**

### 현재 시스템 정보

| 항목 | 값 |
|------|-----|
| **환경 이름** | `llm-scratch` (Conda virtual environment) |
| **Python 버전** | 3.11.13 |
| **위치** | `/home/iamhjoo/miniconda3/envs/llm-scratch` |
| **PyTorch** | 2.8.0+cu128 (CUDA 12.8) |
| **GPU** | NVIDIA L40S (46GB VRAM) |
| **TensorFlow** | 2.20.0 (GPU 지원) |

### IDE 설정 (VSCode / PyCharm)

인터프리터를 다음으로 설정하세요:
```
/home/iamhjoo/miniconda3/envs/llm-scratch/bin/python
```

---

## 환경 설정

### 의존성 정보

**현재 설치된 주요 라이브러리:**
```
torch==2.8.0+cu128       # PyTorch (CUDA 12.8)
jupyterlab==4.4.9        # Jupyter Lab
tiktoken==0.12.0         # GPT-2 토크나이저
matplotlib==3.10.7       # 시각화
tensorflow==2.20.0       # 가중치 로딩
tqdm==4.67.1            # 진행 표시
numpy==2.3.3            # 수치 연산
pandas==2.3.3           # 데이터 처리
psutil==7.1.0           # 시스템 모니터링
```

**새 환경 생성 (이미 완료됨):**
```bash
conda create -n llm-scratch python=3.11 -y
conda activate llm-scratch
pip install -r requirements.txt
```

### GPU 설정 확인

```bash
conda activate llm-scratch
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

**예상 출력:**
```
CUDA available: True
GPU: NVIDIA L40S
```

### 환경 검증

```bash
# 전체 환경 검증
python setup/02_installing-python-libraries/python_environment_check.py

# 특정 테스트 실행
pytest ch02/05_bpe-from-scratch/tests.py
pytest ch05/07_gpt_to_llama/tests/tests.py
```

---

## 프로젝트 구조

### 디렉토리 레이아웃

```
llm-from-scratch/
├── ch02/                    # 텍스트 데이터 처리
│   ├── 01_main-chapter-code/
│   │   ├── ch02.ipynb       # 메인 노트북
│   │   └── ...
│   └── 05_bpe-from-scratch/
│
├── ch03/                    # 어텐션 메커니즘
│   ├── 01_main-chapter-code/
│   │   ├── ch03.ipynb
│   │   └── ...
│   └── 02_bonus_efficient-multihead-attention/
│
├── ch04/                    # GPT 모델 구현
│   ├── 01_main-chapter-code/
│   │   ├── ch04.ipynb
│   │   ├── gpt.py          # GPT 모델 클래스
│   │   └── ...
│   └── 03_kv-cache/
│
├── ch05/                    # 사전 훈련 (Pretraining)
│   ├── 01_main-chapter-code/
│   │   ├── ch05.ipynb
│   │   ├── gpt_train.py    # 훈련 스크립트
│   │   └── ...
│   ├── 07_gpt_to_llama/
│   ├── 11_qwen3/
│   └── 12_gemma3/
│
├── ch06/                    # 분류 미세 튜닝 (Classification Fine-tuning)
│   ├── 01_main-chapter-code/
│   │   ├── ch06.ipynb
│   │   ├── gpt_class_finetune.py
│   │   └── ...
│
├── ch07/                    # 지시 따르기 미세 튜닝 (Instruction Fine-tuning)
│   ├── 01_main-chapter-code/
│   │   ├── ch07.ipynb
│   │   ├── gpt_instruction_finetuning.py
│   │   └── ...
│   └── 04_preference-tuning-with-dpo/
│
├── setup/                   # 환경 설정 관련
├── study/                   # 학습 자료 및 실험
└── requirements.txt         # 의존성 파일
```

### 장(Chapter) 별 구조

각 장(`chXX/`)은 다음과 같은 구조를 가집니다:

- **`01_main-chapter-code/`**: 메인 챕터 코드
  - `.ipynb`: Jupyter 노트북 (학습 및 실습용)
  - `.py`: 독립 실행 Python 스크립트
  - `previous_chapters.py`: 이전 장의 재사용 코드

- **`0X_bonus_*/`**: 보너스 컨텐츠 및 고급 실험

### previous_chapters.py 패턴

각 장의 `previous_chapters.py`는 이전 장에서 구현한 핵심 함수와 클래스를 재사용 가능하도록 모아둔 파일입니다.

**사용 예시:**
```python
# ch05에서 ch04의 코드를 재사용
import sys
sys.path.append("../ch04/01_main-chapter-code")
from previous_chapters import GPTModel, generate_text_simple

# ch06에서 ch05의 훈련 함수 재사용
from previous_chapters import train_model_simple, calc_loss_batch
```

---

## 개발 워크플로우

### 1️⃣ Jupyter 노트북 실행 (학습 및 탐색용)

```bash
conda activate llm-scratch
jupyter lab
```

**메인 챕터 노트북:**
- `ch02/01_main-chapter-code/ch02.ipynb` - 텍스트 데이터 다루기
- `ch03/01_main-chapter-code/ch03.ipynb` - 어텐션 메커니즘
- `ch04/01_main-chapter-code/ch04.ipynb` - GPT 모델 구현
- `ch05/01_main-chapter-code/ch05.ipynb` - 사전 훈련
- `ch06/01_main-chapter-code/ch06.ipynb` - 분류 미세 튜닝
- `ch07/01_main-chapter-code/ch07.ipynb` - 지시 따르기 미세 튜닝

### 2️⃣ 독립 실행 스크립트 (프로덕션용)

모든 스크립트는 GPU를 자동으로 감지하여 사용합니다:

```bash
conda activate llm-scratch

# GPT 모델 생성 및 추론 테스트
python ch04/01_main-chapter-code/gpt.py

# 사전 훈련 실행 (훈련 시간: GPU 필수)
python ch05/01_main-chapter-code/gpt_train.py

# 분류 작업 미세 튜닝
python ch06/01_main-chapter-code/gpt_class_finetune.py

# 지시 따르기 미세 튜닝
python ch07/01_main-chapter-code/gpt_instruction_finetuning.py
```

### 3️⃣ 터미널에서 대화형 Python

```bash
conda activate llm-scratch
python

>>> import torch
>>> print(f"CUDA: {torch.cuda.is_available()}")
>>> print(f"GPU: {torch.cuda.get_device_name(0)}")
```

### 4️⃣ 모델 저장 및 로드

```python
# 모델 저장
torch.save(model.state_dict(), "model.pth")

# 모델 로드
model = GPTModel(GPT_CONFIG_124M)
model.load_state_dict(torch.load("model.pth", weights_only=True))
```

---

## 핵심 코드 모듈

### GPT 모델 구현 (ch04/01_main-chapter-code/gpt.py)

- **`GPTModel`**: 메인 GPT 아키텍처 클래스
- **`MultiHeadAttention`**: 멀티헤드 어텐션 메커니즘
- **`TransformerBlock`**: 트랜스포머 블록 (어텐션 + FFN)
- **`LayerNorm`, `GELU`, `FeedForward`**: 기본 빌딩 블록
- **`generate_text_simple()`**: 텍스트 생성 함수

### 데이터 로딩 (ch02-ch05)

- **`GPTDatasetV1`**: 슬라이딩 윈도우 방식으로 텍스트를 청크로 분할
- **`create_dataloader_v1()`**: 데이터로더 생성 유틸리티

### 훈련 함수 (ch05/01_main-chapter-code/gpt_train.py)

- **`train_model_simple()`**: 기본 훈련 루프
- **`calc_loss_batch()`, `calc_loss_loader()`**: 손실 계산
- **`evaluate_model()`**: 모델 평가
- **`plot_losses()`**: 손실 곡선 시각화

### 미세 튜닝 (ch06, ch07)

- **분류 작업** (ch06/01_main-chapter-code/gpt_class_finetune.py)
  - 감성 분석 등 분류 태스크 미세 튜닝

- **지시 따르기** (ch07/01_main-chapter-code/gpt_instruction_finetuning.py)
  - `InstructionDataset`: 지시-응답 쌍 데이터셋
  - `custom_collate_fn()`: 패딩 및 타겟 마스킹 처리

---

## 주요 설정 (Configuration)

### GPT-124M 기본 설정

```python
GPT_CONFIG_124M = {
    "vocab_size": 50257,      # GPT-2 토크나이저 어휘 크기
    "context_length": 1024,   # 최대 시퀀스 길이 (훈련 시 256으로 단축 가능)
    "emb_dim": 768,           # 임베딩 차원
    "n_heads": 12,            # 어텐션 헤드 수
    "n_layers": 12,           # 트랜스포머 레이어 수
    "drop_rate": 0.1,         # 드롭아웃 비율
    "qkv_bias": False         # Query-Key-Value 바이어스 사용 여부
}
```

### 훈련 하이퍼파라미터

```python
OTHER_SETTINGS = {
    "learning_rate": 5e-4,
    "num_epochs": 10,
    "batch_size": 2,          # 매우 보수적인 설정 (L40S로 8~16 권장)
    "weight_decay": 0.1
}
```

---

## GPU & 배치 크기 가이드

### 메모리 사용량 추정 (L40S 46GB 기준)

| 모델 | Context | Batch Size | 예상 VRAM | 비고 |
|------|---------|-----------|----------|------|
| GPT-124M | 256 | 2 | ~4GB | 기본 설정 (매우 보수적) |
| GPT-124M | 256 | 8 | ~12GB | 권장 설정 |
| GPT-124M | 256 | 16 | ~22GB | 빠른 훈련 |
| GPT-124M | 1024 | 2 | ~8GB | 긴 시퀀스 훈련 |
| GPT-124M | 1024 | 8 | ~28GB | 긴 시퀀스 + 배치 |

### GPU 메모리 관리

```python
import torch

# GPU 메모리 확인
if torch.cuda.is_available():
    print(f"GPU 메모리 할당량: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
    print(f"GPU 메모리 예약량: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")

# GPU 메모리 정리 (필요 시)
torch.cuda.empty_cache()

# 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
```

### GPU 성능 모니터링

```bash
# 훈련 중 GPU 사용률 모니터링
nvidia-smi
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,nounits,noheader -l 1
```

---

## 아키텍처 개념

### 데이터 처리 파이프라인

1. 원본 텍스트 → `tiktoken`으로 토큰화
2. `GPTDatasetV1`로 슬라이딩 윈도우 청킹 (stride 파라미터로 오버랩 조절)
3. `DataLoader`로 배치 생성

### GPT 모델 포워드 패스

1. 토큰 임베딩 (`tok_emb`) + 위치 임베딩 (`pos_emb`)
2. 임베딩 드롭아웃 적용
3. 여러 `TransformerBlock` 순차 통과
4. 최종 레이어 정규화
5. 출력 헤드로 로짓 생성

### TransformerBlock 구조

1. **첫 번째 서브블록** (자기 어텐션):
   - 레이어 정규화 → 멀티헤드 어텐션 → 드롭아웃 → 잔차 연결

2. **두 번째 서브블록** (피드포워드):
   - 레이어 정규화 → 피드포워드 네트워크 → 드롭아웃 → 잔차 연결

---

## 보너스 자료 및 고급 토픽

### 추천 학습 경로

| 보너스 자료 | 위치 | 목적 | 난이도 |
|----------|------|------|--------|
| **BPE 토크나이저** | `ch02/05_bpe-from-scratch/` | 토큰화 과정 이해 및 커스텀 토크나이저 구현 | 중 |
| **효율적인 어텐션** | `ch03/02_bonus_efficient-multihead-attention/` | 어텐션 최적화 기법 비교 (Flash Attention 등) | 중상 |
| **KV 캐시** | `ch04/03_kv-cache/` | 텍스트 생성 속도 최적화 | 중 |
| **GPT to Llama** | `ch05/07_gpt_to_llama/` | 다른 모델 아키텍처로의 변환 | 상 |
| **Qwen3** | `ch05/11_qwen3/` | 밀집 모델과 혼합 전문가(MoE) 모델 구현 | 상 |
| **Gemma 3** | `ch05/12_gemma3/` | Google의 Gemma 모델 아키텍처 | 상 |
| **DPO 미세 튜닝** | `ch07/04_preference-tuning-with-dpo/` | Direct Preference Optimization 기법 | 상 |

### 사용 시나리오

- **BPE 토크나이저**: 자신의 데이터셋에 맞춤형 토크나이저가 필요할 때
- **효율적인 어텐션**: 훈련 속도를 더 높이고 싶을 때
- **KV 캐시**: 추론 속도 개선이 필요할 때 (챗봇, API 서버)
- **GPT to Llama**: 다른 모델 아키텍처를 학습하고 싶을 때
- **Qwen3 & Gemma3**: 최신 SOTA 모델 아키텍처를 이해하고 싶을 때
- **DPO**: 모델의 응답을 선호도 학습으로 개선하고 싶을 때

---

## 문제 해결

### 환경 관련 문제

```bash
# 환경이 활성화되었는지 확인
conda env list
# 현재 환경 옆에 * 표시가 있어야 함

# Python 인터프리터 경로 확인
which python
# /home/iamhjoo/miniconda3/envs/llm-scratch/bin/python이어야 함

# 패키지 설치 확인
pip list | grep torch
pip list | grep tiktoken

# 환경 재설치 (필요 시)
conda activate llm-scratch
pip install -r requirements.txt --upgrade
```

### GPU 관련 문제

```bash
# GPU 감지 확인
python -c "import torch; print(torch.cuda.is_available())"

# 현재 GPU 메모리 상태 확인
nvidia-smi

# PyTorch 버전 확인
python -c "import torch; print(torch.__version__)"
```

---

## 코딩 규칙

### Linting (Ruff)

```bash
# pyproject.toml에 설정됨
# line-length: 140
# ignored rules: C406, E226, E402, E702, E703, E722, E731, E741
```

`.venv` 디렉토리는 린팅에서 제외됩니다.

---

## 학습 노트 (Obsidian)

**Obsidian 볼트 위치:** `/home/iamhjoo/Documents/IAMHJOO`

학습 중 중요한 개념이나 코드 차이점을 Obsidian에 기록할 때:

- 마크다운 형식으로 노트 생성
- 태그 활용 (`#python`, `#llm-scratch`, `#coding-tip` 등)
- 날짜 및 관련 챕터 링크 포함
- 코드 예시와 설명을 함께 작성
- 카테고리 잘 분류해서, 노트 저장 
---

## 참고사항

### 일반 사항

- 메인 챕터 코드는 일반 노트북에서 합리적인 시간 내에 실행 가능하도록 설계됨
- GPU 없이도 작동하지만, ch05-ch07의 훈련은 GPU로 실행하면 훨씬 빠름
- 모든 독립 실행 스크립트는 `if __name__ == "__main__":` 블록을 포함

### 성능 최적화 팁

- 기본 `batch_size=2`는 매우 보수적인 설정
- L40S 46GB VRAM으로 `batch_size=8~16`까지 증가 가능
- 더 큰 배치로 훈련 시간 대폭 단축 가능
- 훈련 중 `nvidia-smi`로 GPU 메모리 모니터링 권장

### 시스템 체크리스트

워크플로우 시작 전 다음을 확인하세요:

- [ ] `conda activate llm-scratch` 실행 완료
- [ ] `which python` 결과가 `/home/iamhjoo/miniconda3/envs/llm-scratch/bin/python` 포함
- [ ] `pytorch.cuda.is_available()` 반환값이 `True`
- [ ] IDE의 Python 인터프리터가 올바른 경로로 설정됨
