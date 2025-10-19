# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 프로젝트 개요

이 저장소는 "밑바닥부터 만들면서 배우는 LLM" (Build a Large Language Model From Scratch) 책의 공식 코드 저장소입니다. GPT와 유사한 대규모 언어 모델(LLM)을 PyTorch로 밑바닥부터 구현하고, 사전 훈련(pretraining) 및 미세 튜닝(fine-tuning)하는 방법을 다룹니다.

## 환경 설정

### ⚠️ 중요: 현재 시스템의 개발 환경

**이 프로젝트는 전용 Conda 환경을 사용합니다:**

```bash
# 환경 활성화 (작업 시작 시 반드시 필요!)
conda activate llm-scratch

# 환경 비활성화 (작업 완료 시)
conda deactivate
```

**설치된 환경 정보:**
- **환경 이름**: `llm-scratch` (Conda virtual environment)
- **Python 버전**: 3.11.13
- **위치**: `/home/iamhjoo/miniconda3/envs/llm-scratch`
- **PyTorch**: 2.8.0+cu128 (CUDA 12.8 지원)
- **GPU**: NVIDIA L40S (46GB VRAM) - GPU 가속 완벽 지원
- **TensorFlow**: 2.20.0 (GPU 지원)

**환경 활성화 없이 Python 스크립트나 노트북을 실행하면 올바른 라이브러리를 찾지 못합니다!**

### 의존성 설치

**새 환경 생성 (이미 완료됨):**
```bash
# 새 Conda 환경 생성
conda create -n llm-scratch python=3.11 -y

# 환경 활성화
conda activate llm-scratch

# 의존성 설치
pip install -r requirements.txt
```

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

### GPU 설정 확인

**PyTorch GPU 지원 확인:**
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
# 환경 활성화 확인
conda activate llm-scratch

# 전체 환경 검증
python setup/02_installing-python-libraries/python_environment_check.py

# 특정 테스트 실행
pytest ch02/05_bpe-from-scratch/tests.py
pytest ch05/07_gpt_to_llama/tests/tests.py
```

## 코드 구조

### 장(Chapter) 별 구조
각 장은 다음과 같은 구조를 가집니다:
- `01_main-chapter-code/`: 메인 챕터 코드 (노트북 + Python 스크립트)
- `0X_bonus_*/`: 보너스 컨텐츠 및 추가 실험

### 핵심 모듈

#### GPT 모델 구현 (ch04/01_main-chapter-code/gpt.py)
- `GPTModel`: 메인 GPT 아키텍처 클래스
- `MultiHeadAttention`: 멀티헤드 어텐션 메커니즘
- `TransformerBlock`: 트랜스포머 블록 (어텐션 + FFN)
- `LayerNorm`, `GELU`, `FeedForward`: 기본 빌딩 블록
- `generate_text_simple()`: 텍스트 생성 함수

#### 데이터 로딩 (ch02-ch05)
- `GPTDatasetV1`: 슬라이딩 윈도우 방식으로 텍스트를 청크로 분할
- `create_dataloader_v1()`: 데이터로더 생성 유틸리티

#### 훈련 (ch05/01_main-chapter-code/gpt_train.py)
- `train_model_simple()`: 기본 훈련 루프
- `calc_loss_batch()`, `calc_loss_loader()`: 손실 계산
- `evaluate_model()`: 모델 평가
- `plot_losses()`: 손실 곡선 시각화

#### 미세 튜닝
- **분류 작업** (ch06/01_main-chapter-code/gpt_class_finetune.py): 감성 분석 등 분류 태스크
- **지시 따르기** (ch07/01_main-chapter-code/gpt_instruction_finetuning.py): instruction-following 모델
  - `InstructionDataset`: 지시-응답 쌍 데이터셋
  - `custom_collate_fn()`: 패딩 및 타겟 마스킹 처리

### previous_chapters.py 패턴
각 장의 `previous_chapters.py`는 이전 장에서 구현한 핵심 함수와 클래스를 재사용 가능하도록 모아둔 파일입니다. 새로운 장에서 이전 구현을 임포트할 때 사용합니다.

## 개발 워크플로우

### ⚠️ 모든 작업 전 필수: 환경 활성화

```bash
# 항상 먼저 실행!
conda activate llm-scratch
```

### Jupyter 노트북 실행

```bash
# 환경 활성화 후 Jupyter Lab 시작
conda activate llm-scratch
jupyter lab

# 메인 챕터 노트북 위치:
# ch02/01_main-chapter-code/ch02.ipynb - 텍스트 데이터 다루기
# ch03/01_main-chapter-code/ch03.ipynb - 어텐션 메커니즘
# ch04/01_main-chapter-code/ch04.ipynb - GPT 모델 구현
# ch05/01_main-chapter-code/ch05.ipynb - 사전 훈련
# ch06/01_main-chapter-code/ch06.ipynb - 분류 미세 튜닝
# ch07/01_main-chapter-code/ch07.ipynb - 지시 따르기 미세 튜닝
```

### 독립 실행 스크립트

**모든 스크립트는 환경 활성화 후 실행해야 합니다:**

```bash
# 환경 활성화
conda activate llm-scratch

# GPT 모델 생성 및 추론 테스트
python ch04/01_main-chapter-code/gpt.py

# 사전 훈련 실행 (GPU 자동 사용)
python ch05/01_main-chapter-code/gpt_train.py

# 분류 작업 미세 튜닝
python ch06/01_main-chapter-code/gpt_class_finetune.py

# 지시 따르기 미세 튜닝
python ch07/01_main-chapter-code/gpt_instruction_finetuning.py
```

### 터미널에서 대화형 Python 사용

```bash
conda activate llm-scratch
python

>>> import torch
>>> print(f"CUDA: {torch.cuda.is_available()}")
>>> print(f"GPU: {torch.cuda.get_device_name(0)}")
```

### 모델 저장 및 로드
```python
# 모델 저장
torch.save(model.state_dict(), "model.pth")

# 모델 로드
model = GPTModel(GPT_CONFIG_124M)
model.load_state_dict(torch.load("model.pth", weights_only=True))
```

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
    "batch_size": 2,
    "weight_decay": 0.1
}
```

## 아키텍처 개념

### 데이터 처리 파이프라인
1. 원본 텍스트 → tiktoken으로 토큰화
2. `GPTDatasetV1`로 슬라이딩 윈도우 청킹 (stride 파라미터로 오버랩 조절)
3. `DataLoader`로 배치 생성

### GPT 모델 포워드 패스
1. 토큰 임베딩 (`tok_emb`) + 위치 임베딩 (`pos_emb`)
2. 임베딩 드롭아웃 적용
3. 여러 `TransformerBlock` 순차 통과
4. 최종 레이어 정규화
5. 출력 헤드로 로짓 생성

### TransformerBlock 구조
1. 레이어 정규화 → 멀티헤드 어텐션 → 드롭아웃 → 잔차 연결
2. 레이어 정규화 → 피드포워드 네트워크 → 드롭아웃 → 잔차 연결

## 디바이스 처리

### GPU 자동 감지 및 사용

코드는 CUDA GPU가 사용 가능하면 자동으로 활용하도록 작성되어 있습니다:
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
```

**현재 시스템의 GPU 정보:**
- **GPU 모델**: NVIDIA L40S
- **VRAM**: 46GB
- **CUDA 버전**: 12.8
- **PyTorch CUDA 지원**: 활성화됨 (torch 2.8.0+cu128)

**GPU 메모리 사용 최적화:**
```python
import torch

# GPU 메모리 확인
if torch.cuda.is_available():
    print(f"GPU 메모리 할당량: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
    print(f"GPU 메모리 예약량: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")

# GPU 메모리 정리 (필요 시)
torch.cuda.empty_cache()
```

**배치 크기 권장사항:**
- L40S 46GB VRAM으로 큰 배치 크기 사용 가능
- GPT-124M 모델: batch_size=8~16 권장 (기본 설정은 2)
- 더 큰 모델이나 긴 시퀀스: batch_size 조정 필요

## 보너스 자료

### 주요 고급 토픽
- **ch02/05_bpe-from-scratch/**: BPE 토크나이저 구현
- **ch03/02_bonus_efficient-multihead-attention/**: 효율적인 어텐션 구현 비교
- **ch04/03_kv-cache/**: KV 캐시 최적화
- **ch05/07_gpt_to_llama/**: GPT를 Llama로 변환
- **ch05/11_qwen3/**: Qwen3 (Dense 및 MoE) 구현
- **ch05/12_gemma3/**: Gemma 3 구현
- **ch07/04_preference-tuning-with-dpo/**: DPO(Direct Preference Optimization)

## 코딩 규칙

### Linting (Ruff)
```bash
# pyproject.toml에 설정됨
# line-length: 140
# ignored rules: C406, E226, E402, E702, E703, E722, E731, E741
```

### 디렉토리 제외
`.venv` 디렉토리는 린팅에서 제외됩니다.

## Obsidian 학습 노트

**Obsidian 볼트 위치:** `/home/iamhjoo/Documents/IAMHJOO`

학습 중 중요한 개념이나 코드 차이점을 Obsidian에 기록할 때:
- 마크다운 형식으로 노트 생성
- 태그 활용 (`#python`, `#llm-scratch`, `#coding-tip` 등)
- 날짜 및 관련 챕터 링크 포함
- 코드 예시와 설명을 함께 작성

## 참고사항

### 일반 사항
- 메인 챕터 코드는 일반 노트북에서 합리적인 시간 내에 실행 가능하도록 설계됨
- GPU 없이도 작동하지만, ch05-ch07의 훈련은 GPU로 실행하면 훨씬 빠름
- 모든 독립 실행 스크립트는 `if __name__ == "__main__":` 블록을 포함

### 현재 시스템 특화 사항

**⚠️ 중요: 환경 관리**
- **항상** `conda activate llm-scratch` 실행 후 작업 시작
- 다른 프로젝트의 Python 환경과 격리되어 있음
- VSCode나 IDE에서 작업 시 인터프리터를 `/home/iamhjoo/miniconda3/envs/llm-scratch/bin/python`으로 설정

**GPU 활용**
- 현재 시스템은 NVIDIA L40S GPU가 있어 훈련 속도가 매우 빠름
- PyTorch가 자동으로 GPU를 감지하고 사용함
- 훈련 중 `nvidia-smi` 명령으로 GPU 사용률 모니터링 가능

**성능 최적화**
- 기본 batch_size=2는 매우 보수적인 설정
- L40S 46GB VRAM으로 batch_size를 8~16까지 증가 가능
- 더 큰 배치로 훈련 시간 대폭 단축 가능

**문제 해결**
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
```
