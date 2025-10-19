# MultiHeadAttention 동작 분석

이 문서는 `ch03/debug_multihead_attention.py` 실행 결과를 바탕으로 멀티헤드 어텐션의 각 단계를 상세히 설명합니다.

## 실행 방법

```bash
conda activate llm-scratch
python ch03/debug_multihead_attention.py
```

## 모델 설정

| 파라미터 | 값 | 설명 |
|---------|-----|------|
| `batch_size` | 2 | 배치 크기 |
| `num_tokens` | 4 | 시퀀스 길이 (토큰 개수) |
| `d_in` | 3 | 입력 차원 |
| `d_out` | 2 | 출력 차원 |
| `num_heads` | 2 | 어텐션 헤드 개수 |
| `head_dim` | 1 | 헤드당 차원 (= d_out / num_heads) |

## 입력 데이터

```python
입력 shape: (batch_size=2, num_tokens=4, d_in=3)
```

배치 1의 4개 토큰:
```
token 0: [ 0.3374, -0.1778, -0.3035]
token 1: [-0.5880,  0.3486,  0.6603]
token 2: [-0.2196, -0.3792, -0.1606]
token 3: [-0.4015,  0.6957, -1.8061]
```

---

## 9단계 Forward Pass 분석

### STEP 1: Query, Key, Value 투영

**동작:** 입력을 3개의 Linear 레이어를 통과시켜 Q, K, V 생성

```python
keys    = W_key(x)    # (2, 4, 3) -> (2, 4, 2)
queries = W_query(x)  # (2, 4, 3) -> (2, 4, 2)
values  = W_value(x)  # (2, 4, 3) -> (2, 4, 2)
```

**결과 shape:** `(batch_size, num_tokens, d_out)` = `(2, 4, 2)`

**의미:**
- 각 토큰의 3차원 입력을 2차원 Q, K, V로 투영
- 이 투영은 학습 가능한 가중치로 이루어짐

**예시:**
```
첫 번째 토큰의 Query: [0.0346, 0.1871]
```

---

### STEP 2: 멀티헤드로 분할 (Reshape)

**동작:** `d_out` 차원을 `num_heads`와 `head_dim`으로 분할

```python
# (2, 4, 2) -> (2, 4, 2, 1)
keys    = keys.view(b, num_tokens, num_heads, head_dim)
queries = queries.view(b, num_tokens, num_heads, head_dim)
values  = values.view(b, num_tokens, num_heads, head_dim)
```

**결과 shape:** `(batch_size, num_tokens, num_heads, head_dim)` = `(2, 4, 2, 1)`

**의미:**
- 2차원 출력을 2개의 헤드로 분할
- 각 헤드는 1차원 벡터를 다룸
- **중요:** 실제로는 메모리 재배치가 아니라 텐서의 "뷰"만 변경

**예시:**
```
첫 번째 토큰의 Query -> 2개 헤드로 분할:
  헤드 0: [0.0346]
  헤드 1: [0.1871]
```

---

### STEP 3: 차원 전치 (Transpose)

**동작:** `num_heads`와 `num_tokens` 차원을 교환

```python
# (2, 4, 2, 1) -> (2, 2, 4, 1)
keys    = keys.transpose(1, 2)
queries = queries.transpose(1, 2)
values  = values.transpose(1, 2)
```

**결과 shape:** `(batch_size, num_heads, num_tokens, head_dim)` = `(2, 2, 4, 1)`

**의미:**
- 각 헤드가 모든 토큰을 독립적으로 처리할 수 있도록 재구성
- 이제 각 헤드는 `(batch_size, num_heads, num_tokens, head_dim)` 형태
- 배치 행렬 곱셈을 효율적으로 수행 가능

---

### STEP 4: 어텐션 점수 계산 (Q @ K^T)

**동작:** Query와 Key의 전치 행렬을 곱함

```python
# (2, 2, 4, 1) @ (2, 2, 1, 4) -> (2, 2, 4, 4)
attn_scores = queries @ keys.transpose(2, 3)
```

**결과 shape:** `(batch_size, num_heads, num_tokens, num_tokens)` = `(2, 2, 4, 4)`

**의미:**
- 각 토큰이 다른 모든 토큰과 얼마나 관련 있는지 계산
- 결과 행렬의 `[i, j]` 원소 = i번째 토큰이 j번째 토큰에 대한 유사도

**예시 (배치1, 헤드1의 어텐션 점수):**
```
        token0   token1   token2   token3
token0  -0.0070   0.0147  -0.0034  -0.0287
token1   0.0161  -0.0336   0.0077   0.0657
token2  -0.0303   0.0634  -0.0145  -0.1241
token3   0.0157  -0.0328   0.0075   0.0642
```

---

### STEP 5: 인과적 마스크 적용

**동작:** 미래 토큰을 볼 수 없도록 상삼각 부분을 `-inf`로 채움

```python
mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
attn_scores.masked_fill_(mask_bool, -torch.inf)
```

**마스크 패턴:**
```
        token0   token1   token2   token3
token0  False    True     True     True
token1  False    False    True     True
token2  False    False    False    True
token3  False    False    False    False
```
- `False` = 볼 수 있음 (현재 및 과거 토큰)
- `True` = 볼 수 없음 (미래 토큰, -inf로 마스킹)

**마스킹 후 어텐션 점수:**
```
        token0   token1   token2   token3
token0  -0.0070   -inf     -inf     -inf
token1   0.0161  -0.0336   -inf     -inf
token2  -0.0303   0.0634  -0.0145   -inf
token3   0.0157  -0.0328   0.0075   0.0642
```

**의미:**
- GPT와 같은 자기회귀 모델에서 필수적
- i번째 토큰은 i번째 이전 토큰만 참조 가능
- `-inf`는 softmax 후 0이 됨

---

### STEP 6: Softmax + Scaling

**동작:**
1. 점수를 `sqrt(head_dim)`으로 나누어 스케일링
2. 각 행에 softmax 적용

```python
attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
```

**스케일링:**
- `head_dim = 1` → 스케일링 팩터 = `sqrt(1) = 1.0`
- 일반적으로 `sqrt(head_dim)`을 사용하여 gradient 안정화

**Softmax 후 어텐션 가중치:**
```
        token0   token1   token2   token3
token0  1.0000   0.0000   0.0000   0.0000   (합계: 1.0)
token1  0.5124   0.4876   0.0000   0.0000   (합계: 1.0)
token2  0.3211   0.3527   0.3262   0.0000   (합계: 1.0)
token3  0.2504   0.2385   0.2483   0.2628   (합계: 1.0)
```

**의미:**
- 각 행의 합 = 1.0 (확률 분포)
- token0은 자기 자신만 100% 참조
- token3은 모든 이전 토큰을 골고루 참조 (약 25%씩)
- 마스킹된 위치는 0.0

---

### STEP 7: Value와 곱하기

**동작:** 어텐션 가중치로 Value의 가중 평균 계산

```python
# (2, 2, 4, 4) @ (2, 2, 4, 1) -> (2, 2, 4, 1)
context_vec = (attn_weights @ values).transpose(1, 2)
# -> (2, 4, 2, 1)
```

**결과 shape:** `(batch_size, num_tokens, num_heads, head_dim)` = `(2, 4, 2, 1)`

**의미:**
- 각 토큰에 대해 관련 있는 다른 토큰들의 Value를 가중 평균
- 어텐션 메커니즘의 핵심: "어디를 볼지" 결정 후 "무엇을 가져올지" 계산

**예시:**
```
첫 번째 토큰, 첫 번째 헤드의 컨텍스트 벡터: [0.0289]
```

---

### STEP 8: 헤드 결합 (Concatenate)

**동작:** 모든 헤드의 출력을 하나로 합침

```python
# (2, 4, 2, 1) -> (2, 4, 2)
context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
```

**결과 shape:** `(batch_size, num_tokens, d_out)` = `(2, 4, 2)`

**의미:**
- 2개 헤드의 1차원 출력 → 2차원으로 결합
- `[head0_output, head1_output]` 형태로 concat
- `.contiguous()`는 메모리 레이아웃 최적화

**예시:**
```
첫 번째 토큰의 결합된 컨텍스트 벡터: [0.0289, -0.2671]
  = [헤드0 출력: 0.0289, 헤드1 출력: -0.2671]
```

---

### STEP 9: 출력 투영 (Final Linear)

**동작:** 결합된 컨텍스트 벡터를 한 번 더 Linear 레이어 통과

```python
context_vec = self.out_proj(context_vec)  # (2, 4, 2) -> (2, 4, 2)
```

**결과 shape:** `(batch_size, num_tokens, d_out)` = `(2, 4, 2)`

**의미:**
- 멀티헤드 어텐션의 마지막 단계
- 결합된 헤드 출력을 최종 표현으로 변환
- 학습 가능한 가중치로 헤드 간 상호작용 학습

**최종 출력 예시:**
```
첫 번째 토큰: [-0.5140, -0.6289]
```

---

## 전체 요약

### Shape 변환 흐름

```
입력:
(2, 4, 3) - batch_size=2, num_tokens=4, d_in=3

↓ STEP 1: Q, K, V 투영
(2, 4, 2) - d_out=2로 투영

↓ STEP 2: 멀티헤드 분할
(2, 4, 2, 1) - num_heads=2, head_dim=1로 분할

↓ STEP 3: 전치
(2, 2, 4, 1) - 헤드와 토큰 차원 교환

↓ STEP 4: 어텐션 점수 계산
(2, 2, 4, 4) - 토큰 간 유사도 행렬

↓ STEP 5-6: 마스킹 + Softmax
(2, 2, 4, 4) - 확률 분포로 변환

↓ STEP 7: Value와 곱하기
(2, 2, 4, 1) -> (2, 4, 2, 1) - 컨텍스트 벡터 계산 및 전치

↓ STEP 8: 헤드 결합
(2, 4, 2) - 헤드 concat

↓ STEP 9: 출력 투영
(2, 4, 2) - 최종 출력

출력:
(2, 4, 2) - batch_size=2, num_tokens=4, d_out=2
```

### 핵심 개념

1. **멀티헤드의 장점**
   - 각 헤드가 다른 패턴을 학습
   - 병렬 처리로 효율성 향상
   - 표현력 증가

2. **인과적 마스킹**
   - 미래 정보 누출 방지
   - 자기회귀 생성에 필수
   - 상삼각 행렬을 `-inf`로 마스킹

3. **스케일링의 중요성**
   - `sqrt(head_dim)`으로 나누어 gradient 안정화
   - 큰 값으로 인한 softmax 포화 방지

4. **행렬 곱셈 효율성**
   - 전치(transpose)로 배치 행렬 곱셈 활용
   - GPU 병렬화에 최적화된 구조

## 추가 실험

스크립트를 수정하여 다양한 설정 테스트 가능:

```python
# 헤드 개수 변경
num_heads = 4  # d_out=4일 때, head_dim=1

# 출력 차원 변경
d_out = 8  # num_heads=4일 때, head_dim=2

# 시퀀스 길이 변경
context_length = 8  # 더 긴 시퀀스
```

## 참고

- 원본 코드: [ch03/01_main-chapter-code/ch03.ipynb](ch03/01_main-chapter-code/ch03.ipynb)
- 디버깅 스크립트: [ch03/debug_multihead_attention.py](ch03/debug_multihead_attention.py)
