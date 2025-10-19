"""
Q @ K^T 계산에서 마지막 차원이 head_dim이 아니라 num_tokens인 이유
====================================================================
"""

import torch

print("=" * 80)
print("Q @ K^T 계산 상세 분석")
print("=" * 80)

batch_size = 1
num_heads = 2
num_tokens = 3
head_dim = 1

# 전치 후의 텐서들
queries_T = torch.randn(batch_size, num_heads, num_tokens, head_dim)
keys_T = torch.randn(batch_size, num_heads, num_tokens, head_dim)

print(f"\n📦 전치 후 상태:")
print(f"queries_T shape: {queries_T.shape}")
print(f"  (batch, num_heads, num_tokens, head_dim)")
print(f"  = ({batch_size}, {num_heads}, {num_tokens}, {head_dim})")

print(f"\nkeys_T shape: {keys_T.shape}")
print(f"  (batch, num_heads, num_tokens, head_dim)")
print(f"  = ({batch_size}, {num_heads}, {num_tokens}, {head_dim})")

print("\n" + "=" * 80)
print("❌ 흔한 오해: 마지막이 head_dim?")
print("=" * 80)

print(f"""
아니요! 어텐션 공식을 보면:

Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V

여기서:
- Q shape: (num_tokens, head_dim)     = ({num_tokens}, {head_dim})
- K^T shape: (head_dim, num_tokens)   = ({head_dim}, {num_tokens})

따라서 Q @ K^T의 결과:
- ({num_tokens}, {head_dim}) @ ({head_dim}, {num_tokens})
- = ({num_tokens}, {num_tokens}) ✅

마지막 차원이 num_tokens인 이유:
K^T의 마지막 차원이 num_tokens이기 때문!
""")

print("=" * 80)
print("✅ 수학적으로 증명해보자")
print("=" * 80)

print("\n🔢 1단계: 마지막 두 차원만 보기")
print(f"queries_T의 마지막 두 차원: ({num_tokens}, {head_dim})")
print(f"keys_T의 마지막 두 차원: ({num_tokens}, {head_dim})")

print("\n🔢 2단계: keys_T를 전치해야 Q @ K^T를 계산 가능")
keys_T_transposed = keys_T.transpose(2, 3)
print(f"keys_T.transpose(2, 3) shape: {keys_T_transposed.shape}")
print(f"  마지막 두 차원: ({head_dim}, {num_tokens})")

print("\n🔢 3단계: 행렬 곱셈 규칙")
print("""
행렬 곱셈 A @ B에서:
- A의 마지막 차원 = B의 첫 번째-끝에서-두번째 차원
- 결과 = A의 모든 앞 차원 + A의 첫-끝에서-두번째 차원 + B의 마지막 차원

따라서:
Q @ K^T = (batch, num_heads, num_tokens, head_dim) @ (batch, num_heads, head_dim, num_tokens)

배치 차원: (batch, num_heads) 유지
행렬 부분: (num_tokens, head_dim) @ (head_dim, num_tokens)

결과의 마지막 두 차원:
- (num_tokens, head_dim) @ (head_dim, num_tokens)
- head_dim끼리 곱해져서 사라짐!
- 결과 = (num_tokens, num_tokens) ← num_tokens이 남음!
""")

print("=" * 80)
print("🎯 실제 계산 예제")
print("=" * 80)

# 간단한 숫자로 계산
q = torch.tensor([[[0.1], [0.2], [0.3]]], dtype=torch.float32)  # (1, 3, 1)
k = torch.tensor([[[0.1], [0.2], [0.3]]], dtype=torch.float32)  # (1, 3, 1)

print(f"\nQ shape: {q.shape}")
print(f"K shape: {k.shape}")

print(f"\nQ @ K.T 계산:")
print(f"K.T shape: {k.transpose(1, 2).shape}")

result = q @ k.transpose(1, 2)
print(f"Q @ K.T shape: {result.shape}")
print(f"  = (batch=1, num_tokens=3, num_tokens=3) ✅")

print(f"\n결과 (각 토큰 쌍의 유사도):")
print(result[0])

print("\n" + "=" * 80)
print("📊 차원별 의미 정리")
print("=" * 80)

print("""
어텐션 점수 행렬: (batch, num_heads, num_tokens, num_tokens)

예: (1, 2, 3, 3)이면
├─ 1: 배치 1개
├─ 2: 헤드 2개
├─ 3: 쿼리 토큰 3개 (현재 위치)
└─ 3: 키 토큰 3개 (참조 위치)

행 i: 쿼리 토큰 i
열 j: 키 토큰 j
값 [i, j]: 토큰 i가 토큰 j와 얼마나 관련 있는가?

예를 들어 [0, 2] = 토큰 0이 토큰 2에 어느 정도 집중?
         [2, 1] = 토큰 2가 토큰 1에 어느 정도 집중?

→ 이것이 왜 (num_tokens, num_tokens)인지 명확!
""")

print("\n" + "=" * 80)
print("❓ 만약 마지막이 head_dim이라면?")
print("=" * 80)

print(f"""
만약 결과가 (1, 2, 3, {head_dim})이라면:
- {head_dim}은 뭘 의미하나?
- head_dim은 이미 (batch, num_heads)에 포함됨
- 어텐션 점수는 "모든 토큰 쌍"에 대한 유사도
- 따라서 (num_tokens, num_tokens) 필요!

head_dim은 Q, K, V의 내부 차원일 뿐,
최종 어텐션 점수의 구조와는 무관함!
""")

print("\n" + "=" * 80)
print("✅ 최종 결론")
print("=" * 80)

print("""
Q @ K^T의 최종 shape:
(batch_size, num_heads, num_tokens, num_tokens)

마지막 두 차원:
- 행(3번째): num_tokens - 쿼리 토큰 개수
- 열(4번째): num_tokens - 키 토큰 개수

이것이 "어텐션 점수"를 만드는 것!
각 쿼리 토큰이 각 키 토큰에 얼마나 집중하는지!
""")
