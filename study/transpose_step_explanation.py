"""
STEP 3의 전치가 왜 필요한가?
================================

멀티헤드 어텐션에서 전치는 "헤드별 병렬 처리"를 위한 것입니다!
"""

import torch
import torch.nn as nn

print("=" * 80)
print("STEP 3 전치의 의미를 이해해보자!")
print("=" * 80)

# 간단한 데이터 준비
batch_size = 1
num_tokens = 3
num_heads = 2
head_dim = 1

# 예제: queries의 모양
# (1, 3, 2) → STEP 2에서 reshape → (1, 3, 2, 1)
queries_multi = torch.tensor([
    [
        [[0.1], [0.3]],    # 토큰 0: [헤드0_값, 헤드1_값]
        [[0.15], [0.35]],  # 토큰 1: [헤드0_값, 헤드1_값]
        [[0.2], [0.4]]     # 토큰 2: [헤드0_값, 헤드1_값]
    ]
], dtype=torch.float32)

print(f"\n📦 STEP 2 후 queries_multi의 shape: {queries_multi.shape}")
print(f"   (batch_size=1, num_tokens=3, num_heads=2, head_dim=1)")
print(f"\nqueries_multi 내용:")
print(queries_multi)

print("\n" + "=" * 80)
print("❌ 만약 전치하지 않고 바로 행렬 곱셈을 한다면?")
print("=" * 80)

# 전치 없이 직접 곱하기 시도
print(f"\n현재 shape: {queries_multi.shape}")
print(f"행렬 곱셈을 하려면: (1, 3, 2, 1) @ (1, 3, 1, 2) 필요")
print(f"하지만 현재 구조로는 이게 불가능합니다! 왜?")
print(f"""
  왜냐하면:
  - 토큰(dim=1)과 헤드(dim=2)가 분리되어 있음
  - 각 헤드별로 독립적인 어텐션 계산을 하기 어려움
  - 배치 행렬 곱셈은 **마지막 두 차원**만 행렬 곱하기 수행
    (첫 차원들은 "배치" 처리로 취급)
""")

print("=" * 80)
print("✅ 전치하는 이유: 헤드를 배치 차원으로 승격!")
print("=" * 80)

# STEP 3: 전치
queries_T = queries_multi.transpose(1, 2)  # (1, 3, 2, 1) → (1, 2, 3, 1)

print(f"\n전치 후 queries_T shape: {queries_T.shape}")
print(f"(batch_size=1, num_heads=2, num_tokens=3, head_dim=1)")
print(f"\nqueries_T 내용:")
print(queries_T)

print(f"""
[핵심 아이디어]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

전치 전: (1, 3, 2, 1)
         batch, tokens, heads, head_dim

전치 후: (1, 2, 3, 1)
         batch, heads, tokens, head_dim

이제 PyTorch는 이렇게 해석합니다:
┌─────────────────────────────────────┐
│ "2개 헤드"를 배치로 처리해야 한다!  │
│                                     │
│ 즉, 2개의 (1, 3, 1) 배치            │
│    각각에 대해 독립적으로 어텐션    │
└─────────────────────────────────────┘
""")

print("=" * 80)
print("🔍 구체적 예: 헤드별로 분리되는 모습")
print("=" * 80)

print(f"\n헤드 0만 추출:")
print(f"  queries_T[0, 0, :, :] shape: {queries_T[0, 0, :, :].shape}")
print(f"  데이터: {queries_T[0, 0, :, :]}")
print(f"  의미: 토큰 3개의 헤드0 쿼리값 [0.1, 0.15, 0.2]")

print(f"\n헤드 1만 추출:")
print(f"  queries_T[0, 1, :, :] shape: {queries_T[0, 1, :, :].shape}")
print(f"  데이터: {queries_T[0, 1, :, :]}")
print(f"  의미: 토큰 3개의 헤드1 쿼리값 [0.3, 0.35, 0.4]")

print("\n" + "=" * 80)
print("📐 STEP 4에서 행렬 곱셈이 어떻게 작동하는가?")
print("=" * 80)

keys_T = queries_T.clone()  # 간단히 같다고 가정

print(f"\nqueries_T shape: {queries_T.shape}")
print(f"keys_T shape: {keys_T.shape}")

# keys_T를 또 전치
keys_T_transposed = keys_T.transpose(2, 3)
print(f"keys_T.transpose(2, 3) shape: {keys_T_transposed.shape}")
print(f"  (batch, num_heads, head_dim, num_tokens)")

print(f"""
🔢 행렬 곱셈 작동:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

queries_T @ keys_T.transpose(2, 3)
= (1, 2, 3, 1) @ (1, 2, 1, 3)

PyTorch의 배치 @ 연산 규칙:
- 첫 번째 배치 차원들 유지: (1, 2)
- 마지막 두 차원만 행렬 곱하기: (3, 1) @ (1, 3) = (3, 3)

결과: (1, 2, 3, 3)
      배치, 헤드, 쿼리토큰, 키토큰

즉, 2개 헤드 각각에 대해:
┌──────────────────────────────────────┐
│ 헤드 0: (3, 1) @ (1, 3) = (3, 3)    │
│ 헤드 1: (3, 1) @ (1, 3) = (3, 3)    │
└──────────────────────────────────────┘

이렇게 **병렬로** 2개 헤드를 동시에 처리!
""")

print("=" * 80)
print("❌ 전치하지 않으면 어떻게 될까?")
print("=" * 80)

print(f"""
전치 없이 (1, 3, 2, 1)로 바로 곱하려면:

  (1, 3, 2, 1) @ (1, 3, 1, 2)

PyTorch는 이렇게 해석:
  - 배치 차원: (1, 3)
  - 행렬 곱하기: (2, 1) @ (1, 2) = (2, 2)

결과: (1, 3, 2, 2) ❌ 이건 우리가 원하는 구조가 아님!

원하는 것: 각 헤드별로 독립적으로 어텐션 계산
얻어지는 것: 토큰별로 혼잡한 구조
""")

print("\n" + "=" * 80)
print("💡 최종 정리")
print("=" * 80)

print("""
1️⃣ STEP 2 (reshape):
   (batch, tokens, d_out)
   → (batch, tokens, num_heads, head_dim)

   '출력 차원'을 '헤드'로 분해

2️⃣ STEP 3 (transpose):
   (batch, tokens, num_heads, head_dim)
   → (batch, num_heads, tokens, head_dim)

   헤드를 배치 차원으로 승격!
   → 이제 "num_heads개의 배치"로 처리 가능
   → PyTorch의 배치 행렬 곱셈 활용 가능

3️⃣ STEP 4 (matmul):
   (batch, num_heads, tokens, head_dim) @ (batch, num_heads, head_dim, tokens)
   = (batch, num_heads, tokens, tokens)

   각 헤드별로 독립적인 어텐션 점수 계산!
   → 병렬 처리로 효율적!

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
결론: 전치는 "헤드별 병렬 처리"를 위한 필수 구조 변환!
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")

print("\n" + "=" * 80)
print("🎯 실제 계산 예제")
print("=" * 80)

# 간단한 숫자로 직접 계산
print("\n[손으로 계산하는 예]")
print(f"헤드 0만 봤을 때:")
print(f"  queries_T[0, 0, :, :] = {queries_T[0, 0, :, :].squeeze()}")
print(f"  keys_T[0, 0, :, :] = {keys_T[0, 0, :, :].squeeze()}")
print(f"  queries_T[0, 0, :, :] @ keys_T[0, 0, :, :].T =")

result_head0 = queries_T[0, 0, :, :] @ keys_T[0, 0, :, :].transpose(0, 1)
print(f"  {result_head0}")

print(f"\n헤드 1만 봤을 때:")
print(f"  queries_T[0, 1, :, :] @ keys_T[0, 1, :, :].T =")
result_head1 = queries_T[0, 1, :, :] @ keys_T[0, 1, :, :].transpose(0, 1)
print(f"  {result_head1}")

print(f"\n배치 연산으로 한 번에:")
result_batch = queries_T @ keys_T.transpose(2, 3)
print(f"  queries_T @ keys_T.transpose(2, 3) =")
print(f"  {result_batch}")
print(f"\n✅ 동일한 결과!")
