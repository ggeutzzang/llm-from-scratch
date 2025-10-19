"""
멀티헤드 어텐션 - 초간단 예제로 완벽히 이해하기
=====================================

설정을 최소한으로 줄여서:
- batch_size = 1  (배치 없이 한 샘플만)
- num_tokens = 3  (3개 토큰만)
- d_in = 2        (입력 차원 2)
- d_out = 2       (출력 차원 2)
- num_heads = 2   (2개 헤드)
- head_dim = 1    (헤드당 1차원)

이제 모든 행렬을 손으로 계산할 수 있습니다!
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# 재현 가능하게 시드 고정
torch.manual_seed(42)


def print_section(title):
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def simple_multihead_attention():
    """
    가장 간단한 멀티헤드 어텐션 예제
    """

    print_section("🎯 초간단 멀티헤드 어텐션 예제")

    # ============================================
    # 1. 간단한 입력 데이터 준비
    # ============================================
    print_section("1️⃣ 입력 데이터 준비")

    batch_size = 1
    num_tokens = 3
    d_in = 2

    # 입력 텐서: (1, 3, 2)
    x = torch.tensor([
        [[1.0, 0.0],      # 토큰 0
         [0.5, 0.5],      # 토큰 1
         [0.0, 1.0]]      # 토큰 2
    ], dtype=torch.float32)

    print(f"입력 텐서 x: shape {x.shape}")
    print(f"  토큰 0: {x[0, 0, :]}")
    print(f"  토큰 1: {x[0, 1, :]}")
    print(f"  토큰 2: {x[0, 2, :]}")

    # ============================================
    # 2. Q, K, V 가중치 (수동으로 간단히 설정)
    # ============================================
    print_section("2️⃣ Q, K, V 변환 가중치 설정")

    d_out = 2
    num_heads = 2
    head_dim = d_out // num_heads  # = 1

    # 간단하게 고정된 가중치 사용
    W_query = torch.tensor([
        [0.1, 0.2],
        [0.3, 0.4]
    ], dtype=torch.float32).T  # (2, 2)

    W_key = torch.tensor([
        [0.2, 0.1],
        [0.4, 0.3]
    ], dtype=torch.float32).T  # (2, 2)

    W_value = torch.tensor([
        [0.15, 0.25],
        [0.35, 0.45]
    ], dtype=torch.float32).T  # (2, 2)

    print(f"W_query shape: {W_query.shape}")
    print(f"W_query:\n{W_query}")
    print(f"\nW_key shape: {W_key.shape}")
    print(f"W_key:\n{W_key}")
    print(f"\nW_value shape: {W_value.shape}")
    print(f"W_value:\n{W_value}")

    # ============================================
    # 3. STEP 1: Q, K, V 계산
    # ============================================
    print_section("3️⃣ STEP 1: Q, K, V 투영 계산 (x @ W)")

    # Q = x @ W_query : (1, 3, 2) @ (2, 2) = (1, 3, 2)
    queries = x @ W_query
    keys = x @ W_key
    values = x @ W_value

    print(f"Queries shape: {queries.shape}")
    print(f"Queries:\n{queries[0]}")
    print(f"\nKeys shape: {keys.shape}")
    print(f"Keys:\n{keys[0]}")
    print(f"\nValues shape: {values.shape}")
    print(f"Values:\n{values[0]}")

    # 수동 계산 확인
    print("\n[수동 계산 확인] Q 계산 (토큰 0):")
    print(f"  [1.0, 0.0] @ [[0.1, 0.3], [0.2, 0.4]] = {x[0, 0:1] @ W_query}")

    # ============================================
    # 4. STEP 2: 멀티헤드로 분할 (reshape)
    # ============================================
    print_section("4️⃣ STEP 2: 헤드로 분할 (reshape)")

    # (1, 3, 2) -> (1, 3, 2, 1)
    # 2를 (num_heads=2, head_dim=1)로 분할
    queries_multi = queries.view(batch_size, num_tokens, num_heads, head_dim)
    keys_multi = keys.view(batch_size, num_tokens, num_heads, head_dim)
    values_multi = values.view(batch_size, num_tokens, num_heads, head_dim)

    print(f"Queries 분할 후: {queries_multi.shape}")
    print(f"  (batch_size, num_tokens, num_heads, head_dim)")
    print(f"  = (1, 3, 2, 1)\n")

    print("Queries 분할 후 데이터:")
    print(f"  헤드 0의 쿼리: {queries_multi[0, :, 0, 0]}")
    print(f"  헤드 1의 쿼리: {queries_multi[0, :, 1, 0]}")
    print(f"\nKeys 분할 후 데이터:")
    print(f"  헤드 0의 키: {keys_multi[0, :, 0, 0]}")
    print(f"  헤드 1의 키: {keys_multi[0, :, 1, 0]}")
    print(f"\nValues 분할 후 데이터:")
    print(f"  헤드 0의 값: {values_multi[0, :, 0, 0]}")
    print(f"  헤드 1의 값: {values_multi[0, :, 1, 0]}")

    # ============================================
    # 5. STEP 3: 전치 (헤드 우선)
    # ============================================
    print_section("5️⃣ STEP 3: 전치 - 헤드를 우선 차원으로")

    # (1, 3, 2, 1) -> transpose(1,2) -> (1, 2, 3, 1)
    queries_T = queries_multi.transpose(1, 2)
    keys_T = keys_multi.transpose(1, 2)
    values_T = values_multi.transpose(1, 2)

    print(f"전치 후 shape: {queries_T.shape}")
    print(f"  (batch_size, num_heads, num_tokens, head_dim)")
    print(f"  = (1, 2, 3, 1)\n")

    print("전치 후 Queries:")
    print(f"  헤드 0: {queries_T[0, 0, :, 0]}")
    print(f"  헤드 1: {queries_T[0, 1, :, 0]}")

    # ============================================
    # 6. STEP 4: 어텐션 점수 계산 (Q @ K^T)
    # ============================================
    print_section("6️⃣ STEP 4: 어텐션 점수 = Q @ K^T")

    # (1, 2, 3, 1) @ (1, 2, 1, 3) = (1, 2, 3, 3)
    attn_scores = queries_T @ keys_T.transpose(2, 3)

    print(f"어텐션 점수 shape: {attn_scores.shape}")
    print(f"  (batch_size, num_heads, num_tokens, num_tokens)")
    print(f"  = (1, 2, 3, 3)\n")

    print("어텐션 점수 (각 토큰이 모든 토큰과의 유사도):")
    print(f"헤드 0:\n{attn_scores[0, 0, :, :]}")
    print(f"\n헤드 1:\n{attn_scores[0, 1, :, :]}")

    print("\n[의미 해석]")
    print("행: 쿼리 토큰 (현재 토큰)")
    print("열: 키 토큰 (참조할 토큰)")
    print("값: 유사도 점수 (높을수록 집중도 높음)")

    # ============================================
    # 7. STEP 5: 인과적 마스킹
    # ============================================
    print_section("7️⃣ STEP 5: 인과적 마스크 적용 (미래 토큰 가리기)")

    # 상삼각 행렬 (미래 토큰을 True로 표시)
    mask = torch.triu(torch.ones(num_tokens, num_tokens), diagonal=1).bool()

    print(f"마스크 패턴 (True = 마스킹할 위치):\n{mask}\n")

    print("설명:")
    print("  토큰 0: 토큰 1, 2 마스킹 (미래 불가)")
    print("  토큰 1: 토큰 2 마스킹")
    print("  토큰 2: 마스킹 없음 (자신만 가능)")

    # 마스킹 전 점수
    print(f"\n마스킹 전 (헤드 0):\n{attn_scores[0, 0, :, :]}")

    # 마스킹 적용
    attn_scores_masked = attn_scores.clone()
    attn_scores_masked[0, :, mask] = -torch.inf

    print(f"\n마스킹 후 (헤드 0):\n{attn_scores_masked[0, 0, :, :]}")
    print("\n주의: -inf는 softmax에서 0으로 변환됨")

    # ============================================
    # 8. STEP 6: Softmax + 스케일링
    # ============================================
    print_section("8️⃣ STEP 6: Softmax 적용 (확률로 변환)")

    # 스케일링 팩터: sqrt(head_dim) = sqrt(1) = 1.0
    scale_factor = head_dim ** 0.5

    print(f"스케일링 팩터: sqrt({head_dim}) = {scale_factor}")
    print(f"  (head_dim이 1이므로 특별한 효과 없음)\n")

    # Softmax 적용
    attn_weights = torch.softmax(attn_scores_masked / scale_factor, dim=-1)

    print(f"어텐션 가중치 shape: {attn_weights.shape}")
    print(f"어텐션 가중치 (헤드 0):\n{attn_weights[0, 0, :, :]}")
    print(f"\n각 행의 합 (1.0이어야 함):")
    print(attn_weights[0, 0, :, :].sum(dim=-1))

    print("\n[의미 해석]")
    print("각 토큰이 다른 토큰들에 할당한 가중치")
    print("값의 범위: 0 ~ 1, 합계: 1.0 (확률 분포)")

    # ============================================
    # 9. STEP 7: Value와 곱하기
    # ============================================
    print_section("9️⃣ STEP 7: 가중합 계산 = Weights @ Values")

    # (1, 2, 3, 3) @ (1, 2, 3, 1) = (1, 2, 3, 1)
    context_vectors = attn_weights @ values_T

    print(f"컨텍스트 벡터 shape: {context_vectors.shape}")
    print(f"  (batch_size, num_heads, num_tokens, head_dim)")
    print(f"  = (1, 2, 3, 1)\n")

    print("컨텍스트 벡터:")
    print(f"헤드 0 결과: {context_vectors[0, 0, :, 0]}")
    print(f"헤드 1 결과: {context_vectors[0, 1, :, 0]}")

    print("\n[수동 계산 예시] 헤드 0, 토큰 0:")
    print(f"  가중치: {attn_weights[0, 0, 0, :]}")
    print(f"  값들: {values_T[0, 0, :, 0]}")
    result = torch.dot(attn_weights[0, 0, 0, :], values_T[0, 0, :, 0])
    print(f"  가중합: {result.item():.6f}")

    # ============================================
    # 10. STEP 8: 헤드 결합
    # ============================================
    print_section("🔟 STEP 8: 헤드 결합 (concatenate)")

    # (1, 2, 3, 1) -> transpose(1,2) -> (1, 3, 2, 1) -> view -> (1, 3, 2)
    context_vectors_T = context_vectors.transpose(1, 2)
    output = context_vectors_T.contiguous().view(batch_size, num_tokens, d_out)

    print(f"전치 후 shape: {context_vectors_T.shape}")
    print(f"  (batch_size, num_tokens, num_heads, head_dim)")
    print(f"  = (1, 3, 2, 1)\n")

    print(f"최종 결합 후 shape: {output.shape}")
    print(f"  (batch_size, num_tokens, d_out)")
    print(f"  = (1, 3, 2)\n")

    print("결합된 출력 (모든 헤드의 결과 연결):")
    print(f"  토큰 0: {output[0, 0, :]}")
    print(f"  토큰 1: {output[0, 1, :]}")
    print(f"  토큰 2: {output[0, 2, :]}")

    print("\n[의미]")
    print("  첫 번째 값: 헤드 0의 결과")
    print("  두 번째 값: 헤드 1의 결과")

    return output, attn_weights, attn_scores_masked


def visualize_attention_flow():
    """
    어텐션의 흐름을 시각적으로 이해하기
    """
    print_section("📊 어텐션 흐름 시각화")

    print("""
    입력 3개 토큰의 처리 과정:

    입력 텐서 (1, 3, 2):
    ┌─────────────────────┐
    │  토큰0: [1.0, 0.0]  │
    │  토큰1: [0.5, 0.5]  │
    │  토큰2: [0.0, 1.0]  │
    └─────────────────────┘
         ↓ (Q, K, V 투영)
    ┌─────────────────────┐
    │  Q, K, V 계산      │
    │  shape: (1,3,2)    │
    └─────────────────────┘
         ↓ (reshape)
    ┌─────────────────────┐
    │  2개 헤드로 분할     │
    │  shape: (1,3,2,1)   │
    └─────────────────────┘
         ↓ (transpose)
    ┌─────────────────────┐
    │  헤드별 정렬         │
    │  shape: (1,2,3,1)   │
    └─────────────────────┘
         ↓ (Q @ K^T)
    ┌─────────────────────┐
    │  어텐션 점수 (3x3)  │
    │  각 토큰 쌍의 유사도 │
    └─────────────────────┘
         ↓ (마스킹)
    ┌─────────────────────┐
    │  미래 토큰 제외      │
    │  인과적 마스크 적용  │
    └─────────────────────┘
         ↓ (softmax)
    ┌─────────────────────┐
    │  어텐션 가중치 (3x3) │
    │  확률 분포 (행의합=1)│
    └─────────────────────┘
         ↓ (@ Values)
    ┌─────────────────────┐
    │  컨텍스트 벡터      │
    │  각 토큰의 집중 결과 │
    └─────────────────────┘
         ↓ (헤드 결합)
    ┌─────────────────────┐
    │  최종 출력 (1,3,2)  │
    │  2개 헤드 결합      │
    └─────────────────────┘
    """)


def compare_heads():
    """
    두 헤드가 어떻게 다르게 집중하는지 비교
    """
    print_section("🧠 헤드 간 집중 패턴 비교")

    output, attn_weights, attn_scores = simple_multihead_attention()

    print("\n" + "=" * 80)
    print("헤드 0 vs 헤드 1의 어텐션 가중치")
    print("=" * 80)

    print("\n[헤드 0] 어텐션 가중치:")
    print(attn_weights[0, 0, :, :])

    print("\n[헤드 1] 어텐션 가중치:")
    print(attn_weights[0, 1, :, :])

    print("\n[해석]")
    print("- 각 헤드는 다른 패턴으로 집중!")
    print("- 헤드 0: 어떤 토큰에 집중할까?")
    print("- 헤드 1: 다른 토큰에 집중할까?")
    print("- 여러 헤드를 합치면 더 풍부한 표현 가능!")


if __name__ == "__main__":
    print("\n" * 2)
    print("█" * 80)
    print("█  멀티헤드 어텐션 - 손으로 계산하면서 배우기")
    print("█" * 80)

    # 메인 예제 실행
    output, attn_weights, attn_scores = simple_multihead_attention()

    # 흐름 시각화
    visualize_attention_flow()

    # 헤드 비교
    compare_heads()

    print("\n" + "=" * 80)
    print("✅ 완료! 모든 단계를 이해했습니다!")
    print("=" * 80 + "\n")
