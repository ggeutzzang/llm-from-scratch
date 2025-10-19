"""
텐서 Shape 계산 - 대화형 학습 도구
==================================
다양한 Shape 조합을 직접 테스트해보세요!
"""

import torch


def explain_matmul_shape(a_shape, b_shape, show_broadcasting=True):
    """
    행렬 곱셈 결과 shape를 계산하고 설명합니다.

    Args:
        a_shape: A 텐서의 shape (tuple)
        b_shape: B 텐서의 shape (tuple)
    """
    print("\n" + "=" * 80)
    print(f"🧮 Shape 계산: {a_shape} @ {b_shape}")
    print("=" * 80)

    try:
        # 실제 텐서 생성하여 계산
        A = torch.randn(*a_shape)
        B = torch.randn(*b_shape)
        result = A @ B

        print(f"\n✅ 성공!")
        print(f"   A shape: {tuple(A.shape)}")
        print(f"   B shape: {tuple(B.shape)}")
        print(f"   A @ B = {tuple(result.shape)}")

    except RuntimeError as e:
        print(f"\n❌ 오류: {e}")
        print(f"   마지막 차원이 일치하지 않습니다!")
        print(f"   A의 마지막 차원: {a_shape[-1]}")
        print(f"   B의 첫 번째 차원: {b_shape[0]}")
        return

    # 상세 설명
    print("\n" + "-" * 80)
    print("📋 상세 분석")
    print("-" * 80)

    # 배치 차원 분석
    a_batch = a_shape[:-2] if len(a_shape) > 2 else ()
    b_batch = b_shape[:-2] if len(b_shape) > 2 else ()

    print(f"\n[1] 배치 차원 (앞의 모든 차원)")
    if a_batch:
        print(f"    A의 배치: {a_batch}")
    else:
        print(f"    A의 배치: 없음 (2D 행렬)")

    if b_batch:
        print(f"    B의 배치: {b_batch}")
    else:
        print(f"    B의 배치: 없음 (2D 행렬)")

    # 행렬 연산 분석
    print(f"\n[2] 행렬 연산 (마지막 2개 차원)")
    a_matrix = a_shape[-2:]
    b_matrix = b_shape[-2:] if len(b_shape) >= 2 else b_shape

    print(f"    A의 마지막 2개: {a_matrix}")
    print(f"    B의 형태: {b_shape}")

    if len(b_shape) == 1:
        print(f"    ⚠️  B가 1D입니다!")
        print(f"       해석: ({a_shape[-2]}, {a_shape[-1]}) @ ({b_shape[0]},)")
        print(f"       결과: ({a_shape[-2]},)")
    elif len(b_shape) == 2:
        print(f"    A @ B = ({a_shape[-2]}, {a_shape[-1]}) @ ({b_shape[0]}, {b_shape[1]})")
        print(f"    = ({a_shape[-2]}, {b_shape[1]})")
    else:
        print(f"    B의 마지막 2개: {b_shape[-2:]}")
        print(f"    연산: ({a_shape[-2]}, {a_shape[-1]}) @ ({b_shape[-2]}, {b_shape[-1]})")
        print(f"    결과: ({a_shape[-2]}, {b_shape[-1]})")

    # 최종 결과
    print(f"\n[3] 최종 결과")
    result_batch = result.shape[:-2] if len(result.shape) > 2 else ()
    result_matrix = result.shape[-2:]

    if result_batch:
        print(f"    배치: {result_batch} (A에서 보존)")
    print(f"    행렬: {result_matrix} (새로 계산됨)")
    print(f"    → 최종 shape: {tuple(result.shape)}")

    # 규칙 요약
    print(f"\n[4] 규칙 적용")
    print(f"    \"마지막 2개 차원만 행렬 곱셈, 나머지는 배치\"")
    if a_batch or b_batch:
        print(f"    배치 보존: {result_batch}")
    print(f"    행렬 연산: ({a_shape[-2]}, {a_shape[-1]}) @ ... = ({a_shape[-2]}, {b_shape[-1]})")


def test_common_patterns():
    """
    자주 사용되는 패턴 테스트
    """
    print("\n\n" + "█" * 80)
    print("█  자주 사용되는 패턴들")
    print("█" * 80)

    patterns = [
        # 기본 2D
        ((3, 2), (2, 5), "기본 행렬 곱"),

        # 배치 1개
        ((2, 3, 2), (2, 5), "배치 1개"),
        ((2, 3, 2), (2, 5), "배치 1개"),

        # 배치 2개
        ((2, 3, 4, 2), (2, 5), "배치 2개"),
        ((5, 2, 3, 4, 2), (2, 6), "배치 3개"),

        # 멀티헤드 어텐션
        ((2, 2, 4, 1), (2, 2, 1, 4), "어텐션: Q @ K^T"),
        ((2, 2, 4, 4), (2, 2, 4, 1), "어텐션: Weights @ V"),

        # 1D 벡터
        ((3, 2), (2,), "2D @ 1D"),
        ((2, 3, 2), (2,), "3D @ 1D"),
    ]

    for a_shape, b_shape, description in patterns:
        print(f"\n{description}: {a_shape} @ {b_shape}")
        print("-" * 40)
        try:
            A = torch.randn(*a_shape)
            B = torch.randn(*b_shape)
            result = A @ B
            print(f"✅ {tuple(result.shape)}")
        except Exception as e:
            print(f"❌ {str(e)[:50]}...")


def interactive_calculator():
    """
    대화형 계산기
    """
    print("\n\n" + "█" * 80)
    print("█  대화형 Shape 계산기")
    print("█" * 80)
    print("\n다음 예제들을 직접 실행해보세요:\n")

    examples = [
        ((1, 3, 2), (2, 2), "간단한 예제"),
        ((2, 4, 3), (3, 5), "배치 있는 예제"),
        ((2, 3, 4, 5), (5, 6), "배치 2개"),
        ((2, 2, 4, 1), (2, 2, 1, 4), "멀티헤드 어텐션 Q@K^T"),
    ]

    print("예제:")
    for i, (a_shape, b_shape, desc) in enumerate(examples, 1):
        print(f"  {i}. {desc}")
        print(f"     {a_shape} @ {b_shape}")

    print("\n" + "=" * 80)
    print("각 예제를 직접 계산해봅시다:")
    print("=" * 80)

    for a_shape, b_shape, desc in examples:
        explain_matmul_shape(a_shape, b_shape)


def show_mental_model():
    """
    정신 모델 설명
    """
    print("\n\n" + "█" * 80)
    print("█  정신 모델 (Mental Model)")
    print("█" * 80)

    print("""
🧠 "마지막 2개 차원만 생각하는" 정신 모델:

┌─────────────────────────────────────────────────────┐
│  A @ B 계산 시                                      │
├─────────────────────────────────────────────────────┤
│                                                     │
│  1️⃣ A shape를 3부분으로 나눔                      │
│     A: [배치 부분 | 마지막 2개]                    │
│        (d1, d2, ..., n, m)                         │
│                                                     │
│  2️⃣ B shape를 3부분으로 나눔                      │
│     B: [배치 부분 | 마지막 2개]                    │
│        (e1, e2, ..., m, k)                         │
│                                                     │
│  3️⃣ 검증: A의 마지막 = B의 -2번째? (m = m) ✓   │
│                                                     │
│  4️⃣ 결과 계산                                      │
│     배치: max(배치부분들)                           │
│     행렬: (n, k)                                    │
│     결과: [배치 | (n, k)]                          │
│                                                     │
└─────────────────────────────────────────────────────┘

💡 핵심: 배치는 자동 처리, 마지막 2개만 신경 쓰면 됨!
    """)


if __name__ == "__main__":
    print("\n" * 2)
    print("█" * 80)
    print("█  텐서 Shape 계산 - 완벽 가이드")
    print("█" * 80)

    # 정신 모델 설명
    show_mental_model()

    # 공통 패턴
    test_common_patterns()

    # 대화형 계산기
    interactive_calculator()

    # 추가 설명
    print("\n\n" + "=" * 80)
    print("🎯 핵심 정리")
    print("=" * 80)
    print("""
✨ 텐서 곱셈 shape 계산 5초 컷:

  "마지막 2개 차원만 행렬 곱셈 규칙 적용,
   나머지는 배치로 보존!"

예시:
  (2, 3, 4, 5) @ (5, 6)
   └─배치──┬──┘  └─┬──┘
         마지막 2개

  마지막 2개: (4, 5) @ (5, 6) = (4, 6)
  배치: (2, 3) 보존
  결과: (2, 3, 4, 6)

Done! 🚀
    """)
