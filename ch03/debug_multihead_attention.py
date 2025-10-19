"""
MultiHeadAttention 디버깅 스크립트
각 단계별로 텐서의 shape과 값을 출력하여 동작을 분석합니다.
"""

import torch
import torch.nn as nn


class MultiHeadAttentionDebug(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert (d_out % num_heads == 0), \
            "d_out은 num_heads로 나누어 떨어져야 합니다"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads  # 원하는 출력 차원에 맞도록 투영 차원을 낮춥니다.

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)  # Linear 층을 사용해 헤드의 출력을 결합합니다.
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length),
                       diagonal=1)
        )

        print("=" * 80)
        print("MultiHeadAttention 초기화")
        print("=" * 80)
        print(f"입력 차원 (d_in): {d_in}")
        print(f"출력 차원 (d_out): {d_out}")
        print(f"헤드 개수 (num_heads): {num_heads}")
        print(f"헤드당 차원 (head_dim): {self.head_dim}")
        print(f"컨텍스트 길이: {context_length}")
        print(f"드롭아웃: {dropout}")
        print(f"QKV 바이어스: {qkv_bias}")
        print()

    def forward(self, x, verbose=True):
        b, num_tokens, d_in = x.shape

        if verbose:
            print("\n" + "=" * 80)
            print("FORWARD PASS 시작")
            print("=" * 80)
            print(f"입력 shape: {x.shape} (batch_size={b}, num_tokens={num_tokens}, d_in={d_in})")
            print(f"입력 예시 (첫 번째 배치, 첫 번째 토큰):\n{x[0, 0, :]}")
            print()

        # Step 1: Query, Key, Value 계산
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        if verbose:
            print("-" * 80)
            print("STEP 1: Query, Key, Value 투영")
            print("-" * 80)
            print(f"Keys shape: {keys.shape}")
            print(f"Queries shape: {queries.shape}")
            print(f"Values shape: {values.shape}")
            print(f"Queries 예시 (첫 번째 배치, 첫 번째 토큰):\n{queries[0, 0, :]}")
            print()

        # Step 2: 멀티헤드로 분할
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        if verbose:
            print("-" * 80)
            print("STEP 2: 멀티헤드로 분할 (reshape)")
            print("-" * 80)
            print(f"Keys shape: {keys.shape}")
            print(f"  -> (batch_size, num_tokens, num_heads, head_dim)")
            print(f"Queries shape: {queries.shape}")
            print(f"Values shape: {values.shape}")
            print(f"Queries 예시 (첫 번째 배치, 첫 번째 토큰, 첫 번째 헤드):\n{queries[0, 0, 0, :]}")
            print()

        # Step 3: 전치 (transpose)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        if verbose:
            print("-" * 80)
            print("STEP 3: 차원 전치 (num_heads와 num_tokens 교환)")
            print("-" * 80)
            print(f"Keys shape: {keys.shape}")
            print(f"  -> (batch_size, num_heads, num_tokens, head_dim)")
            print(f"Queries shape: {queries.shape}")
            print(f"Values shape: {values.shape}")
            print()

        # Step 4: 어텐션 점수 계산
        attn_scores = queries @ keys.transpose(2, 3)

        if verbose:
            print("-" * 80)
            print("STEP 4: 어텐션 점수 계산 (Q @ K^T)")
            print("-" * 80)
            print(f"Attention scores shape: {attn_scores.shape}")
            print(f"  -> (batch_size, num_heads, num_tokens, num_tokens)")
            print(f"첫 번째 배치, 첫 번째 헤드의 어텐션 점수:\n{attn_scores[0, 0, :, :]}")
            print()

        # Step 5: 마스킹
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        if verbose:
            print("-" * 80)
            print("STEP 5: 인과적 마스크 적용")
            print("-" * 80)
            print(f"마스크 shape: {mask_bool.shape}")
            print(f"마스크 (상삼각 행렬, True=마스킹할 위치):\n{mask_bool}")
            print()

        attn_scores.masked_fill_(mask_bool, -torch.inf)

        if verbose:
            print("마스킹 후 어텐션 점수 (첫 번째 배치, 첫 번째 헤드):")
            print(attn_scores[0, 0, :, :])
            print()

        # Step 6: Softmax + Scaling
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)

        if verbose:
            print("-" * 80)
            print("STEP 6: Softmax 적용 (스케일링 포함)")
            print("-" * 80)
            print(f"스케일링 팩터: sqrt(head_dim) = sqrt({keys.shape[-1]}) = {keys.shape[-1]**0.5:.4f}")
            print(f"Attention weights shape: {attn_weights.shape}")
            print(f"첫 번째 배치, 첫 번째 헤드의 어텐션 가중치:")
            print(attn_weights[0, 0, :, :])
            print(f"각 행의 합 (1.0이어야 함): {attn_weights[0, 0, :, :].sum(dim=-1)}")
            print()

        attn_weights = self.dropout(attn_weights)

        # Step 7: Value와 곱하기
        context_vec = (attn_weights @ values).transpose(1, 2)

        if verbose:
            print("-" * 80)
            print("STEP 7: Value와 곱하기 (Attention Weights @ V)")
            print("-" * 80)
            print(f"Context vector shape (전치 후): {context_vec.shape}")
            print(f"  -> (batch_size, num_tokens, num_heads, head_dim)")
            print(f"첫 번째 배치, 첫 번째 토큰, 첫 번째 헤드의 컨텍스트 벡터:")
            print(context_vec[0, 0, 0, :])
            print()

        # Step 8: 헤드 결합
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)

        if verbose:
            print("-" * 80)
            print("STEP 8: 헤드 결합 (concat)")
            print("-" * 80)
            print(f"Context vector shape: {context_vec.shape}")
            print(f"  -> (batch_size, num_tokens, d_out)")
            print(f"첫 번째 배치, 첫 번째 토큰의 결합된 컨텍스트 벡터:")
            print(context_vec[0, 0, :])
            print()

        # Step 9: 최종 투영
        context_vec = self.out_proj(context_vec)

        if verbose:
            print("-" * 80)
            print("STEP 9: 출력 투영 (Linear)")
            print("-" * 80)
            print(f"최종 출력 shape: {context_vec.shape}")
            print(f"첫 번째 배치, 첫 번째 토큰의 최종 출력:")
            print(context_vec[0, 0, :])
            print()
            print("=" * 80)
            print("FORWARD PASS 완료")
            print("=" * 80)

        return context_vec


def main():
    # 시드 설정
    torch.manual_seed(123)

    # 샘플 입력 데이터 생성
    batch_size = 2
    context_length = 4
    d_in = 3

    print("\n샘플 입력 데이터 생성")
    print("=" * 80)
    batch = torch.randn(batch_size, context_length, d_in)
    print(f"입력 텐서 shape: {batch.shape}")
    print(f"입력 텐서:\n{batch}")
    print()

    # 모델 설정
    d_out = 2
    num_heads = 2

    # MultiHeadAttention 생성 및 실행
    mha = MultiHeadAttentionDebug(
        d_in=d_in,
        d_out=d_out,
        context_length=context_length,
        dropout=0.0,
        num_heads=num_heads
    )

    # Forward pass (verbose=True로 모든 디버깅 메시지 출력)
    context_vecs = mha(batch, verbose=True)

    print("\n최종 결과")
    print("=" * 80)
    print(f"출력 텐서:\n{context_vecs}")
    print(f"출력 shape: {context_vecs.shape}")
    print()

    # 간단한 실행 (verbose=False)
    print("\n" + "=" * 80)
    print("간단한 실행 (verbose=False)")
    print("=" * 80)
    context_vecs_simple = mha(batch, verbose=False)
    print(f"출력 shape: {context_vecs_simple.shape}")
    print(f"출력:\n{context_vecs_simple}")


if __name__ == "__main__":
    main()
