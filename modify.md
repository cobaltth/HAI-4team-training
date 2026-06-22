# ---------------------------------------------------------
    # [수정됨] Lane Loss 및 Uniform Regularization (Soft Cross-Entropy)
    # ---------------------------------------------------------
    valid_lane_logits = lane_logits.reshape(-1, 4)[flat_mask]
    loss_lane = F.cross_entropy(valid_lane_logits, lane_t)

    # 1. 모델이 예측한 레인의 Softmax 확률값 계산
    lane_probs = F.softmax(valid_lane_logits, dim=-1) # (Valid_N, 4)
    
    # 2. 배치 내 전체 노트에 대한 각 레인별 예측 확률의 평균 계산
    mean_lane_probs = lane_probs.mean(dim=0)          # (4,)
    
    # 3. 이상적인 타겟 분포 설정 (4키이므로 0.25씩)
    uniform_target = torch.full_like(mean_lane_probs, 0.25)
    
    # 4. 예측 분포와 타겟 분포 간의 Soft Cross-Entropy 계산
    # 수식: - sum(target * log(prediction))
    # 1e-8은 log(0)으로 인해 Loss가 무한대(NaN)로 튀는 것을 방지하는 안전장치입니다.
    loss_lane_reg = - (uniform_target * torch.log(mean_lane_probs + 1e-8)).sum()
    # ---------------------------------------------------------