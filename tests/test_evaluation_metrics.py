from evaluation.evaluate_attack import quality_constrained_asr


def test_quality_constrained_asr_uses_fixed_original_eligible_denominator():
    rows = [
        {"ground_truth_label": "FAKE", "initial_label": "FAKE", "attack_success": True, "ssim": 0.92},
        {"ground_truth_label": "FAKE", "initial_label": "FAKE", "attack_success": True, "ssim": 0.82},
        {"ground_truth_label": "FAKE", "initial_label": "REAL", "attack_success": True, "ssim": 1.0},
        {"ground_truth_label": "REAL", "initial_label": "FAKE", "attack_success": True, "ssim": 1.0},
    ]
    result = quality_constrained_asr(rows)
    assert result["ssim_gte_0.90"] == {"successes": 1, "eligible": 2, "asr": 0.5}
    assert result["ssim_gte_0.80"] == {"successes": 2, "eligible": 2, "asr": 1.0}
