import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.degradation import (
    linear_regression, build_degradation_model, predict_failure_round,
    analyze_loop_degradation, analyze_all_loops, generate_maintenance_suggestions,
    get_degradation_arrow
)
import numpy as np


def test_linear_regression():
    print("=== 测试线性回归 ===")
    x = np.arange(10, dtype=float)
    y = 100 - 2 * x + np.random.randn(10) * 0.5
    slope, intercept, r2 = linear_regression(x, y)
    print(f"斜率: {slope:.4f}, 截距: {intercept:.4f}, R²: {r2:.4f}")
    assert slope < -1.5 and slope > -2.5, "斜率应该接近-2"
    assert r2 > 0.9, "R²应该很高"
    print("✓ 线性回归测试通过\n")


def test_build_degradation_model():
    print("=== 测试劣化模型构建 ===")
    scores = [95, 93, 91, 89, 87, 85, 83, 81, 79, 77]
    model = build_degradation_model(scores)
    print(f"斜率: {model['slope']:.4f}")
    print(f"R²: {model['r_squared']:.4f}")
    print(f"是否显著: {model['is_significant']}")
    print(f"数据点数: {model['n_points']}")
    print(f"当前值: {model['current_value']:.2f}")
    assert model["is_significant"], "这个序列应该判定为显著劣化"
    print("✓ 劣化模型测试通过\n")


def test_predict_failure_round():
    print("=== 测试故障预测 ===")
    scores = [95, 93, 91, 89, 87, 85, 83, 81, 79, 77]
    model = build_degradation_model(scores)
    warning_rounds = predict_failure_round(model, 60)
    danger_rounds = predict_failure_round(model, 40)
    print(f"跌破60分需要: {warning_rounds:.2f} 轮")
    print(f"跌破40分需要: {danger_rounds:.2f} 轮")
    assert warning_rounds is not None and warning_rounds > 0, "应该能预测到警戒线"
    assert danger_rounds is not None and danger_rounds > warning_rounds, "危险线应该在警戒线之后"
    print("✓ 故障预测测试通过\n")


def test_degradation_arrow():
    print("=== 测试方向箭头 ===")
    print(f"快速下降 (slope=-3, R²=0.8): {get_degradation_arrow(-3, 0.8)}")
    print(f"缓慢下降 (slope=-1, R²=0.5): {get_degradation_arrow(-1, 0.5)}")
    print(f"稳定 (slope=0, R²=0.8): {get_degradation_arrow(0, 0.8)}")
    print(f"上升 (slope=2, R²=0.7): {get_degradation_arrow(2, 0.7)}")
    print(f"无趋势 (slope=-0.1, R²=0.1): {get_degradation_arrow(-0.1, 0.1)}")
    print("✓ 方向箭头测试通过\n")


def test_maintenance_suggestions():
    print("=== 测试维护建议生成 ===")
    scores = [95, 93, 91, 89, 87, 85, 83, 81, 79, 77]
    model = build_degradation_model(scores)

    analysis = {
        "health_model": model,
        "warning_rounds": 15.0,
        "danger_rounds": 25.0,
        "is_significant": True,
        "fastest_degrading_key": "harris_index",
        "fastest_degrading_label": "Harris指标",
        "fastest_degrading_rate": 2.5,
    }

    suggestions = generate_maintenance_suggestions(analysis, 3600)
    print(f"生成了 {len(suggestions)} 条建议")
    for i, s in enumerate(suggestions):
        print(f"  {i+1}. [{s['priority']}] {s['action']}")
        print(f"     {s['description']}")
        print(f"     维护窗口: {s['maintenance_window_hours']:.1f} 小时")
    assert len(suggestions) <= 3, "建议数量不应该超过3条"
    print("✓ 维护建议测试通过\n")


def test_full_analysis():
    print("=== 测试完整分析 ===")

    inspection_history = []
    for round_idx in range(15):
        loop_results = []
        for loop_idx in range(3):
            base_health = 90 - loop_idx * 10
            health = base_health - round_idx * (1.5 + loop_idx * 0.5)
            loop_results.append({
                "loop_idx": loop_idx,
                "loop_name": f"回路-{loop_idx}",
                "health_score": health,
                "harris_index": 0.8 - round_idx * 0.02,
                "osc_amplitude_pct": 3 + round_idx * 0.5,
                "stiction_severity": "无" if round_idx < 5 else ("轻微" if round_idx < 10 else "中度"),
            })
        inspection_history.append({
            "timestamp": f"2024-01-{round_idx+1:02d} 00:00:00",
            "loop_results": loop_results,
        })

    loops = [
        {"name": "回路-0", "loop_idx": 0},
        {"name": "回路-1", "loop_idx": 1},
        {"name": "回路-2", "loop_idx": 2},
    ]

    all_analysis = analyze_all_loops(inspection_history, loops)
    print(f"分析了 {len(all_analysis)} 个回路")

    for i, analysis in enumerate(all_analysis):
        print(f"\n{i+1}. {analysis['loop_name']}:")
        print(f"   当前健康分: {analysis['health_model']['current_value']:.1f}")
        print(f"   劣化速率: {analysis['health_model']['slope']:.3f}/轮")
        print(f"   R²: {analysis['health_model']['r_squared']:.3f}")
        print(f"   劣化显著: {analysis['is_significant']}")
        print(f"   最快劣化指标: {analysis['fastest_degrading_label']}")
        if analysis["warning_rounds"]:
            print(f"   警戒线剩余: {analysis['warning_rounds']:.1f} 轮")
        if analysis["danger_rounds"]:
            print(f"   危险线剩余: {analysis['danger_rounds']:.1f} 轮")

    assert len(all_analysis) == 3, "应该分析3个回路"
    assert all_analysis[0]["loop_name"] == "回路-2", "最差的回路应该排最前"
    print("\n✓ 完整分析测试通过\n")


def main():
    print("=" * 60)
    print("劣化建模引擎 - 单元测试")
    print("=" * 60 + "\n")

    try:
        test_linear_regression()
        test_build_degradation_model()
        test_predict_failure_round()
        test_degradation_arrow()
        test_maintenance_suggestions()
        test_full_analysis()

        print("=" * 60)
        print("✓ 所有测试通过!")
        print("=" * 60)
        return 0
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
