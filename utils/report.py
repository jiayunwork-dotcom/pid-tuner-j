import io
import numpy as np
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm, cm
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT


def generate_report(loop_data, metrics, harris_result, oscillation_result,
                    stiction_result, model_result, tuning_results, sim_result,
                    logo_path=None, company_name=""):
    buf = io.BytesIO()

    doc = SimpleDocTemplate(buf, pagesize=A4,
                            topMargin=20 * mm, bottomMargin=20 * mm,
                            leftMargin=15 * mm, rightMargin=15 * mm)

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("CustomTitle", parent=styles["Title"],
                                  fontSize=18, spaceAfter=10, alignment=TA_CENTER)
    heading_style = ParagraphStyle("CustomHeading", parent=styles["Heading2"],
                                    fontSize=14, spaceAfter=8, spaceBefore=12)
    body_style = ParagraphStyle("CustomBody", parent=styles["Normal"],
                                 fontSize=10, spaceAfter=4, leading=14)
    center_style = ParagraphStyle("CustomCenter", parent=styles["Normal"],
                                   fontSize=10, alignment=TA_CENTER)

    elements = []

    if company_name:
        elements.append(Paragraph(company_name, title_style))
        elements.append(Spacer(1, 5 * mm))

    elements.append(Paragraph("控制回路性能审计报告", title_style))
    elements.append(Spacer(1, 8 * mm))

    elements.append(Paragraph("1. 回路基本信息", heading_style))
    info_data = [
        ["回路名称", loop_data.get("name", "N/A")],
        ["采样周期", f"{loop_data.get('sampling_period', 'N/A')} s"],
        ["控制器类型", loop_data.get("controller_type", "N/A")],
        ["动作方向", loop_data.get("action_direction", "N/A")],
        ["数据点数", str(loop_data.get("n_points", "N/A"))],
        ["工艺区域", loop_data.get("area", "N/A")],
    ]
    info_table = Table(info_data, colWidths=[40 * mm, 80 * mm])
    info_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.lightgrey),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("PADDING", (0, 0), (-1, -1), 4),
    ]))
    elements.append(info_table)
    elements.append(Spacer(1, 5 * mm))

    elements.append(Paragraph("2. 性能指标", heading_style))
    if metrics:
        metrics_data = [["指标", "值"]]
        metric_labels = {
            "IAE": "IAE (绝对误差积分)",
            "ISE": "ISE (误差平方积分)",
            "ITAE": "ITAE (时间加权绝对误差积分)",
            "overshoot_pct": "过冲量 (%)",
            "settling_time": "调节时间 (s)",
            "steady_state_error": "稳态误差",
            "oscillation_period": "振荡周期 (s)",
            "decay_ratio": "衰减比",
        }
        for key, label in metric_labels.items():
            val = metrics.get(key, "N/A")
            if isinstance(val, float):
                val = f"{val:.4f}"
            metrics_data.append([label, str(val)])

        mt = Table(metrics_data, colWidths=[60 * mm, 60 * mm])
        mt.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("PADDING", (0, 0), (-1, -1), 4),
        ]))
        elements.append(mt)
    elements.append(Spacer(1, 5 * mm))

    elements.append(Paragraph("3. Harris最小方差基准", heading_style))
    if harris_result:
        harris_val = harris_result.get("harris", 0)
        improvement = harris_result.get("improvement", 0)
        status = "优秀" if harris_val >= 0.8 else ("有提升空间" if harris_val >= 0.5 else "急需调优")
        harris_data = [
            ["Harris指标", f"{harris_val:.4f}"],
            ["评估状态", status],
            ["潜在改善", f"{improvement:.1f}%"],
            ["实际方差", f"{harris_result.get('actual_var', 0):.4f}"],
        ]
        ht = Table(harris_data, colWidths=[40 * mm, 80 * mm])
        ht.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (0, -1), colors.lightgrey),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("PADDING", (0, 0), (-1, -1), 4),
        ]))
        elements.append(ht)
    elements.append(Spacer(1, 5 * mm))

    elements.append(Paragraph("4. 振荡诊断结论", heading_style))
    if oscillation_result:
        diag = oscillation_result.get("diagnosis", {})
        osc_data = [
            ["是否振荡", "是" if oscillation_result.get("is_oscillating") else "否"],
            ["根因", diag.get("root_cause", "N/A")],
            ["诊断说明", diag.get("description", "N/A")],
            ["建议措施", diag.get("recommendation", "N/A")],
        ]
        if oscillation_result.get("period"):
            osc_data.insert(1, ["振荡周期", f"{oscillation_result['period']:.2f} s"])
            osc_data.insert(2, ["振荡幅值", f"{oscillation_result.get('amplitude', 0):.4f}"])

        ot = Table(osc_data, colWidths=[30 * mm, 130 * mm])
        ot.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (0, -1), colors.lightgrey),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("PADDING", (0, 0), (-1, -1), 4),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ]))
        elements.append(ot)
    elements.append(Spacer(1, 5 * mm))

    elements.append(Paragraph("5. 阀门粘滞检测", heading_style))
    if stiction_result:
        st_data = [
            ["是否存在粘滞", "是" if stiction_result.get("has_stiction") else "否"],
            ["粘滞程度", stiction_result.get("stiction_severity", "N/A")],
            ["死区宽度", f"{stiction_result.get('dead_band_width', 0):.4f}"],
            ["CO反转次数", str(stiction_result.get("reversal_count", 0))],
            ["建议", stiction_result.get("recommendation", "N/A")],
        ]
        st = Table(st_data, colWidths=[35 * mm, 125 * mm])
        st.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (0, -1), colors.lightgrey),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("PADDING", (0, 0), (-1, -1), 4),
        ]))
        elements.append(st)
    elements.append(Spacer(1, 5 * mm))

    elements.append(Paragraph("6. 模型参数", heading_style))
    if model_result:
        model_data = [
            ["过程增益 K", f"{model_result.get('K', 0):.4f}"],
            ["时间常数 T (s)", f"{model_result.get('T', 0):.2f}"],
            ["纯滞后 L (s)", f"{model_result.get('L', 0):.2f}"],
            ["拟合优度 R²", f"{model_result.get('r_squared', 0):.4f}"],
            ["辨识方法", model_result.get("method", "N/A")],
        ]
        mdt = Table(model_data, colWidths=[40 * mm, 80 * mm])
        mdt.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (0, -1), colors.lightgrey),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("PADDING", (0, 0), (-1, -1), 4),
        ]))
        elements.append(mdt)
    elements.append(Spacer(1, 5 * mm))

    elements.append(Paragraph("7. PID整定建议", heading_style))
    if tuning_results:
        tuning_data = [["方法", "Kp", "Ki", "Kd"]]
        for method, params in tuning_results.items():
            tuning_data.append([
                params.get("name", method),
                f"{params.get('Kp', 0):.4f}",
                f"{params.get('Ki', 0):.4f}",
                f"{params.get('Kd', 0):.4f}",
            ])

        tt = Table(tuning_data, colWidths=[45 * mm, 35 * mm, 35 * mm, 35 * mm])
        tt.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("PADDING", (0, 0), (-1, -1), 4),
        ]))
        elements.append(tt)

    doc.build(elements)
    buf.seek(0)
    return buf
