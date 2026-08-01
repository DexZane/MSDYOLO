"""分类、中心、尺度和角度四分量稀疏蒸馏损失。"""

import torch
import torch.nn.functional as F


def weightedmean(values, weights, zero):
    """以匹配数归一化，路由权重只控制样本贡献。"""
    if values.numel() == 0:
        return zero
    return (values * weights).mean()


def temperaturekl(studentlogits, teacherlogits, temperature):
    """计算带温度平方修正的逐样本 KL。"""
    studentlogprobability = F.log_softmax(studentlogits / temperature, -1)
    teacherprobability = F.softmax(teacherlogits.detach() / temperature, -1)
    divergence = F.kl_div(
        studentlogprobability,
        teacherprobability,
        reduction="none",
    ).sum(-1)
    return divergence * temperature * temperature


def computefourcomponentloss(
    student,
    teacher,
    matches,
    routing,
    imagesize,
    classtemperature=2.0,
    angletemperature=2.0,
):
    """计算分量损失；教师始终 detach，学生保持梯度。"""
    zero = student.values.sum() * 0.0
    if len(matches) == 0:
        return {
            "classification": zero,
            "center": zero,
            "scale": zero,
            "angle": zero,
            "total": zero,
        }
    if imagesize <= 0:
        raise ValueError("imagesize must be positive")

    studentvalues = student.values[matches.batchindex, matches.studentindex]
    teachervalues = teacher.values[matches.batchindex, matches.teacherindex].detach()
    classend = studentvalues.shape[-1] - 180

    classificationraw = temperaturekl(
        studentvalues[:, 5:classend],
        teachervalues[:, 5:classend],
        classtemperature,
    )
    centerraw = F.smooth_l1_loss(
        studentvalues[:, :2] / imagesize,
        teachervalues[:, :2] / imagesize,
        reduction="none",
    ).sum(-1)
    scaleraw = F.smooth_l1_loss(
        studentvalues[:, 2:4].clamp_min(1e-6).log(),
        teachervalues[:, 2:4].clamp_min(1e-6).log(),
        reduction="none",
    ).sum(-1)
    angleraw = temperaturekl(
        studentvalues[:, classend:],
        teachervalues[:, classend:],
        angletemperature,
    )

    classification = weightedmean(classificationraw, routing.classification, zero)
    center = weightedmean(centerraw, routing.center, zero)
    scale = weightedmean(scaleraw, routing.scale, zero)
    angle = weightedmean(angleraw, routing.angle, zero)
    total = classification + center + scale + angle
    return {
        "classification": classification,
        "center": center,
        "scale": scale,
        "angle": angle,
        "total": total,
    }
