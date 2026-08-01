"""教师、学生稀疏候选与 DOTA 目标的一对一匹配。"""

import math
from dataclasses import dataclass

import torch

from msdyolo.utils.rotatednms import rotatediou


@dataclass
class DistillationMatch:
    """显式保存师生双方独立索引，禁止错误复用候选顺序。"""

    batchindex: torch.Tensor
    studentindex: torch.Tensor
    teacherindex: torch.Tensor
    targetindex: torch.Tensor

    def __len__(self):
        return self.batchindex.numel()


def emptymatches(device):
    """创建位于指定设备的空匹配。"""
    empty = torch.empty(0, dtype=torch.long, device=device)
    return DistillationMatch(empty, empty, empty, empty)


def anglefromlogits(logits):
    """把 180 维 CSL logits 解码为 [-pi/2, pi/2) 弧度。"""
    indices = logits.argmax(-1).to(logits.dtype)
    return (indices - 90.0) * math.pi / 180.0


def predictionconfidence(values):
    """计算 objectness 与最大类别概率的乘积。"""
    classend = values.shape[-1] - 180
    return values[..., 4].sigmoid() * values[..., 5:classend].sigmoid().amax(-1)


def greedyunique(candidates):
    """按成本从小到大选取双方均不重复的匹配。"""
    chosen = []
    usedleft = set()
    usedright = set()
    for cost, left, right in sorted(candidates, key=lambda item: item[0]):
        if left in usedleft or right in usedright:
            continue
        chosen.append((left, right))
        usedleft.add(left)
        usedright.add(right)
    return chosen


def matchpredictions(
    student,
    teacher,
    targets,
    confidencethreshold=0.25,
    iouthreshold=0.1,
    distancethreshold=2.0,
    verbose=False,
):
    """教师按类/置信度/旋转 IoU 匹配 GT，学生按尺度归一化中心距离匹配。"""
    device = student.values.device
    if targets.numel() == 0:
        return emptymatches(device)
    if student.values.shape[0] != teacher.values.shape[0]:
        raise ValueError("Student and teacher batch sizes must match")

    batchmatches = []
    studentmatches = []
    teachermatches = []
    targetmatches = []
    classend = teacher.values.shape[-1] - 180

    # 诊断统计
    totalteacherpreds = 0
    filteredbyconfidence = 0
    filteredbyclass = 0
    filteredbyiou = 0
    teachergtpairs = 0

    for batchindex in range(student.values.shape[0]):
        targetindices = torch.where(targets[:, 0].long() == batchindex)[0]
        if targetindices.numel() == 0:
            continue

        teachervalues = teacher.values[batchindex]
        confidence = predictionconfidence(teachervalues)
        teacherclasses = teachervalues[:, 5:classend].argmax(-1)
        teacherangles = anglefromlogits(teachervalues[:, classend:])
        teachercandidates = []

        totalteacherpreds += teachervalues.shape[0]

        if verbose:
            print(f"\n[Batch {batchindex}] Teacher predictions: {teachervalues.shape[0]}")
            print(f"  Confidence: min={confidence.min():.6f} max={confidence.max():.6f} mean={confidence.mean():.6f}")
            print(f"  Targets: {len(targetindices)}")

        for teacherindex in range(teachervalues.shape[0]):
            if confidence[teacherindex].item() < confidencethreshold:
                filteredbyconfidence += 1
                continue
            for targetindex in targetindices.tolist():
                target = targets[targetindex]
                if teacherclasses[teacherindex].item() != int(target[1].item()):
                    filteredbyclass += 1
                    continue
                teacherbox = [
                    *teachervalues[teacherindex, :4].detach().cpu().tolist(),
                    teacherangles[teacherindex].item(),
                ]
                targetbox = target[2:7].detach().cpu().tolist()
                iou = rotatediou(teacherbox, targetbox)
                if iou >= iouthreshold:
                    teachercandidates.append((-iou, teacherindex, targetindex))
                else:
                    filteredbyiou += 1

        teacherpairs = greedyunique(teachercandidates)
        teachergtpairs += len(teacherpairs)

        if verbose and teacherpairs:
            print(f"  Teacher-GT pairs after matching: {len(teacherpairs)}")

        if not teacherpairs:
            continue

        studentvalues = student.values[batchindex]
        studentcandidates = []
        for pairindex, teacherpair in enumerate(teacherpairs):
            targetindex = teacherpair[1]
            target = targets[targetindex]
            scale = torch.minimum(target[4], target[5]).clamp_min(1.0)
            distances = torch.linalg.vector_norm(studentvalues[:, :2] - target[2:4], dim=-1) / scale
            for studentindex, distance in enumerate(distances.detach().cpu().tolist()):
                if distance <= distancethreshold:
                    studentcandidates.append((distance, studentindex, pairindex))

        studentpairs = greedyunique(studentcandidates)
        for studentindex, pairindex in studentpairs:
            teacherindex, targetindex = teacherpairs[pairindex]
            batchmatches.append(batchindex)
            studentmatches.append(studentindex)
            teachermatches.append(teacherindex)
            targetmatches.append(targetindex)

    if verbose:
        print(f"\n[Matching Summary]")
        print(f"  Total teacher predictions: {totalteacherpreds}")
        print(f"  Filtered by confidence<{confidencethreshold}: {filteredbyconfidence}")
        print(f"  Filtered by class mismatch: {filteredbyclass}")
        print(f"  Filtered by IoU<{iouthreshold}: {filteredbyiou}")
        print(f"  Teacher-GT pairs (after IoU): {teachergtpairs}")
        print(f"  Final student-teacher matches: {len(batchmatches)}")

    if not batchmatches:
        return emptymatches(device)
    return DistillationMatch(
        batchindex=torch.tensor(batchmatches, dtype=torch.long, device=device),
        studentindex=torch.tensor(studentmatches, dtype=torch.long, device=device),
        teacherindex=torch.tensor(teachermatches, dtype=torch.long, device=device),
        targetindex=torch.tensor(targetmatches, dtype=torch.long, device=device),
    )
