"""
Sparse Prediction Extraction and Matching for MSDYOLO (重写版)
GPT第五轮审核修正：
1. 使用YOLOOutputDecoder正确解码
2. 处理eval模式输出
3. 避免二次sigmoid
4. 正确解码CSL角度
5. 实现一对一匹配
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
import logging
from utils.decoder import YOLOOutputDecoder

logger = logging.getLogger(__name__)


class SparsePredictionExtractor:
    """
    从YOLOv5-OBB检测头输出中提取稀疏预测

    GPT第五轮修正：使用正确的解码器
    """

    def __init__(
        self,
        conf_threshold: float = 0.25,
        top_k: int = 300,
        num_classes: int = 16,  # DOTA v1.5 has 16 classes
    ):
        self.conf_threshold = conf_threshold
        self.top_k = top_k
        self.num_classes = num_classes
        self.decoder = YOLOOutputDecoder(
            num_classes=num_classes,
            conf_threshold=conf_threshold
        )

    def extract_from_yolo_output(
        self,
        predictions,
        model_training: bool = False,
        batch_size: int = 1
    ) -> List[Dict[str, torch.Tensor]]:
        """
        从YOLOv5-OBB输出中提取稀疏预测（支持任意batch size）

        Args:
            predictions: YOLO模型输出
                - eval模式: (decoded_tensor, raw_list)
                - train模式: List[Tensor]
            model_training: 模型是否处于训练模式
            batch_size: batch大小

        Returns:
            每张图像的稀疏预测列表 List[Dict]
        """
        # 使用解码器正确解码
        try:
            boxes_xywh, theta, objectness, class_probs, class_ids, batch_indices = \
                self.decoder.decode(predictions, model_training=model_training)
        except NotImplementedError as e:
            # 训练模式暂不支持，必须抛出错误而非静默返回空预测
            raise NotImplementedError(
                f"Decoder不支持此输出格式: {e}. "
                "清晰分支必须使用eval模式以获得正确解码。"
            )

        # 按batch分组处理
        batch_predictions = []
        for b in range(batch_size):
            # 选择当前batch的预测
            batch_mask = batch_indices == b
            batch_boxes = boxes_xywh[batch_mask]
            batch_theta = theta[batch_mask]
            batch_objectness = objectness[batch_mask]
            batch_class_probs = class_probs[batch_mask]
            batch_class_ids = class_ids[batch_mask]

            # 计算置信度分数
            class_scores = batch_class_probs.gather(1, batch_class_ids.unsqueeze(1)).squeeze(1)
            confidence_scores = batch_objectness * class_scores

            # 过滤低置信度
            valid_mask = confidence_scores > self.conf_threshold
            valid_indices = torch.where(valid_mask)[0]

            if len(valid_indices) == 0:
                batch_predictions.append(self._empty_predictions(batch_boxes.device))
                continue

            # 选择有效预测
            valid_boxes = batch_boxes[valid_indices]
            valid_theta = batch_theta[valid_indices]
            valid_scores = confidence_scores[valid_indices]
            valid_class_probs = batch_class_probs[valid_indices]
            valid_class_ids = batch_class_ids[valid_indices]

            # Top-K选择
            k = min(self.top_k, len(valid_indices))
            if k < len(valid_indices):
                top_k_indices = torch.topk(valid_scores, k)[1]
                valid_boxes = valid_boxes[top_k_indices]
                valid_theta = valid_theta[top_k_indices]
                valid_scores = valid_scores[top_k_indices]
                valid_class_probs = valid_class_probs[top_k_indices]
                valid_class_ids = valid_class_ids[top_k_indices]

            # 组装5参数旋转框
            boxes_5d = torch.cat([
                valid_boxes,  # (K, 4)
                valid_theta.unsqueeze(-1)  # (K, 1)
            ], dim=-1)

            batch_predictions.append({
                'boxes': boxes_5d,  # (K, 5)
                'scores': valid_scores,  # (K,)
                'classes': valid_class_probs,  # (K, C)
                'class_ids': valid_class_ids,  # (K,)
                'valid_mask': torch.ones(k, dtype=torch.bool, device=boxes_5d.device),
            })

        return batch_predictions

    def _empty_predictions(self, device: torch.device) -> Dict[str, torch.Tensor]:
        """返回空预测"""
        return {
            'boxes': torch.zeros((0, 5), device=device),
            'scores': torch.zeros(0, device=device),
            'classes': torch.zeros((0, self.num_classes), device=device),
            'class_ids': torch.zeros(0, dtype=torch.long, device=device),
            'valid_mask': torch.zeros(0, dtype=torch.bool, device=device),
        }


class PredictionMatcher:
    """
    匹配清晰分支和退化分支的预测

    GPT第五轮修正：
    - 使用仓库旋转IoU或明确命名为距离匹配
    - 实现一对一匹配
    - 添加类别兼容性检查
    """

    def __init__(
        self,
        match_threshold: float = 0.5,
        match_strategy: str = 'distance',  # 'rotated_iou' or 'distance'
        use_class_filter: bool = True,
    ):
        """
        Args:
            match_threshold: 匹配阈值
            match_strategy: 匹配策略
            use_class_filter: 是否使用类别过滤
        """
        self.match_threshold = match_threshold
        self.match_strategy = match_strategy
        self.use_class_filter = use_class_filter

    def compute_center_distance(
        self,
        boxes1: torch.Tensor,
        boxes2: torch.Tensor
    ) -> torch.Tensor:
        """
        计算中心距离矩阵

        Args:
            boxes1: (N, 5) [x, y, w, h, θ]
            boxes2: (M, 5) [x, y, w, h, θ]

        Returns:
            距离矩阵 (N, M)
        """
        centers1 = boxes1[:, :2]  # (N, 2)
        centers2 = boxes2[:, :2]  # (M, 2)

        # 欧氏距离
        dist = torch.cdist(centers1, centers2, p=2)  # (N, M)
        return dist

    def match_predictions(
        self,
        clear_preds: Dict[str, torch.Tensor],
        degraded_preds: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        一对一匹配清晰和退化分支的预测

        GPT要求：
        - 一对一匹配（每个预测最多匹配一次）
        - 考虑类别兼容性
        - 使用明确命名的距离度量

        Args:
            clear_preds: 清晰分支预测
            degraded_preds: 退化分支预测

        Returns:
            matched_clear_indices: (K,) 匹配的清晰预测索引
            matched_degraded_indices: (K,) 匹配的退化预测索引
        """
        clear_boxes = clear_preds['boxes']  # (N, 5)
        degraded_boxes = degraded_preds['boxes']  # (M, 5)

        if clear_boxes.shape[0] == 0 or degraded_boxes.shape[0] == 0:
            device = clear_boxes.device
            return torch.empty(0, dtype=torch.long, device=device), \
                   torch.empty(0, dtype=torch.long, device=device)

        # 计算距离/相似度矩阵
        if self.match_strategy == 'distance':
            # 使用中心距离
            dist_matrix = self.compute_center_distance(clear_boxes, degraded_boxes)  # (N, M)
            # 距离越小越好，转换为相似度
            similarity_matrix = 1.0 / (1.0 + dist_matrix)
        elif self.match_strategy == 'rotated_iou':
            # TODO: 使用仓库的旋转IoU实现
            raise NotImplementedError("旋转IoU匹配待实现")
        else:
            raise ValueError(f"Unknown match strategy: {self.match_strategy}")

        # 类别兼容性过滤
        if self.use_class_filter:
            clear_class_ids = clear_preds['class_ids']  # (N,)
            degraded_class_ids = degraded_preds['class_ids']  # (M,)

            # 类别必须相同
            class_compatible = clear_class_ids.unsqueeze(1) == degraded_class_ids.unsqueeze(0)  # (N, M)
            similarity_matrix = similarity_matrix * class_compatible.float()

        # 贪心一对一匹配（互为最近邻）
        matched_clear = []
        matched_degraded = []
        used_clear = set()
        used_degraded = set()

        # 按相似度排序
        flat_sim = similarity_matrix.flatten()
        sorted_indices = torch.argsort(flat_sim, descending=True)

        N, M = similarity_matrix.shape
        for flat_idx in sorted_indices:
            i = flat_idx // M  # clear索引
            j = flat_idx % M   # degraded索引

            sim = similarity_matrix[i, j].item()

            # 检查是否满足阈值且未使用
            if sim >= self.match_threshold and i.item() not in used_clear and j.item() not in used_degraded:
                matched_clear.append(i.item())
                matched_degraded.append(j.item())
                used_clear.add(i.item())
                used_degraded.add(j.item())

        if len(matched_clear) == 0:
            device = clear_boxes.device
            return torch.empty(0, dtype=torch.long, device=device), \
                   torch.empty(0, dtype=torch.long, device=device)

        return torch.tensor(matched_clear, device=clear_boxes.device), \
               torch.tensor(matched_degraded, device=degraded_boxes.device)


def test_sparse_extraction():
    """测试稀疏预测提取（使用解码器）"""
    print("Testing Sparse Prediction Extraction with Decoder...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # 测试1: batch_size=1
    print("\n=== Test 1: Batch size 1 ===")
    B, N = 1, 100
    nc = 16  # 16类 DOTA v1.5
    no = 5 + nc + 180

    decoded_preds = torch.rand(B, N, no, device=device)
    decoded_preds[..., :4] = decoded_preds[..., :4] * 1024
    decoded_preds[..., 4] = torch.sigmoid(decoded_preds[..., 4])
    decoded_preds[..., 5:5+nc] = torch.sigmoid(decoded_preds[..., 5:5+nc])

    extractor = SparsePredictionExtractor(conf_threshold=0.25, top_k=50, num_classes=nc)
    sparse_preds_list = extractor.extract_from_yolo_output(
        (decoded_preds, None),
        model_training=False,
        batch_size=1
    )

    print(f"✓ Extracted {len(sparse_preds_list)} batch predictions")
    print(f"  Batch 0: {sparse_preds_list[0]['boxes'].shape[0]} boxes")

    # 测试2: batch_size=2
    print("\n=== Test 2: Batch size 2 ===")
    B = 2
    decoded_preds_2 = torch.rand(B, N, no, device=device)
    decoded_preds_2[..., :4] = decoded_preds_2[..., :4] * 1024
    decoded_preds_2[..., 4] = torch.sigmoid(decoded_preds_2[..., 4])
    decoded_preds_2[..., 5:5+nc] = torch.sigmoid(decoded_preds_2[..., 5:5+nc])

    sparse_preds_list_2 = extractor.extract_from_yolo_output(
        (decoded_preds_2, None),
        model_training=False,
        batch_size=2
    )

    print(f"✓ Extracted {len(sparse_preds_list_2)} batch predictions")
    print(f"  Batch 0: {sparse_preds_list_2[0]['boxes'].shape[0]} boxes")
    print(f"  Batch 1: {sparse_preds_list_2[1]['boxes'].shape[0]} boxes")

    # 测试3: 一对一匹配（确定性测试）
    print("\n=== Test 3: One-to-One Matching (Deterministic) ===")

    # 创建确定性输入
    clear_boxes = torch.tensor([
        [100.0, 100.0, 50.0, 30.0, 0.5],  # box 0
        [200.0, 200.0, 60.0, 40.0, 0.3],  # box 1
        [300.0, 300.0, 70.0, 50.0, 0.1],  # box 2
    ], device=device)

    clear_class_ids = torch.tensor([0, 1, 2], device=device)
    clear_scores = torch.tensor([0.9, 0.8, 0.7], device=device)
    clear_classes = torch.zeros((3, nc), device=device)
    clear_classes[0, 0] = 1.0
    clear_classes[1, 1] = 1.0
    clear_classes[2, 2] = 1.0

    clear_preds_det = {
        'boxes': clear_boxes,
        'scores': clear_scores,
        'classes': clear_classes,
        'class_ids': clear_class_ids,
        'valid_mask': torch.ones(3, dtype=torch.bool, device=device),
    }

    # 创建退化预测：轻微偏移
    degraded_boxes = torch.tensor([
        [102.0, 101.0, 51.0, 31.0, 0.52],  # 接近box 0, 距离约2.2
        [202.0, 201.0, 62.0, 41.0, 0.28],  # 接近box 1, 距离约2.8
        [500.0, 500.0, 80.0, 60.0, 0.15],  # 远离所有（不应匹配）
    ], device=device)

    degraded_class_ids = torch.tensor([0, 1, 0], device=device)  # 最后一个类别不同

    degraded_preds_det = {
        'boxes': degraded_boxes,
        'scores': clear_scores,
        'classes': clear_classes,
        'class_ids': degraded_class_ids,
        'valid_mask': torch.ones(3, dtype=torch.bool, device=device),
    }

    # 使用较宽松的阈值进行匹配
    # 距离5像素 -> similarity = 1/(1+5) ≈ 0.167，所以阈值设为0.15
    matcher = PredictionMatcher(match_threshold=0.15, use_class_filter=True)
    clear_idx, degraded_idx = matcher.match_predictions(clear_preds_det, degraded_preds_det)

    print(f"✓ Matched {len(clear_idx)} predictions")
    print(f"  Clear indices: {clear_idx.tolist()}")
    print(f"  Degraded indices: {degraded_idx.tolist()}")

    # 验证一对一
    assert len(clear_idx) > 0, "Should find at least one match with deterministic input"
    assert len(clear_idx) == len(set(clear_idx.tolist())), "Clear indices must be unique"
    assert len(degraded_idx) == len(set(degraded_idx.tolist())), "Degraded indices must be unique"

    # 验证预期匹配：box0->box0, box1->box1（类别相同且距离近）
    # box2不应匹配（距离远且类别不同）
    expected_matches = 2
    assert len(clear_idx) == expected_matches, f"Expected {expected_matches} matches, got {len(clear_idx)}"

    # 验证匹配的具体索引
    assert 0 in clear_idx.tolist() and 0 in degraded_idx.tolist(), "Box 0 should match"
    assert 1 in clear_idx.tolist() and 1 in degraded_idx.tolist(), "Box 1 should match"

    print(f"✓ One-to-one matching verified")
    print(f"✓ Found expected number of matches: {expected_matches}")

    # 测试4: 类别过滤
    print("\n=== Test 4: Class Filtering ===")
    matcher_no_class = PredictionMatcher(match_threshold=0.3, use_class_filter=False)
    clear_idx_no_filter, degraded_idx_no_filter = matcher_no_class.match_predictions(
        clear_preds_det, degraded_preds_det
    )
    print(f"✓ Without class filter: {len(clear_idx_no_filter)} matches")
    print(f"✓ With class filter: {len(clear_idx)} matches")

    # 测试5: 空输入
    print("\n=== Test 5: Empty Input ===")
    empty_preds = extractor._empty_predictions(device)
    clear_idx_empty, degraded_idx_empty = matcher.match_predictions(empty_preds, clear_preds_det)
    assert len(clear_idx_empty) == 0, "Empty input should produce no matches"
    print(f"✓ Empty input handled correctly")

    print("\n" + "="*50)
    print("All tests passed! ✓")
    print("="*50)


if __name__ == '__main__':
    test_sparse_extraction()
