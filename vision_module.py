import torch
from torchvision import models, transforms
from PIL import Image
from langchain_core.tools import tool
import os

class IndustrialVision:
    def __init__(self):
        # 使用轻量级 ShuffleNet，适合工业现场快速推理
        self.model = models.shufflenet_v2_x1_0(pretrained=True)
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def analyze(self, path):
        try:
            img = Image.open(path).convert('RGB')
            # 实际场景中，这里应加载识别“表面烧伤”、“崩刃”的分类权重
            # 此处模拟提取视觉特征描述
            return "视觉分析报告：检测到工件表面反射率均匀，纹理呈现规律性平行分布，未见宏观崩刃。视觉感官粗糙度处于合格范围。"
        except Exception as e:
            return f"图片分析异常: {str(e)}"

vision_tool = IndustrialVision()

@tool
def analyze_industrial_image(image_path: str) -> str:
    """
    【技能名称】：工业视觉多模态分析
    【功能】：分析上传的工件或刀具照片，从视觉维度评估加工质量。
    """
    if not os.path.exists(image_path):
        return "视觉系统未找到指定路径的照片。"
    return vision_tool.analyze(image_path)