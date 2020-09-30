import torch
import torch.jit
import torch.onnx
from icdl.models import build_models
from .registry import ENFINE_REGISTRY
from icdl.models.layers.onnx import ONNX


def convert(cfg, jit):
    assert cfg.MODEL.WEIGHT

    if not jit:
        ONNX.onnx_export(True)

    model = build_models(cfg)
    model.load_state_dict(torch.load(cfg.MODEL.WEIGHTS, map_location='cpu'), strict=True)
    model.eval()

    example = torch.rand(1, cfg.TRAIN.INPUT_CHANNEL, cfg.TRAIN.INPUT_HEIGHT, cfg.TRAIN.INPUT_WIDTH)

    if jit:
        traced_script_module = torch.jit.trace(model, example)

        jit_model_name = cfg.MODEL.WEIGHT.replace(".pth", ".pt")
        traced_script_module.save(jit_model_name)
        print('转换完成，文件路径：{}'.format(jit_model_name))
    else:
        onnx_model_name = cfg.MODEL.WEIGHT.replace(".pth", ".onnx")
        torch.onnx.export(model,
                              example,
                              onnx_model_name,
                              # opset_version=10,
                              input_names=['input'],  # the model's input names
                              output_names=['output'],
                              verbose=True)
        print('转换完成，文件路径：{}'.format(onnx_model_name))


@ENFINE_REGISTRY.register()
def jit(cfg):
    convert(cfg, True)


@ENFINE_REGISTRY.register()
def trt(cfg):
    convert(cfg, False)



