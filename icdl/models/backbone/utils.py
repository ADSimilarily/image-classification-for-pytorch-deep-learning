try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url


def load_weight_without_strict(model, pretrained_dict):
    model_dict = model.state_dict()
    dis_match = [k for k, v in pretrained_dict.items() if not (k in model_dict and v.shape == model_dict[k].shape)]
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    if dis_match:
        print("注意，以下预训练参数加载失败 : {}".format(dis_match))
    else:
        print("预训练参数加载成功！")
    return model