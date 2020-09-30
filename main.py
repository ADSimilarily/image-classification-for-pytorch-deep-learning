import argparse
from icdl.config import _cfg
from icdl.engine.build import build_runner

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', '-cfg', type=str, default="config/cls_config.yaml", help='指定配置文件，必选项')
    parser.add_argument('--opt', '-opt', type=str, default='train', help='指定操作(one of ==> train, infer, jit, onnx)')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    assert args.cfg, '必须指定cfg配置选项'
    print(args)

    _cfg.merge_from_file(args.cfg)
    _cfg.MODEL.OPT = args.opt
    _cfg.MODEL.STRICT = args.opt != "train"
    _cfg.freeze()

    build_runner(_cfg, args.opt)

