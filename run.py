#!/usr/bin/env python3
"""ALPR 项目主入口 — PyCharm 直接 Run/Debug"""
import sys, os

def main():
    if len(sys.argv) > 1 and sys.argv[1] == 'cli':
        # CLI 三阶段流水线模式
        # 文件名带连字符, 不能常规 import, 用 importlib 加载
        import importlib.util

        def load_module(filename):
            path = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
            spec = importlib.util.spec_from_file_location(
                filename.replace('-', '_').replace('.py', ''), path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod

        vd = load_module('vehicle-detection.py').main
        lpd = load_module('license-plate-detection.py').main
        ocr = load_module('license-plate-ocr.py').main
        print('=' * 60)
        print('  ALPR CLI 流水线 (PyCharm)')
        print('=' * 60)
        if len(sys.argv) < 4:
            print('用法: python run.py cli <input_dir> <output_dir>')
            sys.exit(1)
        input_dir = sys.argv[2]
        output_dir = sys.argv[3]
        sys.argv = ['vehicle-detection.py', input_dir, output_dir]
        vd()
        sys.argv = ['license-plate-detection.py', output_dir, 'data/lp-detector/wpod-net_update1.h5']
        lpd()
        sys.argv = ['license-plate-ocr.py', output_dir]
        ocr()
    else:
        # 默认模式：Web 界面 (PyCharm 直接 Run 这一种)
        from webapp import app, get_models
        print('=' * 60)
        print('  ALPR Web UI - PyCharm 启动')
        print('  浏览器访问: http://127.0.0.1:5000')
        print('=' * 60)
        print('[1/2] 预加载模型 (首次约 1-2 分钟)...')
        get_models()
        print('[2/2] 启动 Flask...')
        # PyCharm 自动启用 debug, 可以从 Run 窗口中断点
        app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)


if __name__ == '__main__':
    main()
