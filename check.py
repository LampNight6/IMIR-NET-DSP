import sys
import importlib

def check_package(package_name, import_name=None):
    """
    检查包是否安装，并打印版本号
    :param package_name: pip安装时的名称 (例如 opencv-python)
    :param import_name: 代码中import的名称 (例如 cv2)，如果不填则默认与package_name相同
    """
    if import_name is None:
        import_name = package_name
    
    print(f"正在检查 {package_name} ... ", end="")
    
    try:
        # 尝试导入模块
        lib = importlib.import_module(import_name)
        
        # 尝试获取版本号
        version = getattr(lib, '__version__', '未知版本')
        print(f"✅ 已安装 (版本: {version})")
        return True
    
    except ImportError:
        print(f"❌ 未安装")
        return False
    except OSError as e:
        # 捕捉类似 WinError 126 的 DLL 加载错误
        print(f"⚠️ 安装了但在加载时崩溃 (可能是DLL缺失或版本不匹配)")
        print(f"    -> 错误详情: {e}")
        return False
    except Exception as e:
        print(f"⚠️ 发生未知错误: {e}")
        return False

def check_torch_environment():
    print("\n" + "="*30)
    print(">>> 深度学习核心环境检查 (PyTorch & CUDA)")
    print("="*30)
    
    # 1. 检查 PyTorch 是否能导入
    try:
        import torch
        print(f"PyTorch 版本: {torch.__version__}")
    except ImportError:
        print("❌ PyTorch 未安装！请先安装 PyTorch。")
        return
    except OSError as e:
        print("❌ PyTorch 安装损坏！(这就是你之前遇到的 WinError 126)")
        print(f"   错误信息: {e}")
        print("\n   [解决方案]:")
        print("   1. 访问 https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist")
        print("   2. 下载并安装 'vc_redist.x64.exe'")
        print("   3. 重启电脑后再试")
        print("   4. 如果还不行，请卸载 PyTorch 并重新安装 CPU 版本进行调试")
        return

    # 2. 检查 CUDA (显卡支持)
    print("检查 CUDA 支持... ", end="")
    try:
        if torch.cuda.is_available():
            print("✅ 可用")
            print(f"   - 当前显卡: {torch.cuda.get_device_name(0)}")
            print(f"   - CUDA 版本: {torch.version.cuda}")
        else:
            print("⚠️ 不可用 (当前使用的是 CPU 版本)")
            print("   注意：代码可以运行，但训练速度会非常慢。")
    except Exception as e:
        print(f"检查出错: {e}")

def main():
    print(f"Python 解释器路径: {sys.executable}")
    print(f"Python 版本: {sys.version.split()[0]}")
    print("\n" + "="*30)
    print(">>> 依赖库检查清单")
    print("="*30)

    # 这里的列表是根据你的 mydataset.py 分析出来的
    required_packages = [
        ("numpy", "numpy"),
        ("scipy", "scipy"),
        ("Pillow", "PIL"),        # 代码中是 import PIL
        ("pandas", "pandas"),
        ("imageio", "imageio"),
        ("opencv-python", "cv2")  # 代码中是 import cv2
    ]

    missing_packages = []
    
    for pkg_name, import_name in required_packages:
        if not check_package(pkg_name, import_name):
            missing_packages.append(pkg_name)

    # 检查 PyTorch
    check_torch_environment()

    # 总结
    print("\n" + "="*30)
    print(">>> 检查结果汇总")
    print("="*30)
    
    if not missing_packages:
        print("🎉 基础依赖库看起来都齐全了！(请重点关注上面的 PyTorch 检查结果)")
    else:
        print("🚨 发现缺少以下库，请运行下方命令进行安装：")
        print(f"\npip install {' '.join(missing_packages)}")
        print("\n(建议加上镜像源加速: -i https://pypi.tuna.tsinghua.edu.cn/simple)")

if __name__ == '__main__':
    main()