python demo.py --checkpoint_path data/weights/graspnet.tar

# 安装
## 安装 graspnet
## 安装 piper_sdk
## 安装 moge
按照说明安装[moge](https://github.com/microsoft/moge)
修改/moge/utils/tools.py中line235的decorator为：
```python
    def decorator(fn: Callable):
        # 在 Python 3.8 中，需要嵌套使用嵌套的 with 语句而非括号组合
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            with pbar:
                pbar.refresh()  
                @catch_exception
                @suppress_traceback
                def _fn(input):
                    ret = fn(input)
                    pbar.update()
                    return ret
                    
                executor.map(_fn, inputs)
                executor.shutdown(wait=True)
    
    return decorator
```

在moge/model/v2.py中添加：
```python
from typing import Union, Optional, Dict, Any, IO
```
# pipeline

给定场景图像和用户输入

生成场景抓取位姿

生成mask

过滤抓取姿态

对抓取姿态排序并返回最优抓取

一个抓取姿态我需要将其变换到机械臂坐标下

执行


# Piper 常用命令
apt install ethtool
apt install can-utils
apt install iproute2

can_piper
- 查找can：bash ./scripts/find_all_can_port.sh
- 激活can：bash ./scripts/can_activate.sh can_piper 1000000 "3-1.1:1.0"

