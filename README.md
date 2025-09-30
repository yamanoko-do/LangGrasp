
使用piper和d435执行抓取的存储库，此环境已经在以下环境进行了测试
1. cuda11.8+ubuntu20.04+python3.8
2. cuda12.1+ubuntu22.04+python3.10
# 安装

- 克隆存储库：
```
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/yamanoko-do/LangGrasp.git --recurse-submodules
```
- 下载权重：
```
git lfs pull
```
- 创建conda环境：
```
cd LangGrasp
conda create -n LangGrasp python=3.10
conda activate LangGrasp
```
## 安装子模块
按照子模块README.md说明安装：[graspnet](https://github.com/yamanoko-do/graspnet) && [piper_sdk](https://github.com/agilexrobotics/piper_sdk) && [moge](https://github.com/microsoft/MoGe)
```bash
cd langgrasp/thirdpart/graspnet/
cd langgrasp/thirdpart/piper_sdk/
cd langgrasp/thirdpart/moge/
```
### 修改moge

- 修改/moge/utils/tools.py中line235的decorator为：
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
### 为piper_sdk安装系统包
```bash
apt install ethtool
apt install can-utils
apt install iproute2
```

# 使用
- 首先执行标定，手眼标定&&工具坐标系标定代码见[d435_test](https://github.com/yamanoko-do/d435_test.git)，将结果写入langrasp/config.py
- 连接piper:

```bash
bash ./scripts/find_all_can_port.sh
bash ./scripts/can_activate.sh can_piper 1000000 "3-1.1:1.0"
```
- 设置qwen2.5vl的APIKEY:
```bash
export DASHSCOPE_API_KEY='YOUR_DASHSCOPE_API_KEY'
```
- 执行抓取:
```bash
python main.py
```
