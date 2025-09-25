
使用piper和d435执行抓取的存储库，此环境已经在cuda11.8+ubuntu20.04+python3.8进行了测试
# 安装

- 克隆存储库
```
git clone https://github.com/yamanoko-do/Langrasp.git --recurse-submodules
```
- 创建conda环境
```
cd Langrasp
conda create -n Langrasp python=3.8
conda activate Langrasp
```
## 安装子模块
按照子模块README.md说明安装：graspnet && piper_sdk && moge
```bash
cd langrasp/thirdpart/graspnet/
cd langrasp/thirdpart/piper_sdk/
cd langrasp/thirdpart/moge/
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
1. 首先执行标定，手眼标定&&工具坐标系标定代码见[d435_test](https://github.com/yamanoko-do/d435_test.git)，将结果写入langrasp/config.py
2. 查找can：bash ./scripts/find_all_can_port.sh
3. 激活can：bash ./scripts/can_activate.sh can_piper 1000000 "3-1.1:1.0"
4. 执行抓取：python main.py