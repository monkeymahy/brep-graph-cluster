# 安装指南

## 方法一：使用批处理脚本（Windows推荐）

双击运行 `setup_env.bat`，或者在命令行中执行：

```cmd
cd E:\mhy\brep-graph-match
setup_env.bat
```

## 方法二：手动分步安装

### 1. 创建并激活conda环境

```cmd
D:\mhy\miniforge\Scripts\conda.exe create -n brep-graph python=3.10 -y
D:\mhy\miniforge\Scripts\conda.exe activate brep-graph
```

### 2. 安装OpenCASCADE相关依赖

```cmd
conda install -c conda-forge pythonocc-core=7.5.1 occt=7.5.1 -y
```

### 3. 安装其他conda包

```cmd
conda install numpy scipy h5py tqdm networkx -y
```

### 4. 安装PyTorch和DGL

根据你的系统选择合适的PyTorch版本：

```cmd
# CPU版本或根据你的CUDA版本选择
pip install torch dgl scikit-learn
```

### 5. 安装occwl

```cmd
pip install git+https://github.com/AutodeskAILab/occwl.git
```

如果安装失败，可以尝试：

```cmd
git clone https://github.com/AutodeskAILab/occwl.git
cd occwl
pip install -e .
```

注意：不要在这个环境里额外安装 `numba`，它会和 `pythonocc-core=7.5.1` 依赖的 `tbb` 版本打架，导致 conda 解不出来。

## 验证安装

激活环境后运行：

```python
python -c "
import numpy
import dgl
import torch
from OCC.Core.STEPControl import STEPControl_Reader
from occwl.solid import Solid
print('All imports successful!')
"
```

或者运行示例脚本：

```cmd
python example.py
```

## 常见问题

### pythonocc-core安装失败

如果conda-forge安装失败，可以尝试：
- 降低版本要求
- 使用不同的conda channel
- 检查网络连接

### occwl安装失败

可以直接将occwl的代码放到项目目录中使用，不需要pip安装。

### pip报错 `InvalidVersion: Invalid version: '-PKG-VERSION'`

这是环境里某个已安装包的元数据坏了，日志里通常会指向 `vtk`。`pip` 在安装 `occwl` 时会先扫描当前环境中已安装的包，结果被这个坏掉的版本号卡住。

处理方式：

```cmd
conda remove vtk -y
conda install -c conda-forge vtk -y
```

如果还报同样错误，检查并删除这个坏掉的元数据文件：

```cmd
del D:\mhy\miniforge\envs\brep-graph\Lib\site-packages\vtk-9.0.1.egg-info
```

这个文件里的 `Version` 现在是 `$PKG_VERSION`，不是合法版本号。删掉后再重装 `vtk`，然后重新执行 `pip install`。

如果还不行，最稳妥的办法是新建一个干净环境重新装，不要复用带有旧 `vtk` 的环境。

### Windows路径问题

确保使用正确的conda路径：`D:\mhy\miniforge\Scripts\conda.exe`
