# Python 学习笔记

## 虚拟环境

每个项目都应该用独立的虚拟环境，避免依赖互相污染。创建方式：`python -m venv .venv`，Windows 上激活用 `.venv\Scripts\activate`。

## 列表推导式

列表推导式比 for 循环更简洁：`squares = [x**2 for x in range(10)]`。带条件的写法：`evens = [x for x in range(20) if x % 2 == 0]`。

## 异常处理

用 try/except 捕获具体的异常类型，不要裸 except。finally 块无论是否出错都会执行，适合做资源清理。更推荐用 with 语句管理文件等资源。

## argparse

命令行工具用 argparse 标准库。add_subparsers 可以实现 git 那样的子命令结构，每个子命令可以有自己的参数。
