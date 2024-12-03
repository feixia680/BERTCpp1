# 基于 Ubuntu 基础镜像
FROM ubuntu:20.04

# 更新软件包列表并安装 g++
RUN apt-get update && apt-get install -y g++

# 设置默认工作目录
WORKDIR /workspace