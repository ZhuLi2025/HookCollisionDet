#!/bin/bash
# 获取当前时间（示例格式：2025-12-05 10:23:45）
TIME_STR=$(date +"%Y-%m-%d %H:%M:%S")

git add .
git commit -m "$TIME_STR"
git pull
git push origin main