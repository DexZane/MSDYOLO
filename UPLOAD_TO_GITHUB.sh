#!/bin/bash
# MSDYOLO GitHub上传脚本
# 使用方法：bash UPLOAD_TO_GITHUB.sh YOUR_GITHUB_USERNAME

set -e

if [ -z "$1" ]; then
    echo "错误：请提供GitHub用户名"
    echo "使用方法：bash UPLOAD_TO_GITHUB.sh YOUR_GITHUB_USERNAME"
    exit 1
fi

USERNAME=$1
REPO_NAME="MSDYOLO"

echo "=========================================="
echo "MSDYOLO GitHub 上传脚本"
echo "=========================================="
echo ""
echo "目标仓库: https://github.com/$USERNAME/$REPO_NAME"
echo ""

# 检查是否在正确的目录
if [ ! -f "trainmsd.py" ]; then
    echo "错误：请在MSDYOLO项目根目录运行此脚本"
    exit 1
fi

# 检查git状态
echo "[1/5] 检查Git状态..."
git status --short

# 移除旧的remote
echo ""
echo "[2/5] 移除旧的remote..."
if git remote | grep -q "origin"; then
    git remote remove origin
    echo "✓ 已移除旧的origin"
else
    echo "✓ 没有旧的origin"
fi

# 添加新的remote
echo ""
echo "[3/5] 添加新的GitHub仓库..."
git remote add origin "https://github.com/$USERNAME/$REPO_NAME.git"
echo "✓ 已添加: https://github.com/$USERNAME/$REPO_NAME.git"

# 验证remote
echo ""
echo "[4/5] 验证remote配置..."
git remote -v

# 推送代码
echo ""
echo "[5/5] 推送代码到GitHub..."
echo "注意：请确保已在GitHub创建 $REPO_NAME 仓库"
echo "创建地址：https://github.com/new"
echo ""
read -p "已创建GitHub仓库？按回车继续，Ctrl+C取消..."

git push -u origin master

echo ""
echo "=========================================="
echo "上传完成！"
echo "=========================================="
echo ""
echo "仓库地址：https://github.com/$USERNAME/$REPO_NAME"
echo ""
echo "下一步："
echo "1. 访问仓库页面"
echo "2. 添加Description和Topics"
echo "3. 检查README是否正确显示"
echo "4. 配置仓库设置（可选）"
echo ""
