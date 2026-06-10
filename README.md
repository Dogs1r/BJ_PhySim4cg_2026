# BJ_PhySim4Sci_2026

master branch contain the final work of all labs, while other branches help to optimize different labs.
run before a new lab 
```
git checkout master          # 确保在主分支
git checkout -b lab0         # 创建并切换到 lab0 分支
```
and after
```
git add .
git commit -m "Finish part of Lab0"
git push origin lab0         # 推送到 GitHub 的 lab0 分支
```

```
# 1. 查看当前分支和状态
git branch --show-current
git status --short

# 2. 只添加本次要提交的文件
git add src/VCX/Labs/Final/README.md \
        src/VCX/Labs/Final/changes.md \
        src/VCX/Labs/Final/pipeline.md \
        src/VCX/Labs/Final/simulate.py \
        src/VCX/Labs/Final/gs_particle_pipeline/*.py

# 3. 确认暂存内容
git status --short
git diff --cached --stat

# 4. 提交
git commit -m "Add CUDA GS cloth rendering pipeline"

# 5. 推送当前分支
git push origin final
```
