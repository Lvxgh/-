# Git 常用命令

## 暂存与提交

git stash 可以临时收起未提交的改动，git stash pop 恢复。git add -p 可以只暂存文件的一部分改动，逐块确认。

## 分支

新建分支用 git switch -c <名字>。合并用 git merge；想要线性历史用 git rebase，但公共分支上别用 rebase。

## 回滚

git restore <文件> 丢弃工作区改动。git reset --soft HEAD~1 撤销最近一次提交但保留改动在暂存区。已经推送的提交要用 git revert 生成一个反向提交，不要改历史。

## 远程

git remote -v 查看远程仓库地址。git pull = fetch + merge 两步合一。git push -u origin main 首次推送并建立分支跟踪。
