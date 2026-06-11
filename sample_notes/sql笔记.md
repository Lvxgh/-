# SQL 笔记

## 基础查询

SELECT 列 FROM 表 WHERE 条件。模糊匹配用 LIKE '%关键词%'。去重用 DISTINCT，排序用 ORDER BY ... DESC。

## 连接

INNER JOIN 只保留两表都匹配的行；LEFT JOIN 保留左表全部行，右表没匹配上的填 NULL。

## 聚合

GROUP BY 分组后用 COUNT/SUM/AVG 聚合。过滤聚合结果要用 HAVING 不能用 WHERE——WHERE 在分组前执行。

## 索引

索引加速查询但拖慢写入。最左前缀原则：联合索引 (a,b,c) 只对以 a 开头的查询条件生效。

## 事务

事务的 ACID：原子性、一致性、隔离性、持久性。BEGIN 开启，COMMIT 提交，ROLLBACK 回滚。
