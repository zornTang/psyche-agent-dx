# RAG Todo

## 已完成

1. 建立了可运行的诊断 MVP pipeline，包含 intake、risk、retrieval、diagnostic、coordinator 流程。
2. 增加了 FastAPI 接口与启动入口，可通过 `/health` 和 `/diagnose` 调用。
3. 补了基础 smoke tests，覆盖 pipeline、API 和 ChatGLM fallback。
4. 从 DSM-5 中文 PDF 中抽取并整理了一批高价值中文知识条目，而不是直接塞整本 OCR 原文。
5. 扩展了中文检索分词，支持中文 token 和简单 CJK n-gram。
6. 将默认知识库从硬编码 Python 列表迁移到持久化数据文件：
   [`data/knowledge/default_corpus.jsonl`](/home/chein/Documents/code/psyche-agent-dx/data/knowledge/default_corpus.jsonl)
7. 将检索从简单词重叠升级为本地 BM25：
   [`src/psyche_agent_dx/knowledge.py`](/home/chein/Documents/code/psyche-agent-dx/src/psyche_agent_dx/knowledge.py)
8. 增加了扫描 PDF -> JSONL chunk 的 OCR 导入脚本：
   [`scripts/ocr_pdf_to_jsonl.py`](/home/chein/Documents/code/psyche-agent-dx/scripts/ocr_pdf_to_jsonl.py)
9. 当前全量测试已通过：`14 tests, OK`

## 当前知识库覆盖

1. 重性抑郁障碍
2. 惊恐障碍
3. 广泛性焦虑障碍
4. 创伤后应激障碍
5. 适应障碍
6. 强迫症
7. 双相 I 型躁狂发作
8. 社交焦虑障碍
9. 广场恐怖症
10. 精神分裂症
11. 躯体症状障碍
12. 疾病焦虑障碍
13. 少量英文 DSM/CBT/安全条目

## 还没做

1. 没有把整本 DSM-5 全量 OCR 并切块导入知识库。
2. 没有 embedding 检索，当前只有 BM25。
3. 没有向量数据库或持久化向量索引。
4. 没有 reranker。
5. 没有 chunk 级页码引用进入 API 响应模型。
6. 没有知识库构建命令或批处理流水线把多个 PDF 自动入库。
7. 没有检索评测集，无法系统比较召回率和误召回。
8. 没有把 `RetrievalAgent` 做成可配置的多检索器架构。

## 建议下一步

1. 用 `scripts/ocr_pdf_to_jsonl.py` 把 DSM-5 剩余章节分批转成 JSONL。
2. 将 `default_corpus.jsonl` 拆成多个语料文件，例如 `dsm5_zh.jsonl`、`cbt.jsonl`、`safety.jsonl`。
3. 给每个 chunk 增加更稳定的元数据：
   `page_start`、`page_end`、`chapter`、`section`、`language`
4. 在 `knowledge.py` 上增加 dense retrieval 接口，形成 BM25 + embedding hybrid search。
5. 引入 reranker，对 top-k chunk 重新排序。
6. 修改 schema 和 coordinator，把证据页码和章节标题一起返回。
7. 增加一组 retrieval regression tests，固定查询与预期命中条目。
8. 如果要吃整本 DSM-5，优先先做“导入流水线 + 评测”，再做 UI。

## 后续建议顺序

1. 先推送当前提交 `b3335a5`
2. 批量导入 DSM-5 剩余章节
3. 做 retrieval regression tests
4. 加 embedding 检索
5. 做 hybrid retrieval
6. 加 reranker
7. 把页码和章节引用返回到 API

## 常用入口文件

1. 知识库实现：
   [`src/psyche_agent_dx/knowledge.py`](/home/chein/Documents/code/psyche-agent-dx/src/psyche_agent_dx/knowledge.py)
2. 默认知识语料：
   [`data/knowledge/default_corpus.jsonl`](/home/chein/Documents/code/psyche-agent-dx/data/knowledge/default_corpus.jsonl)
3. OCR 导入脚本：
   [`scripts/ocr_pdf_to_jsonl.py`](/home/chein/Documents/code/psyche-agent-dx/scripts/ocr_pdf_to_jsonl.py)
4. Pipeline 装配：
   [`src/psyche_agent_dx/pipeline.py`](/home/chein/Documents/code/psyche-agent-dx/src/psyche_agent_dx/pipeline.py)
5. 检索测试：
   [`tests/test_knowledge.py`](/home/chein/Documents/code/psyche-agent-dx/tests/test_knowledge.py)
