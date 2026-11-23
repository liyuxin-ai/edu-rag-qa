# archive

这里存放**不参与运行**的内容，保留仅供追溯：

- `old_main.py`、`rag_qa/rag_main.py`、`rag_qa/core/rag_system.py` —— RAG 系统的早期版本，已被 `new_main.py` / `rag_qa/core/new_rag_system.py` 取代（旧版 `generate_answer` 返回字符串而非生成器，不支持流式）。
- `rag_qa/demo_parse_args.py`、`rag_qa/**/review.py`、`mysql_qa/retrieval/bm25_search_demo.py`、`demo/`、`notebooks/` —— 学习过程中的练习脚本与探索性 Notebook。
- `static/old_index.html` —— 前端旧版界面。

这些文件不在任何 import 链路上，删除或忽略都不影响服务启动。
