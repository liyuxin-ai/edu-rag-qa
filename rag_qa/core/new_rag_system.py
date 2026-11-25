from .prompts import RAGPrompts
#   导入 time 模块，用于计算时间
import time
from base import logger, Config
from .query_classifier import QueryClassifier  # 导入查询分类器
from .strategy_selector import StrategySelector  # 导入策略选择器

conf = Config()


#   定义 RAGSystem 类，封装 RAG 系统的核心逻辑
class RAGSystem:
    #   初始化方法，设置 RAG 系统的基本参数
    def __init__(self, vector_store, llm):
        #   设置向量数据库对象
        self.vector_store = vector_store
        #   设置大语言模型调用函数
        self.llm = llm
        #   获取 RAG 提示模板
        self.rag_prompt = RAGPrompts.rag_prompt()
        #   初始化查询分类器
        self.query_classifier = QueryClassifier(conf.bert_intent_cls)
        #   初始化策略选择器
        self.strategy_selector = StrategySelector()
        # 最大 prompt 长度（字符）；固定指令 + 问题必须完整保留，剩余额度分配给 context 和 history
        self.max_prompt_length = 4096

    def _build_prompt_within_limit(self, context_docs, history_list, question, phone):
        """按块粒度裁剪上下文和历史，保证 system 指令 + 问题完整，不做硬截断。

        优先级：system 指令 > 问题 > context（按文档块砍尾）> history（砍旧留新）。
        这样极端超长场景下，被切掉的是最不相关的文档块或最旧的历史，而不会把指令或问题切坏。
        """
        # 1. 算出模板的固定成本（context 和 history 都为空时）
        fixed_prompt = self.rag_prompt.format(context="", history="", question=question, phone=phone)
        fixed_cost = len(fixed_prompt)

        if fixed_cost > self.max_prompt_length:
            # 极端情况：固定部分本身就超长（问题极长），直接返回（让 LLM 截断问题）
            logger.warning(
                f"固定 prompt 长度 {fixed_cost} 已超过上限 {self.max_prompt_length}，"
                f"问题本身可能过长"
            )
            return fixed_prompt

        remaining = self.max_prompt_length - fixed_cost

        # 2. 先装 context（RAG 核心，reranker 已按相关性排序，靠前的更重要）
        context_parts = []
        used = 0
        for doc in context_docs:
            chunk = doc.page_content
            # 预留 "\n\n" 分隔符（第一个块不需要前缀）
            separator = "\n\n" if context_parts else ""
            cost = len(chunk) + len(separator)
            if used + cost > remaining:
                logger.info(
                    f"context 额度用尽：已装 {len(context_parts)} 块，"
                    f"剩余 {remaining - used} 字符"
                )
                break
            context_parts.append(separator + chunk)
            used += cost

        context_text = "".join(context_parts)

        # 3. 剩余额度装 history（砍旧留新，最近的轮次更重要）
        history_text = ""
        history_budget = remaining - used
        if history_list and history_budget > 0:
            # 从最后一轮往前装，直到超出额度
            history_lines = []
            history_used = 0
            for h in reversed(history_list):
                line = f"Q: {h['question']}\nA: {h['answer']}\n"
                line_cost = len(line)
                if history_used + line_cost > history_budget:
                    break
                history_lines.append(line)
                history_used += line_cost
            # 恢复正序（从旧到新）
            history_text = "".join(reversed(history_lines))
            if history_lines:
                logger.info(
                    f"history 装入 {len(history_lines)} 轮，"
                    f"占用 {history_used}/{history_budget} 字符"
                )

        # 4. 最终拼 prompt（此时长度必定 <= max_prompt_length）
        return self.rag_prompt.format(
            context=context_text,
            history=history_text,
            question=question,
            phone=phone,
        )

    #   定义私有方法，使用假设文档进行检索（HyDE）
    def _retrieve_with_hyde(self, query):
        logger.info(f"使用 HyDE 策略进行检索 (查询: '{query}')")
        #   获取假设问题生成的 Prompt 模板
        hyde_prompt_template = RAGPrompts.hyde_prompt()  # 使用 template 后缀区分
        #   调用大语言模型生成假设答案
        try:
            hypo_answer = self.llm(hyde_prompt_template.format(query=query))
            hypo_answer = "".join(hypo_answer)
            logger.info(f"HyDE 生成的假设答案: '{hypo_answer}'")
            #   使用假设答案进行检索，并返回检索结果
            #   注意：HyDE 通常只用于生成检索向量，不一定需要 rerank 这一步，但这里复用了
            return self.vector_store.hybrid_search_with_rerank(
                hypo_answer, k=conf.RETRIEVAL_K  # 使用 K 而非 M
            )
        except Exception as e:
            logger.error(f"HyDE 策略执行失败: {e}")
            return []

    #   定义私有方法，使用子查询进行检索
    def _retrieve_with_subqueries(self, query):
        logger.info(f"使用子查询策略进行检索 (查询: '{query}')")
        #   获取子查询生成的 Prompt 模板
        subquery_prompt_template = RAGPrompts.subquery_prompt()  # 使用 template 后缀区分
        try:
            #   调用大语言模型生成子查询列表
            subqueries_text = self.llm(subquery_prompt_template.format(query=query))
            subqueries_text = "".join(subqueries_text)
            subqueries = [q.strip() for q in subqueries_text.split("\n") if q.strip()]
            logger.info(f"生成的子查询: {subqueries}")
            if not subqueries:
                logger.warning("未能生成有效的子查询")
                return []

            #   初始化空列表，用于存储所有子查询的检索结果
            all_docs = []
            #   遍历每个子查询
            for sub_q in subqueries:
                #   使用子查询进行检索，并将结果添加到列表中
                #   这里对每个子查询都执行了 hybrid search + rerank，开销可能较大
                docs = self.vector_store.hybrid_search_with_rerank(
                    sub_q, k=conf.RETRIEVAL_K  # 使用 K
                )
                all_docs.extend(docs)
                logger.info(f"子查询 '{sub_q}' 检索到 {len(docs)} 个文档")

            #   对所有检索结果进行去重 (基于对象内存地址，如果 Document 内容相同但对象不同则无法去重)
            #   更可靠的去重方式是基于文档内容或 ID
            unique_docs_dict = {doc.page_content: doc for doc in all_docs}  # 基于内容去重
            unique_docs = list(unique_docs_dict.values())

            logger.info(f"所有子查询共检索到 {len(all_docs)} 个文档, 去重后剩 {len(unique_docs)} 个")
            #   返回去重后的文档，限制数量 (是否需要在此处限制? retrieve_and_merge 末尾会限制)
            # return unique_docs[: Config.CANDIDATE_M]
            return unique_docs  # 返回所有唯一文档，让 retrieve_and_merge 处理数量

        except Exception as e:
            logger.error(f"子查询策略执行失败: {e}")
            return []

    #   定义私有方法，使用回溯问题进行检索
    def _retrieve_with_backtracking(self, query):
        logger.info(f"使用回溯问题策略进行检索 (查询: '{query}')")
        #   获取回溯问题生成的 Prompt 模板
        backtrack_prompt_template = RAGPrompts.backtracking_prompt()  # 使用 template 后缀区分
        try:
            #   调用大语言模型生成回溯问题
            simplified_query = self.llm(backtrack_prompt_template.format(query=query))
            simplified_query = "".join(simplified_query)
            logger.info(f"生成的回溯问题: '{simplified_query}'")
            #   使用回溯问题进行检索，并返回检索结果
            return self.vector_store.hybrid_search_with_rerank(
                simplified_query, k=conf.RETRIEVAL_K  # 使用 K
            )
        except Exception as e:
            logger.error(f"回溯问题策略执行失败: {e}")
            return []

    #   定义方法，检索并合并相关文档
    def retrieve_and_merge(self, query, source_filter=None, strategy=None):  # 新增 strategy 参数
        # 如果未指定检索策略，则使用策略选择器选择
        if not strategy:
            strategy = self.strategy_selector.select_strategy(query)

        #   根据检索策略选择不同的检索方式
        ranked_sub_chunks = []  # 初始化
        if strategy == "回溯问题检索":
            ranked_sub_chunks = self._retrieve_with_backtracking(query)
        elif strategy == "子查询检索":
            ranked_sub_chunks = self._retrieve_with_subqueries(query)  # 返回的是唯一文档列表
            # 注意：子查询返回的是已 rerank 过的父文档或子块列表，后续合并逻辑可能需要调整
            # 当前实现中，子查询返回的是初步检索（可能已rerank）的块，再进行合并
        elif strategy == "假设问题检索":
            ranked_sub_chunks = self._retrieve_with_hyde(query)
        else:  # 默认或“直接检索”
            logger.info(f"使用直接检索策略 (查询: '{query}')")
            ranked_sub_chunks = self.vector_store.hybrid_search_with_rerank(
                query, k=conf.RETRIEVAL_K, source_filter=source_filter
            )  # 注意 hybrid_search_with_rerank 返回的是 rerank 后的父文档

        logger.info(f"策略 '{strategy}' 检索到 {len(ranked_sub_chunks)} 个候选文档 (可能已是父文档)")
        final_context_docs = ranked_sub_chunks[:conf.CANDIDATE_M]
        logger.info(f"最终选取 {len(final_context_docs)} 个文档作为上下文")
        return final_context_docs

    # #   定义方法，生成答案
    # def generate_answer(self, query, source_filter=None):
    #     #   记录查询开始时间
    #     start_time = time.time()
    #     logger.info(f"开始处理查询: '{query}', 学科过滤: {source_filter}")
    #
    #     #   判断查询类型
    #     query_category = self.query_classifier.predict_category(query)
    #     logger.info(f"查询分类结果：{query_category} (查询: '{query}')")
    #
    #     #   如果查询属于“通用知识”类别，则直接使用 LLM 回答
    #     if query_category == "通用知识":
    #         logger.info("查询为通用知识，直接调用 LLM")
    #         prompt_input = self.rag_prompt.format(
    #             context="", question=query, phone=conf.CUSTOMER_SERVICE_PHONE
    #         )  #   不使用上下文
    #         try:
    #             answer = self.llm(prompt_input)
    #         except Exception as e:
    #             logger.error(f"直接调用 LLM 失败: {e}")
    #             answer = f"抱歉，处理您的通用知识问题时出错。请联系人工客服：{conf.CUSTOMER_SERVICE_PHONE}"
    #         processing_time = time.time() - start_time
    #         logger.info(
    #             f"通用知识查询处理完成 (耗时: {processing_time:.2f}s, 查询: '{query}')"
    #         )
    #         return answer
    #
    #     #   否则，进行 RAG 检索并生成答案
    #     logger.info("查询为专业咨询，执行 RAG 流程")
    #     #   选择检索策略
    #     strategy = self.strategy_selector.select_strategy(query)
    #
    #     #   检索相关文档
    #     context_docs = self.retrieve_and_merge(
    #         query, source_filter=source_filter, strategy=strategy
    #     )  #   传递 strategy
    #
    #     #   准备上下文
    #     if context_docs:
    #         context = "\n\n".join([doc.page_content for doc in context_docs]) # 使用换行符分隔文档
    #         logger.info(f"构建上下文完成，包含 {len(context_docs)} 个文档块")
    #         # logger.debug(f"上下文内容:\n{context[:500]}...") # Debug 日志可以打印部分上下文
    #     else:
    #         context = ""
    #         logger.info("未检索到相关文档，上下文为空")
    #
    #     #   构造 Prompt，调用大语言模型生成答案
    #     prompt_input = self.rag_prompt.format(
    #         context=context, question=query, phone=conf.CUSTOMER_SERVICE_PHONE
    #     )
    #     # logger.debug(f"最终生成的 Prompt:\n{prompt_input}") # Debug 日志
    #
    #     try:
    #         answer = self.llm(prompt_input)
    #     except Exception as e:
    #         logger.error(f"调用 LLM 生成最终答案失败: {e}")
    #         answer = f"抱歉，处理您的专业咨询问题时出错。请联系人工客服：{conf.CUSTOMER_SERVICE_PHONE}"
    #
    #
    #     #   记录查询处理完成的日志
    #     processing_time = time.time() - start_time
    #     logger.info(f"查询处理完成 (耗时: {processing_time:.2f}s, 查询: '{query}')")
    #     return answer

    # 添加完历史消息后，定义方法，生成答案
    # 修改 rag_system.py 中的 generate_answer 方法
    def generate_answer(self, query, source_filter=None, history=None):
        """生成答案，支持对话历史，限制提示长度，并支持流式输出"""
        start_time = time.time()
        logger.info(f"开始处理查询: '{query}', 学科过滤: {source_filter}")

        # 验证历史格式
        if history is not None and not isinstance(history, list):
            logger.warning(f"无效的历史格式: {type(history)}，忽略历史")
            history = []
        elif history:
            history = history[-5:]  # 限制最多5轮
            for h in history:
                if not (isinstance(h, dict) and "question" in h and "answer" in h):
                    logger.warning(f"无效的历史条目: {h}，忽略历史")
                    history = []
                    break

        # 判断查询类型，通用知识还是专业知识
        query_category = self.query_classifier.predict_category(query)
        logger.info(f"查询分类结果: {query_category} (查询: '{query}')")

        # 准备上下文（保持列表形式，交给 _build_prompt_within_limit 按块裁剪）
        try:
            if query_category == "通用知识":
                logger.info("查询为通用知识，直接调用 LLM")
                context_docs = []
            else:
                logger.info("查询为专业咨询，执行 RAG 流程")
                strategy = self.strategy_selector.select_strategy(query)
                context_docs = self.retrieve_and_merge(
                    query, source_filter=source_filter, strategy=strategy
                )
                logger.info(f"构建上下文完成，包含 {len(context_docs)} 个文档块")

            # 智能裁剪：system 指令 + 问题完整保留，context 按块砍尾、history 砍旧留新
            prompt_input = self._build_prompt_within_limit(
                context_docs=context_docs,
                history_list=history or [],
                question=query,
                phone=conf.CUSTOMER_SERVICE_PHONE,
            )
            logger.info(
                f"最终 prompt 长度: {len(prompt_input)} / {self.max_prompt_length}，"
                f"context 块={len(context_docs)}，history 轮={len(history or [])}"
            )

            processing_time = time.time() - start_time
            logger.info(f"检索时间 (耗时: {processing_time:.2f}s, 查询: '{query}')")
            start_time = time.time()

            # 调用 LLM（流式输出）
            try:
                # 使用 generator 获取流式输出
                for token in self.llm(prompt_input):
                    yield token  # 直接将每个 token 传递给调用者

                # 记录处理时间
                processing_time = time.time() - start_time
                logger.info(f"LLM查询处理完成 (耗时: {processing_time:.2f}s, 查询: '{query}')")
            except Exception as e:
                logger.error(f"调用 LLM 失败: {e}")
                yield f"抱歉，处理您的问题时出错。请联系人工客服：{conf.CUSTOMER_SERVICE_PHONE}"

        except Exception as e:
            logger.error(f"答案生成失败: {e}")
            yield f"抱歉，处理您的问题时出错。请联系人工客服：{conf.CUSTOMER_SERVICE_PHONE}"

# query 意图识别 使用bert 分类
# 改进：
# 1. 用LLM做意图识别，考虑最近几轮历史问答
# 2. 最近几轮的query 拼接([CLS]current_query[SEP] history_query[SEP] ) 使用多轮对话数据训练Bert
