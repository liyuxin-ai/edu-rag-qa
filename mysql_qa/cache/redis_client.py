# 导入 Redis 客户端
import redis
# 导入 JSON 处理
import json
# 导入配置和日志
from base import Config, logger


# 答案缓存 TTL：7 天（秒），教学项目 FAQ 数据不常变，7 天后自动失效
ANSWER_CACHE_TTL_SECONDS = 60 * 60 * 24 * 7


class RedisClient:
    def __init__(self):
        # 初始化日志
        self.logger = logger
        try:
            # 连接 Redis
            self.client = redis.StrictRedis(
                host=Config().REDIS_HOST,
                port=Config().REDIS_PORT,
                password=Config().REDIS_PASSWORD,
                db=Config().REDIS_DB,
                decode_responses=True
            )
            # 记录连接成功
            self.logger.info("Redis 连接成功")
        except redis.RedisError as e:
            # 记录连接失败
            self.logger.error(f"Redis 连接失败: {e}")
            raise

    # ---- BM25 语料缓存（列表 / 字典，需要 JSON 序列化） ----

    def set_data(self, key, value, ttl=None):
        """缓存结构化数据（BM25 语料等），JSON 序列化"""
        try:
            payload = json.dumps(value, ensure_ascii=False)
            if ttl:
                self.client.setex(key, ttl, payload)
            else:
                self.client.set(key, payload)  # BM25 语料永久缓存，启动时重建
            self.logger.info(f"存储数据到 Redis: {key}")
        except redis.RedisError as e:
            self.logger.error(f"Redis 存储失败: {e}")

    def get_data(self, key):
        """获取结构化缓存，JSON 反序列化"""
        try:
            data = self.client.get(key)
            return json.loads(data) if data else None
        except redis.RedisError as e:
            self.logger.error(f"Redis 获取失败: {e}")
            return None

    # ---- FAQ 答案缓存（纯字符串，直接存取 + TTL） ----

    def set_answer(self, query, answer):
        """缓存 FAQ 命中的标准答案，原子写入 + TTL"""
        try:
            key = f"answer:{query.strip()}"
            self.client.setex(key, ANSWER_CACHE_TTL_SECONDS, answer)
            self.logger.info(f"缓存答案: {key} (TTL={ANSWER_CACHE_TTL_SECONDS}s)")
        except redis.RedisError as e:
            self.logger.error(f"Redis 缓存答案失败: {e}")

    def get_answer(self, query):
        """获取 FAQ 缓存答案（纯字符串，与 set_answer 对称）"""
        try:
            key = f"answer:{query.strip()}"
            answer = self.client.get(key)
            if answer is not None:
                self.logger.info(f"从 Redis 获取答案: {query}")
                return answer
            return None
        except redis.RedisError as e:
            self.logger.error(f"Redis 查询失败: {e}")
            return None


import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    # 初始化 Redis 客户端
    redis_client = RedisClient()
    # 示例数据
    key = "user:1"
    value = {"name": "Alice", "age": 25}
    # 存储数据
    redis_client.set_data(key, value)
    # 获取数据
    result = redis_client.get_data(key)
    if result:
        logger.info(f"查询结果: {result}")
    else:
        logger.info("未找到数据")
    # 示例查询缓存
    query = "test_query"
    answer = redis_client.get_answer(query)
    if answer:
        logger.info(f"缓存答案: {answer}")
    else:
        logger.info("未找到缓存答案")


if __name__ == "__main__":
    main()

# if __name__ == '__main__':
#     redcli = RedisClient()
#     print(redcli)
