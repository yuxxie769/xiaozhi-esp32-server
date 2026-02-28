-- 扩展 OpenAI(兼容) LLM 供应器字段：支持 OpenRouter extra_body/default_headers/stream 等扩展参数
UPDATE `ai_model_provider`
SET `fields` = '[{"key":"base_url","label":"基础URL","type":"string"},{"key":"model_name","label":"模型名称","type":"string"},{"key":"api_key","label":"API密钥","type":"string"},{"key":"temperature","label":"温度","type":"number"},{"key":"max_tokens","label":"最大令牌数","type":"number"},{"key":"top_p","label":"top_p值","type":"number"},{"key":"top_k","label":"top_k值","type":"number"},{"key":"frequency_penalty","label":"频率惩罚","type":"number"},{"key":"extra_body","label":"额外参数(extra_body)","type":"dict","dict_name":"extra_body"},{"key":"default_headers","label":"默认请求头","type":"dict","dict_name":"default_headers"},{"key":"stream","label":"是否流式","type":"boolean"},{"key":"allow_message_extras","label":"允许消息扩展字段","type":"boolean"}]'
WHERE `id` = 'SYSTEM_LLM_openai';

