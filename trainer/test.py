import torch
from torch import nn
from tokenizers import models, ByteLevelBPETokenizer, pre_tokenizers
from transformers import AutoTokenizer

def test_tokenizer():
    tn = ByteLevelBPETokenizer()

    # print(tn.pre_tokenizer)
    # print(tn.pre_tokenizer.pre_tokenize_str("hello world"))
    # print(tn.pre_tokenizer.pre_tokenize_str("你好"))
    # print(pre_tokenizers.ByteLevel.alphabet()[32])

    text = "鉴别一组中文文章的风格和特点，例如官方、口语、文言等。需要提供样例文章才能准确鉴别不同的风格和特点。<|im_end|> 好的，现在帮我查一下今天的天气怎么样?今天的天气依据地区而异。请问你需要我帮你查询哪个地区的天气呢？<|im_end|> 打开闹钟功能，定一个明天早上七点的闹钟。好的，我已经帮您打开闹钟功能，闹钟将在明天早上七点准时响起。<|im_end|> 为以下场景写一句话描述：一个孤独的老人坐在公园长椅上看着远处。一位孤独的老人坐在公园长椅上凝视远方。<|im_end|> 非常感谢你的回答。请告诉我，这些数据是关于什么主题的？这些数据是关于不同年龄段的男女人口比例分布的。<|im_end|> 帮我想一个有趣的标题。这个挺有趣的：\"如何成为一名成功的魔术师\" 调皮的标题往往会吸引读者的注意力。<|im_end|> 回答一个问题，地球的半径是多少？地球的平均半径约为6371公里，这是地球自赤道到两极的距离的平均值。<|im_end|> 识别文本中的语气，并将其分类为喜悦、悲伤、惊异等。\n文本：“今天是我的生日！”这个文本的语气是喜悦。<|im_end|>"
    # text = "你好<|im_end|>世界"

    print(tn.pre_tokenizer.pre_tokenize_str(text))

    trained_tn = AutoTokenizer.from_pretrained("../model_learn_tokenizer")
    model_inputs = trained_tn(text)
    # print(model_inputs)

    token_buffer = []
    for tid in model_inputs['input_ids']:
        token_buffer.append(tid)
        current_decode = trained_tn.decode(token_buffer)
        if current_decode and '\ufffd' not in current_decode:
            ids = token_buffer[0] if len(token_buffer) == 1 else token_buffer
            raw_tokens = [trained_tn.convert_ids_to_tokens(int(t)) for t in token_buffer]
            print(f'Token ID: {str(ids):15} -> Raw: {str(raw_tokens):20} -> Decode Str: {current_decode}')

            token_buffer = []

def test_nn_linear():
    # Create a Linear layer: input size 4, output size 2
    linear = nn.Linear(in_features=4, out_features=2)
    print("Weight: ", linear.weight) # out_features x in_features
    print("Bias: ", linear.bias) # 1 x out_features

    x = torch.randn(10, 100, 4)

    # Pass input through the linear layer
    output = linear(x)

    print("Input shape:", x.shape)
    print("Output shape:", output.shape)
    print("Output shape:", output.shape)
    # print("Output:", output)

if __name__ == '__main__':
    pass
    # test_tokenizer()
    test_nn_linear()
