# abstractive_summarization_models

环境要求：

python3.6

torch 1.0.0+

------

basic seq2seq模型 是由【 https://github.com/ChenRocks/fast_abs_rl 】的模型代码修改而得，提供的就是最为基本的seq2seq+attenion模型
+ train.sh 用于training 模型
+ decode.sh 用于生成test的decode结果
+ eval部分参考fast_abs_rl的readme

-----

hierarchical seq2seq模型也是由【https://github.com/ChenRocks/fast_abs_rl】代码修改而成，提供word，sentence level的seq2seq模型
目前为乞丐版，beam search还没填上，attention还没加好，参数都是裸的，真乞丐版

----


未完待续：

copy_summ

