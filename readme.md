## 说明

主文件在 src/main.py。也有 ipynb 版本的，debug 的时候用更方便
modelLoaders 里面是模型加载器
promptGenerators 里面是 prompt 模板，目前只有一个常规对话模板
settings/character 是人物设定 yaml 文件
settings/featured_chates 是人物特色聊天记录的 txt 版本的文件，实际使用的时候是调用的已经转换成向量库的版本
vertorDB 下面就是保存的各人物实际使用的向量库
src/utils 里面是将 txt 文件转换成向量库的工具

## 安装依赖项的注意事项

bitsandbytes-windows 使用的不是`pip install bitsandbytes`，而是`pip install git+https://github.com/Keith-Hon/bitsandbytes-windows.git`

Pytorch 的安装使用的是`pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117`，其它的版本是否有用并没有测试过。

其它的依赖库见 requirements.txt

requirements-webui.txt 是从 webui 的环境库里提取出来的，当时安装这个项目的依赖时用来参考的。

## 批量调用

原生 Llama 使用批量调用的示范，见`https://github1s.com/facebookresearch/llama/blob/main/example_chat_completion.py`
langchain 也支持批量处理请求，见`https://python.langchain.com/docs/modules/model_io/models/llms/`和`https://python.langchain.com/docs/modules/model_io/models/chat/`,搜索`batch calls`

批量调用的好处是一次性处理的内容多，平均处理时间比较少。
但坏处是，对内存有要求。而且这一批信息是同进同出，单个信息的处理时间其实是长的。
另外，信息出来的顺序不是按照输入的顺序，而是按照处理完成的顺序。所以除非出来的信息带有自识别的标志，不然无法知道哪个信息对应哪个输入。
