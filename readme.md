# debug
bitsandbytes-windows使用的不是`pip install bitsandbytes`，而是`pip install git+https://github.com/Keith-Hon/bitsandbytes-windows.git`

Pytorch的安装使用的是`pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117`，其它的版本是否有用并没有测试过。

其它的依赖库见requirements.txt

requirements-webui.txt是从webui的环境库里提取出来的，当时安装这个项目的依赖时用来参考的。

# 批量调用
原生Llama使用批量调用的示范，见`https://github1s.com/facebookresearch/llama/blob/main/example_chat_completion.py`
langchain也支持批量处理请求，见`https://python.langchain.com/docs/modules/model_io/models/llms/`和`https://python.langchain.com/docs/modules/model_io/models/chat/`,搜索`batch calls`
