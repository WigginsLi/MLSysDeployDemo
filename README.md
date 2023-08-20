### 高级计算机系统结构大作业
## Demo: [http://47.120.38.50:10012/](http://47.120.38.50:10012/)
# 深度学习模型部署(使用GPU推理)
**Plan**
- [x] 选定模型 
  - 参考框架 [so-vits-svc](https://github.com/svc-develop-team/so-vits-svc/tree/4.1-Stable)
  - ~~使用预训练模型 [Trump](https://huggingface.co/Nardicality/so-vits-svc-4.0-models/tree/main/Trump18.5k)~~
  - 使用预训练模型 [Chtholly](https://huggingface.co/overload7015/So-Vits-SukaSuka-Chtholly)
- [x] 推理测试
- [x] -> ~~onnx~~ -> ~~tensorRT~~
  - 无法联合Kmeans及Diffusion，效果不理想
- [x] 后端部署
  - ~~GPU服务器~~ 太贵
  - 内网穿透(frp)
- [x] 前端UI
  - Gradio

# 设备参数
OS： Ubuntu 22.04.2 LTS 64bit

GPU： GeForce GTX 1050 Mobile

CPU： Intel® Core™ i5-8250U CPU @ 1.60GHz × 8

内存： 8G

硬盘： 512G


# 选择模型
- 选择框架[so-vits-svc](https://github.com/svc-develop-team/so-vits-svc/tree/4.1-Stable)
> 歌声音色转换模型，通过 SoftVC 内容编码器提取源音频语音特征，与 F0 同时输入 VITS 替换原本的文本输入达到歌声转换的效果。同时，更换声码器为 NSF HiFiGAN 解决断音问题。
- 简单来说，收集某个角色的音声数据进行训练得到模型，即可使用ta的声线唱出指定歌曲。

- 模型选择（显卡限制无法训练，选择现成模型）
  - ~~[Trump](https://huggingface.co/Nardicality/so-vits-svc-4.0-models/tree/main/Trump18.5k)~~ 效果不好
  - [Chtholly](https://huggingface.co/overload7015/So-Vits-SukaSuka-Chtholly)

# 推理测试
推理参数 （音频文件40s）
  - GPU ~8.8s
```bash
python3 inference_main.py -m "logs/44k/Chtholly_V5Co-1076epoch-80800step-Vec768-Layer12_compressed.pth" -c "configs/Chtholly_V5_config.json" -n "slice.wav" -t 0 -s "Chtholly_V5" -cm "logs/44k/Chtholly_V5_kmeans_10000.pt" -cr 0.1 -f0p "crepe" -wf "wav" -dm "logs/44k/diffusion/Chtholly_V5_model_52000.pt" -dc "configs/Chtholly_V5_config.yaml" -shd
```
  - ![GPU推理结果](/imgs/inference_gpu_result.png)
  - CPU ~77.7s
```bash
python3 inference_main.py -m "logs/44k/Chtholly_V5Co-1076epoch-80800step-Vec768-Layer12_compressed.pth" -c "configs/Chtholly_V5_config.json" -n "slice.wav" -t 0 -s "Chtholly_V5" -cm "logs/44k/Chtholly_V5_kmeans_10000.pt" -cr 0.1 -f0p "crepe" -wf "wav" -dm "logs/44k/diffusion/Chtholly_V5_model_52000.pt" -dc "configs/Chtholly_V5_config.yaml" -shd -d "cpu"
```
  - ![CPU推理结果](/imgs/inference_cpu_result.png)
  
# 转换onnx模型
- 导出Onnx模型
  - 使用`so-vits-svc`中提供的`onnx_export.py`进行导出[参考链接](https://github.com/svc-develop-team/so-vits-svc/blob/4.1-Stable/README_zh_CN.md#-onnx-%E5%AF%BC%E5%87%BA)
- ~~使用MoeSS项目提供的API~~ 
  - 使用了Win32API，且需由VS编译构建，不适合移植ubuntu
- 重写`so-vits-svc`中`inference/infer_tool.py`中SVC类，使其支持Onnx模型导入及推理，大致代码如下：
```python
class OnnxSvc(object):
    def __init__(self, onnx_model_path, config_path,
                 device=None,
                 cluster_model_path="logs/44k/kmeans_10000.pt"):
        # 导入Onnx模型
        self.onnx_session = onnxruntime.InferenceSession(onnx_model_path)
        if device is None:
            self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.dev = torch.device(device)
        self.hps_ms = utils.get_hparams_from_file(config_path)
        self.target_sample = self.hps_ms.data.sampling_rate
        self.hop_size = self.hps_ms.data.hop_length
        self.spk2id = self.hps_ms.spk
        # 加载hubert
        self.hubert_model = utils.get_hubert_model().to(self.dev)
        if os.path.exists(cluster_model_path):
            self.cluster_model = cluster.get_cluster_model(cluster_model_path)

    def get_unit_f0(self, in_path, tran, cluster_infer_ratio, speaker):
      # 无需改动，省略

    def infer(self, speaker, tran, raw_path,
          cluster_infer_ratio=0,
          auto_predict_f0=False,
          noice_scale=0.4):
        speaker_id = self.spk2id[speaker]
        sid = np.array([[int(speaker_id)]], dtype=np.int64)
        c, f0, uv = self.get_unit_f0(raw_path, tran, cluster_infer_ratio, speaker)

        c = c.numpy()  
        f0 = f0.numpy()  
        uv = uv.numpy() 

        inputs = {
            'input_c': c,
            'input_f0': f0,
            'input_g': sid,
            'input_uv': uv,
            'input_predict_f0': np.array([auto_predict_f0], dtype=np.bool),
            'input_noice_scale': np.array([noice_scale], dtype=np.float32)
        }

        start = time.time()
        # 使用onnx模型进行推理
        outputs = self.onnx_session.run(None, inputs)
        use_time = time.time() - start
        print("ONNX inference use time:{}".format(use_time))

        audio = outputs[0][0]  
        return audio, audio.shape[-1]

    def slice_inference(self,raw_audio_path, spk, tran, slice_db,cluster_infer_ratio, auto_predict_f0,noice_scale, pad_seconds=0.5):
        # 无需改动，省略
```
- 导出Kmeans和Diffusion的Onnx模型时出错
  - 有不少 onnx 不支持的算子和涉及到常量的东西，在导出时会报错或者导出的模型输入输出 shape 和结果都有问题
  - 只使用裸模型效果不好

# 后端部署
- ~~CPU推理过慢~~
- ~~GPU服务器按小时计费~~
- 内网穿透
  - 使用frp，阿里云服务器作为跳板提供公网IP

# 前端设计
使用Gradio开源库构建前端
- [x] 上传音频
  - [x] Example: 1. 歌曲切片（40s）； 2. TTS语音（来源https://ttsmaker.com/zh-cn ， 高级设置选择导出WAV）
- [ ] 推理选项
- [x] 命令行推理结果
- [x] 音频试听