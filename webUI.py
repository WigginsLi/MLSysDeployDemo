import os   

import gradio as gr
import numpy as np

import wave
import subprocess

def main_note(audio):
    sr, data = audio
    with wave.open("raw/temp.wav", "w") as f:
        num_channels = data.shape[1] if len(data.shape) > 1 else 1

        print(num_channels)
        f.setnchannels(num_channels)
        f.setsampwidth(2)
        f.setframerate(sr)
        f.writeframes(data.tobytes())
        print(sr, data)

    
    command = 'python3 inference_main.py -m "logs/44k/Chtholly_V5Co-1076epoch-80800step-Vec768-Layer12_compressed.pth" -c "configs/Chtholly_V5_config.json"\
                -n "temp.wav" -t 0 -s "Chtholly_V5"\
                -cm "logs/44k/Chtholly_V5_kmeans_10000.pt" -cr 0.1 -f0p "crepe" -wf "wav" \
                -dm "logs/44k/diffusion/Chtholly_V5_model_52000.pt" -dc "configs/Chtholly_V5_config.yaml" -shd'

    # 使用 subprocess 调用系统命令
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # 等待命令执行完成
    stdout, stderr = process.communicate()


    with wave.open("results/temp.wav", "rb") as f:
        params = f.getparams()
        print(params)
        sr = f.getframerate()
        nframes = f.getnframes()
        data = f.readframes(nframes)
        data = np.frombuffer(data, dtype=np.int16)
        # data.shape = -1, 2

    audio = (sr, data)
    text = stdout.decode("utf-8")+"\n"+stderr.decode("utf-8")+"\n"+str(process.returncode)
    print("finish")
    return audio, text
    
md_desc = "可使用歌曲切片（需小于40s）,亦可选择TTS语音（来源https://ttsmaker.com/zh-cn ， 高级设置选择导出WAV）"

demo = gr.Interface(
    main_note,
    gr.Audio(source="upload", interactive=True),
    outputs=["audio", "text"],
    examples=[
        [os.path.join(os.path.dirname(__file__),"raw/陈奕迅切片.wav")],
        [os.path.join(os.path.dirname(__file__),"raw/tts_test.wav")],
        [os.path.join(os.path.dirname(__file__),"raw/静夜思（男声效果不好）.wav")],
        [os.path.join(os.path.dirname(__file__),"raw/静夜思（女声）.wav")],
    ],
    description=md_desc # "上传或者使用example的wav文件，点击submit开始推理，右侧得到音色转换后的结果",
)

demo.queue(max_size=20)

if __name__ == "__main__":
    demo.launch()
