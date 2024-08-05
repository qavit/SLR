import os
import random

# 使用 pipeline 作為高級輔助函式
from transformers import pipeline

# 創建一個 image-classification pipeline 並指定模型
pipe = pipeline("image-classification", model="RavenOnur/Sign-Language")

URL = 'https://huggingface.co/RavenOnur/Sign-Language/resolve/main/images'

alphabets = [ f"{URL}/{chr(i)}.jpg" for i in range(65, 90)] 
# print(alphabets)
random_sample = random.sample(alphabets, 10)
print(random_sample)

for pic in random_sample:
    result = pipe(pic)
    print(pic, '---->', result[0]['label'])
