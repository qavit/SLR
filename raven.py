import random
from transformers import pipeline

pipe = pipeline("image-classification", model="RavenOnur/Sign-Language")

URL = 'https://huggingface.co/RavenOnur/Sign-Language/resolve/main/images'

alphabets = [chr(i) for i in range(65, 90)] 
print(f'{alphabets = }')

random_samples = random.sample(alphabets, 10)
print(f'{random_samples = }')

random_pics = [f"{URL}/{a}.jpg" for a in random_samples]

for pic in random_pics:
    result = pipe(pic)
    print(pic[-5:] + ' ----> ' + result[0]['label'])
