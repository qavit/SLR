import random
from transformers import pipeline

pipe = pipeline("image-classification", model="RavenOnur/Sign-Language")

URL = 'https://huggingface.co/RavenOnur/Sign-Language/resolve/main/images'

alphabets = [chr(i) for i in range(65, 90)] 
print(f'{alphabets = }')

random_samples = random.sample(alphabets, 10)
print(f'{random_samples = }')

random_pics_pool = [f"{URL}/{a}.jpg" for a in random_samples]

random_pics_pool2 = {
    'C': 'https://datasets-server.huggingface.co/cached-assets/aliciiavs/sign_language_image_dataset/--/8ba2bb7833c82f072251f47f1c315148503e4990/--/default/train/1201/image/image.jpg?Expires=1722875592&Signature=tQfKm2H5FWNtaQyTlX-vNIzNMAIDT4VF49Zarheb6G6kpcJXhbT24axKfBeZXDhNZoOncY0~-ar5yB8mzgzcRVM0IVza1upaTsAWUnjhj03w7rZTX2vY18iBZE5ToqZ2WZg2TngujtDgA5JfJrfYrzALEqeAYHqugMG5hobJzqYiYFuYpepC89c1FTptqKQ8F6WzJc1w5L~NHFeriPRxMcUIAimIWkD3D1FrnujABUSB4VEvRPSM-gVkjTKvMAcpgtyWek7OYoaHCUWKedqUucY2n8-VKigKDGWpYru1wnIkkyMubybN-87oZeXIWDN1qwBChzb~70n1VE6LiQoSLQ__&Key-Pair-Id=K3EI6M078Z3AC3'
}

# for pic in random_pics_pool:
#     result = pipe(pic)
#     print(pic[-5:] + ' ----> ' + result[0]['label'])

for key, val in random_pics_pool2.items():
    result = pipe(val)
    print(key + ' ----> ' + result[0]['label'])