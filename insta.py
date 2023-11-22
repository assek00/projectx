from typing import List, Any

import numpy as np
import requests
import cv2
from collections import Counter

access_token = 'IGQWROa0xPN3d4cUpEb2dfaDBqTEZAFQ0EyeThhSDlJMTUtaDFHem51Y181RURzMFRHUGcyRlV4STlyR09NdEw0TVAyOWk2bnRHTEI3T1o2LVRvNmhEVThhT0pyTC12VWJyRHRxbFZAqT0dXUVY2c2RrZAzVEMG1mMFUZD'
def get_user_posts(user_id):
    try:
        url = f'https://graph.instagram.com/v12.0/{user_id}/media?fields=id,caption&access_token={access_token}'
        response = requests.get(url)
        response.raise_for_status()  # Это вызовет исключение для неудачных запросов.
        data = response.json()
        return data['data']
    except requests.exceptions.RequestException as e:
        print(f"Ошибка запроса: {e}")
        return []

def extract_hashtags(posts):
    hashtags = []
    for post in posts:
        caption = post.get('caption', '')
        hashtags.extend(tag.strip('#') for tag in caption.split() if tag.startswith('#'))
    return hashtags

def get_latest_post_url(user_id):
    posts = get_user_posts(user_id)
    if not posts:
        return None
    latest_post_id = posts[0]['id']
    media_url = f'https://graph.instagram.com/{latest_post_id}?fields=media_url&access_token={access_token}'
    try:
        response = requests.get(media_url)
        response.raise_for_status()
        data = response.json()
        return data['media_url']
    except requests.exceptions.RequestException as e:
        print(f"Ошибка запроса: {e}")
        return None

def load_image(url):
    resp = requests.get(url)
    image_array = np.asarray(bytearray(resp.content), dtype="uint8")
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    return image

    file = open(r'C:/Users/turlybekmussabayev/Desktop/ProjectX/yolo/yolov3.weights')
    file = open(r'/Users/turlybekmussabayev/Desktop/ProjectX/darknet-master/cfg/yolov3.cfg')
    net = cv2.dnn.readNet(yolov3.weights, yolov3.cfg)
    layer_names = net.getUnconnectedOutLayersNames()

def detect_objects(image, net=None):
    height, width, _ = image.shape
    blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(layer_names)
    confidences = []
    class_ids = []
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x, center_y, w, h = (detection[0:4] * np.array([width, height, width, height])).astype(int)
                x, y = center_x - w // 2, center_y - h // 2
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    return boxes, confidences, class_ids
user_id = '%3A"17841403914817220"%2C"nonce'
latest_post_url = get_latest_post_url(user_id)
if latest_post_url:
    image = load_image(latest_post_url)
    boxes, confidences, class_ids = detect_objects(image)

    for i in range(len(boxes)):
        x, y, w, h = boxes[i]
        label = str(class_ids[i])
        confidence = confidences[i]
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, f"{label}: {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("YOLO Output", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

posts = get_user_posts(user_id)
hashtags = extract_hashtags(posts)
top5hashtags = Counter(hashtags).most_common(5)

print("Топ хэштегов:")
for hashtag, count in top5hashtags:
    print(f"{hashtag}: {count} раз")
