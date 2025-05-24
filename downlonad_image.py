import requests
import datetime


def download_image(image_id, header):
    res_image = requests.get(f"https://api-data.line.me/v2/bot/message/{image_id}/content",
                             headers=header,
                             stream=False)
    if res_image.status_code == 200:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"image_{timestamp}.jpg"
        with open(filename, "wb") as f:
            f.write(res_image.content)
        print(f"Image saved as {filename}")
        return filename
    else:
        print(f"Failed to download image: {res_image.status_code}")
