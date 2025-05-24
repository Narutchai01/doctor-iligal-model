import numpy as np


def GLT(image, method='log', coeff=5.0):
    # Convert to float for processing
    img_float = image.astype(np.float32) / 255.0

    if method == 'log':
        # Log transform: s = c * log(1 + r)
        img_transformed = coeff * np.log1p(img_float)
        # Normalize back to 0-255
        img_transformed = (img_transformed /
                           np.max(img_transformed) * 255).astype(np.uint8)
    else:
        # Default (no transformation)
        img_transformed = (img_float * 255).astype(np.uint8)

    return img_transformed
