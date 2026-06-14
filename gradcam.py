import numpy as np
import tensorflow as tf
import cv2


def get_gradcam_heatmap(model, img_array):

    # Run model once to build it
    predictions = model(img_array)

    # Access MobileNetV2 base model
    base_model = model.layers[0]

    # Create model for Grad-CAM
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[
            base_model.get_layer("out_relu").output,
            model.outputs[0],
        ],
    )

    # Gradient computation
    with tf.GradientTape() as tape:

        conv_outputs, predictions = grad_model(img_array)

        top_class_idx = tf.argmax(predictions[0])

        top_class_score = predictions[:, top_class_idx]

    # Compute gradients
    grads = tape.gradient(top_class_score, conv_outputs)

    # Average gradients spatially
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Remove batch dimension
    conv_outputs = conv_outputs[0]

    # Weight channels
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]

    heatmap = tf.squeeze(heatmap)

    # Apply ReLU
    heatmap = tf.maximum(heatmap, 0)

    # Normalize
    max_val = tf.reduce_max(heatmap)

    if max_val > 0:
        heatmap /= max_val

    return heatmap.numpy(), predictions.numpy()[0]


def overlay_heatmap_on_image(original_img_rgb, heatmap, alpha=0.45):

    h, w = original_img_rgb.shape[:2]

    # Resize heatmap
    heatmap_resized = cv2.resize(heatmap, (w, h))

    # Convert heatmap to color
    heatmap_uint8 = np.uint8(255 * heatmap_resized)

    heatmap_colored = cv2.applyColorMap(
        heatmap_uint8,
        cv2.COLORMAP_JET
    )

    heatmap_colored = cv2.cvtColor(
        heatmap_colored,
        cv2.COLOR_BGR2RGB
    )

    # Overlay
    overlay = cv2.addWeighted(
        original_img_rgb,
        1 - alpha,
        heatmap_colored,
        alpha,
        0
    )

    return overlay, heatmap_colored


def get_severity_from_heatmap(heatmap):

    threshold = 0.5

    affected = np.mean(heatmap > threshold) * 100

    if affected < 5:
        return "Minimal / Healthy", affected

    elif affected < 20:
        return "Mild", affected

    elif affected < 45:
        return "Moderate", affected

    else:
        return "Severe", affected
