from ray.serve.drivers import DAGDriver
from ray import serve
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
import numpy as np

@serve.deployment()
class ImageClassifier:
    def __init__(self):
        self.model = MobileNetV2(weights='imagenet')
    # Users can send HTTP requests with an image. The classifier will return
    # the top 1 prediction.
    # Sample output: {"prediction":["n02099601","golden_retriever",0.17944198846817017]}
    async def __call__(self, http_request):
        request = await http_request.form()
        image_file = await request["image"].read()
        import tempfile

        # Create a temporary file in binary write mode
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
            temp_file.write(image_file)
            temp_file.close()
            temp_file_path = temp_file.name
            img = image.load_img(temp_file_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        preds = self.model.predict(x)
        decoded_preds = decode_predictions(preds, top=1)[0]
        return {"prediction": decoded_preds[0]}

image_classifier_handle = ImageClassifier.bind()
graph = DAGDriver.bind(image_classifier_handle)
