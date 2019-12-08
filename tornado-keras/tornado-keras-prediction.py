# import the necessary packages
# for predictions
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications import imagenet_utils
from PIL import Image
import numpy as np
# for web framework
from tornado.web import Application, RequestHandler
from tornado.ioloop import IOLoop
import json
import io

items = []

model = None

def load_model():
    # load the pre-trained Keras model (here we are using a model
    # pre-trained on ImageNet and provided by Keras, but you can
    # substitute in your own networks just as easily)
    global model
    model = ResNet50(weights="imagenet")

def prepare_image(image, target):
    # if the image mode is not RGB, convert it
    if image.mode != "RGB":
        image = image.convert("RGB")

    # resize the input image and preprocess it
    image = image.resize(target)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = imagenet_utils.preprocess_input(image)

    # return the processed image
    return image


class TodoItem(RequestHandler):
    def post(self):
        # items.append(json.loads(self.request.body))
        # self.write({'message': 'new item added'})
        # initialize the data dictionary that will be returned from the
        # view
        data = {"success": False}

        # ensure an image was properly uploaded to our endpoint
        if self.request.method == "POST":
            if self.request.files.get("image"):
                # read the image in PIL format
                image = self.request.files["image"][0].body # without .read(), again [0] as first image
                image = Image.open(io.BytesIO(image))

                # preprocess the image and prepare it for classification
                image = prepare_image(image, target=(224, 224))

                # classify the input image and then initialize the list
                # of predictions to return to the client
                preds = model.predict(image)
                results = imagenet_utils.decode_predictions(preds)
                data["predictions"] = []
                # loop over the results and add them to the list of
                # returned predictions
                for (imagenetID, label, prob) in results[0]:
                    r = {"label": label, "probability": float(prob)}
                    data["predictions"].append(r)

                # indicate that the request was a success
                data["success"] = True
            self.write(data)


def make_app():
    urls = [
        ("/predict/", TodoItem),
    ]
    return Application(urls, debug=True)


if __name__ == '__main__':
    app = make_app()
    app.listen(8000)
    print("Load the model")
    load_model()
    print("Tornado server is app and listen on port 8000")
    IOLoop.instance().start()