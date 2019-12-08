# import the necessary packages
# for predictions
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications import imagenet_utils
from PIL import Image
import numpy as np
import io
# for web framework
# import asyncio
from aiohttp import web, MultipartReader, hdrs
# import aiohttp

model = None

def load_model():
    # load the pre-trained Keras model (here we are using a model
    # pre-trained on ImageNet and provided by Keras, but you can
    # substitute in your own networks just as easily)
    global model
    model = ResNet50(weights="imagenet")
    return model

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

routes = web.RouteTableDef()

@routes.post("/predict/")
async def post_request(request):
    if request.method == 'POST':
        # initialize the data dictionary that will be returned from the
        # view

        data = {"success": False}

        reader = MultipartReader.from_response(request)
        while True:
            part = await reader.next()
            if part is None:
                break
            if part.headers[hdrs.CONTENT_TYPE] == 'application/json':
                metadata = await part.json()
                continue
            filedata = await part.read(decode=False)
            # ensure an image was properly uploaded to our endpoint
            # read the image in PIL format
            image = Image.open(io.BytesIO(filedata))

            # preprocess the image and prepare it for classification
            image = prepare_image(image, target=(224, 224))

            # model = load_model()

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

    return web.json_response(data)

if __name__ == '__main__':
    print("Load model")
    model = load_model()
    app = web.Application()
    app.add_routes(routes)
    print("Aiohttp is up!")
    web.run_app(app, host="localhost", port=8000)