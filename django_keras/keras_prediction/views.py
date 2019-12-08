# import packages for framework part
from django.shortcuts import render
from rest_framework.views import APIView
from django.conf import settings
from django.http import JsonResponse
import base64
# import the necessary packages for prediction
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications import imagenet_utils
from PIL import Image
import numpy as np
import io


class Predict(APIView):

    def post(self, request, filename='lol', format=None):
        data = {"success": False}
        model = None

        def prepare_image(image, target):
            # if the image mode is not RGB, convert it
            if image.mode != "RGB":
                image = image.convert("RGB")

            # resize the input image and preprocess it
            image = image.resize(target)
            image = img_to_array(image)
            image = np.expand_dims(image, axis=0)
            image = imagenet_utils.preprocess_input(image) # ResNet
            # return the processed image
            return image

        # ensure an image was properly uploaded to our endpoint
        if request.FILES.get("image"):
            # read the image in PIL format
            image = request.FILES["image"].read()
            image = Image.open(io.BytesIO(image))
            # preprocess the image and prepare it for classification
            image = prepare_image(image, target=(224, 224)) # for ResNet
            # Use already loaded model from settings
            model = settings.MODEL
            preds = model.predict(image)
            # After prediction
            results = imagenet_utils.decode_predictions(preds)
            data["predictions"] = []
            # loop over the results and add them to the list of
            # returned predictions
            for (imagenetID, label, prob) in results[0]:
                r = {"label": label, "probability": float(prob)}
                data["predictions"].append(r)

            # indicate that the request was a success
            data["success"] = True

        return JsonResponse(data)