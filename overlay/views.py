from django.http import HttpResponse
from django.shortcuts import render
from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.views.decorators.csrf import csrf_exempt
import cv2
import numpy as np


#Load the jewellery image
jewellery_img = cv2.imread('jewellery8.png', -1)


# Initialize the face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


# Create your views here.
@api_view(['POST'])
@csrf_exempt

def overlay_jewellery(request):
    
    frame_bytes = request.data.get('frame')
    nparr = np.frombuffer(frame_bytes.read(), np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Detect the face in the frame
    faces = face_cascade.detectMultiScale(frame, scaleFactor=1.3, minNeighbors=5)

    # Loop through each face and overlay the jewellery
    for (x, y, w, h) in faces:
        # Calculate the size of the jewellery
        jewellery_height = int(h) - 50
        jewellery_width = int(jewellery_height * jewellery_img.shape[1] / jewellery_img.shape[0]) + 30

        # Resize the jewellery to fit the neck region
        resized_jewellery = cv2.resize(jewellery_img, (jewellery_width, jewellery_height))

        # Calculate the position of the jewellery
        cx = int(x + w/2) + 5  # Center x-coordinate of the face
        cy = int(y + h) + 70  # Center y-coordinate of the neck
        jewellery_x = cx - int(jewellery_width/2)  # Leftmost x-coordinate of the jewellery
        jewellery_y = cy - int(jewellery_height/2)  # Topmost y-coordinate of the jewellery

        # Overlay the jewellery on the neck region
        for i in range(jewellery_height):
            for j in range(jewellery_width):
                if resized_jewellery[i,j][3] != 0:  # Check alpha channel
                    frame[jewellery_y+i, jewellery_x+j] = resized_jewellery[i,j][:3]

    retval, buffer = cv2.imencode('.jpg', frame)
    response = Response(buffer.tobytes(), content_type='image/jpeg')
    return response
